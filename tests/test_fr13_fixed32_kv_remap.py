from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


KERNEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "fr10_gdn_tree_kernel.py"
)
KERNEL_FUNCTIONS = {
    "_fr13_fixed32_device_assert",
    "_validate_fixed32_kv16_contract",
    "_launch_attn_kv_linear_remap_syncfree_fixed16_impl",
}


def _load_kv16_impl():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in KERNEL_FUNCTIONS
    ]
    assert {node.name for node in definitions} == KERNEL_FUNCTIONS
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    return namespace["_launch_attn_kv_linear_remap_syncfree_fixed16_impl"]


def _mixed_inputs():
    # Full rows are: prefill(7), spec(32), prefill(5), spec(32).
    query_start_loc = torch.tensor([0, 7, 39, 44, 76], dtype=torch.int32)
    slot_mapping = torch.arange(76, dtype=torch.int64)
    base_cache = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128, 1)
    kv_caches = tuple(base_cache.clone() for _ in range(16))
    accepted_paths = torch.zeros((2, 16), dtype=torch.int64)
    accepted_paths[0, 0] = 3
    accepted_paths[1, 0] = 4
    accepted_lens = torch.ones(2, dtype=torch.int64)
    return {
        "kv_caches": kv_caches,
        "slot_mapping": slot_mapping,
        "query_start_loc": query_start_loc,
        "accepted_paths": accepted_paths,
        "num_accepted_tokens": accepted_lens,
        "num_spec_decodes": 2,
    }


def test_fixed32_kv16_mixed_rows_use_full_batch_qsl_indices() -> None:
    launch = _load_kv16_impl()
    inputs = _mixed_inputs()
    before = inputs["kv_caches"][0].clone()

    launch(
        **inputs,
        batch_indices=torch.tensor([1, 3], dtype=torch.int64),
    )

    for cache in inputs["kv_caches"]:
        # Full row 1: qsl=7, accepted node=3, committed destination=1.
        torch.testing.assert_close(cache[:, 8, 0], before[:, 10, 0])
        # Full row 3: qsl=44, accepted node=4, committed destination=1.
        torch.testing.assert_close(cache[:, 45, 0], before[:, 48, 0])
        # The obsolete prefix interpretation would have written full row 0.
        torch.testing.assert_close(cache[:, 1, 0], before[:, 1, 0])


def test_fixed32_kv16_rejects_unordered_mixed_row_indices() -> None:
    launch = _load_kv16_impl()
    inputs = _mixed_inputs()

    with pytest.raises(RuntimeError, match="dynamic contract violation"):
        launch(
            **inputs,
            batch_indices=torch.tensor([3, 1], dtype=torch.int64),
        )
