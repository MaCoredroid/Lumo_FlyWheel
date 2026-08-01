from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_SPEC = importlib.util.spec_from_file_location(
    "fr13_gdn_pad_slot_binding_patcher",
    PATCHER_PATH,
)
assert PATCHER_SPEC is not None and PATCHER_SPEC.loader is not None
patcher = importlib.util.module_from_spec(PATCHER_SPEC)
PATCHER_SPEC.loader.exec_module(patcher)


BROKEN_GENERATED_GDN_SOURCE = """\
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)


def fill_eager_tail(tensor):
    tensor.fill_(PAD_SLOT_ID)


fr10_tree_parent = None
"""


def _pinned_utils_bound_names(source: str) -> set[str]:
    bound: set[str] = set()
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "vllm.v1.attention.backends.utils"
        ):
            bound.update(alias.asname or alias.name for alias in node.names)
    return bound


def test_generated_gdn_source_binds_pad_slot_id_from_pinned_utils(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_path = tmp_path / "gdn_attn.py"
    generated_path.write_text(BROKEN_GENERATED_GDN_SOURCE, encoding="utf-8")
    monkeypatch.setattr(patcher, "GDN_ATTN_PATH", generated_path)

    assert patcher._patch_gdn_attn() is True
    generated = generated_path.read_text(encoding="utf-8")

    assert "PAD_SLOT_ID" in _pinned_utils_bound_names(generated)
    assert (
        "from vllm.v1.attention.backends.utils import (\n"
        "    NULL_BLOCK_ID,\n"
        "    PAD_SLOT_ID,\n"
    ) in generated
    assert generated.count("    PAD_SLOT_ID,\n") == 1
    assert patcher._patch_gdn_attn() is False
    compile(generated, "<generated-gdn-attn>", "exec")
