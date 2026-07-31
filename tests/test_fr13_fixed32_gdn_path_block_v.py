from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _nested_function(
    function: ast.FunctionDef, name: str
) -> ast.FunctionDef:
    return next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _literal_assignment(tree: ast.Module, name: str) -> object:
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )


def test_fixed32_path_block_v_is_decoupled_from_monolith() -> None:
    text = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"BV", "PATH_BLOCK_V"}
    }
    assert constants == {"BV": 16, "PATH_BLOCK_V": 16}

    launch = _function(tree, "launch_tree_gdn_prepared")
    launch_paths = _nested_function(launch, "_launch_paths")
    launch_monolith = _nested_function(launch, "_launch")
    path_source = ast.get_source_segment(text, launch_paths)
    monolith_source = ast.get_source_segment(text, launch_monolith)
    assert path_source is not None
    assert monolith_source is not None

    assert "triton.cdiv(dim_v, _path_bv)" in path_source
    assert "BLOCK_V=_path_bv" in path_source
    assert "triton.cdiv(dim_v, _bv)" not in path_source
    assert "BLOCK_V=_bv" not in path_source
    assert "_tree_gdn_kernel[grid]" in monolith_source
    assert "BLOCK_V=_bv" in monolith_source
    assert "_path_bv" not in monolith_source

    prefix = text[
        text.index("    _path_bv = _bv", text.index("def launch_tree_gdn_prepared("))
        : text.index("    def _launch_paths", text.index("def launch_tree_gdn_prepared("))
    ]
    assert "if _FR13_FIXED32_MODE is not None:" in prefix
    assert '_subtree_state.get("schedule") != "fixed32"' in prefix
    assert "_path_bv = PATH_BLOCK_V" in prefix


def test_bv16_tiles_exact_value_lanes_without_changing_fixed32_nodes() -> None:
    dim_v = 128
    lanes = [
        lane
        for tile in range((dim_v + 16 - 1) // 16)
        for lane in range(tile * 16, min((tile + 1) * 16, dim_v))
    ]
    assert lanes == list(range(dim_v))
    assert len(lanes) == len(set(lanes))

    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    levels = _literal_assignment(tree, "_FR13_FIXED32_SUBTREE_LEVELS")
    covered = sorted(
        node
        for level in levels
        for path, _parent in level
        for node in path
    )
    path_counts = tuple(len(level) for level in levels)
    max_lengths = tuple(
        max(len(path) for path, _parent in level) for level in levels
    )
    padded_slots = sum(
        count * length
        for count, length in zip(path_counts, max_lengths, strict=True)
    )
    assert covered == list(range(32))
    assert path_counts == (1, 11)
    assert sum(path_counts) == 12
    assert padded_slots == 82


try:
    import triton  # noqa: F401

    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only test hosts
    _TRITON_OK = False


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.skipif(
    not (_TRITON_OK and torch.cuda.is_available()),
    reason="fixed32 GDN path BLOCK_V byte gate requires CUDA + triton",
)
def test_fixed32_path_bv16_is_byte_identical_to_bv8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel

    monkeypatch.setenv("FR13_TREE_GDN_GEOM_OVERRIDE", "BV=8")
    monkeypatch.delenv("FR13_SCAN_ALIGN", raising=False)
    monkeypatch.delenv("FR13_PARENT_GATHER", raising=False)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_ROUTE_REQUESTED", True)
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_SELFCHECK_REQUESTED", False)
    kernel._FR13_SUBTREE_CACHE.clear()

    device = torch.device("cuda", torch.cuda.current_device())
    n_actual = 32
    n_pad = 32
    num_kh = 16
    num_vh = 48
    dim_k = 128
    dim_v = 128
    torch.manual_seed(13016)

    parent = kernel._FR13_FIXED32_PARENT
    kernel.subtree_preseed(
        parent, n_actual, num_vh, dim_v, dim_k, device
    )
    state = kernel.subtree_get(
        n_actual, num_vh, dim_v, dim_k, device
    )

    q = torch.randn(
        (n_actual, num_kh, dim_k), device=device, dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        (n_actual, num_vh, dim_v), device=device, dtype=torch.bfloat16
    )
    raw_a = torch.randn(
        (n_actual, num_vh), device=device, dtype=torch.bfloat16
    )
    raw_b = torch.randn_like(raw_a)
    g = torch.zeros(
        (n_actual, num_vh), device=device, dtype=torch.float32
    )
    beta = torch.zeros_like(g)
    a_log = torch.randn(
        (num_vh,), device=device, dtype=torch.float32
    ).abs()
    dt_bias = torch.randn(
        (num_vh,), device=device, dtype=torch.float32
    )
    h0 = torch.randn(
        (4, num_vh, dim_v, dim_k), device=device, dtype=torch.float32
    )
    h0_indices = torch.tensor([2], device=device, dtype=torch.int64)
    h0_num_accepted_tokens = torch.zeros(
        (1,), device=device, dtype=torch.int32
    )
    strict_mask = torch.zeros(
        (n_pad, n_pad), device=device, dtype=torch.int32
    )
    visible_mask = torch.zeros_like(strict_mask)

    def run(path_block_v: int) -> dict[str, torch.Tensor]:
        monkeypatch.setattr(kernel, "PATH_BLOCK_V", path_block_v)
        state["export"].zero_()
        out = torch.zeros(
            (n_pad, num_vh, dim_v), device=device, dtype=torch.bfloat16
        )
        ring_k = torch.zeros_like(q)
        ring_v = torch.zeros_like(v)
        ring_a = torch.zeros_like(raw_a)
        ring_b = torch.zeros_like(raw_b)
        flags = torch.zeros((2,), device=device, dtype=torch.int32)
        invocation_counter = torch.zeros(
            (), device=device, dtype=torch.int32
        )
        kernel.launch_tree_gdn_prepared(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            raw_a=raw_a,
            raw_b=raw_b,
            A_log=a_log,
            dt_bias=dt_bias,
            h0=h0,
            h0_indices=h0_indices,
            h0_num_accepted_tokens=h0_num_accepted_tokens,
            h0_is_bank=True,
            h0_index_row=0,
            h0_batch_index=0,
            h0_use_accepted_column=False,
            n_actual=n_actual,
            n_pad=n_pad,
            strict_mask=strict_mask,
            visible_mask=visible_mask,
            out=out,
            state=None,
            output_scale=dim_k**-0.5,
            use_qk_l2norm_in_kernel=True,
            invocation_counter=invocation_counter,
            ring_k=ring_k,
            ring_v=ring_v,
            ring_a=ring_a,
            ring_b=ring_b,
            staging_flags=flags,
            staging_rows=n_actual,
        )
        torch.cuda.synchronize()
        return {
            "out": out.clone(),
            "export": state["export"].clone(),
            "ring_k": ring_k.clone(),
            "ring_v": ring_v.clone(),
            "ring_a": ring_a.clone(),
            "ring_b": ring_b.clone(),
            "flags": flags.clone(),
            "counter": invocation_counter.clone(),
        }

    bv8 = run(8)
    bv16 = run(16)
    assert bv8.keys() == bv16.keys()
    mismatches = [
        name for name in bv8 if not _byte_equal(bv8[name], bv16[name])
    ]
    assert not mismatches, f"BV8/BV16 byte mismatches: {mismatches}"
