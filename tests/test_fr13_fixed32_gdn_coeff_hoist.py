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


def _load_layout_function():
    text = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    function = _function(tree, "_fr13_fixed32_coeff_scratch_layout")
    module = ast.Module(
        body=[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Constant(value=value),
            )
            for name, value in (
                ("QK_HEADS", 16),
                ("V_HEADS", 48),
                ("V", 128),
                ("K", 128),
                ("_FR13_FIXED32_COEFF_SCRATCH_NODE", 31),
            )
        ]
        + [function],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), "<coeff-layout>", "exec"), namespace)
    return namespace["_fr13_fixed32_coeff_scratch_layout"]


def test_fixed32_coefficient_scratch_is_unexported_and_in_bounds() -> None:
    layout_fn = _load_layout_function()
    layout = layout_fn(
        n_actual=32,
        num_kh=16,
        num_vh=48,
        dim_v=128,
        dim_k=128,
        export_or_mask=16915,
    )
    row_elems = 48 * 128 * 128
    assert layout == {
        "scratch_node": 31,
        "row_elems": row_elems,
        "q_offset": 31 * row_elems,
        "k_offset": 31 * row_elems + 32 * 16 * 128,
        "decay_offset": 31 * row_elems + 2 * 32 * 16 * 128,
        "beta_offset": 31 * row_elems + 2 * 32 * 16 * 128 + 32 * 48,
        "payload_elems": 2 * 32 * 16 * 128 + 2 * 32 * 48,
        "capacity_elems": row_elems,
    }
    assert not (16915 & (1 << layout["scratch_node"]))
    assert layout["payload_elems"] < layout["capacity_elems"]

    with pytest.raises(ValueError, match="production geometry"):
        layout_fn(
            n_actual=31,
            num_kh=16,
            num_vh=48,
            dim_v=128,
            dim_k=128,
            export_or_mask=16915,
        )
    with pytest.raises(RuntimeError, match="exported parent"):
        layout_fn(
            n_actual=32,
            num_kh=16,
            num_vh=48,
            dim_v=128,
            dim_k=128,
            export_or_mask=16915 | (1 << 31),
        )


def test_coefficient_hoist_removes_fixed32_duplicate_math_by_construction() -> None:
    n_actual = 32
    num_kh = 16
    num_vh = 48
    dim_v = 128
    block_v = 8
    path_node_programs = n_actual * num_vh * (dim_v // block_v)
    qk_precompute_programs = n_actual * num_kh
    scalar_precompute_programs = n_actual * num_vh
    assert path_node_programs // qk_precompute_programs == 48
    assert path_node_programs // scalar_precompute_programs == 16

    text = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    precompute = ast.get_source_segment(
        text, _function(tree, "_tree_gdn_coeff_precompute_kernel")
    )
    node_step = ast.get_source_segment(text, _function(tree, "_gdn_node_step"))
    launch = ast.get_source_segment(
        text, _function(tree, "launch_tree_gdn_prepared")
    )
    assert precompute is not None
    assert node_step is not None
    assert launch is not None
    assert "tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)" in precompute
    assert "tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)" in precompute
    assert "softplus_x = tl.where" in precompute
    assert "decay = tl.exp(-tl.exp(b_a_log) * softplus_x)" in precompute
    assert "b_beta = tl.sigmoid(b_raw_b)" in precompute
    assert "if not COEFFICIENTS_PRECOMPUTED:" in node_step
    assert "state_i *= b_g" in node_step
    assert "_tree_gdn_coeff_precompute_kernel[(n_actual, num_vh)]" in launch
    assert "COEFFICIENTS_PRECOMPUTED=_use_coeff" in launch
    assert "FR13_FIXED32_GDN_COEFF_SELFCHECK MISMATCH" in launch


try:
    import triton  # noqa: F401

    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only test hosts
    _TRITON_OK = False


@pytest.mark.skipif(
    not (_TRITON_OK and torch.cuda.is_available()),
    reason="fixed32 GDN coefficient byte gate requires CUDA + triton",
)
def test_fixed32_coefficient_hoist_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel

    monkeypatch.setenv("FR13_TREE_GDN_GEOM_OVERRIDE", "BV=8")
    monkeypatch.setenv("FR13_FIXED32_GDN_COEFF_HOIST", "1")
    monkeypatch.setenv("FR13_FIXED32_GDN_COEFF_SELFCHECK", "1")
    monkeypatch.delenv("FR13_SCAN_ALIGN", raising=False)
    monkeypatch.delenv("FR13_PARENT_GATHER_SELFCHECK", raising=False)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_ROUTE_REQUESTED", True)
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_SELFCHECK_REQUESTED", False)
    kernel._FR13_SUBTREE_CACHE.clear()

    device = torch.device("cuda", torch.cuda.current_device())
    n_actual = 32
    num_kh = 16
    num_vh = 48
    dim_k = 128
    dim_v = 128
    torch.manual_seed(13048)
    kernel.subtree_preseed(
        kernel._FR13_FIXED32_PARENT,
        n_actual,
        num_vh,
        dim_v,
        dim_k,
        device,
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
    g = torch.zeros((n_actual, num_vh), device=device, dtype=torch.float32)
    beta = torch.zeros_like(g)
    a_log = torch.randn((num_vh,), device=device, dtype=torch.float32)
    dt_bias = torch.randn((num_vh,), device=device, dtype=torch.float32)
    h0 = torch.randn(
        (4, num_vh, dim_v, dim_k), device=device, dtype=torch.float32
    )
    out = torch.zeros(
        (n_actual, num_vh, dim_v), device=device, dtype=torch.bfloat16
    )
    ring_k = torch.zeros_like(q)
    ring_v = torch.zeros_like(v)
    ring_a = torch.zeros_like(raw_a)
    ring_b = torch.zeros_like(raw_b)
    flags = torch.zeros((2,), device=device, dtype=torch.int32)
    counter = torch.zeros((), device=device, dtype=torch.int32)
    descriptor = torch.zeros(
        (n_actual, n_actual), device=device, dtype=torch.int32
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
        h0_indices=torch.tensor([2], device=device, dtype=torch.int64),
        h0_num_accepted_tokens=torch.zeros(
            (1,), device=device, dtype=torch.int32
        ),
        h0_is_bank=True,
        h0_index_row=0,
        h0_batch_index=0,
        h0_use_accepted_column=False,
        n_actual=n_actual,
        n_pad=n_actual,
        strict_mask=descriptor,
        visible_mask=descriptor,
        out=out,
        state=None,
        output_scale=dim_k**-0.5,
        use_qk_l2norm_in_kernel=True,
        invocation_counter=counter,
        ring_k=ring_k,
        ring_v=ring_v,
        ring_a=ring_a,
        ring_b=ring_b,
        staging_flags=flags,
        staging_rows=n_actual,
    )
    torch.cuda.synchronize()
    assert int(counter.item()) == 1
    assert flags.tolist() == [1, n_actual]
