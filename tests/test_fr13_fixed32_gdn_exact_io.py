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


def test_fixed32_schedule_licenses_exact_path_io_specialization() -> None:
    text = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    levels = _literal_assignment(tree, "_FR13_FIXED32_SUBTREE_LEVELS")

    assert len(levels) == 2
    assert len(levels[0]) == 1
    root_path, root_parent = levels[0][0]
    assert root_parent == -1
    assert set(root_path) == {
        parent for _path, parent in levels[1]
    }
    assert sorted(
        node
        for level in levels
        for path, _parent in level
        for node in path
    ) == list(range(32))

    launch_start = text.index("    def _launch_paths")
    launch_end = text.index("\n    def _launch(", launch_start)
    launch = text[launch_start:launch_end]
    assert "_state_source = 1 if _li == 0 else 2" in launch
    assert "_export_mode = 1 if _li == 0 else 2" in launch
    assert "STATE_SOURCE=_state_source" in launch
    assert "EXPORT_MODE=_export_mode" in launch


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
    reason="fixed32 GDN path I/O byte gate requires CUDA + triton",
)
def test_fixed32_exact_io_is_byte_identical_to_dynamic_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel

    monkeypatch.setenv("FR13_TREE_GDN_GEOM_OVERRIDE", "BV=8")
    monkeypatch.delenv("FR13_SCAN_ALIGN", raising=False)
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_ROUTE_REQUESTED", True)
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_SELFCHECK_REQUESTED", False)
    kernel._FR13_SUBTREE_CACHE.clear()

    device = torch.device("cuda", torch.cuda.current_device())
    n_actual = n_pad = 32
    num_kh, num_vh = 16, 48
    dim_k = dim_v = 128
    torch.manual_seed(13032)

    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", None)
    kernel.subtree_preseed(
        kernel._FR13_FIXED32_PARENT,
        n_actual,
        num_vh,
        dim_v,
        dim_k,
        device,
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

    def run(*, specialized: bool) -> dict[str, torch.Tensor]:
        monkeypatch.setattr(
            kernel,
            "_FR13_FIXED32_MODE",
            "tail6_fixed32" if specialized else None,
        )
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
            visible_mask=strict_mask,
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

    dynamic = run(specialized=False)
    exact_io = run(specialized=True)
    mismatches = [
        name
        for name in dynamic
        if not _byte_equal(dynamic[name], exact_io[name])
    ]
    assert not mismatches, f"dynamic/exact-I/O byte mismatches: {mismatches}"
