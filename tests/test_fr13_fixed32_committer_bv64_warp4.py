from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
RUNNER_PATH = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
CODEGEN_PATH = ROOT / "scripts/fr13_codegen_committer_bv64_warp4.py"
SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _text(name: str) -> str:
    node = _function(name)
    lines = SOURCE.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_bv64_warp4_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_bv64_warp4_requested")
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), KERNEL_PATH, "exec"),
        namespace,
    )
    requested = namespace[node.name]
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_BV64_WARP4", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False
    monkeypatch.setenv("FR13_FIXED32_COMMITTER_BV64_WARP4", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_BV64_WARP4")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/logs/fr13_fixed32_committer_bv64_warp4.arm",
    )
    assert requested() is True


def test_candidate_changes_only_independent_value_row_tiling() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launcher = _text("_fr13_fixed32_committer_native_layer_batch")

    assert "bv64_warp4" not in kernel
    assert "block_v = 64 if bv64_warp4 else triton.next_power_of_2(dim_v)" in launcher
    assert "kernel_warps = 4 if bv64_warp4 else 8" in launcher
    assert "grid = (1, triton.cdiv(dim_v, block_v)," in launcher
    assert "o_v = i_v * BV + tl.arange(0, BV)" in kernel
    assert "b_v -= tl.sum(b_h * b_k[None, :], 1)" in kernel
    assert "b_h += b_v[:, None] * b_k[None, :]" in kernel
    assert "tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)" in kernel
    assert kernel.count("tl.store(") == 1


def test_candidate_is_exact_hydra27_physical32_k64_root1_b1_b4_only() -> None:
    preseed = _text("preseed_fixed32_committer_graph")

    for requirement in (
        'not layer_batch',
        '_FR13_FIXED32_MODE != "hydra27_fixed32"',
        'batch not in (1, 4)',
        'int(k_rings.shape[2]) != 32',
        'os.environ.get("FR13_DRAFT_VOCAB_ROOT") != "1"',
        'os.environ.get("FR13_DRAFT_VOCAB_K") != "65536"',
    ):
        assert requirement in preseed
    assert '"bv64_warp4": bv64_warp4' in preseed
    assert '"value_tile": 64 if bv64_warp4 else 128' in preseed
    assert '"kernel_warps": 4 if bv64_warp4 else 8' in preseed
    assert '"state_elements_per_thread_before_compiler_effects": 64' in preseed


def test_runner_keeps_default_off_and_creates_only_hydra_sidecar() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "FR13_FIXED32_COMMITTER_BV64_WARP4=${" in runner
    assert "FR13_FIXED32_COMMITTER_BV64_WARP4:-0" in runner
    assert "FR13_FIXED32_COMMITTER_BV64_WARP4 must be exactly 0 or 1" in runner
    assert '[[ "$KIND" == "hydra27_fixed32" \\' in runner
    assert '"$FR13_FIXED32_COMMITTER_LAYER_BATCH" == "1"' in runner
    assert "fr13_fixed32_committer_bv64_warp4.arm" in runner


def test_codegen_contract_is_static_exact_b1_b4_sm121a() -> None:
    codegen = CODEGEN_PATH.read_text(encoding="utf-8")

    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in codegen
    assert 'GPUTarget("cuda", 121, 32)' in codegen
    assert '"incumbent_bv128_warp8": (128, 8)' in codegen
    assert '"candidate_bv64_warp4": (64, 4)' in codegen
    assert "for batch in (1, 4)" in codegen
    assert '"DECAY_REUSE": True' in codegen
    assert '"K_NORM_REUSE": True' in codegen
    assert '"GATE_REUSE": True' in codegen
    assert 'kwargs["maxnreg"] = 167' in codegen
