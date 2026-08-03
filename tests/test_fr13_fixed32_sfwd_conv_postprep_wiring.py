from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
VARIANT = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
RUNNER = ROOT / "scripts/run_swe_bench_q36_a.py"
SELECTOR = "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"


def _load_patcher(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_literals() -> str:
    source = PATCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    values: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
    return "\n".join(values)


def test_selector_is_default_off_and_baked_only_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SELECTOR, raising=False)
    patcher = _load_patcher("fr13_sfwd_conv_postprep_default_off")
    assert patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION == "0"
    assert patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_IMPORT == ""
    bindings = patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = False\n" in bindings

    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv("MAX_NUM_SEQS", "4")
    patcher = _load_patcher("fr13_sfwd_conv_postprep_explicit")
    assert "launch_fixed32_sfwd_conv_postprep_fusion" in (
        patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_IMPORT
    )
    bindings = patcher._fr13_fixed32_runtime_bindings("tail6_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = True\n" in bindings


@pytest.mark.parametrize("raw", ("", "true", "2"))
def test_selector_rejects_every_noncanonical_value(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(SELECTOR, raw)
    monkeypatch.setenv("MAX_NUM_SEQS", "1")
    patcher = _load_patcher(f"fr13_sfwd_conv_postprep_bad_{raw!r}")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")


def test_patch_contract_rejects_non_k64_or_graph_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher = _load_patcher("fr13_sfwd_conv_postprep_contract")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", "1"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", ""
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", ""
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_fr13_fixed32_eager_boot_warm_contract", lambda: None
    )
    exact = {
        "MAX_NUM_SEQS": "1",
        "SWE_CONCURRENCY": "1",
        "ENFORCE_EAGER": "1",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "32768",
        "FR13_FIXED32_CONV_SOURCE_BATCH": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
    }
    for name, value in exact.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="eager physical32 B1-or-B4 K64/root1"):
        patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("ENFORCE_EAGER", "0")
    with pytest.raises(RuntimeError, match="eager physical32 B1-or-B4 K64/root1"):
        patcher._fr13_fixed32_validate_patch_env()


def test_generated_route_serves_all_direct_outputs_and_has_no_fallback() -> None:
    generated = _generated_literals()
    assert "launch_fixed32_sfwd_conv_postprep_fusion(" in generated
    assert "_fr13_conv_postprep_b not in (1, 4)" in generated
    assert "int(_fr10_tree_n) != 32" in generated
    assert "qualification_profile=\"k64_root\"" in generated
    assert "draft_vocab_k=65536" in generated
    assert "draft_vocab_root=1" in generated
    assert "physical32_guarded=True" in generated
    assert "source_only_qualification=True" in generated
    assert "conv_tap=None" in generated
    assert "(1, _fr13_conv_postprep_rows, 16, 128)" in generated
    assert "(1, _fr13_conv_postprep_rows, 48, 128)" in generated
    assert "(_fr13_conv_postprep_rows, 48)" in generated
    assert "if not _fr13_conv_postprep_active:\n                        mixed_qkv_spec = _fr10_tree_conv_out" in generated
    assert "if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:" in generated
    assert "value_tree = _fr13_conv_postprep_value_tree" in generated
    assert "g_tree = _fr13_conv_postprep_g" in generated
    assert "beta_tree = _fr13_conv_postprep_beta" in generated
    assert "_fr13_conv_postprep_active\n                        or (" in generated


def test_launcher_and_real_task_runner_forward_the_selector() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    variant = VARIANT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert f"{SELECTOR}=${{{SELECTOR}:-0}}" in launcher
    assert f'-e {SELECTOR}="${SELECTOR}"' in launcher
    assert f"{SELECTOR}=${{{SELECTOR}:-0}}" in variant
    assert SELECTOR in runner
    for source in (launcher, variant):
        assert f"{SELECTOR} must be exactly 0 or 1" in source
        assert source.count(SELECTOR) >= 7
