from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda function=None, **_: (
        (lambda decorated: decorated) if function is None else function
    )
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CANDIDATE_PATH = ROOT / "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
TIMING_PATH = ROOT / "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_timing.py"
PASS_PATH = ROOT / "scripts/fr13_sfwd_prior_reuse_timing_pass.py"
GATE_PATH = (
    ROOT
    / "results/fr13_fixed32_sfwd_prior_reuse_k64_root_b1_gate_20260802/gate_summary.json"
)


from lumo_flywheel_serving import fr13_sfwd_prior_reuse_timing as timing  # noqa: E402


def _pass_module():
    spec = importlib.util.spec_from_file_location("prior_timing_pass", PASS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset_state() -> None:
    timing._CREDENTIAL_IDS.clear()
    timing._STATE.update(
        gate_sha256=None,
        candidate_source_sha256=None,
        task_marker=None,
        layers=set(),
        launches=0,
        emitted=False,
    )


def test_qualified_candidate_source_and_reduced_gate_are_exact() -> None:
    assert hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest() == (
        timing.QUALIFIED_CANDIDATE_SOURCE_SHA256
    )
    assert hashlib.sha256(GATE_PATH.read_bytes()).hexdigest() == (
        timing.QUALIFIED_REDUCED_GATE_SHA256
    )
    result = _pass_module().validate_gate(
        GATE_PATH,
        expected_gate_sha256=timing.QUALIFIED_REDUCED_GATE_SHA256,
        candidate_source=CANDIDATE_PATH,
    )
    assert result["candidate_serving_permitted"] is True
    assert result["candidate_source_sha256"] == timing.QUALIFIED_CANDIDATE_SOURCE_SHA256


def test_timing_control_is_default_off_and_rejects_gate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = tmp_path / "timing.arm"
    gate = tmp_path / "gate.json"
    gate.write_bytes(GATE_PATH.read_bytes())
    assert (
        timing.fixed32_sfwd_prior_reuse_timing_control(
            environ={}, arm_path=str(arm), gate_path=str(gate)
        )
        is None
    )
    arm.write_text("1\n", encoding="ascii")
    monkeypatch.setattr(timing, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    env = {
        "FR13_FIXED32_SFWD_PRIOR_REUSE_PRODUCTION": "1",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB": "1",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_GATE_SHA256": timing.QUALIFIED_REDUCED_GATE_SHA256,
    }
    credential = timing.fixed32_sfwd_prior_reuse_timing_control(
        environ=env, arm_path=str(arm), gate_path=str(gate)
    )
    assert credential is not None
    assert credential["runtime_candidate_source_sha256"] == (
        timing.QUALIFIED_CANDIDATE_SOURCE_SHA256
    )
    gate.write_bytes(GATE_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        timing.fixed32_sfwd_prior_reuse_timing_control(
            environ=env, arm_path=str(arm), gate_path=str(gate)
        )


def test_engagement_attests_sole_candidate_producer_and_closes_at_48(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_state()
    credential = {
        "reduced_gate_sha256": timing.QUALIFIED_REDUCED_GATE_SHA256,
        "runtime_candidate_source_sha256": timing.QUALIFIED_CANDIDATE_SOURCE_SHA256,
        "task_marker": timing.TASK_MARKER,
    }
    timing._CREDENTIAL_IDS.add(id(credential))
    engagement = tmp_path / "engagement.json"
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_ENGAGEMENT_PATH", str(engagement)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        timing, "_authenticated_real_event", lambda _: timing.TASK_MARKER
    )
    for layer in range(48):
        timing.fixed32_sfwd_prior_reuse_timing_engagement(
            credential=credential, layer_key=layer, batch_size=1
        )
    payload = json.loads(engagement.read_text(encoding="ascii"))
    assert payload["candidate_served"] is True
    assert payload["sole_conv_source_producer"] is True
    assert payload["incumbent_conv_launches_per_layer"] == 0
    assert payload["fallback_permitted"] is False
    assert payload["layer_count"] == 48
    with pytest.raises(RuntimeError, match="more than 48 layers"):
        timing.fixed32_sfwd_prior_reuse_timing_engagement(
            credential=credential, layer_key=49, batch_size=1
        )


def test_route_wiring_is_exclusive_timed_and_source_bound() -> None:
    patcher = (ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()
    launcher = (ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    bigdenom = (ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh").read_text()
    swe = (ROOT / "scripts/run_swe_bench_q36_a.py").read_text()
    runner = (ROOT / "scripts/fr13_run_b1_sfwd_prior_reuse_timing.sh").read_text()
    assert "_FR13_FIXED32_SFWD_PRODUCTION" in patcher
    assert "launch_fixed32_sfwd_prior_reuse" in patcher
    assert "fixed32_sfwd_prior_reuse_timing_engagement" in patcher
    assert "for _fr10_b in range(" in patcher
    assert "if _fr13_sfwd_production is not None" in patcher
    assert "FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB" in launcher
    assert "fr13_fixed32_sfwd_prior_reuse.timing_gate.json" in launcher
    assert "fr13_fixed32_sfwd_prior_reuse.timing.arm" in launcher
    assert "FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB" in bigdenom
    assert 'sfwd_prior_timing_text == "1"' in swe
    assert "FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0" in runner
    assert "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0" in runner
    assert "unset FR10_ALLOW_LINEAR_FALLBACK" in runner
    assert timing.QUALIFIED_REDUCED_GATE_SHA256 in PASS_PATH.read_text()


def test_launcher_cleans_all_sfwd_routes_before_publishing_timing_arm() -> None:
    launcher = (ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    candidate_start = launcher.index(
        'if [[ "$_fr13_sfwd_prior_production" == "1" ]]; then'
    )
    candidate_end = launcher.index(
        'elif [[ "$_fr13_sfwd_prior_reuse" == "1" ]]; then',
        candidate_start,
    )
    candidate = launcher[candidate_start:candidate_end]
    cleanup = candidate.index("  rm -f \\\n")
    gate_publish = candidate.index(
        '  cp -- "$FR13_FIXED32_SFWD_PRIOR_REUSE_GATE_JSON"'
    )
    arm_publish = candidate.index(
        "  printf '1\\n' > \"$LOG_DIR/fr13_fixed32_sfwd_prior_reuse.timing.arm\""
    )
    assert cleanup < gate_publish < arm_publish
    for sidecar in (
        "fr13_fixed32_sfwd_state_fusion_byte_ab.enabled",
        "fr13_fixed32_sfwd_prior_reuse_byte_ab.enabled",
        "fr13_fixed32_sfwd_state_fusion.production.arm",
        "fr13_fixed32_sfwd_prior_reuse.timing.arm",
    ):
        assert f'$LOG_DIR/{sidecar}' in candidate

    stock_start = launcher.index(
        'elif [[ "$_fr13_sfwd_prior_timing" == "1" ]]; then',
        candidate_end,
    )
    stock_end = launcher.index("\nelse\n", stock_start)
    stock = launcher[stock_start:stock_end]
    for sidecar in (
        "fr13_fixed32_sfwd_state_fusion_byte_ab.enabled",
        "fr13_fixed32_sfwd_prior_reuse_byte_ab.enabled",
        "fr13_fixed32_sfwd_state_fusion.production.arm",
        "fr13_fixed32_sfwd_prior_reuse.timing.arm",
    ):
        assert f'$LOG_DIR/{sidecar}' in stock
    assert "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH=" in stock
