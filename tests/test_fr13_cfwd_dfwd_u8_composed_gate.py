from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh"
GENERIC = ROOT / "scripts/fr13_run_b1_kernel_live_gate.sh"
MANIFEST = ROOT / "scripts/fr13_runtime_manifest.py"
compose = importlib.import_module("scripts.fr13_cfwd_dfwd_u8_composed_gate")


def _write(path: Path, payload: dict) -> bytes:
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(raw)
    return raw


def test_runner_admits_only_default_off_full_graph_cfwd_u8_composition() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "FR13_GATE_COMPOSE_CFWD_U8:-0" in source
    assert 'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$COMPOSE_CFWD"' in source
    assert 'FR13_CFWD_LOGIT_DIRECT_BYTE_AB="$COMPOSE_CFWD"' in source
    assert "fr13_device_multidraft_cfwd_packed_v3.py" in source
    assert "scripts/fr13_taw_b1_credential.py validate-production" in source
    assert "scripts/fr13_run_b1_kernel_live_gate.sh" in source
    assert "scripts/fr13_cfwd_logit_direct_gate.py issue" in source
    assert "scripts/fr13_cfwd_dfwd_u8_composed_gate.py" in source
    assert "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0" in source
    assert "ENFORCE_EAGER=0" in source
    assert "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in source
    generic = GENERIC.read_text(encoding="ascii")
    assert "scripts/fr13_dfwd_k64_m1_r64_u8_gate.py" in generic
    for assignment in (
        'FR13_DEVICE_MULTIDRAFT_KERNEL="${FR13_DEVICE_MULTIDRAFT_KERNEL:-/workspace/scripts/fr13_device_multidraft_kernel.py}"',
        'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION:-0}"',
        'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON:-}"',
        'FR13_CFWD_LOGIT_DIRECT_BYTE_AB="${FR13_CFWD_LOGIT_DIRECT_BYTE_AB:-0}"',
        'FR13_CFWD_LOGIT_DIRECT_PRODUCTION="${FR13_CFWD_LOGIT_DIRECT_PRODUCTION:-0}"',
        'FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON="${FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON:-}"',
        'FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256="${FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256:-}"',
    ):
        assert assignment in generic
    manifest = MANIFEST.read_text(encoding="ascii")
    assert '"scripts/fr13_cfwd_dfwd_u8_composed_gate.py"' in manifest


def test_composed_validator_reexecutes_both_component_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = "a" * 40
    final_raw = b"final\n"
    boundary_raw = b"boundary\n"
    traffic_raw = b"traffic\n"
    (tmp_path / "final.json").write_bytes(final_raw)
    (tmp_path / "boundary.json").write_bytes(boundary_raw)
    (tmp_path / "traffic.json").write_bytes(traffic_raw)
    cfwd_payload = {
        "integration_source_commit": source_commit,
        "task_ids": [compose.TASK_ID],
        "qualified_batch": 1,
        "complete_work_census_events": 7,
        "final_flush_sha256": hashlib.sha256(final_raw).hexdigest(),
        "boundary_snapshot_sha256": hashlib.sha256(boundary_raw).hexdigest(),
        "traffic_audit_sha256": hashlib.sha256(traffic_raw).hexdigest(),
        "reference_always_served": True,
        "timing_eligible": False,
    }
    dfwd_payload = {
        "source_commit": source_commit,
        "completed_events": 7,
        "final_flush_sha256": hashlib.sha256(final_raw).hexdigest(),
        "boundary_snapshot_sha256": hashlib.sha256(boundary_raw).hexdigest(),
        "chat_traffic_audit_sha256": hashlib.sha256(traffic_raw).hexdigest(),
        "reference_always_served": True,
        "candidate_returned": False,
        "timing_eligible": False,
    }
    _write(tmp_path / "cfwd.json", cfwd_payload)
    _write(tmp_path / "dfwd.json", dfwd_payload)
    for name in ("cfwd-live.json", "dfwd-live.json", "candidate.so", "fa2.so"):
        (tmp_path / name).write_bytes(b"x")
    calls = {"cfwd": 0, "dfwd": 0}

    def fake_cfwd(**_kwargs):
        calls["cfwd"] += 1
        return cfwd_payload

    def fake_dfwd(**_kwargs):
        calls["dfwd"] += 1
        return dfwd_payload

    monkeypatch.setattr(compose.cfwd, "issue", fake_cfwd)
    monkeypatch.setattr(compose.dfwd, "validate_gate", fake_dfwd)
    result = compose.validate_composed_gate(
        repo=ROOT,
        source_commit=source_commit,
        cfwd_credential=tmp_path / "cfwd.json",
        cfwd_live_result=tmp_path / "cfwd-live.json",
        dfwd_gate=tmp_path / "dfwd.json",
        dfwd_live_result=tmp_path / "dfwd-live.json",
        candidate_so=tmp_path / "candidate.so",
        fa2_so=tmp_path / "fa2.so",
        final_flush=tmp_path / "final.json",
        boundary_snapshot=tmp_path / "boundary.json",
        traffic_audit=tmp_path / "traffic.json",
    )
    assert calls == {"cfwd": 1, "dfwd": 1}
    assert result["status"] == "PASS"
    assert result["shared_complete_work_census_events"] == 7
    assert result["component_validators_reexecuted"] is True
    assert result["performance_measurement"] is False
    assert result["sfwd_requires_separate_eager_qrow16_boot"] is True

    forged_cfwd = dict(cfwd_payload)
    forged_cfwd["qualified_batch"] = True
    _write(tmp_path / "cfwd.json", forged_cfwd)
    with pytest.raises(compose.GateError, match="recorded CFWD credential differs"):
        compose.validate_composed_gate(
            repo=ROOT,
            source_commit=source_commit,
            cfwd_credential=tmp_path / "cfwd.json",
            cfwd_live_result=tmp_path / "cfwd-live.json",
            dfwd_gate=tmp_path / "dfwd.json",
            dfwd_live_result=tmp_path / "dfwd-live.json",
            candidate_so=tmp_path / "candidate.so",
            fa2_so=tmp_path / "fa2.so",
            final_flush=tmp_path / "final.json",
            boundary_snapshot=tmp_path / "boundary.json",
            traffic_audit=tmp_path / "traffic.json",
        )

    _write(tmp_path / "cfwd.json", cfwd_payload)
    forged_dfwd = dict(dfwd_payload)
    forged_dfwd["completed_events"] = 7.0
    _write(tmp_path / "dfwd.json", forged_dfwd)
    with pytest.raises(compose.GateError, match="recorded DFWD U8 result differs"):
        compose.validate_composed_gate(
            repo=ROOT,
            source_commit=source_commit,
            cfwd_credential=tmp_path / "cfwd.json",
            cfwd_live_result=tmp_path / "cfwd-live.json",
            dfwd_gate=tmp_path / "dfwd.json",
            dfwd_live_result=tmp_path / "dfwd-live.json",
            candidate_so=tmp_path / "candidate.so",
            fa2_so=tmp_path / "fa2.so",
            final_flush=tmp_path / "final.json",
            boundary_snapshot=tmp_path / "boundary.json",
            traffic_audit=tmp_path / "traffic.json",
        )


def test_composed_validator_rejects_component_event_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = "b" * 40
    final = tmp_path / "final.json"
    boundary = tmp_path / "boundary.json"
    traffic = tmp_path / "traffic.json"
    final.write_bytes(b"final\n")
    boundary.write_bytes(b"boundary\n")
    traffic.write_bytes(b"traffic\n")
    shared = {
        "final_flush_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
        "boundary_snapshot_sha256": hashlib.sha256(boundary.read_bytes()).hexdigest(),
    }
    cfwd_payload = {
        "integration_source_commit": source_commit,
        "task_ids": [compose.TASK_ID],
        "complete_work_census_events": 2,
        **shared,
        "traffic_audit_sha256": hashlib.sha256(traffic.read_bytes()).hexdigest(),
        "reference_always_served": True,
        "timing_eligible": False,
    }
    dfwd_payload = {
        "source_commit": source_commit,
        "completed_events": 3,
        **shared,
        "chat_traffic_audit_sha256": hashlib.sha256(traffic.read_bytes()).hexdigest(),
        "reference_always_served": True,
        "candidate_returned": False,
        "timing_eligible": False,
    }
    _write(tmp_path / "cfwd.json", cfwd_payload)
    _write(tmp_path / "dfwd.json", dfwd_payload)
    for name in ("cfwd-live.json", "dfwd-live.json", "candidate.so", "fa2.so"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(compose.cfwd, "issue", lambda **_kwargs: cfwd_payload)
    monkeypatch.setattr(compose.dfwd, "validate_gate", lambda **_kwargs: dfwd_payload)
    with pytest.raises(compose.GateError, match="one shared real event stream"):
        compose.validate_composed_gate(
            repo=ROOT,
            source_commit=source_commit,
            cfwd_credential=tmp_path / "cfwd.json",
            cfwd_live_result=tmp_path / "cfwd-live.json",
            dfwd_gate=tmp_path / "dfwd.json",
            dfwd_live_result=tmp_path / "dfwd-live.json",
            candidate_so=tmp_path / "candidate.so",
            fa2_so=tmp_path / "fa2.so",
            final_flush=final,
            boundary_snapshot=boundary,
            traffic_audit=traffic,
        )
