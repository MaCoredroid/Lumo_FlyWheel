from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEVICE_PATH = ROOT / "scripts" / "fr13_device_multidraft_cfwd_packed_v3.py"
GATE_PATH = ROOT / "scripts" / "fr13_cfwd_logit_direct_gate.py"
LIVE_RUNNER = ROOT / "scripts" / "fr13_run_b1_cfwd_logit_direct_live_gate.sh"
TIMING_RUNNER = ROOT / "scripts" / "fr13_run_b1_cfwd_logit_direct_timing.sh"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
CENSUS_PATH = ROOT / "scripts" / "fr13_fixed32_work_census.py"
CANDIDATE_SOURCE = ROOT / "scripts" / "fr13_cfwd_logit_direct_decision_kernel.py"
ONE_TASK_SUBSET = ROOT / "config" / "fr13_fixed32" / "subset_b1_diagnostic_one.json"
EXACT4_SUBSET = ROOT / "config" / "fr13_fixed32" / "subset_b4_four.json"
EXACT16_SUBSET = ROOT / "config" / "fr13_fixed32" / "subset_b4_sixteen.json"


def _load(path: Path, name: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module._base if path == DEVICE_PATH else module


def _credential(source_commit: str) -> dict:
    return {
        "schema": "fr13.fixed32.cfwd_logit_direct.production_credential.v2",
        "status": "production_timing_ready",
        "candidate": "fixed32_cfwd_logit_direct_packed_physical_slots_v3",
        "candidate_schema": (
            "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
        ),
        "candidate_source_sha256": (
            "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
        ),
        "integration_source_commit": source_commit,
        "integration_source_schema": (
            "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
        ),
        "integration_source_sha256": (
            "421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b"
        ),
        "mode": "hydra27_fixed32",
        "qualified_batch": 1,
        "task_count": 1,
        "task_ids": ["astropy__astropy-12907"],
        "task_marker": "swe_verified:astropy__astropy-12907",
        "reference_always_served": True,
        "decision_mismatches": [0] * 5,
        "walk_mismatches": [0] * 5,
        "candidate_invalid": 0,
        "complete_work_census_events": 7,
        "live_result_sha256": "1" * 64,
        "final_flush_sha256": "2" * 64,
        "boundary_snapshot_sha256": "3" * 64,
        "traffic_audit_sha256": "4" * 64,
        "subset_sha256": (
            "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
        ),
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
    }


def test_production_selector_is_source_bound_and_emits_engagement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(DEVICE_PATH, "fr13_cfwd_direct_production_selector_test")
    source_commit = "a" * 40
    credential_path = tmp_path / "credential.json"
    raw = (
        json.dumps(_credential(source_commit), ensure_ascii=True, sort_keys=True)
        + "\n"
    ).encode("ascii")
    credential_path.write_bytes(raw)
    engagement_path = tmp_path / "engagement.json"
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "1")
    monkeypatch.setenv(
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_PATH", str(credential_path)
    )
    monkeypatch.setenv(
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_SOURCE_COMMIT", source_commit)
    monkeypatch.setenv(
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION_ENGAGEMENT_JSON",
        str(engagement_path),
    )

    assert (
        module._fr13_cfwd_logit_direct_selector(
            mode="hydra27_fixed32", batch_size=1
        )
        == "production"
    )
    engagement = json.loads(engagement_path.read_text(encoding="ascii"))
    assert engagement["served_return"] == "logit-direct candidate products"
    assert engagement["source_commit"] == source_commit
    assert engagement["integration_source_schema"] == (
        module._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA
    )
    assert engagement["integration_source_sha256"] == (
        module._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256
    )
    assert engagement["production_pass_sha256"] == hashlib.sha256(raw).hexdigest()


def test_production_selector_rejects_diagnostic_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(DEVICE_PATH, "fr13_cfwd_direct_overlap_selector_test")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        module._fr13_cfwd_logit_direct_selector(
            mode="hydra27_fixed32", batch_size=1
        )


def test_cfwd_integration_contract_propagates_without_rekeying_taw() -> None:
    device = _load(DEVICE_PATH, "fr13_cfwd_direct_contract_device_test")
    gate = _load(GATE_PATH, "fr13_cfwd_direct_contract_gate_test")
    census = _load(CENSUS_PATH, "fr13_cfwd_direct_contract_census_test")
    contract = device._fr13_cfwd_logit_direct_integration_source_contract()
    assert (
        contract["integration_source_schema"]
        == gate.INTEGRATION_SOURCE_SCHEMA
        == census.TAW_CFWD_LOGIT_DIRECT_SOURCE_SCHEMA
    )
    assert (
        contract["integration_source_sha256"]
        == gate.INTEGRATION_SOURCE_SHA256
        == census.TAW_CFWD_LOGIT_DIRECT_SOURCE_SHA256
    )
    assert (
        device._FR13_FIXED32_TAW_SOURCE_SHA256
        == census.TAW_SOURCE_CONTRACT_SHA256
        == "68b289aee5773edf1134f184c37551a90ec8543430d768a05066bc1341473c6d"
    )


def test_direct_production_work_census_has_no_pytorch_vocab_materialization() -> None:
    census = _load(CENSUS_PATH, "fr13_cfwd_direct_work_census_test")
    event = census.reference_event(
        "hydra27_fixed32", 1, "cfwd-logit-direct-production"
    )
    event["taw"].update(
        {
            "route": census.TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE,
            "child_lanes": 51,
            "target_rows": 17,
            "self_rows": 13,
            "self_cdf_rows": 13,
            "source_cdf_rows": 17,
            "residual_cdf_rows": 17,
            "qmix_rows": 17,
            "residual_rows": 17,
            "exact_commit_launches": 1,
            "exact_commit_programs": 1,
            "source_contract_schema": census.TAW_CFWD_LOGIT_DIRECT_SOURCE_SCHEMA,
            "source_contract_sha256": census.TAW_CFWD_LOGIT_DIRECT_SOURCE_SHA256,
            "tensor_call_census": dict(
                census.TAW_CFWD_LOGIT_DIRECT_PRODUCTION_TENSOR_CALL_CENSUS
            ),
        }
    )
    validated = census.validate_event(event, source="cfwd-direct-production")
    calls = validated.normalized_work["taw"]["tensor_call_census"]
    assert calls["full_vocab_row_gathers"] == 0
    assert calls["full_vocab_softmax_calls"] == 0
    assert calls["residual_subtract_calls"] == 0
    assert calls["exact_commit_launches"] == 1


def test_gate_issues_only_from_canonical_real_task_and_final_flush(
    tmp_path: Path,
) -> None:
    gate = _load(GATE_PATH, "fr13_cfwd_direct_gate_issue_test")
    source_commit = "b" * 40
    boundary_path = tmp_path / "boundary.json"
    boundary_path.write_text('{"boundary":1}\n', encoding="ascii")
    boundary_sha = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
    live = {
        "schema": gate.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "candidate": gate.CANDIDATE,
        "candidate_schema": gate.CANDIDATE_SCHEMA,
        "candidate_source_sha256": gate.CANDIDATE_SOURCE_SHA256,
        "integration_source_schema": gate.INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": gate.INTEGRATION_SOURCE_SHA256,
        "source_commit": source_commit,
        "mode": gate.MODE,
        "instance_id": gate.GATE_TASK_ID,
        "task_marker": "swe_verified:" + gate.GATE_TASK_ID,
        "batch_histogram": {"1": 7, "4": 0},
        "complete_work_census_events": 7,
        "counted_graph_replays": 7,
        "decision_mismatches": [0] * 5,
        "walk_mismatches": [0] * 5,
        "candidate_invalid": 0,
        "served_return": "reference all-parent products unchanged",
        "performance_measurement": False,
        "finalized_by_fixed32_flush": True,
        "flush_generation": 3,
        "flush_nonce": "c" * 64,
        "producer_pid": 99,
        "boundary_snapshot_sha256": boundary_sha,
    }
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live) + "\n", encoding="ascii")
    flush = {
        "schema": "fr13-fixed32-flush-client-result-v1",
        "ack": {
            "schema": "fr13-fixed32-flush-ack-v1",
            "action": "final",
            "status": "ok",
            "mode": gate.MODE,
            "generation": 3,
            "nonce": "c" * 64,
            "producer_pid": 99,
            "counters": {"complete_work_census_events": 7},
        },
    }
    flush_path = tmp_path / "flush.json"
    flush_path.write_text(json.dumps(flush) + "\n", encoding="ascii")
    subset_raw = ONE_TASK_SUBSET.read_bytes()
    traffic = {
        "subset": {
            "task_ids": [gate.GATE_TASK_ID],
            "task_count": 1,
            "sha256": hashlib.sha256(subset_raw).hexdigest(),
        },
        "checks": {"all_authenticated": True},
        "complete_stream": {"complete_work_census_events": 7},
    }
    traffic_path = tmp_path / "traffic.json"
    traffic_path.write_text(json.dumps(traffic) + "\n", encoding="ascii")
    output = tmp_path / "credential.json"

    credential = gate.issue(
        live_result=live_path,
        subset=ONE_TASK_SUBSET,
        final_flush=flush_path,
        boundary_snapshot=boundary_path,
        traffic_audit=traffic_path,
        candidate_source=CANDIDATE_SOURCE,
        source_commit=source_commit,
        output=output,
    )
    assert credential["status"] == "production_timing_ready"
    assert credential["subset_sha256"] == gate.GATE_SUBSET_SHA256
    assert output.is_file()


def test_timing_credential_admits_only_canonical_exact4_and_exact16(
    tmp_path: Path,
) -> None:
    gate = _load(GATE_PATH, "fr13_cfwd_direct_timing_subsets_test")
    observed = {}
    for subset in (EXACT4_SUBSET, EXACT16_SUBSET):
        raw = subset.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw.decode("ascii"))
        observed[digest] = payload["instance_ids"]
        assert gate._validate_timing_subset(subset) == digest
    assert gate.TIMING_SUBSETS == observed

    unknown = tmp_path / "unknown.json"
    unknown.write_text("{}\n", encoding="ascii")
    with pytest.raises(gate.GateError, match="canonical exact4 or exact16"):
        gate._validate_timing_subset(unknown)


def test_live_runner_is_canonical_real_swe_byte_gate() -> None:
    subprocess.run(["bash", "-n", str(LIVE_RUNNER)], check=True)
    text = LIVE_RUNNER.read_text(encoding="ascii")
    assert LIVE_RUNNER.stat().st_mode & stat.S_IXUSR
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in text
    assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in text
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in text
    assert "FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1" in text
    assert "FR13_CFWD_LOGIT_DIRECT_BYTE_AB=1" in text
    assert "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0" in text
    assert (
        "FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/"
        "fr13_device_multidraft_cfwd_packed_v3.py"
    ) in text
    assert "fr13_taw_b1_credential.py validate-production" in text
    assert text.index("fr13_taw_b1_credential.py validate-production") < text.index(
        "docker ps -aq"
    )
    for prerequisite in (
        "TAW_B1_CREDENTIAL",
        "TAW_B1_LIVE_BUNDLE",
        "TAW_REVIEWED_B4_PASS",
        "TAW_REVIEWED_B4_VERDICT",
        "TAW_MERGE_BINDING",
    ):
        assert prerequisite in text
    assert "--traffic-audit \"$TRAFFIC_AUDIT\"" in text
    assert "--boundary-snapshot \"$BOUNDARY\"" in text
    assert "_fr13_cfwd_logit_direct_integration_source_contract" in text
    assert "CFWD integration source contract mismatch" in text
    assert "PROBE_ONLY" not in text and "CAPTURE_ONLY" not in text


def test_timing_runner_is_exact4_stock_then_credentialed_candidate() -> None:
    subprocess.run(["bash", "-n", str(TIMING_RUNNER)], check=True)
    text = TIMING_RUNNER.read_text(encoding="ascii")
    assert TIMING_RUNNER.stat().st_mode & stat.S_IXUSR
    assert "config/fr13_fixed32/subset_b4_four.json" in text
    assert "--timing-subset \"$SUBSET\"" in text
    assert text.index('"$GATE" validate') < text.index("docker ps -aq")
    assert "fr13_taw_b1_credential.py validate-production" in text
    assert text.index("fr13_taw_b1_credential.py validate-production") < text.index(
        "docker ps -aq"
    )
    for prerequisite in (
        "TAW_B1_CREDENTIAL",
        "TAW_B1_LIVE_BUNDLE",
        "TAW_REVIEWED_B4_PASS",
        "TAW_REVIEWED_B4_VERDICT",
        "TAW_MERGE_BINDING",
    ):
        assert prerequisite in text
    assert 'run_arm "$STOCK_ARM" 0' in text
    assert 'run_arm "$CANDIDATE_ARM" 1' in text
    assert (
        "FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/"
        "fr13_device_multidraft_cfwd_packed_v3.py"
    ) in text
    assert "--expected-tok-per-draft 31" in text
    assert '"timing_eligible": True' in text
    assert '"floor_acceptance_eligible": False' in text
    assert "cfwd_gate.INTEGRATION_SOURCE_SCHEMA" in text
    assert "cfwd_gate.INTEGRATION_SOURCE_SHA256" in text
    assert "CFWD integration source contract mismatch" in text
    for field in (
        '"step_wall_ms"',
        '"measured_tps_fullstep_wall"',
        '"accepted_drafts_per_event"',
        '"committed_tokens_per_event"',
        '"sfwd_gpu_ms_per_step"',
        '"dfwd_gpu_ms_per_step"',
        '"cfwd_gpu_ms_per_step"',
        '"other_wall_ms_per_step"',
        '"floor_ratio"',
    ):
        assert field in text
    assert "PROBE_ONLY" not in text and "CAPTURE_ONLY" not in text


def test_launcher_issues_read_only_canonical_marker_and_keeps_modes_exclusive() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "swe_verified:astropy__astropy-12907" in text
    assert 'chmod 444 "$_fr13_cfwd_marker_tmp"' in text
    assert "FR13_CFWD_LOGIT_DIRECT_REAL_EVENT_UID=$(stat -c '%u'" in text
    assert '"$FR13_FIXED32_B1_DIAGNOSTIC:-0"' not in text
    assert '"${FR13_FIXED32_B1_DIAGNOSTIC:-0}" == "1"' in text
    assert "diagnostic and production are mutually exclusive" in text
