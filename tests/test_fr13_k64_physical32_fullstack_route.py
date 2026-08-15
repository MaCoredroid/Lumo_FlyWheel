from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import fr13_taw_b1_credential as credential  # noqa: E402
from scripts import fr13_fixed32_work_census as work_census  # noqa: E402


SOURCE = ROOT / "scripts" / "fr13_device_multidraft_kernel.py"
GATE = ROOT / "scripts" / "fr13_run_b1_k64_taw_source_v7_gate.sh"
PAIR = ROOT / "scripts" / "fr13_run_b1_k64_physical32_fullstack_pair.sh"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
MANIFEST = ROOT / "scripts" / "fr13_runtime_manifest.py"
ARTIFACT = ROOT / "results" / "fr13_k64_physical32_b1_fullstack_route_20260801"


def _source_module():
    module, topology, source_sha256 = credential._load_source(SOURCE)
    return module, topology, source_sha256


def _record(module, topology, *, mode: str, batch: int) -> dict[str, object]:
    marker = credential.TASK_MARKER
    if batch != 1:
        marker = "swe_verified:campaign4_" + "a" * 64
    return {
        "schema": "fr13.fixed32.taw_native_precompute.live_pass.v2",
        "status": "pass",
        "candidate": credential.CANDIDATE,
        "source_contract_schema": credential.SOURCE_SCHEMA,
        "source_contract_sha256": credential.SOURCE_CONTRACT_SHA256,
        "task_marker": marker,
        "mode": mode,
        "valid_mask": int(topology.VALID_MASK_BY_MODE[mode]),
        "topology_binding": module._fr13_fixed32_taw_topology_binding(topology),
        "batch_size": batch,
        "covered_batches": [batch],
        "geometry": dict(module._FR13_FIXED32_TAW_GEOMETRY),
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "evidence_route": "full_graph_replay",
        "reference_returned": True,
        "candidate_returned": False,
    }


def _partial_bundle(module, topology, *, mode: str) -> dict[str, object]:
    return {
        "schema": "fr13.fixed32.taw_native_precompute.pass_bundle.v1",
        "status": "partial",
        "candidate": credential.CANDIDATE,
        "source_contract_schema": credential.SOURCE_SCHEMA,
        "source_contract_sha256": credential.SOURCE_CONTRACT_SHA256,
        "mode": mode,
        "valid_mask": int(topology.VALID_MASK_BY_MODE[mode]),
        "topology_binding": module._fr13_fixed32_taw_topology_binding(topology),
        "required_production_batches": list(
            module._FR13_FIXED32_TAW_REQUIRED_PRODUCTION_BATCHES
        ),
        "qualified_batches": [1],
        "batch_passes": {"1": _record(module, topology, mode=mode, batch=1)},
    }


def _credential_payload(*, mode: str, source_sha256: str, record_sha256: str, live_sha256: str):
    contract = credential.MODE_CONTRACTS[mode]
    payload = {
        "schema": credential.CREDENTIAL_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_k64_b1_graph_byte_gate",
        "candidate": credential.CANDIDATE,
        "source_contract_schema": credential.SOURCE_SCHEMA,
        "source_contract_sha256": credential.SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_sha256,
        "source_commit": "0" * 40,
        "mode": mode,
        "logical_topology": contract["logical_topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": [credential.TASK_ID],
        "task_marker": credential.TASK_MARKER,
        "subset_sha256": credential.B1_SUBSET_SHA256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks": credential.BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": credential.BLOCK_MAP_SHA256,
        "mandatory_weight_bytes": credential.MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": credential.MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": credential.ONE_SIDED_U95_CAP_MS,
        "evidence_route": "full_graph_replay",
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "reference_returned": True,
        "candidate_returned": False,
        "production_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "live_bundle_sha256": live_sha256,
        "live_b1_record_sha256": record_sha256,
        "health_sha256": "2" * 64,
        "authenticated_traffic_audit_sha256": "3" * 64,
        "runtime_manifest_sha256": "4" * 64,
        "runtime_manifest_canonical_sha256": "5" * 64,
        "gate_runner_sha256": "6" * 64,
    }
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(raw)
    return raw


def _b4_verdict(*, mode: str, source_sha256: str, production_sha256: str):
    contract = credential.MODE_CONTRACTS[mode]
    marker = f"swe_verified:campaign4_{credential.EXACT4_SUBSET_SHA256}"
    return {
        "schema": credential.B4_VERDICT_SCHEMAS[mode],
        "status": "pass",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "reference_always_served": True,
        "candidate_returned": False,
        "production_default_enabled": False,
        "raw_prompt_response_published": False,
        "candidate": credential.CANDIDATE,
        "source_commit": "0" * 40,
        "source_contract_schema": credential.SOURCE_SCHEMA,
        "source_contract_sha256": credential.SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_sha256,
        "mode": mode,
        "logical_topology": contract["logical_topology"],
        "active_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "qualified_batches": [1, 2, 3, 4],
        "required_production_batches": list(
            _source_module()[0]._FR13_FIXED32_TAW_REQUIRED_PRODUCTION_BATCHES
        ),
        "independent_b1_record": True,
        "independent_b4_record": True,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks": credential.BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": credential.BLOCK_MAP_SHA256,
        "mandatory_weight_bytes": credential.MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": credential.MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": credential.ONE_SIDED_U95_CAP_MS,
        "subset_sha256": credential.EXACT4_SUBSET_SHA256,
        "task_ids": credential.EXACT4_TASK_IDS,
        "task_marker": marker,
        "stock_fa2_sha256": credential.STOCK_FA2_SHA256,
        "live_bundle_sha256": production_sha256,
        "production_bundle_sha256": production_sha256,
        "campaign_proof_sha256": "7" * 64,
        "runtime_manifest_sha256": "8" * 64,
        "gate_runner_sha256": "9" * 64,
        "probability_mismatches": 0,
        "product_mismatches": 0,
    }


def test_b1_gate_is_real_k64_graph_replay_and_mode_specific() -> None:
    gate = GATE.read_text(encoding="ascii")
    assert GATE.stat().st_mode & 0o111
    assert "subset_b1_diagnostic_one.json" in gate
    assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in gate
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in gate
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in gate
    assert "BLOCK_MAP_SHA256=85dffa58703e42aa" in gate
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1" in gate
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0" in gate
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in gate
    assert "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1" in gate
    assert "FR13_SFWD_GPU_TIMER=0" not in gate
    assert "reference_always_served=1" in gate
    assert "candidate_returned=0" in gate
    assert "tail6_fixed32)" in gate
    assert "hydra27_fixed32)" in gate
    assert "VALID_MASK=0x7a9ce7ff" in gate
    assert "VALID_MASK=0x7abdffff" in gate
    assert "fr13_taw_b1_credential.py issue" in gate
    for forbidden in ("PROBE_ONLY", "ACCEPT_SPEED_PROBE", "synthetic task"):
        assert forbidden not in gate


def test_pair_has_only_all_parent_delta_and_full_wall_breakdown() -> None:
    pair = PAIR.read_text(encoding="ascii")
    helper = (ROOT / "scripts" / "fr13_taw_b1_credential.py").read_text(
        encoding="ascii"
    )
    assert PAIR.stat().st_mode & 0o111
    assert "subset_b4_four.json" in pair
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in pair
    assert "FR13_FIXED32_B1_DIAGNOSTIC=0" in pair
    assert "FR13_FA2_QROW16_PRODUCTION=1" in pair
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=1" in pair
    assert 'run_arm "$STOCK_ARM" 0' in pair
    assert 'run_arm "$CANDIDATE_ARM" 1' in pair
    assert "source_v7_all_parent_committer_production_0_to_1" in pair
    assert "validate-production" in pair
    assert "TAW_REVIEWED_B4_VERDICT" in pair
    assert "TAW_MERGE_BINDING" in pair
    assert "--expected-tok-per-draft 31" in pair
    for field in (
        '"full_step_wall_ms"',
        '"full_wall_tps"',
        '"accepted_drafts_per_event"',
        '"committed_tokens_per_event"',
        '"sfwd_ms"',
        '"dfwd_ms"',
        '"cfwd_ms"',
        '"other_wall_ms"',
        '"wall_over_floor_ratio"',
        '"wall_gap_to_cap_ms"',
    ):
        assert field in helper
    assert '"formal_floor_acceptance_eligible": False' in helper
    assert '"s_per_fwd_gpu_per_forward"' in helper
    assert "fixed32_native_precompute_production_candidate_return" in helper


def test_sfwd_launcher_admits_only_source_gated_taw_production() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    production = launcher.split('elif [[ "$_fr13_sfwd_production" == "1" ]]', 1)[1]
    production = production.split("python3 scripts/fr13_sfwd_state_fusion_pass.py", 1)[0]
    assert '"${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" != "0"' in production
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION:-0" not in production
    assert "source-gated TAW production" in production
    assert "TAW native production requires fixed32 and a regular live PASS JSON" in launcher

    patcher = PATCHER.read_text(encoding="utf-8")
    sfwd = patcher.split('if sfwd_production == "1":', 1)[1]
    sfwd = sfwd.split("if graph_batch_gdn_byte_diagnostic", 1)[0]
    assert 'taw_production not in ("0", "1")' in sfwd
    exact_runtime = sfwd.split("exact_runtime = {", 1)[1].split("}", 1)[0]
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION" not in exact_runtime


def test_runtime_manifest_closes_new_gate_pair_and_b1_subset() -> None:
    manifest = MANIFEST.read_text(encoding="utf-8")
    for required in (
        "scripts/fr13_run_b1_k64_taw_source_v7_gate.sh",
        "scripts/fr13_run_b1_k64_physical32_fullstack_pair.sh",
        "scripts/fr13_taw_b1_credential.py",
        "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    ):
        assert f'"{required}"' in manifest


def test_prepared_campaign_requires_corrected_b4_inputs_and_claims_no_measurement() -> None:
    prepared = ARTIFACT / "prepared_campaign.sh"
    source = prepared.read_text(encoding="ascii")
    readiness = json.loads((ARTIFACT / "readiness.json").read_text(encoding="ascii"))
    assert prepared.stat().st_mode & 0o111
    for required in (
        "TAIL23_REVIEWED_B4_TAW_PASS",
        "TAIL23_REVIEWED_B4_TAW_VERDICT",
        "HYDRA27_REVIEWED_B4_TAW_PASS",
        "HYDRA27_REVIEWED_B4_TAW_VERDICT",
    ):
        assert required in source
    assert source.index("validate-reviewed-b4") < source.index("run_mode()")
    assert readiness["corrected_b4_review_tip"].startswith("e0ac403c2")
    assert readiness["pre_review_b4_artifacts_accepted"] is False
    assert readiness["gpu_campaign_run"] is False
    assert readiness["measurements_present"] is False


@pytest.mark.parametrize("mode", sorted(credential.MODE_CONTRACTS))
def test_live_b1_credential_rejects_cross_mode_reuse(mode: str) -> None:
    module, topology, _ = _source_module()
    live = _partial_bundle(module, topology, mode=mode)
    record = credential._validate_live_bundle(
        live,
        module=module,
        topology=topology,
        mode=mode,
    )
    assert record["mode"] == mode
    other = next(candidate for candidate in credential.MODE_CONTRACTS if candidate != mode)
    with pytest.raises(credential.CredentialError, match="partial pass"):
        credential._validate_live_bundle(
            live,
            module=module,
            topology=topology,
            mode=other,
        )


def test_production_merge_replaces_b1_with_fresh_mode_record(tmp_path: Path) -> None:
    mode = "hydra27_fixed32"
    module, topology, source_sha256 = _source_module()
    live = _partial_bundle(module, topology, mode=mode)
    live_path = tmp_path / "b1.json"
    live_raw = _write_json(live_path, live)
    record = live["batch_passes"]["1"]
    assert isinstance(record, dict)
    record_sha256 = credential._canonical_record_sha256(record)

    credential_payload = _credential_payload(
        mode=mode,
        source_sha256=source_sha256,
        record_sha256=record_sha256,
        live_sha256=hashlib.sha256(live_raw).hexdigest(),
    )
    credential_path = tmp_path / "credential.json"
    _write_json(credential_path, credential_payload)

    campaign_marker = (
        f"swe_verified:campaign4_{credential.EXACT4_SUBSET_SHA256}"
    )
    stale_b1 = dict(record)
    stale_b1["task_marker"] = campaign_marker
    b4_records = {
        str(batch): _record(module, topology, mode=mode, batch=batch)
        for batch in (1, 2, 3, 4)
    }
    for b4_record in b4_records.values():
        b4_record["task_marker"] = campaign_marker
    b4_records["1"] = stale_b1
    production = {
        **{key: value for key, value in live.items() if key not in {"status", "qualified_batches", "batch_passes"}},
        "status": "production_ready",
        "qualified_batches": [1, 2, 3, 4],
        "batch_passes": b4_records,
    }
    b4_path = tmp_path / "b4.json"
    b4_raw = _write_json(b4_path, production)
    verdict_path = tmp_path / "b4-verdict.json"
    _write_json(
        verdict_path,
        _b4_verdict(
            mode=mode,
            source_sha256=source_sha256,
            production_sha256=hashlib.sha256(b4_raw).hexdigest(),
        ),
    )
    output = tmp_path / "merged.json"
    binding = tmp_path / "binding.json"

    args = type(
        "Args",
        (),
        {
            "mode": mode,
            "source": str(SOURCE),
            "credential": str(credential_path),
            "b1_live_bundle": str(live_path),
            "b4_production_pass": str(b4_path),
            "b4_gate_verdict": str(verdict_path),
            "binding_out": str(binding),
            "out": str(output),
        },
    )()
    result = credential.merge_production(args)
    merged = json.loads(output.read_text(encoding="ascii"))
    assert result["status"] == "production_ready"
    assert merged["batch_passes"]["1"] == record
    assert merged["batch_passes"]["4"]["batch_size"] == 4
    merge_binding = json.loads(binding.read_text(encoding="ascii"))
    assert merge_binding["schema"] == credential.MERGE_BINDING_SCHEMA
    assert merge_binding["reviewed_b4_gate_verdict_sha256"]
    validation_args = type(
        "ValidationArgs",
        (),
        {
            "mode": mode,
            "source": str(SOURCE),
            "credential": str(credential_path),
            "b1_live_bundle": str(live_path),
            "b4_production_pass": str(b4_path),
            "b4_gate_verdict": str(verdict_path),
            "merge_binding": str(binding),
            "production_pass": str(output),
        },
    )()
    validation = credential.validate_production(validation_args)
    assert validation["status"] == "bound"


def test_phase_reducer_uses_ms_and_reconciles_full_wall_tps() -> None:
    tasks = [f"task-{index}" for index in range(4)]
    payload = {
        "schema": "fr13.measure.deploy_speed.v1",
        "instrument": "OFF",
        "regime": "deployment",
        "batch_size": 1,
        "n_tasks": 4,
        "task_instance_ids": tasks,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "mandatory_weight_bytes": credential.MANDATORY_WEIGHT_BYTES,
        "step_wall_ms": 200.0,
        "wall_s_per_event": 0.200,
        "events_per_step": 1.0,
        "accept_per_event": 4.0,
        "committed_per_event": 5.0,
        "measured_tps_fullstep_wall": 25.0,
        "s_per_fwd_gpu": 0.120,
        "s_per_fwd_gpu_per_forward": 0.120,
        "drafter_gpu_ms_per_step": 30.0,
        "committer_gpu_ms_per_step": 20.0,
    }
    reduced = credential._validate_measure(
        payload,
        label="candidate",
        task_ids=tasks,
        logical_drafts=27,
    )
    assert reduced["sfwd_ms"] == 120.0
    assert reduced["gpu_component_total_ms"] == 170.0
    assert reduced["other_wall_ms"] == 30.0


@pytest.mark.parametrize(
    ("route", "native"),
    (
        ("fixed32_pytorch_exact_float_triton_integer_commit", False),
        ("fixed32_native_precompute_production_candidate_return", True),
    ),
)
def test_census_validator_accepts_terminal_and_checks_every_event_route(
    tmp_path: Path,
    route: str,
    native: bool,
) -> None:
    mode = "hydra27_fixed32"
    taw = work_census._native_production_taw(1) if native else None
    events = [
        work_census.reference_event(
            mode,
            1,
            f"event-{index}",
            event_index=index,
            forward_step_index=index + 1,
            taw=taw,
        )
        for index in range(2)
    ]
    terminal = work_census.reference_terminal_summary(
        events,
        fixture_synthetic_runtime_proof=True,
    )
    path = tmp_path / "census.jsonl"
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
            for record in [*events, terminal]
        ),
        encoding="ascii",
    )
    validated_raw, validated_events = credential._validate_taw_census(
        path,
        expected_route=route,
        expected_mode=mode,
    )
    assert validated_raw == path.read_bytes()
    assert validated_events == 2

    events[1]["taw"]["route"] = "wrong-route"
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
            for record in [*events, terminal]
        ),
        encoding="ascii",
    )
    with pytest.raises(credential.CredentialError, match="work census"):
        credential._validate_taw_census(
            path,
            expected_route=route,
            expected_mode=mode,
        )
