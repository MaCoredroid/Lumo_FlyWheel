from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GATE = REPO / "scripts" / "fr13_dfwd_k64_m4_r64_u8_gate.py"
CREDENTIAL = REPO / "scripts" / "fr13_dfwd_k64_m4_r64_u8_production_credential.py"
RUNNER = REPO / "scripts" / "fr13_run_b4_dfwd_k64_m4_r64_u8_live_gate.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_namespace(*names: str) -> dict[str, object]:
    source = _load(
        PATCHER, "fr13_m4_patcher_runtime"
    )._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    wanted = set(names)
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in body} == wanted
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<m4-runtime>", "exec"),
        namespace,
    )
    return namespace


def _bridge_namespace() -> dict[str, object]:
    source = _load(
        PATCHER, "fr13_m4_patcher_bridge"
    )._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_fr13_draft_head_m4_u8_worker_env_bridge"
        ):
            body.append(node)
    namespace: dict[str, object] = {
        "_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_REQUIRED": True,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<m4-bridge>", "exec"),
        namespace,
    )
    return namespace


def _worker_payload(patcher) -> dict[str, str]:
    payload = {key: "" for key in patcher._FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS}
    payload.update(
        {
            "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB": "1",
            "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE": "0",
            "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION": "0",
            "FR13_DRAFT_HEAD_M4_R64_U8_SO": "/tmp/fr13_bf16_k64_m4_r64_u8.abi3.so",
            "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT": "b" * 40,
            "FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS": (
                "astropy__astropy-12907,astropy__astropy-13033,"
                "astropy__astropy-13236,astropy__astropy-13398"
            ),
            "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON": "/logs/fr13_dfwd_k64_m4_r64_u8.live.json",
            "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_ROOT": "1",
        }
    )
    for key in payload:
        if key.endswith("SHA256") and not key.endswith(
            "PRODUCTION_PASS_SIDECAR_SHA256"
        ):
            payload[key] = "a" * 64
    return payload


def _live(gate, events: int = 3) -> dict[str, object]:
    identities = {
        "build_attestation_sha256": gate.BUILD_SHA256,
        "candidate_so_bytes": gate.SO_BYTES,
        "candidate_so_sha256": gate.SO_SHA256,
        "candidate_source_sha256": gate.SOURCE_SHA256,
        "fa2_sha256": gate.FA2_SHA256,
        "patch_source_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "source_commit": "b" * 40,
        "subset_sha256": gate.SUBSET_SHA256,
        "taw_source_sha256": "8" * 64,
        "task_ids": list(gate.TASK_IDS),
        "vocab_blocks_sha256": gate.BLOCKS_SHA256,
    }
    taw = {
        "schema": "fr13.fixed32.taw_candidate_acceptance_census.v1",
        "status": "PASS",
        "mode": "hydra27_fixed32",
        "batch_size": 4,
        "completed_events": events,
        "comparison_events": events,
        "events_sha256": "5" * 64,
        "task_marker": "swe_verified:campaign4_" + gate.SUBSET_SHA256,
        "candidate_token_source": {
            "operation": gate.CANDIDATE["operation"],
            "candidate_so_sha256": gate.SO_SHA256,
            "candidate_source_sha256": gate.SOURCE_SHA256,
            "task_ids": list(gate.TASK_IDS),
        },
        "draft_probs": None,
        "target_authority": True,
        "source_contract_schema": "fr13-fixed32-taw-all-parent-v7",
        "source_contract_sha256": gate.TAW_SOURCE_CONTRACT_SHA256,
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "accept_decision_mismatches": 0,
        "reference_returned": True,
    }
    return {
        "schema": gate.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "task_ids": list(gate.TASK_IDS),
        "task_markers": ["swe_verified:" + task for task in gate.TASK_IDS],
        "concurrency": 4,
        "batch_size": 4,
        "source_commit": "b" * 40,
        "identities": identities,
        "worker_env_bridge": {
            "schema": "fr13.fixed32.dfwd_k64_m4_r64_u8_worker_env_bridge.v1",
            "sidecar": "/logs/fr13_draft_head_m4_r64_u8.worker_env.json",
            "sidecar_sha256": "3" * 64,
            "payload_sha256": "4" * 64,
            "hydrated_keys": list(gate.WORKER_ENV_KEYS),
        },
        "topology": gate.TOPOLOGY,
        "geometry": gate.GEOMETRY,
        "candidate": gate.CANDIDATE,
        "completed_events": events,
        "complete_work_census_events": events,
        "work_census_last_event_index": events - 1,
        "events_sha256": "5" * 64,
        "flush_generation": 1,
        "flush_nonce": "6" * 64,
        "producer_pid": 7,
        "boundary_snapshot_sha256": "7" * 64,
        "root_forward_steps": list(range(events)),
        "captured_mtp_depths": [1, 2, 3, 4],
        "per_depth_full_logit_comparisons": {label: events for label in gate.DEPTHS},
        "per_depth_raw_bf16_mismatches": {label: 0 for label in gate.DEPTHS},
        "per_depth_nonfinite_logits": {label: 0 for label in gate.DEPTHS},
        "comparison_scope": gate.COMPARISON_SCOPE,
        "full_logit_comparisons": events * 5,
        "compared_elements": events * 5 * 4 * 65536,
        "compared_bytes": events * 5 * 4 * 65536 * 2,
        "raw_bf16_mismatches": 0,
        "nonfinite_logits": 0,
        "qualification_policy": "lossless_deterministic_proposal_taw_exact_v1",
        "proposal_distribution": gate.PROPOSAL_DISTRIBUTION,
        "taw_exact_acceptance": taw,
        "reference_always_served": False,
        "candidate_returned": True,
        "served_return": "candidate BF16 logits",
        "performance_measurement": False,
        "timing_eligible": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }


class _Counter:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


def test_worker_sidecar_round_trip_and_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _load(PATCHER, "fr13_m4_patcher_sidecar")
    payload = _worker_payload(patcher)
    for key, value in payload.items():
        monkeypatch.setenv(key, value)
    sidecar = tmp_path / "m4.worker_env.json"
    record = patcher._fr13_write_draft_head_m4_u8_worker_env_sidecar(sidecar)
    assert record is not None and record["payload"] == payload
    assert sidecar.stat().st_mode & 0o777 == 0o400
    for key in payload:
        monkeypatch.delenv(key, raising=False)
    namespace = _bridge_namespace()
    bridge = namespace["_fr13_draft_head_m4_u8_worker_env_bridge"](str(sidecar))
    assert bridge["hydrated_keys"] == list(
        patcher._FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS
    )
    assert bridge["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    for key in payload:
        monkeypatch.delenv(key, raising=False)

    decoded = json.loads(sidecar.read_text(encoding="ascii"))
    decoded["payload"]["FR13_DRAFT_VOCAB_K"] = "32768"
    sidecar.chmod(0o600)
    sidecar.write_text(json.dumps(decoded), encoding="ascii")
    sidecar.chmod(0o400)
    with pytest.raises(RuntimeError, match="payload drifted"):
        namespace["_fr13_draft_head_m4_u8_worker_env_bridge"](str(sidecar))


def test_b4_depth_classifier_accounts_four_rows_at_each_mtp_depth() -> None:
    namespace = _runtime_namespace("_fr13_draft_head_m4_u8_live_depth")
    namespace["_FR13_FIXED32_MODE"] = "hydra27_fixed32"
    namespace["_FR13_DRAFT_HEAD_M4_U8_LIVE_STATE"] = {
        "root_forward_steps": [],
        "captured_depths": [],
    }
    namespace["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = {
        "batch_size": 4,
        "mode": "hydra27_fixed32",
        "measured": True,
        "mtp_execution_basis": "unbound",
        "mtp_forward_calls": 0,
        "mtp_forward_rows": 0,
        "forward_step_index": 0,
    }
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"] = None
    classify = namespace["_fr13_draft_head_m4_u8_live_depth"]
    assert classify(4) == 0
    context = {
        "capturing": True,
        "batch_size": 4,
        "mode": "hydra27_fixed32",
        "draft_head_u8_calls": 0,
        "draft_head_u8_rows": 0,
        "mtp_forward_calls": 1,
        "mtp_forward_rows": 4,
    }
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"] = context
    for depth in range(1, 5):
        context["mtp_forward_calls"] = depth
        context["mtp_forward_rows"] = depth * 4
        assert classify(4) == depth
    assert context["draft_head_u8_calls"] == 4
    assert context["draft_head_u8_rows"] == 16


def test_live_reducer_requires_all_five_b4_byte_exact_sites() -> None:
    gate = _load(GATE, "fr13_m4_gate_validation")
    payload = _live(gate)
    assert (
        gate.validate_live_result(payload, expected_source_commit="b" * 40) is payload
    )
    for mutation in (
        lambda value: value["per_depth_full_logit_comparisons"].__setitem__(
            "mtp_depth_4", 2
        ),
        lambda value: value.__setitem__(
            "compared_elements", value["compared_elements"] // 4
        ),
        lambda value: value["identities"].__setitem__("candidate_so_sha256", "0" * 64),
        lambda value: value["task_ids"].reverse(),
        lambda value: value.__setitem__("reference_always_served", True),
    ):
        changed = copy.deepcopy(payload)
        mutation(changed)
        with pytest.raises(ValueError):
            gate.validate_live_result(changed, expected_source_commit="b" * 40)


def test_final_flush_record_is_consumable_by_strict_b4_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _load(GATE, "fr13_m4_gate_finalizer")
    namespace = _runtime_namespace("_fr13_draft_head_m4_u8_live_finalize")
    output = tmp_path / "m4.live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON", str(output))
    base = _live(gate, events=2)
    namespace["_FR13_DRAFT_HEAD_M4_U8_LIVE_STATE"] = {
        "compares": _Counter([2] * 5),
        "mismatches": _Counter([0] * 5),
        "nonfinite": _Counter([0] * 5),
        "geometry": gate.GEOMETRY,
        "candidate": gate.CANDIDATE,
        "identities": base["identities"],
        "worker_env_bridge": base["worker_env_bridge"],
        "root_forward_steps": [0, 1],
        "captured_depths": [1, 2, 3, 4],
    }
    monkeypatch.setitem(
        sys.modules,
        "_fr13_device_multidraft_kernel",
        types.SimpleNamespace(
            fr13_fixed32_taw_candidate_acceptance_census=(
                lambda **_kwargs: base["taw_exact_acceptance"]
            )
        ),
    )
    namespace["_fr13_draft_head_m4_u8_live_finalize"](
        [
            {"batch_size": 4, "forward_step_index": 0},
            {"batch_size": 4, "forward_step_index": 1},
        ],
        {
            "action": "final",
            "boundary_snapshot_sha256": "7" * 64,
            "complete_work_census_events": 2,
            "events_sha256": "5" * 64,
            "generation": 1,
            "nonce": "6" * 64,
            "producer_pid": 7,
        },
    )
    payload = json.loads(output.read_text(encoding="ascii"))
    assert (
        gate.validate_live_result(payload, expected_source_commit="b" * 40) is payload
    )


def test_b4_production_credential_requires_full_candidate_taw_census() -> None:
    credential = _load(CREDENTIAL, "fr13_m4_production_credential")
    gate = credential.gate
    live = _live(gate, events=3)
    inputs = {
        key: live["identities"][key]
        for key in credential.INPUT_KEYS
        if key != "candidate_so_bytes"
    }
    inputs["candidate_so_bytes"] = gate.SO_BYTES
    validated = {
        "schema": gate.GATE_SCHEMA,
        "status": "PASS",
        "source_commit": "b" * 40,
        "task_ids": list(gate.TASK_IDS),
        "all_tasks_resolved": True,
        "topology": gate.TOPOLOGY,
        "geometry": gate.GEOMETRY,
        "candidate": gate.CANDIDATE,
        "inputs": inputs,
        "live_result_sha256": "a" * 64,
        "completed_events": 3,
        "captured_mtp_depths": [1, 2, 3, 4],
        "comparison_scope": gate.COMPARISON_SCOPE,
        "raw_bf16_mismatches": 17,
        "nonfinite_logits": 0,
        "qualification_policy": "lossless_deterministic_proposal_taw_exact_v1",
        "proposal_distribution": gate.PROPOSAL_DISTRIBUTION,
        "taw_exact_acceptance": live["taw_exact_acceptance"],
        "reference_always_served": False,
        "candidate_returned": True,
        "events_sha256": "5" * 64,
        "final_flush_sha256": "c" * 64,
        "boundary_snapshot_sha256": "d" * 64,
        "chat_traffic_audit_sha256": "e" * 64,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": True,
    }
    payload = credential._credential_from_gate(validated)
    assert payload["production_default_enabled"] is False
    assert payload["candidate_returned_during_qualification"] is True
    assert payload["incumbent_served_during_qualification"] is False
    assert payload["taw_exact_acceptance"]["comparison_events"] == 3

    forged = copy.deepcopy(validated)
    forged["taw_exact_acceptance"]["accept_decision_mismatches"] = 1
    with pytest.raises(ValueError, match="exact candidate-served PASS"):
        credential._credential_from_gate(forged)

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "fr13_dfwd_k64_m4_r64_u8_production_credential.py issue" in launcher
    assert "fr13_dfwd_k64_m4_r64_u8_production_credential.py verify" in launcher
    assert "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED=1" in launcher


def test_b4_terminal_validator_requires_four_rows_per_event(tmp_path: Path) -> None:
    gate = _load(GATE, "fr13_m4_gate_terminal")
    live = _live(gate, events=2)
    counters = {
        "pure_decode_forward_steps": 2,
        "complete_work_census_events": 2,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": 1,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    final = {
        "schema": "fr13-fixed32-flush-client-result-v1",
        "ack": {
            "schema": "fr13-fixed32-flush-ack-v1",
            "mode": "hydra27_fixed32",
            "status": "ok",
            "action": "final",
            "generation": 1,
            "nonce": "6" * 64,
            "producer_pid": 7,
            "counters": counters,
        },
    }
    boundary = {
        "schema": "fr13-fixed32-boundary-snapshot-v4",
        "mode": "hydra27_fixed32",
        "action": "final",
        "generation": 1,
        "nonce": "6" * 64,
        "producer_pid": 7,
        "counters": counters,
        "metrics": {
            "fixed32": {
                "pure_decode_forward_steps": 2,
                "complete_work_census_events": 2,
                "complete_spec_rows": 8,
                "spec_drafts": 8,
                "spec_tokens": 248,
                "batch_histogram": {"1": 0, "2": 0, "3": 0, "4": 2},
                "first_forward_step": 0,
                "last_forward_step": 1,
                "events_sha256": "5" * 64,
            },
            "sfwd": {"steps": 2, "drafts": 8},
            "dfwd": {"spans": 2},
            "cfwd": {"spans": 2},
        },
    }
    final_path = tmp_path / "flush.json"
    boundary_path = tmp_path / "boundary.json"
    final_path.write_text(json.dumps(final), encoding="ascii")
    boundary_path.write_text(json.dumps(boundary), encoding="ascii")
    live["boundary_snapshot_sha256"] = hashlib.sha256(
        boundary_path.read_bytes()
    ).hexdigest()
    assert (
        gate._validate_b4_terminal(
            live=live,
            final_flush=final_path,
            boundary_snapshot=boundary_path,
        )["completed_events"]
        == 2
    )

    boundary["metrics"]["fixed32"]["spec_drafts"] = 2
    boundary_path.write_text(json.dumps(boundary), encoding="ascii")
    live["boundary_snapshot_sha256"] = hashlib.sha256(
        boundary_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="B4 terminal"):
        gate._validate_b4_terminal(
            live=live,
            final_flush=final_path,
            boundary_snapshot=boundary_path,
        )


def test_launcher_and_runner_close_exact_b4_runtime_identity() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "gemvx_m4_shuffle_r64_u8_out" in patcher
    assert "_fr13_draft_head_m4_u8_live_depth(4)" not in patcher
    assert '"_fr13_draft_head_m4_u8_live_depth",' in patcher
    assert 'compared_elements": total_compares * 4 * 65536' in patcher
    assert "/tmp/fr13_bf16_k64_m4_r64_u8.abi3.so:ro" in launcher
    assert '"$MAX_NUM_SEQS" == "4"' in launcher
    assert '"${SWE_CONCURRENCY:-}" == "4"' in launcher
    assert "FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256" in launcher
    assert "FR13_RUN_B4_DFWD_M4_U8_LIVE_GATE" in runner
    assert "fr13_bigdenom_swe_serve_variant.sh" in runner
    assert "subset_b4_four.json" in runner
    assert "OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in runner
    assert "performance_measurement" not in runner
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        env={**os.environ, "FR13_RUN_B4_DFWD_M4_U8_LIVE_GATE": "0"},
        check=False,
    )
    assert result.returncode == 2
    assert "disabled" in result.stderr
