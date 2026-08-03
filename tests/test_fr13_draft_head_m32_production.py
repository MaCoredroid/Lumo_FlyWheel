from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SIDECAR_SCRIPT = REPO / "scripts" / "fr13_draft_head_m32_pass.py"
TIMING_RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m32_timing.sh"
LIVE_RUNNER = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
RUNTIME_MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
CONTRACT = REPO / "results" / "fr13_fixed32_draft_head_m32_deployed_contract_20260731" / "contract.json"
INTEGRATION_MANIFEST = (
    REPO
    / "results"
    / "fr13_fixed32_kernel_candidates_integrated_20260801"
    / "manifest.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("draft_head_pass", SIDECAR_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patcher_module():
    spec = importlib.util.spec_from_file_location("draft_head_patcher", PATCHER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_eagle_tree_consumption_verify":
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_M32_PRODUCTION" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head M32 production snippet not found")


def _snippet_functions(*names: str) -> list[ast.FunctionDef]:
    wanted = set(names)
    selected = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(_eagle_snippet())))
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in selected} == wanted
    return selected


def _live(source_sha256: str) -> dict[str, object]:
    module = _module()
    return {
        "schema": module.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": module.EXPECTED_INSTANCE,
        "task_marker": f"swe_verified:{module.EXPECTED_INSTANCE}",
        "concurrency": 1,
        "batch_size": 1,
        "source_commit": "a" * 40,
        "candidate_source_sha256": source_sha256,
        "geometry": module.EXPECTED_GEOMETRY,
        "candidate": module.EXPECTED_CANDIDATE,
        "completed_events": 7,
        "complete_work_census_events": 7,
        "work_census_last_event_index": 6,
        "events_sha256": "c" * 64,
        "flush_generation": 3,
        "flush_nonce": "d" * 64,
        "producer_pid": 257,
        "boundary_snapshot_sha256": "e" * 64,
        "full_logit_comparisons": 35,
        "raw_bf16_mismatches": 0,
        "served_return": "reference BF16 logits unchanged",
        "performance_measurement": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }


def _terminal_evidence(
    tmp_path: Path, live: dict[str, object]
) -> tuple[Path, Path]:
    events = int(live["completed_events"])
    counters = {
        "complete_work_census_events": live["complete_work_census_events"],
        "pure_decode_forward_steps": events,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": events - 1,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    boundary = {
        "schema": "fr13-fixed32-boundary-snapshot-v4",
        "mode": "hydra27_fixed32",
        "action": "final",
        "generation": live["flush_generation"],
        "nonce": live["flush_nonce"],
        "producer_pid": live["producer_pid"],
        "counters": counters,
        "metrics": {
            "fixed32": {
                "complete_work_census_events": live[
                    "complete_work_census_events"
                ],
                "pure_decode_forward_steps": events,
                "spec_drafts": events,
                "complete_spec_rows": events,
                "spec_tokens": events * 31,
                "batch_histogram": {
                    "1": live["completed_events"],
                    "2": 0,
                    "3": 0,
                    "4": 0,
                },
                "first_forward_step": 0,
                "last_forward_step": events - 1,
                "events_sha256": live["events_sha256"],
            },
            "sfwd": {
                "gpu_seconds": 1.0,
                "steps": events,
                "drafts": events,
                "wall_seconds": 1.1,
                "wall_drafts": events - 1,
                "wall_steps": events - 1,
                "wall_rejected": 0,
            },
            "dfwd": {"gpu_seconds": 0.2, "spans": events},
            "cfwd": {"gpu_seconds": 0.1, "spans": events},
            "boot_warm": {},
            "committer": {},
            "conv_pregather": {},
        },
    }
    boundary_path = tmp_path / "boundary.json"
    module = _module()
    boundary_path.write_bytes(module.canonical_bytes(boundary) + b"\n")
    live["boundary_snapshot_sha256"] = module.sha256_file(boundary_path)
    final_flush = {
        "schema": "fr13-fixed32-flush-client-result-v1",
        "ack": {
            "schema": "fr13-fixed32-flush-ack-v1",
            "mode": "hydra27_fixed32",
            "status": "ok",
            "action": "final",
            "generation": live["flush_generation"],
            "nonce": live["flush_nonce"],
            "producer_pid": live["producer_pid"],
            "counters": counters,
        },
    }
    flush_path = tmp_path / "final_flush.json"
    flush_path.write_text(json.dumps(final_flush), encoding="ascii")
    return flush_path, boundary_path


def _traffic_audit(tmp_path: Path, events: int = 7) -> Path:
    module = _module()
    request_sha = _sha(b"request")
    task_set_sha = _sha(b"task-set")
    payload = {
        "schema": "fr13-fixed32-chat-task-provenance-audit-v2",
        "mode": "hydra27_fixed32",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "subset": {
            "sha256": module.EXPECTED_B1_SUBSET_SHA256,
            "task_count": 1,
            "task_ids": [module.EXPECTED_INSTANCE],
        },
        "checks": {key: True for key in module.TRAFFIC_CHECK_KEYS},
        "offload_fetch_status": {
            "path": "/evidence/offload_fetch_status.txt",
            "sha256": _sha(b"ok\n"),
            "bytes": 3,
        },
        "proxy_runtime": {
            "path": "/evidence/offload_proxy_env.txt",
            "sha256": _sha(b"proxy"),
            "bytes": 5,
            "canonical_task_set_sha256": task_set_sha,
            "raw_dump_environment_absent": True,
            "raw_dump_artifacts_absent": True,
        },
        "complete_stream": {
            "pure_decode_forward_steps": events,
            "complete_work_census_events": events,
            "merged_forward_step_intervals": [[0, events]],
        },
        "ingress": {
            "canonical_task_set_sha256": task_set_sha,
            "census": {
                "all_census_requests_authenticated": True,
                "all_census_requests_inside_task_brackets": True,
                "all_successful_requests_present": True,
                "bytes": 1024,
                "event_count": events,
                "event_schema": "fr13-fixed32-work-census-v12",
                "path": "/evidence/fr13_fixed32_work_census.jsonl",
                "per_task_request_step_memberships": {
                    module.EXPECTED_INSTANCE: events
                },
                "request_step_memberships": events,
                "sha256": _sha(b"census"),
                "successful_engine_requests": 1,
                "terminal_schema": "fr13-fixed32-work-census-terminal-v12",
            },
            "engine": {"authenticated": True},
            "exact_proxy_engine_attempt_parity": True,
            "preflight": {"authenticated": True},
            "proxy": {"authenticated": True},
            "zero_campaign_rejections": True,
            "zero_failed_or_aborted_requests": True,
        },
        "tasks": {
            module.EXPECTED_INSTANCE: {
                "task_key_id": _sha(b"task-key"),
                "dataset_record_sha256": module.EXPECTED_DATASET_RECORD_SHA256,
                "trace": {
                    "path": "/evidence/qwen_trace.jsonl",
                    "sha256": _sha(b"trace"),
                    "bytes": 100,
                    "event_count": 2,
                    "completed_logical_model_requests": 1,
                    "model_request_id_sha256s": [request_sha],
                    "model_request_ids_sha256": _sha(b"request-list"),
                },
                "task_auth": {
                    "completed_logical_model_requests": 1,
                    "aborted_logical_requests": 0,
                    "accepted_attempts": 1,
                    "completed_attempts": 1,
                    "failed_attempts": 0,
                    "evidence_before_sha256": _sha(b"before"),
                    "evidence_after_sha256": _sha(b"after"),
                    "evidence_after_ledger_records": 1,
                    "evidence_after_ledger_chain_head_sha256": _sha(b"head"),
                },
                "terminal": {
                    "agent": {
                        "exit_code": 0,
                        "timed_out": False,
                        "offloaded": True,
                        "network_drop": False,
                    },
                    "eval": {
                        "verdict": "resolved",
                        "passed": True,
                        "harness_exit_code": 0,
                    },
                    "eval_artifact": {
                        "path": "/evidence/eval_report.json",
                        "sha256": _sha(b"eval"),
                        "bytes": 100,
                    },
                },
                "boundary": {
                    "path": "/evidence/fixed32_task_boundary.json",
                    "sha256": _sha(b"boundary"),
                    "bytes": 100,
                    "forward_step_interval": [0, events],
                },
            }
        },
    }
    path = tmp_path / "fixed32_chat_traffic_audit.json"
    path.write_bytes(module.canonical_bytes(payload) + b"\n")
    return path


def test_deployed_format_contract_and_production_are_fail_closed() -> None:
    snippet = _eagle_snippet()
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'type(_fr13_dh_sh.quant_method).__name__\n                        != "UnquantizedEmbeddingMethod"' in snippet
    assert "tuple(_fr13_dh_w.shape) != (65536, 5120)" in snippet
    assert "tuple(_fr13_dh_w.stride()) != (5120, 1)" in snippet
    assert "_fr13_dh_batch not in (1, 2, 3, 4)" in snippet
    assert "tuple(_h.shape) != (_fr13_dh_batch, 5120)" in snippet
    assert '"batch_size": 1' in snippet
    assert "tuple(_h.stride()) != (5120, 1)" in snippet
    assert "torch.mm(_fr13_dh_in, _sh.weight.t(), out=_fr13_dh_out)" in snippet
    assert "FR13_DRAFT_HEAD_M32_INTERNAL_PRODUCTION_ATTESTED" in snippet
    assert "FR13 draft-head M32 production failed its strict " in snippet
    assert '"runtime contract"' in snippet
    assert "fallback_calls" in snippet
    assert "_fr13_dh_m32_note_production_replay" in snippet
    assert "observed_measured_replays_at_least" in snippet
    assert "_fr13_draft_head_m32_live_register" in snippet
    assert "_fr13_dh_m32_live_count_enable.fill_" in snippet
    assert "* _fr13_dh_count_enable" in snippet
    assert 'not in ("measured", "unmeasured")' in snippet
    assert 'if _fr13_dh_proposal.get("measured") is not True:' in snippet
    assert 'self._fr13_dh_m32_selected_capture_calls != 0' in snippet

    assert "FR13_DRAFT_HEAD_M32_PRODUCTION=${FR13_DRAFT_HEAD_M32_PRODUCTION:-0}" in launcher
    assert "draft-head M32 live A/B and production are mutually exclusive" in launcher
    assert "fr13_draft_head_m32_pass.py issue" in launcher
    assert '--final-flush "$FR13_DRAFT_HEAD_M32_LIVE_FINAL_FLUSH_JSON"' in launcher
    assert '--boundary-snapshot "$FR13_DRAFT_HEAD_M32_LIVE_BOUNDARY_SNAPSHOT_JSON"' in launcher
    assert "fr13_draft_head_m32_pass.py verify" in launcher
    assert "FR13_DRAFT_HEAD_M32_INTERNAL_PRODUCTION_ATTESTED=1" in launcher
    assert ".lumo.local.env must not override draft-head M32" in launcher
    assert "FR13_DRAFT_HEAD_M32_TIMING_ARM" in launcher
    assert "FR13_DRAFT_HEAD_M32_LIVE_FINAL_FLUSH_JSON" in launcher
    assert "FR13_DRAFT_HEAD_M32_LIVE_BOUNDARY_SNAPSHOT_JSON" in launcher
    assert "FR13_DRAFT_HEAD_M32_LIVE_CHAT_TRAFFIC_AUDIT_JSON" in launcher
    assert '--chat-traffic-audit "$FR13_DRAFT_HEAD_M32_LIVE_CHAT_TRAFFIC_AUDIT_JSON"' in launcher
    assert '${FR13_DRAFT_HEAD_M32_PRODUCTION:-0}' in launcher
    assert '-e FR13_DRAFT_HEAD_M32_PRODUCTION="$FR13_DRAFT_HEAD_M32_PRODUCTION"' in launcher
    assert '"scripts/fr13_draft_head_m32_pass.py"' in RUNTIME_MANIFEST.read_text(
        encoding="utf-8"
    )


def test_live_pass_sidecar_binds_exact_source_and_census(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "patcher.py"
    source.write_bytes(b"exact candidate source")
    source_sha = module.sha256_file(source)
    live_payload = _live(source_sha)
    final_flush, boundary = _terminal_evidence(tmp_path, live_payload)
    traffic_audit = _traffic_audit(tmp_path)
    live = tmp_path / "live.json"
    live.write_bytes(module.canonical_bytes(live_payload) + b"\n")
    live_sha = module.sha256_file(live)
    sidecar = tmp_path / "pass.json"

    issued = module.issue_sidecar(
        live_result=live,
        expected_live_sha256=live_sha,
        final_flush=final_flush,
        boundary_snapshot=boundary,
        chat_traffic_audit=traffic_audit,
        candidate_source=source,
        expected_candidate_source_sha256=source_sha,
        out=sidecar,
    )
    assert issued["status"] == "PASS"
    assert issued["qualified_completed_events"] == 7
    assert issued["qualified_flush_generation"] == 3
    assert issued["final_flush_sha256"] == module.sha256_file(final_flush)
    assert issued["boundary_snapshot_sha256"] == module.sha256_file(boundary)
    assert issued["chat_traffic_audit_sha256"] == module.sha256_file(
        traffic_audit
    )
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    assert module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=module.sha256_file(sidecar),
        expected_live_sha256=live_sha,
        candidate_source=source,
        expected_candidate_source_sha256=source_sha,
    ) == issued

    bad = _live(source_sha)
    bad["raw_bf16_mismatches"] = 1
    with pytest.raises(ValueError, match="comparison census drifted"):
        module.validate_live_result(bad, expected_source_sha256=source_sha)
    bad = _live(source_sha)
    bad["full_logit_comparisons"] = 34
    with pytest.raises(ValueError, match="comparison census drifted"):
        module.validate_live_result(bad, expected_source_sha256=source_sha)


def test_production_engagement_requires_root_plus_four_captured_heads(tmp_path: Path) -> None:
    module = _module()
    source_sha = _sha(b"source")
    sidecar_sha = _sha(b"sidecar")
    payload = {
        "schema": module.ENGAGEMENT_SCHEMA,
        "status": "ENGAGED",
        "source_commit": "b" * 40,
        "candidate_source_sha256": source_sha,
        "production_pass_sidecar_sha256": sidecar_sha,
        "geometry": module.EXPECTED_GEOMETRY,
        "candidate": module.EXPECTED_CANDIDATE,
        "selected_root_calls": 1,
        "captured_loop_calls": 4,
        "fallback_calls": 0,
        "drafter_graph_id": 123,
        "drafter_graph_signature": module.EXPECTED_GRAPH_SIGNATURE,
        "observed_measured_replays_at_least": 1,
        "capture_origin": "unmeasured",
        "execution_basis": "cudagraph_replay",
        "forward_step_index": 0,
        "runtime_mode": "FULL",
    }
    engagement = tmp_path / "engagement.json"
    engagement.write_text(json.dumps(payload), encoding="ascii")
    assert module.validate_engagement(
        engagement_path=engagement,
        expected_source_sha256=source_sha,
        expected_sidecar_sha256=sidecar_sha,
    ) == payload
    payload["captured_loop_calls"] = 3
    engagement.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="engagement drifted"):
        module.validate_engagement(
            engagement_path=engagement,
            expected_source_sha256=source_sha,
            expected_sidecar_sha256=sidecar_sha,
        )


def test_unmeasured_capture_attests_then_measured_replay_engages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    graph_id = 123
    signature = module.EXPECTED_GRAPH_SIGNATURE
    source_sha = _sha(b"source")
    sidecar_sha = _sha(b"sidecar")
    state = SimpleNamespace(
        _fr13_dh_m32_production_active=True,
        _fr13_dh_m32_engagement_written=False,
        _fr13_dh_m32_selected_root_calls=0,
        _fr13_dh_m32_selected_capture_calls=0,
        _fr13_dh_m32_fallback_calls=0,
        _fr13_dh_m32_graph_attestation=None,
    )
    proposal = {
        "batch_size": 1,
        "mode": "hydra27_fixed32",
        "graph_id": graph_id,
        "graph_signature": signature,
        "graph_replays": 1,
        "measured": False,
        "forward_step_index": 0,
    }
    lifecycle = {
        "captures": 1,
        "batch_size": 1,
        "graph_signature": signature,
        "capture_origin": "unmeasured",
        "measured_replays": 0,
    }
    gdn = types.ModuleType("gdn_linear_attn")
    gdn._FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = proposal
    gdn._FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE = {graph_id: lifecycle}
    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    packages["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    for name, package in packages.items():
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setenv("FR13_DRAFT_HEAD_M32_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv(
        "FR13_DRAFT_HEAD_M32_PRODUCTION_PASS_SIDECAR_SHA256", sidecar_sha
    )

    writes: list[dict[str, object]] = []
    namespace = {
        "self": state,
        "os": os,
        "_fr13_dh_source_sha": source_sha,
        "_fr13_dh_m32_contract": lambda: {
            "geometry": module.EXPECTED_GEOMETRY,
            "candidate": module.EXPECTED_CANDIDATE,
        },
        "_fr13_dh_m32_atomic_json": lambda _path, payload: writes.append(
            payload
        ),
    }
    exec(
        compile(
            ast.Module(
                body=_snippet_functions(
                    "_fr13_dh_m32_note_production",
                    "_fr13_dh_m32_note_production_replay",
                ),
                type_ignores=[],
            ),
            "<m32-production-lifecycle>",
            "exec",
        ),
        namespace,
    )
    note = namespace["_fr13_dh_m32_note_production"]
    replay = namespace["_fr13_dh_m32_note_production_replay"]
    note(False)
    for _ in range(4):
        note(True)
    replay(graph_id, signature, 1)
    assert writes == []
    assert state._fr13_dh_m32_graph_attestation == {
        "graph_id": graph_id,
        "graph_signature": signature,
        "capture_origin": "unmeasured",
    }
    assert state._fr13_dh_m32_selected_root_calls == 0
    assert state._fr13_dh_m32_selected_capture_calls == 0

    proposal["measured"] = True
    proposal["forward_step_index"] = 1
    lifecycle["measured_replays"] = 1
    note(False)
    replay(graph_id, signature, 1)
    assert len(writes) == 1
    engagement_path = tmp_path / "engagement.json"
    engagement_path.write_bytes(module.canonical_bytes(writes[0]) + b"\n")
    assert module.validate_engagement(
        engagement_path=engagement_path,
        expected_source_sha256=source_sha,
        expected_sidecar_sha256=sidecar_sha,
    ) == writes[0]


def test_timing_runner_is_exact4_b1_full_wall_and_credentialed() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")

    assert "config/fr13_fixed32/subset_b4_four.json" in text
    assert "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5" in text
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in text
    assert "fr13_draft_head_m32_pass.py validate-live" in text
    assert "LIVE_FINAL_FLUSH_JSON" in text
    assert "LIVE_BOUNDARY_SNAPSHOT_JSON" in text
    assert "LIVE_CHAT_TRAFFIC_AUDIT_JSON" in text
    assert "FR13_DRAFT_HEAD_M32_LIVE_FINAL_FLUSH_JSON" in text
    assert "FR13_DRAFT_HEAD_M32_LIVE_BOUNDARY_SNAPSHOT_JSON" in text
    assert "FR13_DRAFT_HEAD_M32_LIVE_CHAT_TRAFFIC_AUDIT_JSON" in text
    assert "FR13_DRAFT_HEAD_M32_TIMING_ARM=1" in text
    assert 'FR13_DRAFT_HEAD_M32_PRODUCTION="$production"' in text
    assert "scripts/fr13_measure.py deploy-speed" in text
    assert 'record.get("batch_size") != 1' in text
    assert 'record.get("n_tasks") != 4' in text
    assert '"measured_tps_fullstep_wall"' in text
    assert '"step_wall_ms"' in text
    assert '"committed_per_event"' in text
    assert '"wall_steps_measured"' in text
    assert '"events_per_step"' in text
    assert '"mandatory_weight_bytes"' in text
    assert '"weight_floor_bandwidth_bytes_per_s"' in text
    assert '"raw_sha256"' in text
    assert 'sidecar.get("final_flush_sha256")' in text
    assert 'sidecar.get("boundary_snapshot_sha256")' in text
    assert 'sidecar.get("chat_traffic_audit_sha256")' in text
    assert "MIN_RETAINED_WALL_FRACTION = 0.99" in text
    assert "MIN_TASK_COUNTER_STEPS = 64" in text
    assert 'record.get("raw_counter_delta_aggregate")' in text
    assert "floor_acceptance_eligible=0" in text


def test_live_gate_uses_real_swe_b1_and_serves_reference() -> None:
    text = LIVE_RUNNER.read_text(encoding="utf-8")

    assert "FR13_GATE_DRAFT_HEAD_M32=${FR13_GATE_DRAFT_HEAD_M32:-0}" in text
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in text
    assert "astropy__astropy-12907" in text
    assert 'FR13_DRAFT_HEAD_M32_LIVE_AB="$FR13_GATE_DRAFT_HEAD_M32"' in text
    assert "fr13_draft_head_m32_pass.py validate-live" in text
    assert "DRAFT_HEAD_FINAL_FLUSH" in text
    assert "DRAFT_HEAD_BOUNDARY" in text
    assert "DRAFT_HEAD_TRAFFIC_AUDIT" in text


def test_live_pass_reconciles_terminal_flush_and_boundary(tmp_path: Path) -> None:
    module = _module()
    source_sha = _sha(b"source")
    live = _live(source_sha)
    flush_path, boundary_path = _terminal_evidence(tmp_path, live)
    assert module.validate_live_evidence(
        live_payload=live,
        final_flush_path=flush_path,
        boundary_snapshot_path=boundary_path,
    )["completed_events"] == 7
    boundary = json.loads(boundary_path.read_text(encoding="ascii"))
    boundary["metrics"]["fixed32"]["spec_drafts"] = 6
    boundary_path.write_bytes(module.canonical_bytes(boundary) + b"\n")
    live["boundary_snapshot_sha256"] = module.sha256_file(boundary_path)
    with pytest.raises(ValueError, match="terminal flush evidence drifted"):
        module.validate_live_evidence(
            live_payload=live,
            final_flush_path=flush_path,
            boundary_snapshot_path=boundary_path,
        )


def test_live_pass_rejects_unclosed_census_and_wrong_mode(tmp_path: Path) -> None:
    module = _module()
    live = _live(_sha(b"source"))
    flush_path, boundary_path = _terminal_evidence(tmp_path, live)
    final_flush = json.loads(flush_path.read_text(encoding="ascii"))
    final_flush["ack"]["counters"]["sfwd_pending"] = 1
    flush_path.write_text(json.dumps(final_flush), encoding="ascii")
    with pytest.raises(ValueError, match="exact closed B1 event census"):
        module.validate_live_evidence(
            live_payload=live,
            final_flush_path=flush_path,
            boundary_snapshot_path=boundary_path,
        )

    flush_path, boundary_path = _terminal_evidence(tmp_path, live)
    boundary = json.loads(boundary_path.read_text(encoding="ascii"))
    boundary["mode"] = "tail6_fixed32"
    boundary_path.write_bytes(module.canonical_bytes(boundary) + b"\n")
    live["boundary_snapshot_sha256"] = module.sha256_file(boundary_path)
    with pytest.raises(ValueError, match="terminal flush evidence drifted"):
        module.validate_live_evidence(
            live_payload=live,
            final_flush_path=flush_path,
            boundary_snapshot_path=boundary_path,
        )


def test_live_pass_rejects_traffic_audit_stream_mismatch(tmp_path: Path) -> None:
    module = _module()
    traffic_audit = _traffic_audit(tmp_path, events=6)
    with pytest.raises(ValueError, match="provenance drifted"):
        module.validate_chat_traffic_audit(
            audit_path=traffic_audit,
            expected_events=7,
        )


def test_runtime_live_result_is_written_only_from_exact_final_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _patcher_module()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_draft_head_m32_live_finalize"
    ]
    assert len(selected) == 1
    out = tmp_path / "live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_M32_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M32_LIVE_JSON", str(out))

    class Counter:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return self.values

    namespace = {
        "_FR13_DRAFT_HEAD_M32_LIVE_STATE": {
            "compares": Counter([35, 0, 0]),
            "mismatches": Counter([0, 0, 0]),
            "geometry": _module().EXPECTED_GEOMETRY,
            "candidate": _module().EXPECTED_CANDIDATE,
            "source_commit": "a" * 40,
            "candidate_source_sha256": "b" * 64,
            "instance_id": _module().EXPECTED_INSTANCE,
        }
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<m32-live>", "exec"),
        namespace,
    )
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "c" * 64,
        "complete_work_census_events": 7,
        "events_sha256": "d" * 64,
        "generation": 3,
        "nonce": "e" * 64,
        "producer_pid": 257,
    }
    namespace["_fr13_draft_head_m32_live_finalize"](
        [{"batch_size": 1} for _ in range(7)], binding
    )
    payload = json.loads(out.read_text(encoding="ascii"))
    assert payload["status"] == "PASS"
    assert payload["completed_events"] == 7
    assert payload["full_logit_comparisons"] == 35
    assert payload["flush_action"] == "final"

    namespace["_FR13_DRAFT_HEAD_M32_LIVE_STATE"]["compares"].values[0] = 34
    with pytest.raises(RuntimeError, match="comparison/event census mismatch"):
        namespace["_fr13_draft_head_m32_live_finalize"](
            [{"batch_size": 1} for _ in range(7)], binding
        )
    assert json.loads(out.read_text(encoding="ascii"))["status"] == "FAIL"


def test_published_contract_matches_candidate_source_and_exact_math() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="ascii"))
    integration = json.loads(INTEGRATION_MANIFEST.read_text(encoding="ascii"))

    assert payload["performance_claim"] is False
    assert payload["byte_equality_claim"] is False
    assert payload["live_b1_reference_contract"]["gemm_mnk"] == [1, 65536, 5120]
    assert payload["live_b1_reference_contract"]["calls_per_event"] == 5
    assert payload["replacement"]["candidate_gemm_mnk"] == [32, 65536, 5120]
    assert integration["components"]["draft_head_m32"][
        "candidate_source_sha256"
    ] == hashlib.sha256(
        PATCHER.read_bytes()
    ).hexdigest()
    assert integration["components"]["draft_head_m32"]["source_commit"] == (
        "3b06acebbd673466703268bf0b3647f4bf4a3070"
    )
    assert payload["replacement"]["candidate_source_sha256"] == (
        "0ecd359c7ffb211f0212db3a83baabfff327c07286fd808ea99ac68d536798e2"
    )
    assert payload["roofline"]["weight_bytes_per_event"] == 5 * 65536 * 5120 * 2
    assert payload["modeled_hypothesis"]["measurement"] is False
    assert payload["current_floor_gap_context"]["modeled_recovery_fraction_of_gap"] == pytest.approx(
        payload["modeled_hypothesis"]["recovery_ms_per_event"]
        / payload["current_floor_gap_context"]["gap_to_cap_ms"],
        abs=1e-15,
    )
