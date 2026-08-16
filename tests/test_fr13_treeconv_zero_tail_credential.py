from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fr13_fixed32_contract as contract  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402
import fr13_fixed32_work_census as work_census  # noqa: E402
import fr13_runtime_manifest as runtime_manifest  # noqa: E402
import fr13_treeconv_zero_tail_credential as credential  # noqa: E402
from lumo_flywheel_serving import inference_proxy  # noqa: E402


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        raw = _canonical(value) + b"\n"
    else:
        raw = (
            json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
    path.write_bytes(raw)


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _qwen_trace(instance_id: str) -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(instance_id)
    response_id = f"final-{instance_id}"
    return [
        {
            "type": "system",
            "subtype": "init",
            "qwen_code_version": "0.19.4",
            "uuid": f"system-{instance_id}",
            "session_id": session_id,
            "parent_tool_use_id": None,
        },
        {
            "type": "assistant",
            "uuid": response_id,
            "session_id": session_id,
            "parent_tool_use_id": None,
            "message": {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": "qwen3.8-27b-nvfp4-radixark",
                "content": [{"type": "text", "text": "complete"}],
                "stop_reason": None,
                "usage": {"input_tokens": 32, "output_tokens": 8},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "uuid": f"result-{instance_id}",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 100,
            "duration_api_ms": 90,
            "num_turns": 1,
            "result": "complete",
            "usage": {
                "input_tokens": 32,
                "output_tokens": 8,
                "total_tokens": 40,
            },
            "permission_denials": [],
        },
    ]


def _metrics(completed: int) -> bytes:
    labels = 'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"'
    lines = [
        f"vllm:prompt_tokens_total{{{labels}}} {completed * 32}",
        f"vllm:generation_tokens_total{{{labels}}} {completed * 8}",
        f"vllm:request_params_max_tokens_count{{{labels}}} {completed}",
        (
            f"vllm:request_params_max_tokens_sum{{{labels}}} "
            f"{completed * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS}"
        ),
    ]
    for reason in ("stop", "length", "abort", "error", "repetition"):
        reason_labels = (
            f'engine="0",finished_reason="{reason}",'
            'model_name="qwen3.8-27b-nvfp4-radixark"'
        )
        lines.append(
            f"vllm:request_success_total{{{reason_labels}}} "
            f"{completed if reason == 'stop' else 0}"
        )
    for le, count in (
        ("10000.0", 0),
        ("20000.0", 0),
        ("50000.0", completed),
        ("+Inf", completed),
    ):
        bucket_labels = f'engine="0",le="{le}",model_name="qwen3.8-27b-nvfp4-radixark"'
        lines.append(
            f"vllm:request_params_max_tokens_bucket{{{bucket_labels}}} {count}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _task_metrics(batch: int, step: int, qwen_completed: int) -> bytes:
    timing = floor_gate.fixture_metrics(
        [1.0], [2.0], [batch], step, 31
    ).encode("ascii")
    return timing + _metrics(qwen_completed)


def _runtime_snapshot_metrics(
    *, mode: str, batch: int, event: dict[str, Any], step: int
) -> dict[str, Any]:
    histogram = {
        str(candidate): int(step == 1 and candidate == batch)
        for candidate in range(1, 5)
    }
    spec_drafts = batch * step
    zero_by_batch = {str(candidate): 0 for candidate in range(1, 5)}
    capture_by_batch = {
        str(candidate): int(candidate <= batch) for candidate in range(1, 5)
    }
    ready_capacities = {
        str(candidate): batch for candidate in range(1, batch + 1)
    }
    full_coverage = {
        str(candidate): 0x0FFF for candidate in range(1, batch + 1)
    }
    return {
        "fixed32": {
            "pure_decode_forward_steps": step,
            "complete_work_census_events": step,
            "complete_spec_rows": spec_drafts,
            "spec_drafts": spec_drafts,
            "spec_tokens": 31 * spec_drafts,
            "batch_histogram": histogram,
            "first_forward_step": 0 if step else None,
            "last_forward_step": step - 1 if step else None,
            "events_sha256": hashlib.sha256(
                _canonical([event] if step else [])
            ).hexdigest(),
        },
        "sfwd": {
            "gpu_seconds": 0.001 * step,
            "steps": step,
            "drafts": spec_drafts,
            "wall_seconds": 0.002 * step,
            "wall_drafts": spec_drafts,
            "wall_steps": step,
            "wall_rejected": 0,
        },
        "dfwd": {"gpu_seconds": 0.001 * step, "spans": step},
        "cfwd": {"gpu_seconds": 0.002 * step, "spans": step},
        "boot_warm": {
            "schema": "fr13-fixed32-boot-warm-v3",
            "classification": "unmeasured_boot",
            "hardware_scope": "device_postprocess_kernels",
            "wrapper_bookkeeping_warmed": False,
            "copy_source_dtype": "torch.int64",
            "copy_destination_dtype": "torch.int32",
            "mode": mode,
            "capacity": batch,
            "vocab_size": 248320,
            "batches": list(range(1, batch + 1)),
            "taw_executions": batch,
            "output_copy_pairs": batch,
            "slot_copy_pairs": batch * (batch + 1) // 2,
            "spec_copy_pairs": batch,
            "flags_zero_fills": 1,
            "persistent_copy_state_restored": True,
            "flags_state_restored": True,
            "conv_commit_direct_launches": batch,
            "conv_commit_gather_launches": 0,
            "conv_commit_scatter_launches": 0,
            "committer_replays": batch,
            "observed_event_absent": True,
            "pending_event_absent": True,
            "taw_cache_lease_current": True,
            "taw_rng_state_restored": True,
            "taw_staging_state_restored": True,
            "taw_measured_state_restored": True,
            "committer_route_lease_current": True,
            "committer_bank_state_restored": True,
            "committer_conv_bank_state_restored": True,
            "committer_conv_staging_state_restored": True,
            "committer_alias_destination_contract": "exact_alias_only_16x3",
            "committer_input_state_restored": True,
            "committer_measured_state_restored": True,
            "committer_scratch_overwrite_proven": True,
        },
        "committer": {
            "actual_replays_by_batch": dict(histogram),
            "actual_replays_enqueued": step,
            "all_batches_ready": True,
            "captures": batch,
            "fast_route_ready": True,
            "layer_batch_gate_attempts_by_batch": {
                str(candidate): 0 for candidate in range(1, batch + 1)
            },
            "layer_batch_gate_coverage_mask_by_batch": full_coverage,
            "layer_batch_gate_passed_by_batch": {
                str(candidate): 1 for candidate in range(1, batch + 1)
            },
            "maximum_ready_capacity": batch,
            "nonpure_committer_replays_by_batch": zero_by_batch,
            "nonpure_committer_replays_enqueued": 0,
            "nonpure_dispatch": {
                "guarded_steps": 0,
                "piecewise_steps": 0,
                "none_steps": 0,
                "forbidden_full_steps": 0,
            },
            "preseeded_batches": list(range(1, batch + 1)),
            "preseeded_graphs": batch,
            "ready_capacities": ready_capacities,
            "required_capacity": batch,
        },
        "conv_pregather": {
            "actual_stages": 0,
            "actual_stages_by_batch": zero_by_batch,
            "aux_capture_stages": 0,
            "graph_capture_stages": batch,
            "graph_capture_stages_by_batch": capture_by_batch,
            "graph_replay_stages": step,
            "graph_replay_stages_by_batch": dict(histogram),
            "max_batch_size": batch,
            "pointer_entries": 48,
            "preseeded": True,
            "preseeded_batches": list(range(1, batch + 1)),
            "profile_capture_stages": 0,
        },
    }


def _append(
    rows: list[dict[str, Any]],
    *,
    role: str,
    phase: str,
    event: str,
    outcome: str,
    route: str | None = None,
    task_key_id: str | None = None,
    logical_id_sha256: str | None = None,
    wire_id_sha256: str | None = None,
    engine_request_id_sha256: str | None = None,
    status_code: int | None = None,
    reason: str | None = None,
    evidence_sha256: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema": inference_proxy.FIXED32_INGRESS_LEDGER_SCHEMA,
        "seq": len(rows),
        "role": role,
        "phase": phase,
        "event": event,
        "route": route,
        "task_key_id": task_key_id,
        "logical_id_sha256": logical_id_sha256,
        "wire_id_sha256": wire_id_sha256,
        "engine_request_id_sha256": engine_request_id_sha256,
        "status_code": status_code,
        "outcome": outcome,
        "reason": reason,
        "evidence_sha256": evidence_sha256,
        "prev_sha256": rows[-1]["record_sha256"] if rows else "0" * 64,
    }
    row["record_sha256"] = hashlib.sha256(_canonical(row)).hexdigest()
    rows.append(row)
    return row


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _runtime_repo(
    tmp_path: Path, *, batch: int
) -> tuple[Path, str, Path, Path, Path]:
    repo = tmp_path / "repo"
    spec = runtime_manifest.PROFILES["fixed32"]
    paths = {
        *spec.host_script_source,
        *spec.python_package_source,
        *spec.runtime_data_and_config,
        *spec.verdict_tools,
        credential.SEQUENCE,
    }
    for relative in sorted(paths):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "-f", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "canonical runtime"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = runtime_manifest.build_manifest(
        repo, profile="fixed32", sequence=credential.SEQUENCE
    )
    launch_manifest_path = tmp_path / "runtime_manifest.at_launch.json"
    end_manifest_path = tmp_path / "runtime_manifest.at_end.json"
    _write_json(launch_manifest_path, manifest)
    _write_json(end_manifest_path, manifest)
    subset = repo / credential.CANONICAL_SUBSETS[batch]["path"]
    return repo, commit, subset, launch_manifest_path, end_manifest_path


def _make_fixture(tmp_path: Path, *, batch: int) -> dict[str, Any]:
    mode = "tail6_fixed32" if batch == 1 else "hydra27_fixed32"
    tasks = list(credential.CANONICAL_SUBSETS[batch]["task_ids"])
    run = tmp_path / "run"
    task_root = run / "swe_out" / "verified" / "per_task"
    logs = run / "logs"
    logs.mkdir(parents=True)
    task_keys = {
        task: inference_proxy.fixed32_task_key_id(task) for task in tasks
    }
    task_set_sha = inference_proxy.fixed32_canonical_task_set_sha256(tuple(tasks))

    proxy_rows: list[dict[str, Any]] = []
    engine_rows: list[dict[str, Any]] = []
    for role, rows, alternate_reason in (
        ("proxy", proxy_rows, "malformed_bearer"),
        ("engine", engine_rows, "invalid_engine_bearer"),
    ):
        for route in ("chat", "responses"):
            for reason in ("missing_bearer", alternate_reason):
                _append(
                    rows,
                    role=role,
                    phase="preflight",
                    event="request_rejected",
                    route=route,
                    outcome="rejected",
                    reason=reason,
                )
        _append(
            rows,
            role=role,
            phase="preflight",
            event="campaign_begin",
            outcome="begun",
            evidence_sha256=task_set_sha,
        )

    engine_ids: dict[str, str] = {}
    task_auth: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        task_key = task_keys[task]
        before_payload = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key,
            "completed_logical_model_requests": 0,
            "aborted_logical_requests": 0,
            "accepted_attempts": 0,
            "completed_attempts": 0,
            "failed_attempts": 0,
            "phase": "campaign",
            "ledger_records": len(proxy_rows),
            "ledger_chain_head_sha256": proxy_rows[-1]["record_sha256"],
        }
        engine_id = f"chatcmpl-treeconv-{index}-{task}"
        engine_ids[task] = engine_id
        engine_digest = hashlib.sha256(engine_id.encode()).hexdigest()
        wire_digest = hashlib.sha256(f"wire-{task}".encode()).hexdigest()
        logical_digest = hashlib.sha256(f"logical-{task}".encode()).hexdigest()
        evidence_digest = hashlib.sha256(f"evidence-{task}".encode()).hexdigest()
        _append(
            proxy_rows,
            role="proxy",
            phase="campaign",
            event="logical_begin",
            route="chat",
            task_key_id=task_key,
            logical_id_sha256=logical_digest,
            outcome="accepted",
        )
        _append(
            proxy_rows,
            role="proxy",
            phase="campaign",
            event="attempt_begin",
            route="chat",
            task_key_id=task_key,
            logical_id_sha256=logical_digest,
            wire_id_sha256=wire_digest,
            engine_request_id_sha256=engine_digest,
            outcome="dispatched",
            evidence_sha256=evidence_digest,
        )
        _append(
            engine_rows,
            role="engine",
            phase="campaign",
            event="request_accepted",
            route="chat",
            task_key_id=task_key,
            wire_id_sha256=wire_digest,
            engine_request_id_sha256=engine_digest,
            outcome="accepted",
            evidence_sha256=evidence_digest,
        )
        _append(
            engine_rows,
            role="engine",
            phase="campaign",
            event="request_complete",
            route="chat",
            task_key_id=task_key,
            wire_id_sha256=wire_digest,
            engine_request_id_sha256=engine_digest,
            outcome="completed",
            evidence_sha256=evidence_digest,
        )
        _append(
            proxy_rows,
            role="proxy",
            phase="campaign",
            event="attempt_result",
            route="chat",
            task_key_id=task_key,
            logical_id_sha256=logical_digest,
            wire_id_sha256=wire_digest,
            engine_request_id_sha256=engine_digest,
            status_code=200,
            outcome="response",
            evidence_sha256=evidence_digest,
        )
        _append(
            proxy_rows,
            role="proxy",
            phase="campaign",
            event="logical_complete",
            route="chat",
            task_key_id=task_key,
            logical_id_sha256=logical_digest,
            outcome="completed",
        )
        evidence = {
            "completed_logical_model_requests": 1,
            "aborted_logical_requests": 0,
            "accepted_attempts": 1,
            "completed_attempts": 1,
            "failed_attempts": 0,
        }
        after_payload = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key,
            **evidence,
            "phase": "campaign",
            "ledger_records": len(proxy_rows),
            "ledger_chain_head_sha256": proxy_rows[-1]["record_sha256"],
        }
        task_auth[task] = {
            **evidence,
            "task_auth_evidence_before_sha256": hashlib.sha256(
                _canonical(before_payload)
            ).hexdigest(),
            "task_auth_evidence_after_sha256": hashlib.sha256(
                _canonical(after_payload)
            ).hexdigest(),
            "task_auth_evidence_after_ledger_records": len(proxy_rows),
            "task_auth_evidence_after_ledger_chain_head_sha256": proxy_rows[-1][
                "record_sha256"
            ],
        }
    for role, rows in (("proxy", proxy_rows), ("engine", engine_rows)):
        _append(
            rows,
            role=role,
            phase="campaign",
            event="campaign_finalize",
            outcome="finalized",
            evidence_sha256=task_set_sha,
        )
    proxy_path = logs / "fr13_fixed32_proxy_ingress.jsonl"
    engine_path = logs / "fr13_fixed32_engine_ingress.jsonl"
    _write_ledger(proxy_path, proxy_rows)
    _write_ledger(engine_path, engine_rows)

    event = work_census.reference_event(
        mode,
        batch,
        "treeconv-event-0",
        event_index=0,
        forward_step_index=0,
        request_ids=[engine_ids[task] for task in tasks],
    )
    terminal = work_census.reference_terminal_summary(
        [event], fixture_synthetic_runtime_proof=True
    )
    work_path = logs / "fr13_fixed32_work_census.jsonl"
    work_path.write_bytes(_canonical(event) + b"\n" + _canonical(terminal) + b"\n")

    producer_pid = event["producer_pid"]
    generation = 2 * len(tasks) + 1
    nonce = f"{generation:064x}"
    counters = {
        "pure_decode_forward_steps": 1,
        "complete_work_census_events": 1,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": 0,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    ready_ack = {
        "schema": credential.FLUSH_ACK_SCHEMA,
        "mode": mode,
        "producer_pid": producer_pid,
        "generation": 0,
        "nonce": credential.FLUSH_READY_NONCE,
        "action": "ready",
        "status": "ok",
        "counters": {
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
            "work_census_first_forward_step": None,
            "work_census_last_forward_step": None,
            "sfwd_pending": 0,
            "dfwd_pending": 0,
            "cfwd_pending": 0,
        },
    }
    _write_json(run / "fixed32_ready_ack.json", ready_ack)
    boundary_base = logs / "fr13_fixed32_boundary_snapshot"
    boundary_path = Path(str(boundary_base) + f".{generation}.json")
    _write_json(
        boundary_path,
        {
            "schema": credential.BOUNDARY_SCHEMA,
            "mode": mode,
            "producer_pid": producer_pid,
            "generation": generation,
            "nonce": nonce,
            "action": "final",
            "counters": counters,
            "metrics": _runtime_snapshot_metrics(
                mode=mode,
                batch=batch,
                event=event,
                step=1,
            ),
        },
        canonical=True,
    )
    flush_path = run / "fixed32_final_flush.json"
    final_ack = {
        "schema": credential.FLUSH_ACK_SCHEMA,
        "mode": mode,
        "producer_pid": producer_pid,
        "generation": generation,
        "nonce": nonce,
        "action": "final",
        "status": "ok",
        "counters": counters,
    }
    _write_json(
        flush_path,
        {
            "schema": credential.FLUSH_RESULT_SCHEMA,
            "ack": final_ack,
        },
        canonical=True,
    )
    _write_json(logs / "fr13_fixed32_flush_ack.json", final_ack, canonical=True)
    _write_json(
        logs / "fr13_fixed32_flush_request.json",
        {
            "schema": credential.FLUSH_REQUEST_SCHEMA,
            "mode": mode,
            "producer_pid": producer_pid,
            "prev_generation": generation - 1,
            "generation": generation,
            "nonce": nonce,
            "action": "final",
        },
        canonical=True,
    )

    descriptor = credential._topology_descriptor(mode)
    comparison = {
        "schema": credential.RECORD_SCHEMA,
        "mode": mode,
        "event_id": event["event_id"],
        "event_index": 0,
        "forward_step_index": 0,
        "producer_pid": producer_pid,
        "batch_size": batch,
        "request_ids_sha256": event["drafter_runtime"]["request_ids_sha256"],
        "request_id_sha256s": event["drafter_runtime"]["request_id_sha256s"],
        "execution_basis": "cudagraph_full_replay",
        "topology": descriptor,
        "conv_layers": 48,
        "conv_channels": 10240,
        "conv_state_length": 34,
        "source_rows_per_request": 36,
        "candidate_zero_tail": True,
        "reference_zero_tail": False,
        "reference_restored_and_served": True,
        "raw_bf16_byte_comparison": True,
        "compared_bytes": batch * 48 * 10240 * 34 * 2,
        "differing_bytes": 0,
        "byte_equal": True,
        "timing_eligible": False,
    }
    comparison_terminal = {
        "schema": credential.TERMINAL_SCHEMA,
        "status": "PASS",
        "mode": mode,
        "topology": descriptor,
        "complete_work_census_events": 1,
        "first_event_index": 0,
        "last_event_index": 0,
        "first_forward_step_index": 0,
        "last_forward_step_index": 0,
        "producer_pid": producer_pid,
        "counted_graph_replays": 1,
        "total_compared_bytes": comparison["compared_bytes"],
        "total_differing_bytes": 0,
        "comparison_records_sha256": hashlib.sha256(
            _canonical([comparison])
        ).hexdigest(),
        "work_census_events_sha256": terminal["events_sha256"],
        "flush_generation": generation,
        "flush_nonce": nonce,
        "boundary_snapshot_sha256": hashlib.sha256(
            boundary_path.read_bytes()
        ).hexdigest(),
        "flush_action": "final",
        "finalized_by_fixed32_flush": True,
        "reference_always_served": True,
        "timing_eligible": False,
    }
    comparator_path = logs / "fr13_fixed32_treeconv_zero_tail.byte_ab.jsonl"
    comparator_path.write_bytes(
        _canonical(comparison) + b"\n" + _canonical(comparison_terminal) + b"\n"
    )

    trace_events: dict[str, list[dict[str, Any]]] = {}
    trace_paths: dict[str, Path] = {}
    boundaries: dict[str, dict[str, Any]] = {}
    contract_tasks = []
    for index, task in enumerate(tasks):
        task_dir = task_root / task
        trace_path = task_dir / "qwen_trace.jsonl"
        trace = _qwen_trace(task)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in trace))
        trace_events[task] = trace
        trace_paths[task] = trace_path
        (task_dir / "vllm_metrics_pre.txt").write_bytes(
            _task_metrics(batch, 0, 0)
        )
        (task_dir / "vllm_metrics_post.txt").write_bytes(
            _task_metrics(batch, 1, 1)
        )
        pre_ack = {
            "schema": credential.FLUSH_ACK_SCHEMA,
            "mode": mode,
            "producer_pid": producer_pid,
            "generation": index + 1,
            "nonce": f"{index + 1:064x}",
            "action": "snapshot",
            "status": "ok",
            "counters": {
                "pure_decode_forward_steps": 0,
                "complete_work_census_events": 0,
                "work_census_first_forward_step": None,
                "work_census_last_forward_step": None,
                "sfwd_pending": 0,
                "dfwd_pending": 0,
                "cfwd_pending": 0,
            },
        }
        post_ack = {
            **pre_ack,
            "generation": len(tasks) + index + 1,
            "nonce": f"{len(tasks) + index + 1:064x}",
            "counters": {
                "pure_decode_forward_steps": 1,
                "complete_work_census_events": 1,
                "work_census_first_forward_step": 0,
                "work_census_last_forward_step": 0,
                "sfwd_pending": 0,
                "dfwd_pending": 0,
                "cfwd_pending": 0,
            },
        }
        snapshot_refs: dict[str, dict[str, object]] = {}
        for label, ack in (("pre", pre_ack), ("post", post_ack)):
            snapshot_path = Path(
                str(boundary_base) + f".{ack['generation']}.json"
            )
            _write_json(
                snapshot_path,
                {
                    "schema": credential.BOUNDARY_SCHEMA,
                    "mode": mode,
                    "producer_pid": producer_pid,
                    "generation": ack["generation"],
                    "nonce": ack["nonce"],
                    "action": "snapshot",
                    "counters": ack["counters"],
                    "metrics": _runtime_snapshot_metrics(
                        mode=mode,
                        batch=batch,
                        event=event,
                        step=ack["counters"]["pure_decode_forward_steps"],
                    ),
                },
                canonical=True,
            )
            snapshot_refs[label] = {
                "schema": credential.BOUNDARY_SCHEMA,
                "generation": ack["generation"],
                "path": str(snapshot_path),
                "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            }
        boundary = {
            "schema": credential.TASK_BOUNDARY_SCHEMA,
            "instance_id": task,
            "mode": mode,
            "producer_pid": producer_pid,
            "pre": pre_ack,
            "post": post_ack,
            "pre_runtime_snapshot": snapshot_refs["pre"],
            "post_runtime_snapshot": snapshot_refs["post"],
            "forward_step_interval": {
                "start_forward_step": 0,
                "end_forward_step": 1,
                "expected_complete_events": 1,
            },
        }
        boundaries[task] = boundary
        _write_json(task_dir / "fixed32_task_boundary.json", boundary)
        contract_tasks.append(
            {
                "instance_id": task,
                "expected_session_id": contract.fixed32_trace_session_id(task),
                "expected_completed_logical_model_requests": 1,
                "events": trace,
                "budget_capped": False,
            }
        )

    dataset_root = task_root.parent
    qwen_path: Path | None = None
    campaign_identity: dict[str, object] | None = None
    campaign_replay: dict[str, dict[str, Any]] = {}
    if batch == 4:
        campaign_pre = dataset_root / "fixed32_qwen_campaign_metrics_pre.txt"
        campaign_post = dataset_root / "fixed32_qwen_campaign_metrics_post.txt"
        campaign_pre.write_bytes(_metrics(0))
        campaign_post.write_bytes(_metrics(4))
        replay = contract.validate_fixed32_qwen_campaign_metrics(
            contract_tasks,
            metrics_pre=campaign_pre.read_bytes(),
            metrics_post=campaign_post.read_bytes(),
        )
        campaign_replay = replay["tasks"]
        qwen_path = dataset_root / "fixed32_qwen_campaign_provenance.json"
        proof = {
            "schema": credential.QWEN_SCHEMA,
            "metric_scope": "concurrent_campaign_union",
            "concurrency": 4,
            "task_ids": tasks,
            "selection": {
                "basis": "runner_owned_campaign_endpoint_metrics",
                "task_boundary_schema": credential.TASK_BOUNDARY_SCHEMA,
                "task_stream_coverage": {
                    "start_forward_step": 0,
                    "end_forward_step": 1,
                    "complete_stream_forward_steps": 1,
                },
            },
            "metrics_pre": _identity(campaign_pre),
            "metrics_post": _identity(campaign_post),
            "tasks": [
                {
                    "instance_id": task,
                    "task_key_id": task_keys[task],
                    "expected_completed_logical_model_requests": 1,
                    "trace": _identity(trace_paths[task]),
                }
                for task in tasks
            ],
            "metric_evidence_sha256": replay["metric_evidence_sha256"],
            "metric_evidence": replay["metric_evidence"],
        }
        _write_json(qwen_path, proof, canonical=True)
        campaign_identity = _identity(qwen_path)

    for task in tasks:
        task_dir = task_root / task
        trace_path = trace_paths[task]
        if batch == 1:
            metrics_pre = task_dir / "vllm_metrics_pre.txt"
            metrics_post = task_dir / "vllm_metrics_post.txt"
            replay = contract.validate_fixed32_trace_model_requests(
                trace_events[task],
                expected_session_id=contract.fixed32_trace_session_id(task),
                expected_completed_logical_model_requests=1,
                metrics_pre=metrics_pre.read_bytes(),
                metrics_post=metrics_post.read_bytes(),
            )
        else:
            replay = campaign_replay[task]
        request_digests = sorted(
            hashlib.sha256(value.encode()).hexdigest()
            for value in replay["model_request_ids"]
        )
        agent = {
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        }
        provenance = {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": task,
            "task_key_id": task_keys[task],
            **task_auth[task],
            "trace_completed_logical_model_requests": 1,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "trace_bytes": len(trace_path.read_bytes()),
            "event_count": len(trace_events[task]),
            "trace_model_request_ids_sha256": hashlib.sha256(
                _canonical(request_digests)
            ).hexdigest(),
            "hidden_successful_compaction_model_requests": replay.get(
                "hidden_successful_compaction_model_requests", 0
            ),
            "hidden_failed_compaction_model_requests": replay.get(
                "hidden_failed_compaction_model_requests", 0
            ),
            "synthetic_compaction_failure_terminal": replay.get(
                "synthetic_compaction_failure_terminal", False
            ),
            "qwen_metric_scope": "campaign" if batch == 4 else "task",
            "qwen_campaign_metric_proof": campaign_identity,
            "qwen_campaign_metric_evidence_sha256": (
                replay.get("qwen_campaign_metric_evidence_sha256")
                if batch == 4
                else None
            ),
            "qwen_compaction_metric_evidence": replay.get(
                "qwen_compaction_metric_evidence"
            ),
            "agent_terminal": agent,
        }
        metadata = {
            "instance_id": task,
            "agent": agent,
            "fixed32_task_boundary": boundaries[task],
            "fixed32_real_task_provenance": provenance,
        }
        if batch == 4:
            metadata["fixed32_qwen_campaign_proof"] = campaign_identity
        _write_json(task_dir / "runner_metadata.json", metadata)

    health_path = run / "health.json"
    _write_json(
        health_path,
        {
            "swe_orchestrator_rc": 0,
            "tasks": [
                {
                    "instance_id": task,
                    "codex_timed_out": False,
                    "verdict": "resolved",
                }
                for task in tasks
            ],
        },
    )
    (
        repo,
        source_commit,
        subset_path,
        launch_manifest_path,
        end_manifest_path,
    ) = _runtime_repo(tmp_path, batch=batch)
    git_head_path = run / "git_head.txt"
    git_head_path.write_text(source_commit + "\n", encoding="ascii")
    _write_json(
        run / "fixed32_chat_traffic_audit.json",
        {
            "complete_stream": {
                "pure_decode_forward_steps": 1,
                "complete_work_census_events": 1,
                "merged_forward_step_intervals": [[0, 1]],
            }
        },
        canonical=True,
    )
    container_env = run / "container_env.txt"
    container_env.write_text(
        "\n".join(
            f"{key}={value}"
            for key, value in {
                **credential.REQUIRED_CONTAINER_ENV,
                "FR13_FIXED32_MODE": mode,
            }.items()
        )
        + "\n",
        encoding="ascii",
    )
    return {
        "comparator_path": comparator_path,
        "subset_path": subset_path,
        "health_path": health_path,
        "proxy_ledger_path": proxy_path,
        "engine_ledger_path": engine_path,
        "work_census_path": work_path,
        "final_flush_path": flush_path,
        "boundary_snapshot_base": boundary_base,
        "runtime_manifest_launch_path": launch_manifest_path,
        "runtime_manifest_end_path": end_manifest_path,
        "runtime_git_head_path": git_head_path,
        "source_path": repo / credential.SOURCE_RELATIVE,
        "repo_path": repo,
        "container_env_path": container_env,
        "task_root": task_root,
        "source_commit": source_commit,
        "mode": mode,
        "batch_size": batch,
        "qwen_campaign_path": qwen_path,
    }


def _issue_with_audit_stub(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> dict[str, Any]:
    _stub_repo_import(monkeypatch, inputs)
    arm_dir = Path(inputs["task_root"]).parents[2]
    audit_path = arm_dir / "fixed32_chat_traffic_audit.json"
    audit = json.loads(audit_path.read_bytes())
    raw = audit_path.read_bytes()

    def validate(**kwargs: Any) -> tuple[dict[str, Any], bytes]:
        assert kwargs["arm_dir"] == arm_dir
        return audit, raw

    monkeypatch.setattr(credential, "_validate_real_task_audit", validate)
    return credential.issue_credential(**inputs)


def _stub_repo_import(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    source = (
        Path(inputs["repo_path"])
        / "src/lumo_flywheel_serving/inference_proxy.py"
    )

    def validate(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["repo"] == Path(inputs["repo_path"])
        return _identity(source)

    monkeypatch.setattr(
        credential,
        "_validate_inference_proxy_import",
        validate,
    )


@pytest.mark.parametrize("batch", (1, 4))
def test_credential_replays_real_graph_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    payload = _issue_with_audit_stub(
        monkeypatch, _make_fixture(tmp_path, batch=batch)
    )
    assert payload["status"] == "PASS"
    assert payload["work_census_terminal_present"] is True
    assert payload["all_engine_requests_joined_to_comparator"] is True
    assert payload["authenticated_engine_requests"] == batch
    assert payload["topology"]["logical_topology"] == (
        "Tail23" if batch == 1 else "Hydra27"
    )
    assert (payload["qwen_campaign_proof"] is not None) == (batch == 4)


def test_credential_rejects_truncated_work_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    _stub_repo_import(monkeypatch, inputs)
    work = Path(inputs["work_census_path"])
    work.write_bytes(work.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(credential.CredentialError, match="work census is invalid"):
        credential.issue_credential(**inputs)


def test_credential_rejects_comparator_request_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    comparator = Path(inputs["comparator_path"])
    rows = [json.loads(line) for line in comparator.read_text().splitlines()]
    rows[0]["request_id_sha256s"] = ["f" * 64]
    comparator.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    with pytest.raises(credential.CredentialError, match="comparator/work join"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_fake_two_key_manifest(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    _write_json(
        Path(inputs["runtime_manifest_end_path"]),
        {"schema": "fr13-runtime-manifest-v1", "closures": {}},
    )
    with pytest.raises(credential.CredentialError, match="full canonical"):
        credential.issue_credential(**inputs)


def test_credential_rejects_source_commit_tamper_with_fresh_manifest(
    tmp_path: Path,
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    source = Path(inputs["source_path"])
    source.write_text("# modified after source commit\n", encoding="ascii")
    manifest = runtime_manifest.build_manifest(
        Path(inputs["repo_path"]),
        profile="fixed32",
        sequence=credential.SEQUENCE,
    )
    _write_json(Path(inputs["runtime_manifest_launch_path"]), manifest)
    _write_json(Path(inputs["runtime_manifest_end_path"]), manifest)
    with pytest.raises(credential.CredentialError, match="source-commit binding"):
        credential.issue_credential(**inputs)


def test_credential_rejects_qwen_metric_artifact_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=4)
    proof = Path(inputs["qwen_campaign_path"])
    post = proof.parent / "fixed32_qwen_campaign_metrics_post.txt"
    post.write_bytes(post.read_bytes() + b"# tampered\n")
    with pytest.raises(credential.CredentialError, match="campaign replay failed"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_task_auth_provenance_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    task = credential.CANONICAL_SUBSETS[1]["task_ids"][0]
    metadata_path = Path(inputs["task_root"]) / task / "runner_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["fixed32_real_task_provenance"]["completed_attempts"] = 2
    _write_json(metadata_path, metadata)
    with pytest.raises(credential.CredentialError, match="task-auth counters"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_uncommitted_executed_source_with_fresh_manifests(
    tmp_path: Path,
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    repo = Path(inputs["repo_path"])
    source = repo / "src/lumo_flywheel_serving/model_server.py"
    source.write_bytes(source.read_bytes() + b"\n# uncommitted runtime change\n")
    manifest = runtime_manifest.build_manifest(
        repo, profile="fixed32", sequence=credential.SEQUENCE
    )
    _write_json(Path(inputs["runtime_manifest_launch_path"]), manifest)
    _write_json(Path(inputs["runtime_manifest_end_path"]), manifest)
    with pytest.raises(
        credential.CredentialError,
        match=r"source-commit binding differs for .*model_server.py",
    ):
        credential.issue_credential(**inputs)


def test_credential_rejects_runtime_git_head_tamper(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    Path(inputs["runtime_git_head_path"]).write_text(
        "f" * 40 + "\n", encoding="ascii"
    )
    with pytest.raises(credential.CredentialError, match="Git head differs"):
        credential.issue_credential(**inputs)


def test_credential_rejects_synthetic_real_task_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    _stub_repo_import(monkeypatch, inputs)
    with pytest.raises(credential.CredentialError, match="real-task chat audit"):
        credential.issue_credential(**inputs)


def test_real_task_audit_rebuilds_from_independent_pinned_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_dir = tmp_path / "arm"
    repo = tmp_path / "repo"
    subset_path = repo / "subset.json"
    arm_dir.mkdir()
    subset_path.parent.mkdir()
    subset_path.write_text("{}\n", encoding="ascii")
    subset = {"task_ids": ["task"], "sha256": "a" * 64}
    digests = {"task": "b" * 64}
    expected = {
        "complete_stream": {
            "pure_decode_forward_steps": 1,
            "complete_work_census_events": 1,
            "merged_forward_step_intervals": [[0, 1]],
        }
    }
    _write_json(
        arm_dir / "fixed32_chat_traffic_audit.json", expected, canonical=True
    )
    calls: dict[str, Any] = {}

    def validate(path: Path, *, b1_diagnostic: bool) -> dict[str, Any]:
        calls["subset"] = (path, b1_diagnostic)
        return subset

    def pinned(repo_text: str) -> dict[str, str]:
        calls["repo"] = repo_text
        return digests

    def build(
        path: Path,
        *,
        mode: str,
        subset: dict[str, Any],
        dataset_record_digests: dict[str, str],
        concurrency: int,
    ) -> dict[str, Any]:
        calls["build"] = (
            path,
            mode,
            subset,
            dataset_record_digests,
            concurrency,
        )
        return expected

    monkeypatch.setattr(floor_gate, "validate_fixed32_run_subset", validate)
    monkeypatch.setattr(floor_gate, "pinned_dataset_record_digests", pinned)
    monkeypatch.setattr(floor_gate, "build_fixed32_chat_traffic_audit", build)
    replay, raw = credential._validate_real_task_audit(
        arm_dir=arm_dir,
        repo=repo,
        subset_path=subset_path,
        mode="tail6_fixed32",
        batch_size=1,
    )
    assert replay == expected
    assert raw == _canonical(expected) + b"\n"
    assert calls == {
        "subset": (subset_path, True),
        "repo": str(repo),
        "build": (arm_dir, "tail6_fixed32", subset, digests, 1),
    }


def test_credential_rejects_shifted_work_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    _stub_repo_import(monkeypatch, inputs)
    work = Path(inputs["work_census_path"])
    event = json.loads(work.read_text().splitlines()[0])
    event["forward_step_index"] = 7
    event["drafter_runtime"]["forward_step_index"] = 7
    terminal = work_census.reference_terminal_summary(
        [event], fixture_synthetic_runtime_proof=True
    )
    work.write_bytes(_canonical(event) + b"\n" + _canonical(terminal) + b"\n")
    with pytest.raises(credential.CredentialError, match="exact contiguous stream"):
        credential.issue_credential(**inputs)


def test_credential_rejects_task_interval_past_terminal_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    task = credential.CANONICAL_SUBSETS[1]["task_ids"][0]
    task_dir = Path(inputs["task_root"]) / task
    boundary_path = task_dir / "fixed32_task_boundary.json"
    boundary = json.loads(boundary_path.read_bytes())
    boundary["post"]["counters"].update(
        {
            "pure_decode_forward_steps": 2,
            "complete_work_census_events": 2,
            "work_census_last_forward_step": 1,
        }
    )
    boundary["forward_step_interval"] = {
        "start_forward_step": 0,
        "end_forward_step": 2,
        "expected_complete_events": 2,
    }
    snapshot_path = Path(boundary["post_runtime_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["counters"] = boundary["post"]["counters"]
    _write_json(snapshot_path, snapshot, canonical=True)
    boundary["post_runtime_snapshot"]["sha256"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    _write_json(boundary_path, boundary)
    metadata_path = task_dir / "runner_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["fixed32_task_boundary"] = boundary
    _write_json(metadata_path, metadata)
    with pytest.raises(credential.CredentialError, match="task metrics differ"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_final_flush_generation_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    flush_path = Path(inputs["final_flush_path"])
    flush = json.loads(flush_path.read_bytes())
    old_generation = flush["ack"]["generation"]
    generation = old_generation + 1
    nonce = f"{generation:064x}"
    flush["ack"].update({"generation": generation, "nonce": nonce})
    _write_json(flush_path, flush, canonical=True)
    _write_json(
        Path(inputs["boundary_snapshot_base"]).parent
        / "fr13_fixed32_flush_ack.json",
        flush["ack"],
        canonical=True,
    )
    request_path = (
        Path(inputs["boundary_snapshot_base"]).parent
        / "fr13_fixed32_flush_request.json"
    )
    request = json.loads(request_path.read_bytes())
    request.update(
        {
            "prev_generation": generation - 1,
            "generation": generation,
            "nonce": nonce,
        }
    )
    _write_json(request_path, request, canonical=True)

    old_snapshot = Path(
        str(inputs["boundary_snapshot_base"]) + f".{old_generation}.json"
    )
    snapshot = json.loads(old_snapshot.read_bytes())
    snapshot.update({"generation": generation, "nonce": nonce})
    snapshot_path = Path(
        str(inputs["boundary_snapshot_base"]) + f".{generation}.json"
    )
    _write_json(snapshot_path, snapshot, canonical=True)

    comparator_path = Path(inputs["comparator_path"])
    comparator = [
        json.loads(line) for line in comparator_path.read_text().splitlines()
    ]
    comparator[-1].update(
        {
            "flush_generation": generation,
            "flush_nonce": nonce,
            "boundary_snapshot_sha256": hashlib.sha256(
                snapshot_path.read_bytes()
            ).hexdigest(),
        }
    )
    comparator_path.write_bytes(
        b"".join(_canonical(row) + b"\n" for row in comparator)
    )
    with pytest.raises(credential.CredentialError, match="generation chain"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_stale_boundary_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    base = Path(inputs["boundary_snapshot_base"])
    final = Path(str(base) + ".3.json")
    Path(str(base) + ".99.json").write_bytes(final.read_bytes())
    with pytest.raises(credential.CredentialError, match="generation set"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_current_flush_ack_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    ack_path = (
        Path(inputs["boundary_snapshot_base"]).parent
        / "fr13_fixed32_flush_ack.json"
    )
    ack = json.loads(ack_path.read_bytes())
    ack["nonce"] = "f" * 64
    _write_json(ack_path, ack, canonical=True)
    with pytest.raises(credential.CredentialError, match="current flush ack"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_malformed_task_runtime_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    task = credential.CANONICAL_SUBSETS[1]["task_ids"][0]
    task_dir = Path(inputs["task_root"]) / task
    boundary_path = task_dir / "fixed32_task_boundary.json"
    boundary = json.loads(boundary_path.read_bytes())
    snapshot_path = Path(boundary["post_runtime_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["metrics"]["fixed32"]["events_sha256"] = "f" * 64
    _write_json(snapshot_path, snapshot, canonical=True)
    boundary["post_runtime_snapshot"]["sha256"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    _write_json(boundary_path, boundary)
    metadata_path = task_dir / "runner_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["fixed32_task_boundary"] = boundary
    _write_json(metadata_path, metadata)
    with pytest.raises(
        credential.CredentialError,
        match=(
            "runtime snapshot/metrics/census is invalid: "
            ".*census prefix digest mismatch"
        ),
    ):
        _issue_with_audit_stub(monkeypatch, inputs)


@pytest.mark.parametrize("artifact", ("snapshot", "metrics"))
def test_credential_rejects_symlinked_task_runtime_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    task = credential.CANONICAL_SUBSETS[1]["task_ids"][0]
    task_dir = Path(inputs["task_root"]) / task
    boundary = json.loads(
        (task_dir / "fixed32_task_boundary.json").read_bytes()
    )
    path = (
        Path(boundary["post_runtime_snapshot"]["path"])
        if artifact == "snapshot"
        else task_dir / "vllm_metrics_post.txt"
    )
    target = path.with_name(path.name + ".target")
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(credential.CredentialError, match="regular non-symlink"):
        _issue_with_audit_stub(monkeypatch, inputs)


def test_credential_rejects_generation_order_counter_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=4)
    task_ids = list(credential.CANONICAL_SUBSETS[4]["task_ids"])
    first_dir = Path(inputs["task_root"]) / task_ids[0]
    last_dir = Path(inputs["task_root"]) / task_ids[-1]
    first_boundary_path = first_dir / "fixed32_task_boundary.json"
    last_boundary_path = last_dir / "fixed32_task_boundary.json"
    first_boundary = json.loads(first_boundary_path.read_bytes())
    last_boundary = json.loads(last_boundary_path.read_bytes())
    base = Path(inputs["boundary_snapshot_base"])
    generation4_path = Path(str(base) + ".4.json")
    generation5_path = Path(str(base) + ".5.json")
    generation4 = json.loads(generation4_path.read_bytes())
    generation5 = json.loads(generation5_path.read_bytes())

    rewritten4 = {
        **generation5,
        "generation": 4,
        "nonce": f"{4:064x}",
    }
    rewritten5 = {
        **generation4,
        "generation": 5,
        "nonce": f"{5:064x}",
    }
    _write_json(generation4_path, rewritten4, canonical=True)
    _write_json(generation5_path, rewritten5, canonical=True)
    first_boundary["post"].update(
        {"generation": 4, "nonce": f"{4:064x}"}
    )
    first_boundary["post_runtime_snapshot"] = {
        "schema": credential.BOUNDARY_SCHEMA,
        "generation": 4,
        "path": str(generation4_path),
        "sha256": hashlib.sha256(generation4_path.read_bytes()).hexdigest(),
    }
    last_boundary["pre"].update(
        {"generation": 5, "nonce": f"{5:064x}"}
    )
    last_boundary["pre_runtime_snapshot"] = {
        "schema": credential.BOUNDARY_SCHEMA,
        "generation": 5,
        "path": str(generation5_path),
        "sha256": hashlib.sha256(generation5_path.read_bytes()).hexdigest(),
    }
    for task_dir, boundary_path, boundary in (
        (first_dir, first_boundary_path, first_boundary),
        (last_dir, last_boundary_path, last_boundary),
    ):
        _write_json(boundary_path, boundary)
        metadata_path = task_dir / "runner_metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["fixed32_task_boundary"] = boundary
        _write_json(metadata_path, metadata)

    with pytest.raises(credential.CredentialError, match="ACK counters regress"):
        _issue_with_audit_stub(monkeypatch, inputs)


@pytest.mark.parametrize(
    ("ambient", "repo_first", "expected_error"),
    (
        (None, True, "real-task chat audit"),
        ("stale", True, "real-task chat audit"),
        ("stale", False, "git-show-bound repo source"),
    ),
)
def test_credential_cli_uses_repo_first_import_path(
    tmp_path: Path,
    ambient: str | None,
    repo_first: bool,
    expected_error: str,
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    repo = Path(inputs["repo_path"])
    env = os.environ.copy()
    if ambient is None:
        env.pop("PYTHONPATH", None)
        ambient_path = None
    else:
        ambient_path = tmp_path / "stale"
        package = ambient_path / "lumo_flywheel_serving"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="ascii")
        (package / "inference_proxy.py").write_text(
            "STALE_INFERENCE_PROXY = True\n",
            encoding="ascii",
        )
    prefix = str(repo / "src") if repo_first else ""
    suffix = str(ambient_path) if ambient_path is not None else ""
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (prefix, suffix) if value
    )
    command = [
        sys.executable,
        str(repo / "scripts/fr13_treeconv_zero_tail_credential.py"),
        "--comparator",
        str(inputs["comparator_path"]),
        "--subset",
        str(inputs["subset_path"]),
        "--health",
        str(inputs["health_path"]),
        "--proxy-ledger",
        str(inputs["proxy_ledger_path"]),
        "--engine-ledger",
        str(inputs["engine_ledger_path"]),
        "--work-census",
        str(inputs["work_census_path"]),
        "--final-flush",
        str(inputs["final_flush_path"]),
        "--boundary-snapshot-base",
        str(inputs["boundary_snapshot_base"]),
        "--runtime-manifest-launch",
        str(inputs["runtime_manifest_launch_path"]),
        "--runtime-manifest-end",
        str(inputs["runtime_manifest_end_path"]),
        "--runtime-git-head",
        str(inputs["runtime_git_head_path"]),
        "--source",
        str(inputs["source_path"]),
        "--repo",
        str(repo),
        "--container-env",
        str(inputs["container_env_path"]),
        "--task-root",
        str(inputs["task_root"]),
        "--source-commit",
        inputs["source_commit"],
        "--mode",
        inputs["mode"],
        "--batch-size",
        "1",
        "--output",
        str(tmp_path / "credential.json"),
    ]
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert expected_error in completed.stderr
    if repo_first:
        assert "git-show-bound repo source" not in completed.stderr


def test_credential_rejects_duplicate_container_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path, batch=1)
    _stub_repo_import(monkeypatch, inputs)
    env_path = Path(inputs["container_env_path"])
    env_path.write_text(
        env_path.read_text(encoding="ascii") + "FR13_FIXED32_MODE=tail6_fixed32\n",
        encoding="ascii",
    )
    with pytest.raises(credential.CredentialError, match="duplicated"):
        credential.issue_credential(**inputs)


def test_topology_descriptors_share_physical_state_but_not_logical_identity() -> None:
    tail = credential._topology_descriptor("tail6_fixed32")
    hydra = credential._topology_descriptor("hydra27_fixed32")
    assert tail["logical_topology"] == "Tail23"
    assert hydra["logical_topology"] == "Hydra27"
    assert tail["valid_mask"] != hydra["valid_mask"]
    assert tail["physical_parent_sha256"] == hydra["physical_parent_sha256"]
    assert tail["state_src_sha256"] == hydra["state_src_sha256"]
