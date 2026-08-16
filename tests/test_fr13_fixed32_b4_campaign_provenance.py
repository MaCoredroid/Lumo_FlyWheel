from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_contract as contract  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402


TASK_IDS = list(floor_gate.CANONICAL_TASK_IDS[:4])


def _load_runner() -> Any:
    path = SCRIPTS / "run_swe_bench_q36_a.py"
    spec = importlib.util.spec_from_file_location(
        "fixed32_b4_campaign_runner_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qwen_trace(instance_id: str) -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(instance_id)
    tool_id = f"tool-{instance_id}"
    first_id = f"first-{instance_id}"
    final_id = f"final-{instance_id}"
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
            "uuid": first_id,
            "session_id": session_id,
            "parent_tool_use_id": None,
            "message": {
                "id": first_id,
                "type": "message",
                "role": "assistant",
                "model": "qwen3.8-27b-nvfp4-radixark",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "read_file",
                        "input": {},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 32, "output_tokens": 8},
            },
        },
        {
            "type": "user",
            "uuid": f"tool-result-{instance_id}",
            "session_id": session_id,
            "parent_tool_use_id": None,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "done",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": final_id,
            "session_id": session_id,
            "parent_tool_use_id": None,
            "message": {
                "id": final_id,
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
            "num_turns": 2,
            "result": "complete",
            "usage": {
                "input_tokens": 64,
                "output_tokens": 16,
                "total_tokens": 80,
            },
            "permission_denials": [],
        },
    ]


def _qwen_trace_with_request_count(
    instance_id: str,
    completed: int,
    *,
    hidden_input_tokens: int = 0,
    hidden_output_tokens: int = 0,
) -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(instance_id)
    events: list[dict[str, Any]] = [
        {
            "type": "system",
            "subtype": "init",
            "qwen_code_version": "0.19.4",
            "uuid": f"system-{instance_id}",
            "session_id": session_id,
            "parent_tool_use_id": None,
        }
    ]
    for ordinal in range(completed - 1):
        assistant_id = f"assistant-{ordinal}-{instance_id}"
        tool_id = f"tool-{ordinal}-{instance_id}"
        events.extend(
            [
                {
                    "type": "assistant",
                    "uuid": assistant_id,
                    "session_id": session_id,
                    "parent_tool_use_id": None,
                    "message": {
                        "id": assistant_id,
                        "type": "message",
                        "role": "assistant",
                        "model": "qwen3.8-27b-nvfp4-radixark",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": "read_file",
                                "input": {},
                            }
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 32, "output_tokens": 8},
                    },
                },
                {
                    "type": "user",
                    "uuid": f"tool-result-{ordinal}-{instance_id}",
                    "session_id": session_id,
                    "parent_tool_use_id": None,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": "done",
                                "is_error": False,
                            }
                        ],
                    },
                },
            ]
        )
    final_id = f"final-{instance_id}"
    events.append(
        {
            "type": "assistant",
            "uuid": final_id,
            "session_id": session_id,
            "parent_tool_use_id": None,
            "message": {
                "id": final_id,
                "type": "message",
                "role": "assistant",
                "model": "qwen3.8-27b-nvfp4-radixark",
                "content": [{"type": "text", "text": "complete"}],
                "stop_reason": None,
                "usage": {"input_tokens": 32, "output_tokens": 8},
            },
        }
    )
    input_tokens = completed * 32 + hidden_input_tokens
    output_tokens = completed * 8 + hidden_output_tokens
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "uuid": f"result-{instance_id}",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 100,
            "duration_api_ms": 90,
            "num_turns": completed,
            "result": "complete",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "permission_denials": [],
        }
    )
    return events


def _metrics(
    completed: int,
    *,
    compactions: int = 0,
    normal_requests: int | None = None,
    prompt_tokens: int | None = None,
    generation_tokens: int | None = None,
) -> bytes:
    if normal_requests is None:
        normal_requests = completed
    if prompt_tokens is None:
        prompt_tokens = completed * 32
    if generation_tokens is None:
        generation_tokens = completed * 8
    max_tokens_sum = (
        normal_requests * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS
        + compactions * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS
    )
    base_labels = 'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"'
    lines = [
        f"vllm:prompt_tokens_total{{{base_labels}}} {prompt_tokens}",
        f"vllm:generation_tokens_total{{{base_labels}}} {generation_tokens}",
        (
            "vllm:request_params_max_tokens_count"
            f"{{{base_labels}}} {completed}"
        ),
        (
            "vllm:request_params_max_tokens_sum"
            f"{{{base_labels}}} {max_tokens_sum}"
        ),
    ]
    for reason in ("stop", "length", "abort", "error", "repetition"):
        labels = (
            f'engine="0",finished_reason="{reason}",'
            'model_name="qwen3.8-27b-nvfp4-radixark"'
        )
        lines.append(
            f"vllm:request_success_total{{{labels}}} "
            f"{completed if reason == 'stop' else 0}"
        )
    for le, value in (
        ("10000.0", 0),
        ("20000.0", compactions),
        ("50000.0", completed),
        ("+Inf", completed),
    ):
        labels = f'engine="0",le="{le}",model_name="qwen3.8-27b-nvfp4-radixark"'
        lines.append(
            f"vllm:request_params_max_tokens_bucket{{{labels}}} {value}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _real_count_shape_campaign(
    *,
    compactions: int = 2,
    max_tokens_sum_delta: int = 0,
    hidden_input_tokens: int = 20,
    hidden_output_tokens: int = 2,
) -> tuple[list[dict[str, Any]], bytes]:
    trace_counts = (8, 28, 41, 51)
    task_auth_counts = (8, 28, 41, 53)
    tasks = []
    for index, (instance_id, trace_count, task_auth_count) in enumerate(
        zip(TASK_IDS, trace_counts, task_auth_counts, strict=True)
    ):
        is_last = index == len(TASK_IDS) - 1
        tasks.append(
            {
                "instance_id": instance_id,
                "expected_session_id": contract.fixed32_trace_session_id(
                    instance_id
                ),
                "expected_completed_logical_model_requests": task_auth_count,
                "budget_capped": False,
                "events": _qwen_trace_with_request_count(
                    instance_id,
                    trace_count,
                    hidden_input_tokens=(
                        hidden_input_tokens if is_last else 0
                    ),
                    hidden_output_tokens=(
                        hidden_output_tokens if is_last else 0
                    ),
                ),
            }
        )
    normal_requests = sum(trace_counts)
    metrics_post = _metrics(
        sum(task_auth_counts),
        compactions=compactions,
        normal_requests=sum(task_auth_counts) - compactions,
        prompt_tokens=normal_requests * 32 + hidden_input_tokens,
        generation_tokens=normal_requests * 8 + hidden_output_tokens,
    )
    if max_tokens_sum_delta:
        marker = b"vllm:request_params_max_tokens_sum"
        lines = metrics_post.splitlines()
        for index, line in enumerate(lines):
            if line.startswith(marker):
                prefix, value = line.rsplit(b" ", 1)
                lines[index] = prefix + b" " + str(
                    int(value) + max_tokens_sum_delta
                ).encode("ascii")
                break
        metrics_post = b"\n".join(lines) + b"\n"
    return tasks, metrics_post


def _task_auth(task_key_id: str, completed: int) -> dict[str, Any]:
    return {
        "task_key_id": task_key_id,
        "completed_logical_model_requests": completed,
        "aborted_logical_requests": 0,
        "accepted_attempts": completed,
        "completed_attempts": completed,
        "failed_attempts": 0,
    }


def _campaign_fixture(
    tmp_path: Path,
    *,
    tamper_final_metrics: bool = False,
) -> tuple[Any, list[dict[str, Any]], Path, Path]:
    runner = _load_runner()
    dataset_out = tmp_path / "verified"
    per_task_root = dataset_out / "per_task"
    intervals = ((0, 5), (0, 7), (1, 8), (2, 9))
    post_completions = (2, 4, 6, 7 if tamper_final_metrics else 8)
    summaries: list[dict[str, Any]] = []
    for index, (instance_id, interval) in enumerate(
        zip(TASK_IDS, intervals, strict=True)
    ):
        task_dir = (per_task_root / instance_id).resolve()
        task_dir.mkdir(parents=True)
        trace_path = task_dir / "qwen_trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(event) + "\n" for event in _qwen_trace(instance_id)),
            encoding="utf-8",
        )
        metrics_pre_path = task_dir / "vllm_metrics_pre.txt"
        metrics_post_path = task_dir / "vllm_metrics_post.txt"
        metrics_pre_path.write_bytes(_metrics(0 if index < 2 else index))
        metrics_post_path.write_bytes(_metrics(post_completions[index]))
        start, end = interval
        boundary = {
            "schema": "fr13-fixed32-task-boundary-v1",
            "instance_id": instance_id,
            "pre": {
                "generation": index + 1,
                "counters": {"pure_decode_forward_steps": start},
            },
            "post": {
                "generation": index + 5,
                "counters": {"pure_decode_forward_steps": end},
            },
            "forward_step_interval": {
                "start_forward_step": start,
                "end_forward_step": end,
                "expected_complete_events": end - start,
            },
        }
        (task_dir / "fixed32_task_boundary.json").write_text(
            json.dumps(boundary),
            encoding="utf-8",
        )
        for relative_path in (
            "patch.diff",
            "eval/predictions.jsonl",
            "eval/eval_report.json",
            "qwen_runtime_attestation.json",
            "qwen_runtime_attestation_post.json",
            runner._FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME,
        ):
            artifact_path = task_dir / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("{}\n", encoding="ascii")
        task_key_id = floor_gate.fixed32_task_key_id(instance_id)
        summary = {
            "instance_id": instance_id,
            "ended_at": "2026-08-01T00:00:00Z",
            "fixed32_task_boundary": boundary,
            runner._FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY: {
                "instance_id": instance_id,
                "trace_path": trace_path,
                "agent_meta": {},
                "task_key_id": task_key_id,
                "task_auth_before": _task_auth(task_key_id, 0),
                "task_auth_after": _task_auth(task_key_id, 2),
                "metrics_pre_path": metrics_pre_path,
                "metrics_post_path": metrics_post_path,
            },
        }
        pending = {
            key: value
            for key, value in summary.items()
            if key != runner._FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY
        }
        (task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME).write_text(
            json.dumps(pending, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary)
    (dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME).write_bytes(
        _metrics(0)
    )
    (
        dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME
    ).write_bytes(_metrics(7 if tamper_final_metrics else 8))
    return runner, summaries, dataset_out, per_task_root


def _campaign_metric_paths(runner: Any, dataset_out: Path) -> tuple[Path, Path]:
    return (
        dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME,
        dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME,
    )


def _convert_fixture_to_eager_boundaries(
    runner: Any,
    summaries: list[dict[str, Any]],
    per_task_root: Path,
) -> None:
    for summary in summaries:
        instance_id = summary["instance_id"]
        task_dir = per_task_root / instance_id
        pre_path = task_dir / "vllm_metrics_pre.txt"
        post_path = task_dir / "vllm_metrics_post.txt"
        boundary = {
            "schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
            "instance_id": instance_id,
            "mode": "hybrid23",
            "producer_pid": 1234,
            "run_classification": "eager_kernel_byte_diagnostic",
            "acceptance_valid": False,
            "flush_protocol_used": False,
            "pre_metrics": runner._fixed32_artifact_identity(
                pre_path,
                pre_path.read_bytes(),
            ),
            "post_metrics": runner._fixed32_artifact_identity(
                post_path,
                post_path.read_bytes(),
            ),
        }
        summary["fixed32_task_boundary"] = boundary
        pending = {
            key: value
            for key, value in summary.items()
            if key != runner._FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY
        }
        (
            task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME
        ).write_text(json.dumps(pending, indent=2), encoding="utf-8")


def _provenance_for_replay(
    *,
    task_dir: Path,
    proof_identity: dict[str, Any],
    trace_requests: dict[str, Any],
) -> dict[str, Any]:
    trace_path = task_dir / "qwen_trace.jsonl"
    raw = trace_path.read_bytes()
    events = [json.loads(line) for line in raw.decode().splitlines()]
    response_digests = sorted(
        hashlib.sha256(request_id.encode()).hexdigest()
        for request_id in trace_requests["model_request_ids"]
    )
    return {
        "instance_id": task_dir.name,
        "completed_logical_model_requests": trace_requests[
            "completed_logical_model_requests"
        ],
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_bytes": len(raw),
        "event_count": len(events),
        "trace_completed_logical_model_requests": trace_requests[
            "completed_logical_model_requests"
        ],
        "trace_model_request_ids_sha256": hashlib.sha256(
            json.dumps(
                response_digests,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "hidden_successful_compaction_model_requests": trace_requests[
            "hidden_successful_compaction_model_requests"
        ],
        "hidden_failed_compaction_model_requests": trace_requests[
            "hidden_failed_compaction_model_requests"
        ],
        "synthetic_compaction_failure_terminal": trace_requests[
            "synthetic_compaction_failure_terminal"
        ],
        "qwen_metric_scope": "campaign",
        "qwen_campaign_metric_proof": proof_identity,
        "qwen_campaign_metric_evidence_sha256": trace_requests[
            "qwen_campaign_metric_evidence_sha256"
        ],
        "qwen_compaction_metric_evidence": trace_requests[
            "qwen_compaction_metric_evidence"
        ],
    }


def test_b4_campaign_failed_only_compactions_use_global_algebra() -> None:
    tasks, metrics_post = _real_count_shape_campaign()

    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=_metrics(0),
        metrics_post=metrics_post,
    )

    evidence = reconciliation["metric_evidence"]
    assert evidence["completed_engine_requests"] == 130
    assert evidence["normal_requests"] == 128
    assert evidence["successful_compaction_requests"] == 0
    assert evidence["failed_compaction_requests"] == 2
    assert evidence["total_compaction_requests"] == 2
    last = reconciliation["tasks"][TASK_IDS[-1]]
    assert last["completed_logical_model_requests"] == 53
    assert last["hidden_successful_compaction_model_requests"] == 0
    assert last["hidden_failed_compaction_model_requests"] == 2
    assert last["synthetic_compaction_failure_terminal"] is False


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("compaction_bucket", "32768/20000"),
        ("max_tokens_sum", "32768/20000"),
        ("task_auth_count", "completion metrics"),
        ("hidden_input", "hidden compaction token usage"),
        ("hidden_output", "hidden compaction token usage"),
    ),
)
def test_b4_campaign_failed_only_compaction_tamper_fails_closed(
    tamper: str,
    message: str,
) -> None:
    kwargs: dict[str, int] = {}
    if tamper == "compaction_bucket":
        kwargs["compactions"] = 1
    elif tamper == "max_tokens_sum":
        kwargs["max_tokens_sum_delta"] = 1
    elif tamper == "hidden_input":
        kwargs["hidden_input_tokens"] = 0
    elif tamper == "hidden_output":
        kwargs["hidden_output_tokens"] = 0
    tasks, metrics_post = _real_count_shape_campaign(**kwargs)
    if tamper == "task_auth_count":
        tasks[-1]["expected_completed_logical_model_requests"] = 52

    with pytest.raises(contract.ContractError, match=message):
        contract.validate_fixed32_qwen_campaign_metrics(
            tasks,
            metrics_pre=_metrics(0),
            metrics_post=metrics_post,
        )


def test_b4_campaign_union_finalizes_only_after_global_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(tmp_path)
    for task_id in TASK_IDS:
        assert not (per_task_root / task_id / "runner_metadata.json").exists()

    monkeypatch.setattr(
        runner,
        "_fixed32_real_task_provenance",
        lambda **kwargs: {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": kwargs["instance_id"],
        },
    )
    campaign_metrics_pre_path, campaign_metrics_post_path = (
        _campaign_metric_paths(runner, dataset_out)
    )
    proof = runner._finalize_fixed32_qwen_campaign_provenance(
        summaries=summaries,
        instance_ids=TASK_IDS,
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        campaign_metrics_pre_path=campaign_metrics_pre_path,
        campaign_metrics_post_path=campaign_metrics_post_path,
    )

    assert proof["selection"] == {
        "basis": "runner_owned_campaign_endpoint_metrics",
        "task_boundary_schema": "fr13-fixed32-task-boundary-v1",
        "task_stream_coverage": {
            "start_forward_step": 0,
            "end_forward_step": 9,
            "complete_stream_forward_steps": 9,
        },
    }
    assert proof["metric_evidence"]["completed_engine_requests"] == 8
    for task_id in TASK_IDS:
        task_dir = per_task_root / task_id
        metadata = json.loads((task_dir / "runner_metadata.json").read_text())
        assert metadata["fixed32_qwen_campaign_proof"]["sha256"]
        assert not (
            task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME
        ).exists()

    second_events = _qwen_trace(TASK_IDS[1])
    with pytest.raises(contract.ContractError, match="completion metrics"):
        contract.validate_fixed32_trace_model_requests(
            second_events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_IDS[1]),
            expected_completed_logical_model_requests=2,
            metrics_pre=_metrics(0),
            metrics_post=_metrics(4),
        )

    proof_path = dataset_out / runner._FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME
    proof_identity = json.loads(
        (per_task_root / TASK_IDS[0] / "runner_metadata.json").read_text()
    )["fixed32_qwen_campaign_proof"]
    contract_tasks = [
        {
            "instance_id": task_id,
            "expected_session_id": contract.fixed32_trace_session_id(task_id),
            "expected_completed_logical_model_requests": 2,
            "events": _qwen_trace(task_id),
            "budget_capped": False,
        }
        for task_id in TASK_IDS
    ]
    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        contract_tasks,
        metrics_pre=_metrics(0),
        metrics_post=_metrics(8),
    )
    provenance = _provenance_for_replay(
        task_dir=per_task_root / TASK_IDS[0],
        proof_identity=proof_identity,
        trace_requests=reconciliation["tasks"][TASK_IDS[0]],
    )
    replay = floor_gate._fixed32_trace_model_requests(
        per_task_root / TASK_IDS[0] / "qwen_trace.jsonl",
        provenance=provenance,
        require_campaign_scope=True,
        expected_campaign_task_ids=TASK_IDS,
    )
    assert replay["completed_logical_model_requests"] == 2

    tampered = json.loads(proof_path.read_text())
    tampered["metric_evidence"]["prompt_tokens"] += 1
    proof_path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    tampered_raw = proof_path.read_bytes()
    provenance["qwen_campaign_metric_proof"] = {
        "path": str(proof_path.resolve()),
        "sha256": hashlib.sha256(tampered_raw).hexdigest(),
        "bytes": len(tampered_raw),
    }
    with pytest.raises(floor_gate.GateError, match="metric evidence differs"):
        floor_gate._fixed32_trace_model_requests(
            per_task_root / TASK_IDS[0] / "qwen_trace.jsonl",
            provenance=provenance,
            require_campaign_scope=True,
            expected_campaign_task_ids=TASK_IDS,
        )


def test_b4_campaign_metric_tamper_publishes_no_final_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(
        tmp_path,
        tamper_final_metrics=True,
    )
    monkeypatch.setattr(runner, "_fixed32_real_task_provenance", lambda **_: {})

    with pytest.raises(
        runner.Fixed32BoundaryError,
        match="campaign metrics do not reconcile",
    ):
        runner._finalize_fixed32_qwen_campaign_provenance(
            summaries=summaries,
            instance_ids=TASK_IDS,
            dataset_out=dataset_out,
            per_task_root=per_task_root,
            campaign_metrics_pre_path=_campaign_metric_paths(
                runner, dataset_out
            )[0],
            campaign_metrics_post_path=_campaign_metric_paths(
                runner, dataset_out
            )[1],
        )

    assert not (
        dataset_out / runner._FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME
    ).exists()
    for task_id in TASK_IDS:
        task_dir = per_task_root / task_id
        assert not (task_dir / "runner_metadata.json").exists()
        assert (
            task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME
        ).is_file()


def test_b4_autocommit_publishes_proof_and_all_tasks_as_one_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(tmp_path)
    monkeypatch.setattr(
        runner,
        "_fixed32_real_task_provenance",
        lambda **kwargs: {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": kwargs["instance_id"],
        },
    )
    runner._finalize_fixed32_qwen_campaign_provenance(
        summaries=summaries,
        instance_ids=TASK_IDS,
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        campaign_metrics_pre_path=_campaign_metric_paths(
            runner, dataset_out
        )[0],
        campaign_metrics_post_path=_campaign_metric_paths(
            runner, dataset_out
        )[1],
    )

    captured: list[tuple[list[str], str, bool]] = []
    monkeypatch.setattr(
        runner,
        "_autocommit_paths",
        lambda paths, message, *, strict_push=False: captured.append(
            (paths, message, strict_push)
        ),
    )
    monkeypatch.setenv("LUMO_SWE_AUTOCOMMIT", "1")
    taw_campaign_arm = dataset_out / "fixed32_taw_campaign_arm.json"
    taw_campaign_arm.write_text("{}\n", encoding="ascii")
    runner._autocommit_fixed32_campaign_artifacts(
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        instance_ids=TASK_IDS,
        taw_campaign_arm_artifact_path=taw_campaign_arm,
    )

    assert len(captured) == 1
    committed_paths, message, strict_push = captured[0]
    assert strict_push is True
    assert committed_paths[:3] == [
        str(dataset_out / runner._FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME),
        str(dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME),
        str(dataset_out / runner._FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME),
    ]
    assert str(taw_campaign_arm) in committed_paths
    assert {
        str(per_task_root / task_id / "runner_metadata.json")
        for task_id in TASK_IDS
    }.issubset(committed_paths)
    for task_id in TASK_IDS:
        task_dir = per_task_root / task_id
        for relative_path in runner._AUTOCOMMIT_FIXED32_CAMPAIGN_RELS:
            assert str(task_dir / relative_path) in committed_paths
        assert str(task_dir / "fixed32_task_boundary.json") in committed_paths
        assert str(task_dir / "vllm_metrics_pre.txt") in committed_paths
        assert str(task_dir / "vllm_metrics_post.txt") in committed_paths
    assert not any(path.endswith("runner_metadata.pending.json") for path in committed_paths)
    assert "finalized B4 campaign artifacts" in message


def test_b4_autocommit_fails_closed_on_missing_replay_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(tmp_path)
    monkeypatch.setattr(
        runner,
        "_fixed32_real_task_provenance",
        lambda **kwargs: {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": kwargs["instance_id"],
        },
    )
    runner._finalize_fixed32_qwen_campaign_provenance(
        summaries=summaries,
        instance_ids=TASK_IDS,
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        campaign_metrics_pre_path=_campaign_metric_paths(
            runner, dataset_out
        )[0],
        campaign_metrics_post_path=_campaign_metric_paths(
            runner, dataset_out
        )[1],
    )
    (per_task_root / TASK_IDS[2] / "vllm_metrics_pre.txt").unlink()
    monkeypatch.setenv("LUMO_SWE_AUTOCOMMIT", "1")

    with pytest.raises(
        runner.Fixed32BoundaryError,
        match="artifact publication set is incomplete",
    ):
        runner._autocommit_fixed32_campaign_artifacts(
            dataset_out=dataset_out,
            per_task_root=per_task_root,
            instance_ids=TASK_IDS,
        )


def test_b4_eager_campaign_ignores_contaminated_per_task_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(tmp_path)
    _convert_fixture_to_eager_boundaries(
        runner,
        summaries,
        per_task_root,
    )
    monkeypatch.setattr(
        runner,
        "_fixed32_real_task_provenance",
        lambda **kwargs: {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": kwargs["instance_id"],
        },
    )

    second_runtime = summaries[1][runner._FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY]
    with pytest.raises(contract.ContractError, match="completion metrics"):
        contract.validate_fixed32_trace_model_requests(
            _qwen_trace(TASK_IDS[1]),
            expected_session_id=contract.fixed32_trace_session_id(TASK_IDS[1]),
            expected_completed_logical_model_requests=2,
            metrics_pre=second_runtime["metrics_pre_path"].read_bytes(),
            metrics_post=second_runtime["metrics_post_path"].read_bytes(),
        )

    campaign_metrics_pre_path, campaign_metrics_post_path = (
        _campaign_metric_paths(runner, dataset_out)
    )
    proof = runner._finalize_fixed32_qwen_campaign_provenance(
        summaries=summaries,
        instance_ids=TASK_IDS,
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        campaign_metrics_pre_path=campaign_metrics_pre_path,
        campaign_metrics_post_path=campaign_metrics_post_path,
    )

    assert proof["selection"] == {
        "basis": "runner_owned_campaign_endpoint_metrics",
        "task_boundary_schema": (
            "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1"
        ),
        "task_stream_coverage": None,
    }
    assert proof["metric_evidence"]["completed_engine_requests"] == 8


def test_strict_artifact_push_retries_and_surfaces_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        commands.append(command)
        operation = command[1]
        return subprocess.CompletedProcess(
            command,
            1 if operation in {"diff", "push"} else 0,
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="git push failed after 3 attempts"):
        runner._autocommit_paths(
            ["artifact.json"],
            "campaign",
            strict_push=True,
        )

    assert sum(command[1] == "push" for command in commands) == 3


def _synthetic_compaction_failure_trace() -> list[dict[str, Any]]:
    instance_id = TASK_IDS[0]
    events = _qwen_trace(instance_id)
    text = (
        "[API Error: Context is too large to send safely after automatic "
        "compression. Estimated prompt tokens: 78280; hard limit: 75304; "
        "compression status: COMPRESSION_FAILED_EMPTY_SUMMARY. Start a new "
        "session or reduce the resumed history before continuing.]"
    )
    synthetic_id = events[-2]["uuid"]
    events[-2] = {
        "type": "assistant",
        "uuid": synthetic_id,
        "session_id": events[-1]["session_id"],
        "parent_tool_use_id": None,
        "message": {
            "id": synthetic_id,
            "type": "message",
            "role": "assistant",
            "model": "qwen3.8-27b-nvfp4-radixark",
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    events[-1]["result"] = text
    events[-1]["usage"] = {
        "input_tokens": 64,
        "output_tokens": 16,
        "total_tokens": 80,
    }
    return events


def _failed_only_campaign_fixture(
    tmp_path: Path,
) -> tuple[Any, list[dict[str, Any]], Path, Path]:
    runner, summaries, dataset_out, per_task_root = _campaign_fixture(tmp_path)
    first_task_id = TASK_IDS[0]
    first_task_dir = per_task_root / first_task_id
    first_task_dir.joinpath("qwen_trace.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in _synthetic_compaction_failure_trace()
        ),
        encoding="utf-8",
    )
    runtime_args = summaries[0][runner._FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY]
    task_key_id = runtime_args["task_key_id"]
    runtime_args["task_auth_after"] = _task_auth(task_key_id, 5)
    dataset_out.joinpath(
        runner._FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME
    ).write_bytes(
        _metrics(
            11,
            compactions=4,
            normal_requests=7,
            prompt_tokens=256,
            generation_tokens=64,
        )
    )
    return runner, summaries, dataset_out, per_task_root


@pytest.mark.parametrize("tamper", ("usage_key", "result_text"))
def test_synthetic_compaction_failure_exclusion_is_exact(tamper: str) -> None:
    events = _synthetic_compaction_failure_trace()
    baseline = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_IDS[0]),
    )
    assert baseline["completed_logical_model_requests"] == 1
    assert baseline["synthetic_compaction_failure_terminal"] is True

    if tamper == "usage_key":
        events[-2]["message"]["usage"]["total_tokens"] = 0
    else:
        events[-1]["result"] += " "
    near_miss = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_IDS[0]),
    )
    assert near_miss["completed_logical_model_requests"] == 2
    assert near_miss["synthetic_compaction_failure_terminal"] is False


def test_b4_failed_only_compactions_finalize_from_exact_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, summaries, dataset_out, per_task_root = (
        _failed_only_campaign_fixture(tmp_path)
    )
    monkeypatch.setattr(
        runner,
        "_fixed32_real_task_provenance",
        lambda **kwargs: {
            "schema": "fr13-fixed32-real-task-provenance-v3",
            "instance_id": kwargs["instance_id"],
        },
    )

    proof = runner._finalize_fixed32_qwen_campaign_provenance(
        summaries=summaries,
        instance_ids=TASK_IDS,
        dataset_out=dataset_out,
        per_task_root=per_task_root,
        campaign_metrics_pre_path=_campaign_metric_paths(
            runner, dataset_out
        )[0],
        campaign_metrics_post_path=_campaign_metric_paths(
            runner, dataset_out
        )[1],
    )

    assert proof["metric_evidence"]["completed_engine_requests"] == 11
    assert proof["metric_evidence"]["normal_requests"] == 7
    assert proof["metric_evidence"]["successful_compaction_requests"] == 0
    assert proof["metric_evidence"]["failed_compaction_requests"] == 4
    first_task = proof["metric_evidence"]["tasks"][0]
    assert first_task["synthetic_compaction_failure_terminal"] is True
    for task_id in TASK_IDS:
        task_dir = per_task_root / task_id
        assert (task_dir / "runner_metadata.json").is_file()
        assert not (
            task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME
        ).exists()


@pytest.mark.parametrize("tamper", ("terminal", "compaction_metric"))
def test_b4_failed_only_compaction_tamper_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    runner, summaries, dataset_out, per_task_root = (
        _failed_only_campaign_fixture(tmp_path)
    )
    if tamper == "terminal":
        trace_path = per_task_root / TASK_IDS[0] / "qwen_trace.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        events[-1]["result"] += " "
        trace_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
    else:
        dataset_out.joinpath(
            runner._FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME
        ).write_bytes(
            _metrics(
                11,
                compactions=3,
                normal_requests=8,
                prompt_tokens=256,
                generation_tokens=64,
            )
        )
    monkeypatch.setattr(runner, "_fixed32_real_task_provenance", lambda **_: {})

    with pytest.raises(
        runner.Fixed32BoundaryError,
        match="campaign metrics do not reconcile",
    ):
        runner._finalize_fixed32_qwen_campaign_provenance(
            summaries=summaries,
            instance_ids=TASK_IDS,
            dataset_out=dataset_out,
            per_task_root=per_task_root,
            campaign_metrics_pre_path=_campaign_metric_paths(
                runner, dataset_out
            )[0],
            campaign_metrics_post_path=_campaign_metric_paths(
                runner, dataset_out
            )[1],
        )

    assert not (
        dataset_out / runner._FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME
    ).exists()
    for task_id in TASK_IDS:
        task_dir = per_task_root / task_id
        assert not (task_dir / "runner_metadata.json").exists()
        assert (
            task_dir / runner._FIXED32_PENDING_RUNNER_METADATA_FILENAME
        ).is_file()
