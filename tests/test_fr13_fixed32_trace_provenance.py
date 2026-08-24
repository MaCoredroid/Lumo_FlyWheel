from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
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
import fr13_runtime_manifest as runtime_manifest  # noqa: E402


def _load_runner() -> Any:
    path = SCRIPTS / "run_swe_bench_q36_a.py"
    spec = importlib.util.spec_from_file_location("fixed32_trace_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK_A = "astropy__astropy-12907"
TASK_B = "astropy__astropy-13033"
CAP_CHUNK_PATH = Path(
    "npm/lib/node_modules/@qwen-code/qwen-code/chunks/chunk-BFG6OZN7.js"
)


def _assistant_event(
    *,
    response_id: str,
    session_id: str,
    content: list[dict[str, Any]],
    stop_reason: str | None,
    parent_tool_use_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": response_id,
        "session_id": session_id,
        "parent_tool_use_id": parent_tool_use_id,
        "message": {
            "role": "assistant",
            "id": response_id,
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def _context_event(
    *,
    event_type: str,
    event_id: str,
    session_id: str,
    parent_tool_use_id: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": event_type,
        "uuid": event_id,
        "session_id": session_id,
        "parent_tool_use_id": parent_tool_use_id,
    }
    if event_type == "system":
        event["subtype"] = "init"
        event["qwen_code_version"] = "0.19.4"
    return event


def _user_event(
    *,
    event_id: str,
    session_id: str,
    content: list[dict[str, Any]],
    parent_tool_use_id: str | None,
) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": event_id,
        "session_id": session_id,
        "parent_tool_use_id": parent_tool_use_id,
        "message": {
            "role": "user",
            "content": content,
        },
    }


def _qwen_result_trace(
    instance_id: str = TASK_A,
) -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(instance_id)
    events: list[dict[str, Any]] = [
        _context_event(
            event_type="system",
            event_id="system",
            session_id=session_id,
        )
    ]
    for index in range(12):
        events.append(
            _assistant_event(
                response_id=f"tool-turn-{index}",
                session_id=session_id,
                content=[
                    {
                        "type": "tool_use",
                        "id": f"tool-call-{index}",
                        "name": "read_file",
                        "input": {},
                    }
                ],
                stop_reason="tool_use",
            )
        )
        events.append(
            _context_event(
                event_type="user",
                event_id=f"tool-result-{index}",
                session_id=session_id,
            )
        )
    events.extend(
        [
            _assistant_event(
                response_id="final-thinking",
                session_id=session_id,
                content=[{"type": "thinking", "thinking": "complete"}],
                stop_reason=None,
            ),
            _assistant_event(
                response_id="final-text",
                session_id=session_id,
                content=[{"type": "text", "text": "completed"}],
                stop_reason=None,
            ),
            {
                "type": "result",
                "subtype": "success",
                "uuid": "result-uuid",
                "session_id": session_id,
                "is_error": False,
                "duration_ms": 100,
                "duration_api_ms": 90,
                "num_turns": 13,
                "result": "completed",
                "usage": {
                    "input_tokens": 91,
                    "output_tokens": 13,
                    "total_tokens": 104,
                },
                "permission_denials": [],
            },
        ]
    )
    return events


def _top_level_assistant_groups(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    previous_was_top_level_assistant = False
    for event in events:
        is_top_level_assistant = (
            event.get("type") == "assistant"
            and event.get("parent_tool_use_id") is None
        )
        if not is_top_level_assistant:
            previous_was_top_level_assistant = False
            continue
        if previous_was_top_level_assistant:
            groups[-1].append(event)
        else:
            groups.append([event])
        previous_was_top_level_assistant = True
    return groups


def _set_top_level_group_input_tokens(
    events: list[dict[str, Any]],
    values: list[int],
) -> list[list[dict[str, Any]]]:
    groups = _top_level_assistant_groups(events)
    assert len(groups) == len(values)
    for group, value in zip(groups, values, strict=True):
        for event in group:
            event["message"]["usage"]["input_tokens"] = 0
        group[-1]["message"]["usage"]["input_tokens"] = value
    return groups


def _bind_top_level_tool_result(
    events: list[dict[str, Any]],
    *,
    next_group_index: int,
) -> None:
    groups = _top_level_assistant_groups(events)
    previous_group = groups[next_group_index - 1]
    next_group = groups[next_group_index]
    previous_index = events.index(previous_group[-1])
    next_index = events.index(next_group[0])
    assert next_index == previous_index + 2
    tool_uses = [
        item
        for event in previous_group
        for item in event["message"]["content"]
        if item.get("type") == "tool_use"
    ]
    assert len(tool_uses) == 1
    boundary = events[previous_index + 1]
    events[previous_index + 1] = _user_event(
        event_id=boundary["uuid"],
        session_id=boundary["session_id"],
        content=[
            {
                "type": "tool_result",
                "tool_use_id": tool_uses[0]["id"],
                "content": "tool result",
                "is_error": False,
            }
        ],
        parent_tool_use_id=None,
    )


def _task_evidence(task_key_id: str, logical: int, records: int) -> dict[str, Any]:
    return {
        "schema": "fr13-fixed32-task-auth-evidence-v1",
        "task_key_id": task_key_id,
        "completed_logical_model_requests": logical,
        "aborted_logical_requests": 0,
        "accepted_attempts": logical,
        "completed_attempts": logical,
        "failed_attempts": 0,
        "phase": "campaign",
        "ledger_records": records,
        "ledger_chain_head_sha256": f"{records:064x}",
    }


def _qwen_compaction_metrics(
    *,
    completed: int,
    compactions: int,
    normal_requests: int,
    prompt_tokens: int,
    generation_tokens: int,
    before_offset: int = 0,
    overrides: dict[str, int] | None = None,
) -> tuple[bytes, bytes]:
    values = {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "max_tokens_count": completed,
        "max_tokens_sum": (
            normal_requests * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS
            + compactions * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS
        ),
        "max_tokens_le_10000": 0,
        "max_tokens_le_20000": compactions,
        "max_tokens_le_50000": completed,
        "max_tokens_le_inf": completed,
        "request_success_stop": completed,
        "request_success_length": 0,
        "request_success_abort": 0,
        "request_success_error": 0,
        "request_success_repetition": 0,
    }
    values.update(overrides or {})

    def render(deltas: dict[str, int], *, post: bool) -> bytes:
        def value(key: str) -> int:
            return before_offset + (deltas[key] if post else 0)

        base_labels = 'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"'
        lines = [
            (
                f"vllm:prompt_tokens_total{{{base_labels}}} "
                f"{value('prompt_tokens')}"
            ),
            (
                f"vllm:generation_tokens_total{{{base_labels}}} "
                f"{value('generation_tokens')}"
            ),
            (
                "vllm:request_params_max_tokens_count"
                f"{{{base_labels}}} {value('max_tokens_count')}"
            ),
            (
                "vllm:request_params_max_tokens_sum"
                f"{{{base_labels}}} {value('max_tokens_sum')}"
            ),
        ]
        for reason in ("stop", "length", "abort", "error", "repetition"):
            labels = (
                f'engine="0",finished_reason="{reason}",'
                'model_name="qwen3.8-27b-nvfp4-radixark"'
            )
            lines.append(
                f"vllm:request_success_total{{{labels}}} "
                f"{value(f'request_success_{reason}')}"
            )
        for le, key in (
            ("10000.0", "max_tokens_le_10000"),
            ("20000.0", "max_tokens_le_20000"),
            ("50000.0", "max_tokens_le_50000"),
            ("+Inf", "max_tokens_le_inf"),
        ):
            labels = (
                f'engine="0",le="{le}",model_name="qwen3.8-27b-nvfp4-radixark"'
            )
            lines.append(
                "vllm:request_params_max_tokens_bucket"
                f"{{{labels}}} {value(key)}"
            )
        return ("\n".join(lines) + "\n").encode("ascii")

    return render(values, post=False), render(values, post=True)


def _qwen_failed_compaction_trace() -> list[dict[str, Any]]:
    events = _qwen_result_trace()
    _set_top_level_group_input_tokens(
        events,
        [100 * index for index in range(1, 13)] + [500],
    )
    _bind_top_level_tool_result(events, next_group_index=12)
    events[-1]["usage"] = {
        "input_tokens": 8_500,
        "output_tokens": 100,
        "total_tokens": 8_600,
    }
    return events


def _qwen_failed_only_compaction_trace() -> list[dict[str, Any]]:
    events = _qwen_result_trace()
    result = events[-1]
    text = (
        "[API Error: Context is too large to send safely after automatic "
        "compression. Estimated prompt tokens: 78280; hard limit: 75304; "
        "compression status: COMPRESSION_FAILED_EMPTY_SUMMARY. Start a new "
        "session or reduce the resumed history before continuing.]"
    )
    synthetic_id = "synthetic-compaction-failure"
    events[-3:-1] = [
        {
            "type": "assistant",
            "uuid": synthetic_id,
            "session_id": result["session_id"],
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
    ]
    result["result"] = text
    result["usage"] = {
        "input_tokens": 52,
        "output_tokens": 20,
        "total_tokens": 72,
    }
    return events


def _fixed32_bundle_observation(runner: Any) -> dict[str, Any]:
    return {
        "qwen_code_version": runner._FIXED32_QWEN_CODE_VERSION,
        "bundle_tree": copy.deepcopy(
            runner._FIXED32_QWEN_BUNDLE_TREE_EXPECTED
        ),
    }


def _fixed32_agent_meta(
    runner: Any,
    task_dir: Path,
    instance_id: str = TASK_A,
    **overrides: Any,
) -> dict[str, Any]:
    workspace = task_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    attestation = runner._build_fixed32_qwen_runtime_attestation(
        bundle_observation=_fixed32_bundle_observation(runner),
        host_mode="remote",
    )
    digest = runner._persist_fixed32_qwen_runtime_attestation(
        workspace=workspace,
        attestation=attestation,
    )
    post_digest = runner._persist_fixed32_qwen_runtime_attestation(
        workspace=workspace,
        attestation=attestation,
        filename="qwen_runtime_attestation_post.json",
    )
    pinned_image = runner._FIXED32_AGENT_IMAGE_IDENTITIES[instance_id]
    image = pinned_image["repo_digest"].split("@", 1)[0] + ":latest"
    image_identity = runner._validate_fixed32_agent_image_observation(
        {
            "instance_id": instance_id,
            "image": image,
            "id": pinned_image["id"],
            "repo_digest": pinned_image["repo_digest"],
            "architecture": "amd64",
            "os": "linux",
        },
        instance_id=instance_id,
        expected_image=image,
    )
    image_digest = runner._fixed32_canonical_json_sha256(image_identity)
    placement = runner._validate_fixed32_agent_placement_observation(
        copy.deepcopy(runner._FIXED32_AGENT_HOST_IDENTITY),
        measured_observation=copy.deepcopy(
            runner._FIXED32_MEASURED_HOST_IDENTITY
        ),
        remote_host="alienware",
    )
    placement_digest = runner._fixed32_canonical_json_sha256(placement)
    bundle_observation = _fixed32_bundle_observation(runner)
    remote_settings_observation = {
        **runner._fixed32_expected_remote_settings_observation(),
        "file_identity_sha256": "1" * 64,
    }
    remote_settings_digest = runner._fixed32_canonical_json_sha256(
        remote_settings_observation
    )
    mounted_proof = {
        "schema": runner._FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": bundle_observation,
        },
        "system_settings": {
            "container_path": runner._FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **remote_settings_observation,
        },
    }
    mounted_proof_digest = (
        runner._validate_fixed32_mounted_runtime_proof(
            mounted_proof,
            expected_bundle_observation=bundle_observation,
        )
    )
    mounted_proof_path = (
        task_dir / runner._FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
    )
    mounted_proof_path.write_text(
        json.dumps(
            mounted_proof,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    meta = {
        "exit_code": 0,
        "timed_out": False,
        "offloaded": True,
        "network_drop": False,
        "agent_env": "instance_image",
        "instance_image": image,
        "instance_image_identity": image_identity,
        "instance_image_identity_sha256": image_digest,
        "instance_image_postrun_identity_sha256": image_digest,
        "instance_image_run_reference": pinned_image["repo_digest"],
        "agent_placement": placement,
        "agent_placement_sha256": placement_digest,
        "agent_postrun_placement_sha256": placement_digest,
        "qwen_bundle_snapshot": attestation["bundle_snapshot"],
        "qwen_remote_settings_observation": remote_settings_observation,
        "qwen_remote_settings_observation_sha256": remote_settings_digest,
        "qwen_remote_settings_postrun_observation_sha256": (
            remote_settings_digest
        ),
        "qwen_mounted_runtime_proof": mounted_proof,
        "qwen_mounted_runtime_proof_sha256": mounted_proof_digest,
        "qwen_mounted_runtime_proof_file_sha256": (
            runner.hashlib.sha256(mounted_proof_path.read_bytes()).hexdigest()
        ),
        "qwen_runtime_attestation": attestation,
        "qwen_runtime_attestation_sha256": digest,
        "qwen_runtime_postrun_attestation_sha256": post_digest,
    }
    meta.update(overrides)
    return meta


def test_legacy_terminal_records_return_without_a_result_event() -> None:
    events = [
        _assistant_event(
            response_id="legacy-response",
            session_id="legacy-session",
            content=[
                {
                    "type": "tool_use",
                    "id": "legacy-tool",
                    "name": "read_file",
                    "input": {},
                }
            ],
            stop_reason="tool_use",
        )
    ]

    trace_requests = contract.validate_fixed32_trace_model_requests(events)

    assert trace_requests == {
        "trace_format": "legacy_terminal_records",
        "completed_logical_model_requests": 1,
        "model_request_ids": ["legacy-response"],
        "hidden_terminal_model_requests": 0,
        "hidden_compaction_model_requests": 0,
        "engine_id_joinable": True,
    }


def test_qwen_result_trace_counts_the_final_null_stop_turn(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_result_trace()
    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "a" * 64

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )
    assert trace_requests["trace_format"] == "qwen_result"
    assert trace_requests["completed_logical_model_requests"] == 13
    assert trace_requests["hidden_compaction_model_requests"] == 0
    assert trace_requests["engine_id_joinable"] is False
    assert len(trace_requests["model_request_ids"]) == 13
    assert len(set(trace_requests["model_request_ids"])) == 13
    assert all(
        request_id.startswith("qwen-assistant-group-sha256:")
        for request_id in trace_requests["model_request_ids"]
    )
    assert trace_requests["model_request_ids"] == (
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )["model_request_ids"]
    )

    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 13, 53),
    )
    assert provenance["trace_completed_logical_model_requests"] == 13
    assert provenance["completed_logical_model_requests"] == 13

    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 13
    assert len(floor_trace["model_request_id_sha256s"]) == 13
    assert floor_trace["engine_id_joinable"] is False


def test_real_task_provenance_binds_prevalidated_campaign_requests(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_result_trace()
    base = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )
    campaign_digest = "a" * 64
    base_ids_digest = hashlib.sha256(
        json.dumps(
            base["model_request_ids"],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    campaign_requests = {
        **base,
        "qwen_campaign_metric_evidence_sha256": campaign_digest,
        "qwen_compaction_metric_evidence": {
            "schema": contract.QWEN_CAMPAIGN_TASK_METRIC_SCHEMA,
            "campaign_metric_evidence_sha256": campaign_digest,
            "base_model_request_ids_sha256": base_ids_digest,
            "trace_completed_requests_before_failed_compactions": 13,
        },
    }
    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "f" * 64
    proof = {
        "path": str((tmp_path / "campaign.json").resolve()),
        "sha256": "b" * 64,
        "bytes": 100,
    }

    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 13, 53),
        campaign_trace_requests=campaign_requests,
        campaign_metric_binding={
            "artifact": proof,
            "metric_evidence_sha256": campaign_digest,
        },
    )

    assert provenance["qwen_metric_scope"] == "campaign"
    assert provenance["qwen_campaign_metric_proof"] == proof
    assert provenance["qwen_campaign_metric_evidence_sha256"] == campaign_digest


def test_qwen_top_level_usage_drop_counts_hidden_compaction(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_result_trace()
    _set_top_level_group_input_tokens(
        events,
        [100 * index for index in range(1, 13)] + [500],
    )
    _bind_top_level_tool_result(events, next_group_index=12)

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 14
    assert trace_requests["hidden_compaction_model_requests"] == 1
    compaction_ids = [
        request_id
        for request_id in trace_requests["model_request_ids"]
        if request_id.startswith("qwen-hidden-compaction-sha256:")
    ]
    assert len(compaction_ids) == 1
    assert trace_requests["model_request_ids"] == (
        contract.validate_fixed32_trace_model_requests(
            copy.deepcopy(events),
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )["model_request_ids"]
    )

    trace_path = tmp_path / "qwen_compaction_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "d" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 14, 57),
    )
    assert provenance["trace_completed_logical_model_requests"] == 14
    assert provenance["completed_logical_model_requests"] == 14
    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 14
    assert len(floor_trace["model_request_id_sha256s"]) == 14


def test_qwen_failed_compactions_reconcile_from_pinned_metrics(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_failed_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=8_500,
        generation_tokens=100,
        before_offset=17,
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        expected_completed_logical_model_requests=16,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )

    assert trace_requests["completed_logical_model_requests"] == 16
    assert trace_requests["hidden_compaction_model_requests"] == 3
    assert trace_requests["hidden_successful_compaction_model_requests"] == 1
    assert trace_requests["hidden_failed_compaction_model_requests"] == 2
    failed_ids = [
        request_id
        for request_id in trace_requests["model_request_ids"]
        if request_id.startswith(
            "qwen-hidden-failed-compaction-sha256:"
        )
    ]
    assert len(failed_ids) == 2
    assert len(set(failed_ids)) == 2
    assert trace_requests["model_request_ids"] == (
        contract.validate_fixed32_trace_model_requests(
            copy.deepcopy(events),
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=16,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )["model_request_ids"]
    )

    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    metrics_pre_path = tmp_path / "vllm_metrics_pre.txt"
    metrics_post_path = tmp_path / "vllm_metrics_post.txt"
    metrics_pre_path.write_bytes(metrics_pre)
    metrics_post_path.write_bytes(metrics_post)
    task_key_id = "e" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 16, 65),
        metrics_pre_path=metrics_pre_path,
        metrics_post_path=metrics_post_path,
    )
    assert provenance["trace_completed_logical_model_requests"] == 16
    assert provenance["hidden_successful_compaction_model_requests"] == 1
    assert provenance["hidden_failed_compaction_model_requests"] == 2
    evidence = provenance["qwen_compaction_metric_evidence"]
    assert evidence["normal_requests"] == 13
    assert evidence["total_compaction_requests"] == 3
    assert evidence["failed_compaction_requests"] == 2

    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 16
    assert len(floor_trace["model_request_id_sha256s"]) == 16


def test_qwen_failed_only_compactions_require_exact_synthetic_terminal() -> None:
    events = _qwen_failed_only_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=4,
        normal_requests=12,
        prompt_tokens=52,
        generation_tokens=20,
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        expected_completed_logical_model_requests=16,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )

    assert trace_requests["completed_logical_model_requests"] == 16
    assert trace_requests["hidden_successful_compaction_model_requests"] == 0
    assert trace_requests["hidden_failed_compaction_model_requests"] == 4
    assert trace_requests["synthetic_compaction_failure_terminal"] is True


@pytest.mark.parametrize("tamper", ("usage_key", "result_text"))
def test_qwen_failed_only_compaction_terminal_near_miss_fails_closed(
    tamper: str,
) -> None:
    events = _qwen_failed_only_compaction_trace()
    if tamper == "usage_key":
        events[-2]["message"]["usage"]["total_tokens"] = 0
    else:
        events[-1]["result"] += " "
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=52,
        generation_tokens=20,
    )

    with pytest.raises(
        contract.ContractError,
        match="exact synthetic failure terminal",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=16,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


def _qwen_subagent_compaction_trace() -> list[dict[str, Any]]:
    """Delegated conversation whose assistant records carry no usage.

    Production Qwen traces report ``{"input_tokens": 0, "output_tokens": 0}``
    on every assistant record inside a delegated (sub-agent) conversation, so
    a compaction performed there can never surface as a top-level
    input-token drop.
    """
    events = _nested_agent_qwen_result_trace()
    for event in events:
        if (
            event.get("type") == "assistant"
            and event.get("parent_tool_use_id") is not None
        ):
            event["message"]["usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
            }
    _set_top_level_group_input_tokens(events, [10, 20])
    events[-1]["usage"] = {
        "input_tokens": 50,
        "output_tokens": 8,
        "total_tokens": 58,
    }
    return events


def test_qwen_subagent_compaction_reconciles_without_a_visible_drop() -> None:
    events = _qwen_subagent_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=6,
        compactions=1,
        normal_requests=5,
        prompt_tokens=50,
        generation_tokens=8,
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        expected_completed_logical_model_requests=6,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )

    assert trace_requests["completed_logical_model_requests"] == 6
    assert trace_requests["hidden_successful_compaction_model_requests"] == 0
    assert trace_requests["hidden_failed_compaction_model_requests"] == 1
    assert trace_requests["synthetic_compaction_failure_terminal"] is False
    evidence = trace_requests["qwen_compaction_metric_evidence"]
    assert evidence["unobservable_compaction_boundaries"] == 1
    assert evidence["normal_requests"] == 5
    assert evidence["total_compaction_requests"] == 1


def test_qwen_compactions_beyond_unobservable_boundaries_fail_closed() -> None:
    events = _qwen_subagent_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=7,
        compactions=2,
        normal_requests=5,
        prompt_tokens=50,
        generation_tokens=8,
    )

    with pytest.raises(
        contract.ContractError,
        match="exact synthetic failure terminal",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=7,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


def test_top_level_trace_has_no_unobservable_boundaries() -> None:
    events = _qwen_failed_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=8_500,
        generation_tokens=100,
        before_offset=17,
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        expected_completed_logical_model_requests=16,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )

    evidence = trace_requests["qwen_compaction_metric_evidence"]
    assert evidence["unobservable_compaction_boundaries"] == 0


@pytest.mark.parametrize(
    ("overrides", "expected_completed", "message"),
    (
        ({"max_tokens_sum": 485_985}, 16, "max-token algebra does not reconcile"),
        ({"max_tokens_le_20000": 2}, 16, "max-token algebra does not reconcile"),
        ({"max_tokens_le_10000": 1}, 16, "unpinned low"),
        ({"max_tokens_count": 15}, 16, "completion metrics"),
        ({"request_success_stop": 15}, 16, "completion metrics"),
        ({"request_success_error": 1}, 16, "completion metrics"),
        ({"prompt_tokens": 8_499}, 16, "aggregate and vLLM"),
        ({"generation_tokens": 99}, 16, "aggregate and vLLM"),
        ({}, 15, "completion metrics"),
    ),
)
def test_qwen_failed_compaction_metric_tamper_fails_closed(
    overrides: dict[str, int],
    expected_completed: int,
    message: str,
) -> None:
    events = _qwen_failed_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=8_500,
        generation_tokens=100,
        overrides=overrides,
    )

    with pytest.raises(contract.ContractError, match=message):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=expected_completed,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda raw: b"\n".join(raw.splitlines()[1:]) + b"\n",
            "missing",
        ),
        (
            lambda raw: raw + raw.splitlines(keepends=True)[0],
            "duplicated",
        ),
        (
            lambda raw: raw.replace(
                b'engine="0",model_name=',
                b'engine="1",model_name=',
                1,
            ),
            "labels differ",
        ),
        (
            lambda raw: raw.replace(
                b"vllm:prompt_tokens_total"
                b'{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} 8500',
                b"vllm:prompt_tokens_total"
                b'{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} -1',
            ),
            "nonnegative integer",
        ),
    ),
)
def test_qwen_malformed_compaction_metrics_fail_closed(
    mutate: Any,
    message: str,
) -> None:
    events = _qwen_failed_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=8_500,
        generation_tokens=100,
    )

    with pytest.raises(contract.ContractError, match=message):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=16,
            metrics_pre=metrics_pre,
            metrics_post=mutate(metrics_post),
        )


def test_qwen_raw_request_count_subtraction_cannot_create_compactions() -> None:
    events = _qwen_result_trace()
    events[-1]["usage"] = {
        "input_tokens": 200,
        "output_tokens": 40,
        "total_tokens": 240,
    }
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=15,
        compactions=2,
        normal_requests=13,
        prompt_tokens=200,
        generation_tokens=40,
    )

    with pytest.raises(
        contract.ContractError,
        match="lack a trace-visible successful compaction",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=15,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


def test_qwen_ordinary_request_mismatch_cannot_be_reclassified() -> None:
    events = _qwen_result_trace()
    events[-1]["usage"] = {
        "input_tokens": 14,
        "output_tokens": 14,
        "total_tokens": 28,
    }
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=14,
        compactions=0,
        normal_requests=14,
        prompt_tokens=14,
        generation_tokens=14,
    )

    with pytest.raises(
        contract.ContractError, match="max-token algebra does not reconcile"
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=14,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


def test_qwen_nonpinned_hidden_request_algebra_fails_closed() -> None:
    events = _qwen_failed_compaction_trace()
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=16,
        compactions=3,
        normal_requests=13,
        prompt_tokens=8_500,
        generation_tokens=100,
        overrides={
            "max_tokens_sum": 16
            * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS,
        },
    )

    with pytest.raises(
        contract.ContractError, match="max-token algebra does not reconcile"
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
            expected_completed_logical_model_requests=16,
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )


def test_qwen_multiple_top_level_usage_drops_count_exactly() -> None:
    events = _qwen_result_trace()
    _set_top_level_group_input_tokens(
        events,
        [100, 200, 50, 100, 200, 60, 100, 200, 300, 400, 500, 600, 700],
    )
    _bind_top_level_tool_result(events, next_group_index=2)
    _bind_top_level_tool_result(events, next_group_index=5)

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 15
    assert trace_requests["hidden_compaction_model_requests"] == 2


@pytest.mark.parametrize(
    ("previous_input_tokens", "next_input_tokens"),
    ((0, 1), (1, 0), (0, 0)),
)
def test_qwen_zero_usage_cannot_infer_hidden_compaction(
    previous_input_tokens: int,
    next_input_tokens: int,
) -> None:
    events = _qwen_result_trace()
    values = [0] * 11 + [previous_input_tokens, next_input_tokens]
    _set_top_level_group_input_tokens(events, values)
    _bind_top_level_tool_result(events, next_group_index=12)

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["hidden_compaction_model_requests"] == 0


@pytest.mark.parametrize("value", (True, -1, "100", None))
def test_qwen_malformed_usage_cannot_infer_hidden_compaction(
    value: Any,
) -> None:
    events = _qwen_result_trace()
    groups = _set_top_level_group_input_tokens(
        events,
        [100 * index for index in range(1, 13)] + [500],
    )
    _bind_top_level_tool_result(events, next_group_index=12)
    groups[-2][-1]["message"]["usage"]["input_tokens"] = value

    with pytest.raises(contract.ContractError, match="input-token usage"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_qwen_usage_drop_requires_exact_top_level_tool_results() -> None:
    events = _qwen_result_trace()
    _set_top_level_group_input_tokens(
        events,
        [100 * index for index in range(1, 13)] + [500],
    )

    with pytest.raises(contract.ContractError, match="top-level tool results"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def _move_result_before_final_text(events: list[dict[str, Any]]) -> None:
    result = events.pop()
    events.insert(-1, result)


def _remove_final_text(events: list[dict[str, Any]]) -> None:
    """Close the trace on the reasoning-only turn that precedes the text."""
    del events[-2]


def _blank_final_text(events: list[dict[str, Any]]) -> None:
    """Blank the closing text record, leaving the reasoning record before it.

    This is a legal live shape, not tampering -- see
    ``test_qwen_blank_final_text_after_reasoning_is_served``. Blanking text
    cannot forge served work: the group still counts as exactly one logical
    model request and that count must reconcile against the engine's own
    metrics, which is where trace tampering is actually caught.
    """
    events[-2]["message"]["content"] = [{"type": "text", "text": "   "}]


def _duplicate_assistant_identity(events: list[dict[str, Any]]) -> None:
    events[3]["uuid"] = events[1]["uuid"]
    events[3]["message"]["id"] = events[1]["uuid"]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda events: events[-1].__setitem__("num_turns", 12),
        lambda events: events[-1].__setitem__("num_turns", 14),
        _move_result_before_final_text,
        lambda events: events[-1].__setitem__("subtype", "error"),
        lambda events: events[-1].__setitem__("is_error", True),
        lambda events: events[-2]["message"].__setitem__("id", ""),
        lambda events: events[-2]["message"].__setitem__("id", "tool-turn-11"),
    ),
)
def test_qwen_result_trace_tamper_fails_closed(
    mutate: Any,
) -> None:
    events = copy.deepcopy(_qwen_result_trace())
    mutate(events)

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_qwen_blank_final_text_after_reasoning_is_served() -> None:
    """A closing turn may trail its reasoning with a whitespace-only record.

    Observed live: astropy__astropy-13398 closed a 434-event trajectory on
    ``"\\n\\n"`` after a thinking record, having already produced a real
    2216-byte patch. The turn was served, so it counts like any other group.
    """
    canonical = contract.validate_fixed32_trace_model_requests(
        _qwen_result_trace(),
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )
    events = copy.deepcopy(_qwen_result_trace())
    _blank_final_text(events)
    final_group = _top_level_assistant_groups(events)[-1]
    assert [
        item["type"] for item in final_group[0]["message"]["content"]
    ] == ["thinking"]
    assert final_group[-1]["message"]["content"] == [
        {"type": "text", "text": "   "}
    ]

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    # Blanking the visible text changes nothing about how much work was
    # served: same request count, same group identities.
    assert trace_requests["completed_logical_model_requests"] == (
        canonical["completed_logical_model_requests"]
    )
    assert trace_requests["model_request_ids"] == canonical["model_request_ids"]


def test_qwen_blank_final_text_still_reconciles_against_engine_metrics() -> None:
    """The count control, not the text, is what catches a forged trace."""
    events = copy.deepcopy(_qwen_result_trace())
    _blank_final_text(events)
    session_id = contract.fixed32_trace_session_id(TASK_A)

    truthful = contract.validate_fixed32_trace_model_requests(
        events, expected_session_id=session_id
    )
    served = truthful["completed_logical_model_requests"]

    # Accepting the blank close does not let a trace claim work the engine
    # never served: the metric-proven count still has to agree exactly.
    for wrong in (served - 1, served + 1):
        with pytest.raises(contract.ContractError):
            contract.validate_fixed32_trace_model_requests(
                events,
                expected_session_id=session_id,
                expected_completed_logical_model_requests=wrong,
            )


def test_qwen_blank_final_group_without_reasoning_fails_closed() -> None:
    """A bare whitespace close carries no evidence the turn was served."""
    events = copy.deepcopy(_qwen_result_trace())
    # Drop the reasoning record, leaving only the blank text record.
    _blank_final_text(events)
    del events[-3]

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def _reasoning_only_final_qwen_result_trace() -> list[dict[str, Any]]:
    """The astropy__astropy-13398 shape: the task closes on reasoning.

    The final top-level assistant group is one record whose only content is a
    ``thinking`` block, so the run emits no closing text and the Qwen result
    carries the empty string. The engine still served that turn.
    """
    events = copy.deepcopy(_qwen_result_trace())
    _remove_final_text(events)
    events[-2]["message"]["content"] = [
        {
            "type": "thinking",
            "thinking": "This is the key test - test_straight_overhead.",
            "signature": "",
        }
    ]
    events[-1]["result"] = ""
    return events


def test_qwen_reasoning_only_final_group_counts_as_served(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    canonical = contract.validate_fixed32_trace_model_requests(
        _qwen_result_trace(),
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )
    events = _reasoning_only_final_qwen_result_trace()
    final_group = _top_level_assistant_groups(events)[-1]
    assert len(final_group) == 1
    assert [
        item["type"] for item in final_group[0]["message"]["content"]
    ] == ["thinking"]

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["trace_format"] == "qwen_result"
    # The reasoning-only close is served work: it contributes exactly the
    # one logical model request the canonical text close contributes.
    assert trace_requests["completed_logical_model_requests"] == (
        canonical["completed_logical_model_requests"]
    )
    assert trace_requests["completed_logical_model_requests"] == 13
    assert trace_requests["synthetic_compaction_failure_terminal"] is False
    assert trace_requests["hidden_compaction_model_requests"] == 0
    request_ids = trace_requests["model_request_ids"]
    assert len(request_ids) == len(set(request_ids)) == 13
    # Only the final group's identity moves; it is still present.
    assert request_ids[:-1] == canonical["model_request_ids"][:-1]
    assert request_ids[-1] != canonical["model_request_ids"][-1]
    assert request_ids[-1] == contract._fixed32_qwen_group_request_id(
        [final_group[0]["uuid"]]
    )

    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "a" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 13, 53),
    )
    assert provenance["trace_completed_logical_model_requests"] == 13
    assert provenance["completed_logical_model_requests"] == 13

    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 13
    assert len(floor_trace["model_request_id_sha256s"]) == 13


def test_qwen_multi_record_reasoning_only_final_group_is_served() -> None:
    events = _reasoning_only_final_qwen_result_trace()
    session_id = contract.fixed32_trace_session_id(TASK_A)
    events.insert(
        len(events) - 1,
        _assistant_event(
            response_id="final-thinking-continued",
            session_id=session_id,
            content=[{"type": "thinking", "thinking": "still reasoning"}],
            stop_reason=None,
        ),
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=session_id,
    )

    assert trace_requests["completed_logical_model_requests"] == 13
    assert trace_requests["model_request_ids"][-1] == (
        contract._fixed32_qwen_group_request_id(
            ["final-thinking", "final-thinking-continued"]
        )
    )


@pytest.mark.parametrize(
    "final_content",
    (
        [{"type": "text", "text": "   "}],
        [
            {"type": "thinking", "thinking": "complete"},
            {"type": "text", "text": ""},
        ],
        [{"type": "redacted_thinking", "data": "opaque"}],
    ),
)
def test_qwen_final_group_without_text_or_reasoning_fails_closed(
    final_content: list[dict[str, Any]],
) -> None:
    events = _reasoning_only_final_qwen_result_trace()
    events[-2]["message"]["content"] = final_content

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def _parallel_qwen_result_trace() -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(TASK_A)
    events = [
        _context_event(
            event_type="system",
            event_id="parallel-system",
            session_id=session_id,
        ),
        _assistant_event(
            response_id="top-0-thinking",
            session_id=session_id,
            content=[{"type": "thinking", "thinking": "plan"}],
            stop_reason=None,
        ),
        _assistant_event(
            response_id="top-0-text",
            session_id=session_id,
            content=[{"type": "text", "text": "dispatch"}],
            stop_reason=None,
        ),
        _assistant_event(
            response_id="top-0-terminal",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": "parallel-parent-tool",
                    "name": "agent",
                    "input": {
                        "description": "inspect the repository",
                        "prompt": "delegated task",
                        "subagent_type": "Explore",
                    },
                }
            ],
            stop_reason="tool_use",
        ),
        _user_event(
            event_id="parallel-dispatch-result",
            session_id=session_id,
            content=[{"type": "text", "text": "delegated task"}],
            parent_tool_use_id="parallel-parent-tool",
        ),
    ]
    nested_terminal_widths = (3, 2, 2, 1, 1, 1, 2, 1, 1)
    for group_index, terminal_width in enumerate(nested_terminal_widths):
        for terminal_index in range(terminal_width):
            events.append(
                _assistant_event(
                    response_id=(
                        f"nested-{group_index}-terminal-{terminal_index}"
                    ),
                    session_id=session_id,
                    content=[
                        {
                            "type": "tool_use",
                            "id": (
                                f"nested-{group_index}-call-{terminal_index}"
                            ),
                            "name": "read_file",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                    parent_tool_use_id="parallel-parent-tool",
                )
            )
        for terminal_index in range(terminal_width):
            events.append(
                _user_event(
                    event_id=(
                        f"nested-{group_index}-result-{terminal_index}"
                    ),
                    session_id=session_id,
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": (
                                f"nested-{group_index}-call-{terminal_index}"
                            ),
                            "content": "tool result",
                            "is_error": False,
                        }
                    ],
                    parent_tool_use_id="parallel-parent-tool",
                )
            )
    events.append(
        _user_event(
            event_id="parallel-top-resume",
            session_id=session_id,
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "parallel-parent-tool",
                    "content": "nested exploration complete",
                    "is_error": False,
                }
            ],
            parent_tool_use_id=None,
        )
    )
    for group_index in range(1, 9):
        events.extend(
            [
                _assistant_event(
                    response_id=f"top-{group_index}-thinking",
                    session_id=session_id,
                    content=[{"type": "thinking", "thinking": "plan"}],
                    stop_reason=None,
                ),
                _assistant_event(
                    response_id=f"top-{group_index}-text",
                    session_id=session_id,
                    content=[{"type": "text", "text": "act"}],
                    stop_reason=None,
                ),
                _assistant_event(
                    response_id=f"top-{group_index}-terminal",
                    session_id=session_id,
                    content=[
                        {
                            "type": "tool_use",
                            "id": f"top-{group_index}-call",
                            "name": "read_file",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                ),
                _context_event(
                    event_type="user",
                    event_id=f"top-{group_index}-result",
                    session_id=session_id,
                ),
            ]
        )
    events.extend(
        [
            _assistant_event(
                response_id="parallel-final-thinking",
                session_id=session_id,
                content=[{"type": "thinking", "thinking": "complete"}],
                stop_reason=None,
            ),
            _assistant_event(
                response_id="parallel-final-text",
                session_id=session_id,
                content=[{"type": "text", "text": "completed"}],
                stop_reason=None,
            ),
            {
                "type": "result",
                "subtype": "success",
                "uuid": "parallel-result",
                "session_id": session_id,
                "is_error": False,
                "duration_ms": 100,
                "duration_api_ms": 90,
                "num_turns": 10,
                "result": "completed",
                "usage": {"input_tokens": 2, "output_tokens": 2},
                "permission_denials": [],
            },
        ]
    )
    return events


def _insert_nested_error_boundary(events: list[dict[str, Any]]) -> int:
    boundary_index = next(
        index
        for index, event in enumerate(events)
        if event.get("uuid") == "parallel-top-resume"
    )
    events.insert(
        boundary_index,
        {
            "type": "result",
            "subtype": "error_during_execution",
            "uuid": "nested-error-boundary",
            "session_id": contract.fixed32_trace_session_id(TASK_A),
            "is_error": True,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 0,
            "error": {"message": "nested task failed"},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "permission_denials": [],
        },
    )
    return boundary_index


def test_parallel_terminal_records_count_as_20_model_requests(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    session_id = contract.fixed32_trace_session_id(TASK_A)
    events = _parallel_qwen_result_trace()

    assert sum(event.get("type") == "assistant" for event in events) == 43
    assert (
        sum(
            event.get("type") == "assistant"
            and event["message"].get("stop_reason") == "tool_use"
            for event in events
        )
        == 23
    )
    assert events[-1]["num_turns"] == 10

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=session_id,
    )

    assert trace_requests["completed_logical_model_requests"] == 20
    assert trace_requests["hidden_terminal_model_requests"] == 1
    assert len(trace_requests["model_request_ids"]) == 20
    assert len(set(trace_requests["model_request_ids"])) == 20

    trace_path = tmp_path / "qwen_parallel_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "b" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 20, 81),
    )
    assert provenance["trace_completed_logical_model_requests"] == 20
    assert provenance["completed_logical_model_requests"] == 20

    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 20
    assert len(floor_trace["model_request_id_sha256s"]) == 20
    assert floor_trace["engine_id_joinable"] is False


def _nested_agent_qwen_result_trace() -> list[dict[str, Any]]:
    session_id = contract.fixed32_trace_session_id(TASK_A)
    return [
        _context_event(
            event_type="system",
            event_id="nested-system",
            session_id=session_id,
        ),
        _assistant_event(
            response_id="nested-top-agent",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": "nested-agent-tool",
                    "name": "agent",
                    "input": {
                        "description": "delegate task",
                        "prompt": "delegated task",
                    },
                }
            ],
            stop_reason="tool_use",
        ),
        _user_event(
            event_id="nested-agent-prompt",
            session_id=session_id,
            content=[{"type": "text", "text": "delegated task"}],
            parent_tool_use_id="nested-agent-tool",
        ),
        _assistant_event(
            response_id="nested-visible-turn-0",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": "nested-call-0",
                    "name": "read_file",
                    "input": {},
                }
            ],
            stop_reason="tool_use",
            parent_tool_use_id="nested-agent-tool",
        ),
        _user_event(
            event_id="nested-result-0",
            session_id=session_id,
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "nested-call-0",
                    "content": "first result",
                    "is_error": False,
                }
            ],
            parent_tool_use_id="nested-agent-tool",
        ),
        _assistant_event(
            response_id="nested-visible-turn-1",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": "nested-call-1",
                    "name": "read_file",
                    "input": {},
                }
            ],
            stop_reason="tool_use",
            parent_tool_use_id="nested-agent-tool",
        ),
        _user_event(
            event_id="nested-result-1",
            session_id=session_id,
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "nested-call-1",
                    "content": "second result",
                    "is_error": False,
                }
            ],
            parent_tool_use_id="nested-agent-tool",
        ),
        _user_event(
            event_id="nested-agent-result",
            session_id=session_id,
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "nested-agent-tool",
                    "content": "agent terminal response",
                    "is_error": False,
                }
            ],
            parent_tool_use_id=None,
        ),
        _assistant_event(
            response_id="nested-top-final",
            session_id=session_id,
            content=[{"type": "text", "text": "completed"}],
            stop_reason=None,
        ),
        {
            "type": "result",
            "subtype": "success",
            "uuid": "nested-final-result",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 100,
            "duration_api_ms": 90,
            "num_turns": 2,
            "result": "completed",
            "usage": {"input_tokens": 2, "output_tokens": 2},
            "permission_denials": [],
        },
    ]


def _event_by_uuid(
    events: list[dict[str, Any]],
    event_id: str,
) -> dict[str, Any]:
    return next(event for event in events if event.get("uuid") == event_id)


def test_closed_nested_agent_counts_one_hidden_terminal_request(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _nested_agent_qwen_result_trace()
    expected_session_id = contract.fixed32_trace_session_id(TASK_A)

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=expected_session_id,
    )

    assert trace_requests["completed_logical_model_requests"] == 5
    assert trace_requests["hidden_terminal_model_requests"] == 1
    assert trace_requests["hidden_compaction_model_requests"] == 0
    assert len(trace_requests["model_request_ids"]) == 5
    hidden_ids = [
        request_id
        for request_id in trace_requests["model_request_ids"]
        if request_id.startswith("qwen-hidden-agent-terminal-sha256:")
    ]
    assert len(hidden_ids) == 1
    assert trace_requests["model_request_ids"] == (
        contract.validate_fixed32_trace_model_requests(
            copy.deepcopy(events),
            expected_session_id=expected_session_id,
        )["model_request_ids"]
    )

    trace_path = tmp_path / "nested_agent_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "c" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(task_key_id, 5, 21),
    )
    assert provenance["trace_completed_logical_model_requests"] == 5
    assert provenance["completed_logical_model_requests"] == 5
    assert floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )["completed_logical_model_requests"] == 5


def test_nested_to_top_level_usage_drop_is_not_compaction() -> None:
    events = _nested_agent_qwen_result_trace()
    _set_top_level_group_input_tokens(events, [100, 200])
    nested_groups: list[list[dict[str, Any]]] = []
    previous_was_nested_assistant = False
    for event in events:
        is_nested_assistant = (
            event.get("type") == "assistant"
            and event.get("parent_tool_use_id") == "nested-agent-tool"
        )
        if not is_nested_assistant:
            previous_was_nested_assistant = False
            continue
        if previous_was_nested_assistant:
            nested_groups[-1].append(event)
        else:
            nested_groups.append([event])
        previous_was_nested_assistant = True
    assert len(nested_groups) == 2
    for group, input_tokens in zip(
        nested_groups,
        (500, 1000),
        strict=True,
    ):
        for event in group:
            event["message"]["usage"]["input_tokens"] = 0
        group[-1]["message"]["usage"]["input_tokens"] = input_tokens

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 5
    assert trace_requests["hidden_terminal_model_requests"] == 1
    assert trace_requests["hidden_compaction_model_requests"] == 0


def test_nested_non_agent_parent_fails_closed() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-top-agent")["message"]["content"][0][
        "name"
    ] = "read_file"

    with pytest.raises(contract.ContractError, match="non-agent parent"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_prompt_only_agent_counts_hidden_terminal_request() -> None:
    events = _nested_agent_qwen_result_trace()
    events[3:7] = []

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 3
    assert trace_requests["hidden_terminal_model_requests"] == 1
    assert trace_requests["hidden_compaction_model_requests"] == 0


def test_agent_setup_error_without_a_child_prompt_adds_no_request() -> None:
    events = _nested_agent_qwen_result_trace()
    events[2:7] = []
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "is_error"
    ] = True

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 2
    assert trace_requests["hidden_terminal_model_requests"] == 0


def test_schema_rejected_agent_setup_error_adds_no_request() -> None:
    events = _nested_agent_qwen_result_trace()
    events[2:7] = []
    agent_input = _event_by_uuid(events, "nested-top-agent")["message"][
        "content"
    ][0]["input"]
    del agent_input["prompt"]
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "is_error"
    ] = True

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 2
    assert trace_requests["hidden_terminal_model_requests"] == 0


def test_agent_success_without_a_child_prompt_fails_closed() -> None:
    events = _nested_agent_qwen_result_trace()
    events[2:7] = []

    with pytest.raises(contract.ContractError, match="setup-error closure"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    "input_update",
    (
        {"run_in_background": True},
        {"subagent_type": "fork"},
        {"name": "worker"},
        {"isolation": "worktree"},
    ),
)
def test_asynchronous_agent_modes_fail_closed(
    input_update: dict[str, Any],
) -> None:
    events = _nested_agent_qwen_result_trace()
    agent_input = _event_by_uuid(events, "nested-top-agent")["message"][
        "content"
    ][0]["input"]
    agent_input.update(input_update)

    with pytest.raises(contract.ContractError, match="agent invocation"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_in_background", "false"),
        ("isolation", 1),
        ("subagent_type", 1),
        ("subagent_type", " "),
        ("name", 1),
    ),
)
def test_agent_mode_selectors_must_be_typed(
    field: str,
    value: Any,
) -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-top-agent")["message"]["content"][0][
        "input"
    ][field] = value

    with pytest.raises(contract.ContractError, match="selector is invalid"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_agent_input_must_be_an_object() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-top-agent")["message"]["content"][0][
        "input"
    ] = []

    with pytest.raises(contract.ContractError, match="input is not an object"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize("field", ("description", "prompt"))
def test_agent_required_text_fields_must_be_nonempty(field: str) -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-top-agent")["message"]["content"][0][
        "input"
    ][field] = ""

    with pytest.raises(contract.ContractError, match="empty or invalid"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_agent_input_unknown_field_fails_closed() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-top-agent")["message"]["content"][0][
        "input"
    ]["unexpected"] = True

    with pytest.raises(contract.ContractError, match="unknown fields"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_agent_outer_result_content_must_be_nonempty() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "content"
    ] = ""

    with pytest.raises(contract.ContractError, match="content is empty"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    "content",
    (
        "Failed to run subagent: setup failed",
        "Subagent execution failed.",
        "Agent was cancelled by the user. Partial result follows:",
        "(subagent produced no model-visible output)",
    ),
)
def test_agent_failure_owner_results_fail_closed(content: str) -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "content"
    ] = content

    with pytest.raises(contract.ContractError, match="closure is invalid"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_agent_prompt_must_match_the_dispatched_prompt() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-agent-prompt")["message"]["content"][0][
        "text"
    ] = "different task"

    with pytest.raises(contract.ContractError, match="closure is invalid"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def _nested_wrong_terminal_tool_result(
    events: list[dict[str, Any]],
) -> None:
    _event_by_uuid(events, "nested-result-1")["message"]["content"][0][
        "tool_use_id"
    ] = "nested-call-0"


def _nested_missing_terminal_tool_result(
    events: list[dict[str, Any]],
) -> None:
    events.remove(_event_by_uuid(events, "nested-result-1"))


def _nested_outer_wrong_tool_result(
    events: list[dict[str, Any]],
) -> None:
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "tool_use_id"
    ] = "nested-call-1"


def _nested_outer_not_top_level(
    events: list[dict[str, Any]],
) -> None:
    _event_by_uuid(events, "nested-agent-result")[
        "parent_tool_use_id"
    ] = "nested-agent-tool"


def _nested_outer_error_tool_result(
    events: list[dict[str, Any]],
) -> None:
    _event_by_uuid(events, "nested-agent-result")["message"]["content"][0][
        "is_error"
    ] = True


def _nested_interposed_context(
    events: list[dict[str, Any]],
) -> None:
    outer_index = events.index(_event_by_uuid(events, "nested-agent-result"))
    events.insert(
        outer_index,
        _context_event(
            event_type="system",
            event_id="nested-interposed-system",
            session_id=contract.fixed32_trace_session_id(TASK_A),
        ),
    )


def _nested_duplicate_outer_result(
    events: list[dict[str, Any]],
) -> None:
    outer = copy.deepcopy(_event_by_uuid(events, "nested-agent-result"))
    outer["uuid"] = "nested-agent-result-duplicate"
    events.insert(events.index(_event_by_uuid(events, "nested-top-final")), outer)


def _nested_child_event_after_outer_result(
    events: list[dict[str, Any]],
) -> None:
    final_index = events.index(_event_by_uuid(events, "nested-top-final"))
    events.insert(
        final_index,
        _user_event(
            event_id="nested-late-child-event",
            session_id=contract.fixed32_trace_session_id(TASK_A),
            content=[{"type": "text", "text": "late child event"}],
            parent_tool_use_id="nested-agent-tool",
        ),
    )


def _nested_descendant_event_after_outer_result(
    events: list[dict[str, Any]],
) -> None:
    final_index = events.index(_event_by_uuid(events, "nested-top-final"))
    events.insert(
        final_index,
        _user_event(
            event_id="nested-late-descendant-event",
            session_id=contract.fixed32_trace_session_id(TASK_A),
            content=[{"type": "text", "text": "late descendant event"}],
            parent_tool_use_id="nested-call-0",
        ),
    )


@pytest.mark.parametrize(
    "mutate",
    (
        _nested_wrong_terminal_tool_result,
        _nested_missing_terminal_tool_result,
        _nested_outer_wrong_tool_result,
        _nested_outer_not_top_level,
        _nested_outer_error_tool_result,
        _nested_interposed_context,
        _nested_duplicate_outer_result,
        _nested_child_event_after_outer_result,
        _nested_descendant_event_after_outer_result,
    ),
)
def test_nested_agent_closure_tamper_fails_closed(
    mutate: Any,
) -> None:
    events = _nested_agent_qwen_result_trace()
    mutate(events)

    with pytest.raises(contract.ContractError, match="agent"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_failed_child_tool_result_still_reconciles_the_agent_request() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-result-1")["message"]["content"][0][
        "is_error"
    ] = True

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 5
    assert trace_requests["hidden_terminal_model_requests"] == 1


def test_nested_agent_error_boundary_does_not_infer_hidden_request() -> None:
    events = _nested_agent_qwen_result_trace()
    outer_index = events.index(_event_by_uuid(events, "nested-agent-result"))
    events.insert(
        outer_index,
        {
            "type": "result",
            "subtype": "error_during_execution",
            "uuid": "nested-agent-error-boundary",
            "session_id": contract.fixed32_trace_session_id(TASK_A),
            "is_error": True,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 0,
            "error": {"message": "nested task failed"},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "permission_denials": [],
        },
    )

    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert trace_requests["completed_logical_model_requests"] == 4
    assert trace_requests["hidden_terminal_model_requests"] == 0


def test_nested_agent_error_boundary_must_bind_to_that_agent() -> None:
    events = _nested_agent_qwen_result_trace()
    _event_by_uuid(events, "nested-result-1")[
        "parent_tool_use_id"
    ] = "nested-call-1"
    outer_index = events.index(_event_by_uuid(events, "nested-agent-result"))
    events.insert(
        outer_index,
        {
            "type": "result",
            "subtype": "error_during_execution",
            "uuid": "nested-agent-wrong-parent-error-boundary",
            "session_id": contract.fixed32_trace_session_id(TASK_A),
            "is_error": True,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 0,
            "error": {"message": "nested task failed"},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "permission_denials": [],
        },
    )

    with pytest.raises(contract.ContractError, match="belongs to another tool"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_zero_work_nested_error_suppresses_hidden_terminal_request() -> None:
    events = _parallel_qwen_result_trace()
    baseline = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )
    _insert_nested_error_boundary(events)

    with_boundary = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_A),
    )

    assert baseline["completed_logical_model_requests"] == 20
    assert baseline["hidden_terminal_model_requests"] == 1
    assert with_boundary["completed_logical_model_requests"] == 19
    assert with_boundary["hidden_terminal_model_requests"] == 0
    assert with_boundary["model_request_ids"] == [
        request_id
        for request_id in baseline["model_request_ids"]
        if not request_id.startswith("qwen-hidden-agent-terminal-sha256:")
    ]
    assert events[-1]["num_turns"] == 10


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("subtype", "success"),
        ("is_error", False),
        ("num_turns", 1),
        ("duration_ms", 1),
        ("duration_api_ms", 1),
        ("permission_denials", ["denied"]),
        ("result", ""),
        ("parent_tool_use_id", "parallel-parent-tool"),
    ),
)
def test_nested_error_boundary_state_tamper_fails(
    field: str,
    value: Any,
) -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    events[boundary_index][field] = value

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    "usage",
    (
        {"input_tokens": 1, "output_tokens": 0},
        {"input_tokens": 0, "output_tokens": 1},
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        {"input_tokens": False, "output_tokens": 0},
    ),
)
def test_nested_error_boundary_usage_tamper_fails(
    usage: dict[str, Any],
) -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    events[boundary_index]["usage"] = usage

    with pytest.raises(contract.ContractError, match="usage is not zero"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    "error",
    (
        {"message": ""},
        {"message": "failure", "code": "extra"},
        {"message": 1},
    ),
)
def test_nested_error_boundary_message_tamper_fails(
    error: dict[str, Any],
) -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    events[boundary_index]["error"] = error

    with pytest.raises(contract.ContractError, match="message is invalid"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


@pytest.mark.parametrize(
    "identity",
    (
        "",
        "parallel-result",
        "top-0-thinking",
        "parallel-top-resume",
    ),
)
def test_nested_error_boundary_identity_tamper_fails(
    identity: str,
) -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    events[boundary_index]["uuid"] = identity

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def _nested_boundary_previous_parent_is_null(
    events: list[dict[str, Any]],
    index: int,
) -> None:
    events[index - 1]["parent_tool_use_id"] = None


def _nested_boundary_previous_is_not_user(
    events: list[dict[str, Any]],
    index: int,
) -> None:
    events[index - 1]["type"] = "system"


def _nested_boundary_next_parent_is_nested(
    events: list[dict[str, Any]],
    index: int,
) -> None:
    events[index + 1]["parent_tool_use_id"] = "parallel-parent-tool"


def _nested_boundary_next_parent_is_missing(
    events: list[dict[str, Any]],
    index: int,
) -> None:
    del events[index + 1]["parent_tool_use_id"]


def _nested_boundary_next_is_not_user(
    events: list[dict[str, Any]],
    index: int,
) -> None:
    events[index + 1]["type"] = "system"


@pytest.mark.parametrize(
    "mutate",
    (
        _nested_boundary_previous_parent_is_null,
        _nested_boundary_previous_is_not_user,
        _nested_boundary_next_parent_is_nested,
        _nested_boundary_next_parent_is_missing,
        _nested_boundary_next_is_not_user,
    ),
)
def test_nested_error_boundary_transition_tamper_fails(
    mutate: Any,
) -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    mutate(events, boundary_index)

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_nested_error_boundary_session_tamper_fails() -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    events[boundary_index]["session_id"] = "other-session"

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_parallel_response_group_parent_change_fails_closed() -> None:
    events = _parallel_qwen_result_trace()
    nested_records = [
        event
        for event in events
        if event.get("type") == "assistant"
        and event.get("parent_tool_use_id") == "parallel-parent-tool"
    ]
    nested_records[1]["parent_tool_use_id"] = None

    with pytest.raises(
        contract.ContractError,
        match="contiguous assistant group changes parent identity",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_parallel_nested_parent_must_be_ancestral() -> None:
    session_id = contract.fixed32_trace_session_id(TASK_A)
    events = [
        _assistant_event(
            response_id="orphan-nested-terminal",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": "nested-tool-call",
                    "name": "read_file",
                    "input": {},
                }
            ],
            stop_reason="tool_use",
            parent_tool_use_id="future-parent",
        ),
        {
            "type": "result",
            "subtype": "success",
            "uuid": "parallel-result",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 100,
            "duration_api_ms": 90,
            "num_turns": 1,
            "result": "completed",
            "usage": {"input_tokens": 2, "output_tokens": 2},
            "permission_denials": [],
        },
    ]

    with pytest.raises(contract.ContractError, match="non-ancestral parent"):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=session_id,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda events: events[0].__setitem__("session_id", "other-session"),
        lambda events: events[2].__setitem__("session_id", "other-session"),
        lambda events: events[-1].__setitem__("session_id", "other-session"),
        lambda events: events[1].__setitem__("session_id", "other-session"),
        lambda events: events[1].__setitem__("uuid", "other-event"),
        _duplicate_assistant_identity,
        lambda events: events[-1].__setitem__("uuid", events[1]["uuid"]),
    ),
)
def test_qwen_session_or_event_identity_tamper_fails(
    mutate: Any,
) -> None:
    events = copy.deepcopy(_qwen_result_trace())
    mutate(events)

    with pytest.raises(contract.ContractError):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_qwen_multiple_nested_error_records_fail_closed() -> None:
    events = _parallel_qwen_result_trace()
    boundary_index = _insert_nested_error_boundary(events)
    second_boundary = copy.deepcopy(events[boundary_index])
    second_boundary["uuid"] = "second-nested-error-boundary"
    events.insert(boundary_index + 1, second_boundary)

    with pytest.raises(
        contract.ContractError,
        match="at most one nested error boundary",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_A),
        )


def test_same_count_cross_task_trace_swap_fails_session_binding() -> None:
    events = _qwen_result_trace(TASK_A)

    with pytest.raises(
        contract.ContractError,
        match="session does not bind to the task",
    ):
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_B),
        )


def test_qwen_launchers_pin_the_task_session_id() -> None:
    runner = _load_runner()
    session_id = contract.fixed32_trace_session_id(TASK_A)
    command = runner._instance_agent_command(
        container_name="agent",
        image="image",
        endpoint="http://127.0.0.1:8023/v1",
        model="model",
        host_out_dir="/tmp/out",
        bundle_src="/tmp/bundle",
        agents_md_b64="YQ==",
        prompt_b64="Yg==",
        base_commit="deadbeef",
        session_id=session_id,
    )

    assert "--session-id {session_id}" in runner.QWEN_CODE_TEMPLATE
    assert '--session-id "$SWE_SESSION_ID"' in runner._INSTANCE_WRAPPER
    assert f"-e SWE_SESSION_ID='{session_id}'" in command
    assert runner._FIXED32_QWEN_SETTINGS_ENV not in command


def test_fixed32_qwen_launcher_mounts_only_narrow_system_settings() -> None:
    runner = _load_runner()
    command = runner._instance_agent_command(
        container_name="agent",
        image="image",
        endpoint="http://127.0.0.1:8023/v1",
        model="model",
        host_out_dir="/tmp/out",
        bundle_src="/tmp/bundle",
        agents_md_b64="YQ==",
        prompt_b64="Yg==",
        base_commit="deadbeef",
        session_id=contract.fixed32_trace_session_id(TASK_A),
        system_settings_src="/tmp/qwen-system-settings.json",
        bundle_observation=_fixed32_bundle_observation(runner),
    )

    assert (
        "-e QWEN_CODE_SYSTEM_SETTINGS_PATH="
        "/run/fr13/qwen-system-settings.json"
    ) in command
    assert (
        "-v /tmp/qwen-system-settings.json:"
        "/run/fr13/qwen-system-settings.json:ro"
    ) in command
    assert "--bare" not in command
    assert "grep_search" not in command


def test_fixed32_qwen_settings_are_exact_and_auto_skill_only() -> None:
    runner = _load_runner()

    metadata = runner._fixed32_qwen_settings_metadata()

    assert runner._FIXED32_QWEN_SETTINGS_PATH.read_bytes() == (
        b'{"memory":{"enableAutoSkill":false},'
        b'"tools":{"exclude":["web_fetch","web_search","tool_search"]}}\n'
    )
    assert metadata == {
        "source": "config/fr13_fixed32/qwen_system_settings.json",
        "bytes": 98,
        "sha256": (
            "d1c7e744e9febaa96b341e01de24cc6e"
            "a07dd30bdf33352618b2c63de225ee9f"
        ),
        "container_path": "/run/fr13/qwen-system-settings.json",
        "mount_mode": "ro",
        "environment": {
            "name": "QWEN_CODE_SYSTEM_SETTINGS_PATH",
            "value": "/run/fr13/qwen-system-settings.json",
        },
        "remote_file": {
            "mode": "0444",
            "uid": 1000,
            "gid": 1000,
            "nlink": 1,
            "xattrs": [],
        },
        "enable_auto_skill": False,
    }


def test_fixed32_qwen_identity_pins_full_executable_tree() -> None:
    runner = _load_runner()
    tree = runner._FIXED32_QWEN_BUNDLE_TREE_EXPECTED

    assert tree["roots"] == ["**"]
    assert tree["summary"]["entry_count"] == 10_499
    assert tree["summary"]["regular_file_bytes"] == 327_941_291
    assert tree["manifest_sha256"] == (
        "594cac41e2d5ed505e0646f318b263ff"
        "70e200bcffe97326fe1c042fdc220516"
    )
    assert set(tree["entrypoints"]) == set(
        runner._FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS
    )
    assert tree["entrypoints"][
        runner._FIXED32_QWEN_CAP_CHUNK_RELATIVE_PATH
    ]["sha256"] == (
        "d61b71c03180822e875976a721a85614"
        "4b70ae8b7ff687910021a5cb91a7db89"
    )


def test_remote_settings_rehash_expands_tilde_inside_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[list[str]] = []

    def fake_net_retry(
        argv: list[str],
        **_kwargs: Any,
    ) -> Any:
        calls.append(argv)
        return runner.subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    **runner._fixed32_expected_remote_settings_observation(),
                    "file_identity_sha256": "1" * 64,
                }
            ),
            "",
        )

    monkeypatch.setattr(runner, "_net_retry", fake_net_retry)
    runner._verify_fixed32_qwen_settings_remote(
        host="alienware",
        remote_path="~/lumo_proxy_offload/codex_work/task/settings.json",
    )

    remote_command = calls[0][-1]
    assert ".expanduser()" in remote_command
    assert "os.O_NOFOLLOW" in remote_command
    assert "before.st_nlink != 1" in remote_command
    assert "os.listxattr" in remote_command
    assert "identity(after_read) != identity(opened)" in remote_command
    assert "~/lumo_proxy_offload/codex_work/task/settings.json" in remote_command


def test_fixed32_runtime_mode_rejects_legacy_and_non_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setenv("SWE_AGENT", "qwen_code")
    monkeypatch.setenv("SWE_AGENT_ENV", "legacy")
    with pytest.raises(runner.Fixed32BoundaryError, match="instance_image"):
        runner._validate_fixed32_agent_runtime_mode(
            remote_host="alienware"
        )

    monkeypatch.setenv("SWE_AGENT_ENV", "instance_image")
    monkeypatch.setenv("SWE_AGENT", "codex")
    with pytest.raises(runner.Fixed32BoundaryError, match="qwen_code"):
        runner._validate_fixed32_agent_runtime_mode(
            remote_host="alienware"
        )

    monkeypatch.setenv("SWE_AGENT", "qwen")
    with pytest.raises(runner.Fixed32BoundaryError, match="agent-host"):
        runner._validate_fixed32_agent_runtime_mode(remote_host=None)
    runner._validate_fixed32_agent_runtime_mode(remote_host="alienware")


def test_fixed32_main_rejects_local_agent_before_runtime_binding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    monkeypatch.setenv("SWE_AGENT", "qwen")
    monkeypatch.setenv("SWE_AGENT_ENV", "instance_image")
    monkeypatch.setenv("SWE_EMPTY_PATCH_RETRIES", "0")

    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--subset",
                "/does/not/exist.json",
                "--fixed32-container",
                "engine",
                "--fixed32-producer-pid",
                "1",
                "--fixed32-mode",
                "tail6_fixed32",
                "--fixed32-flush-request",
                "/tmp/request",
                "--fixed32-flush-ack",
                "/tmp/ack",
                "--fixed32-boundary-snapshot",
                "/tmp/snapshot",
            ]
        )

    assert "requires --agent-host" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutate",
    (
        lambda runner, value: value.__setitem__(
            "qwen_code_version",
            "0.19.6",
        ),
        lambda runner, value: value["bundle_tree"].__setitem__(
            "manifest_sha256", "0" * 64
        ),
        lambda runner, value: value["bundle_tree"]["summary"].__setitem__(
            "entry_count", 1
        ),
        lambda runner, value: value["bundle_tree"]["entrypoints"].pop(
            "bin/qwen"
        ),
    ),
)
def test_fixed32_qwen_bundle_attestation_fails_closed(
    mutate: Any,
) -> None:
    runner = _load_runner()
    observation = _fixed32_bundle_observation(runner)
    mutate(runner, observation)

    with pytest.raises(runner.Fixed32BoundaryError):
        runner._validate_fixed32_qwen_bundle_observation(observation)


def _write_minimal_qwen_bundle(root: Path) -> Path:
    package_root = root / "npm/lib/node_modules/@qwen-code/qwen-code"
    (root / "bin").mkdir(parents=True)
    (root / "node/bin").mkdir(parents=True)
    (root / "npm/bin").mkdir(parents=True)
    (package_root / "chunks").mkdir(parents=True)
    (root / "bin/qwen").write_text("#!/bin/sh\n", encoding="ascii")
    (root / "node/bin/node").write_bytes(b"node")
    (package_root / "cli-entry.js").write_text("cli\n", encoding="ascii")
    (package_root / "package.json").write_text(
        '{"version":"0.19.4"}\n',
        encoding="ascii",
    )
    (root / CAP_CHUNK_PATH).write_text(
        "var TURN_TOOL_CALL_CAP = 256;\n",
        encoding="ascii",
    )
    runtime_file = package_root / "chunks/unpinned.js"
    runtime_file.write_text(
        "before\n",
        encoding="ascii",
    )
    (root / "npm/bin/qwen").symlink_to(
        "../lib/node_modules/@qwen-code/qwen-code/cli-entry.js"
    )
    for executable in (
        root / "bin/qwen",
        root / "node/bin/node",
        package_root / "cli-entry.js",
    ):
        executable.chmod(0o755)
    return runtime_file


def test_qwen_tree_manifest_covers_previously_unlisted_runtime_file(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    bundle_root = tmp_path / "qwen"
    runtime_file = _write_minimal_qwen_bundle(bundle_root)

    before = runner._observe_fixed32_qwen_bundle_local(bundle_root)
    runtime_file.write_text("after\n", encoding="ascii")
    after = runner._observe_fixed32_qwen_bundle_local(bundle_root)

    assert (
        before["bundle_tree"]["manifest_sha256"]
        != after["bundle_tree"]["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "violation",
    ("hardlink", "xattr", "absolute_symlink", "dangling_symlink", "fifo", "nonascii"),
)
def test_qwen_tree_manifest_rejects_noncanonical_entries(
    tmp_path: Path,
    violation: str,
) -> None:
    runner = _load_runner()
    bundle_root = tmp_path / "qwen"
    runtime_file = _write_minimal_qwen_bundle(bundle_root)
    if violation == "hardlink":
        os.link(runtime_file, runtime_file.with_name("hardlink.js"))
    elif violation == "xattr":
        try:
            os.setxattr(runtime_file, b"user.fr13-test", b"x")
        except OSError:
            pytest.skip("test filesystem does not support user xattrs")
    elif violation in {"absolute_symlink", "dangling_symlink"}:
        runtime_file.unlink()
        runtime_file.symlink_to(
            "/etc/passwd"
            if violation == "absolute_symlink"
            else "missing-target.js"
        )
    elif violation == "fifo":
        runtime_file.unlink()
        os.mkfifo(runtime_file)
    elif violation == "nonascii":
        runtime_file.with_name("runtime-\N{LATIN SMALL LETTER E WITH ACUTE}.js").write_text(
            "noncanonical\n",
            encoding="utf-8",
        )

    with pytest.raises(runner.Fixed32BoundaryError):
        runner._observe_fixed32_qwen_bundle_local(bundle_root)


def test_fixed32_remote_paths_and_command_isolate_read_only_settings() -> None:
    runner = _load_runner()
    container, task_root, out_dir, settings = (
        runner._fixed32_remote_agent_paths(TASK_A, nonce="unit")
    )
    pinned = runner._FIXED32_AGENT_IMAGE_IDENTITIES[TASK_A]
    snapshot = (
        f"{task_root}/qwen_bundle-"
        f"{runner._FIXED32_QWEN_BUNDLE_TREE_SHA256}"
    )
    command = runner._instance_agent_command(
        container_name=container,
        image=pinned["repo_digest"],
        endpoint="http://127.0.0.1:8023/v1",
        model="model",
        host_out_dir=out_dir,
        bundle_src=snapshot,
        agents_md_b64="YQ==",
        prompt_b64="Yg==",
        base_commit="deadbeef",
        session_id=contract.fixed32_trace_session_id(TASK_A),
        system_settings_src=settings,
        bundle_observation=_fixed32_bundle_observation(runner),
    )

    assert settings == f"{task_root}/qwen_system_settings.json"
    assert not settings.startswith(out_dir + "/")
    assert f"-v {settings}:/run/fr13/qwen-system-settings.json:ro" in command
    assert f"-v {out_dir}:/out" in command
    assert f"-v {snapshot}:/opt/qwen:ro" in command
    assert "~/qwen_agent_bundle:/opt/qwen" not in command
    assert "fr13_qwen_tree_scanner.py /opt/qwen" in command
    assert "qwen_mounted_runtime_proof.json" in command
    assert f"-w /testbed {pinned['repo_digest']} bash" in command
    assert "-e QWEN_STREAM_IDLE_TIMEOUT_MS=600000" in command
    for variable in runner._FIXED32_CLEARED_AGENT_ENV:
        assert f"-e {variable}=" in command


def _run_local_remote_trace_command(argv: list[str], **_kwargs: Any) -> Any:
    if argv[0] == "ssh":
        return subprocess.run(
            ["bash", "-c", argv[-1]],
            capture_output=True,
            text=True,
            check=False,
        )
    if argv[0] == "scp":
        _host, remote_path = argv[-2].split(":", 1)
        shutil.copy2(Path(remote_path).expanduser(), Path(argv[-1]))
        return subprocess.CompletedProcess(argv, 0, "", "")
    raise AssertionError(f"unexpected transport command: {argv[0]}")


def test_fixed32_remote_trace_is_written_inside_container_and_pulled_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    remote_out = tmp_path / "remote out"
    remote_out.mkdir(mode=0o700)
    remote_trace = remote_out / runner._REMOTE_AGENT_TRACE_FILENAME
    local_trace = tmp_path / "local" / "qwen_trace.jsonl"
    local_trace.parent.mkdir()
    event_count = 64
    events = [
        {"type": "event", "sequence": index, "payload": "x" * 4096}
        for index in range(event_count)
    ]
    expected = "".join(
        json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        for event in events
    ).encode("ascii")
    assert len(expected) > 258_048

    writer = "\n".join(
        (
            "import json, pathlib, sys",
            "path = pathlib.Path(sys.argv[1])",
            "with path.open('w', encoding='ascii') as stream:",
            f"    for index in range({event_count}):",
            "        event = {'type': 'event', 'sequence': index, "
            "'payload': 'x' * 4096}",
            "        stream.write(json.dumps(event, ensure_ascii=True, "
            "separators=(',', ':')) + '\\n')",
        )
    )
    fake_container = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(writer),
            shlex.quote(str(remote_trace)),
        )
    )
    capture_command = runner._remote_agent_trace_capture_command(
        fake_container,
        remote_trace_path=str(remote_trace),
    )
    completed = subprocess.run(
        ["bash", "-c", capture_command],
        input="task-credential\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert remote_trace.read_bytes() == expected
    assert remote_trace.stat().st_mode & 0o777 == 0o600

    instance_command = runner._instance_agent_command(
        container_name="agent",
        image="image",
        endpoint="http://127.0.0.1:8023/v1",
        model="model",
        host_out_dir=str(remote_out),
        bundle_src="/tmp/bundle",
        agents_md_b64="YQ==",
        prompt_b64="Yg==",
        base_commit="deadbeef",
        session_id=contract.fixed32_trace_session_id(TASK_A),
        trace_output_path=runner._INSTANCE_TRACE_OUTPUT_PATH,
    )
    assert (
        '-p "$PROMPT" > /out/qwen_trace.jsonl; rc=$?; '
        in instance_command
    )
    assert f"{fake_container} > /dev/null" in capture_command
    assert capture_command.count('> "$trace_path"') == 1

    monkeypatch.setattr(runner, "_net_retry", _run_local_remote_trace_command)
    observation = runner._pull_remote_agent_trace(
        host="test-host",
        instance_id=TASK_A,
        remote_trace_path=str(remote_trace),
        trace_path=local_trace,
    )

    assert local_trace.read_bytes() == expected
    assert observation == {
        "schema": runner._REMOTE_AGENT_TRACE_OBSERVATION_SCHEMA,
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
        "event_count": event_count,
    }


def test_nonfixed_instance_wrapper_keeps_legacy_stdout_trace_route() -> None:
    runner = _load_runner()
    command = runner._instance_agent_command(
        container_name="agent",
        image="image",
        endpoint="http://127.0.0.1:8023/v1",
        model="model",
        host_out_dir="/tmp/out",
        bundle_src="/tmp/bundle",
        agents_md_b64="YQ==",
        prompt_b64="Yg==",
        base_commit="deadbeef",
        session_id="ordinary-session",
    )

    assert '-p "$PROMPT"; rc=$?; ' in command
    assert "> /out/qwen_trace.jsonl" not in command
    assert runner._instance_wrapper(trace_output_path=None) == (
        runner._INSTANCE_WRAPPER
    )


@pytest.mark.parametrize(
    ("remote_bytes", "error"),
    (
        (b"", "trace is empty"),
        (
            b'{"type":"event"}\n{"private-secret-value":"truncated',
            "trace is not newline-framed",
        ),
        (
            b'{"type":"event"}\n{"private-secret-value":\n',
            "invalid JSON",
        ),
    ),
)
def test_fixed32_remote_trace_rejection_preserves_exact_malformed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_bytes: bytes,
    error: str,
) -> None:
    runner = _load_runner()
    remote_out = tmp_path / "remote"
    remote_out.mkdir(mode=0o700)
    remote_trace = remote_out / runner._REMOTE_AGENT_TRACE_FILENAME
    remote_trace.write_bytes(remote_bytes)
    remote_trace.chmod(0o600)
    local_trace = tmp_path / "local" / "qwen_trace.jsonl"
    local_trace.parent.mkdir()
    monkeypatch.setattr(runner, "_net_retry", _run_local_remote_trace_command)

    with pytest.raises(runner.Fixed32BoundaryError, match=error) as raised:
        runner._pull_remote_agent_trace(
            host="test-host",
            instance_id=TASK_A,
            remote_trace_path=str(remote_trace),
            trace_path=local_trace,
        )

    assert "private-secret-value" not in str(raised.value)
    assert local_trace.read_bytes() == remote_bytes


def test_fixed32_remote_cleanup_is_fail_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[list[str]] = []

    def fail_cleanup(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        return runner.subprocess.CompletedProcess(argv, 1, "", "failed")

    monkeypatch.setattr(runner, "_net_retry", fail_cleanup)
    with pytest.raises(runner.Fixed32BoundaryError, match="cleanup failed"):
        runner._cleanup_remote_agent_task(
            host="alienware",
            instance_id=TASK_A,
            container_name="swe-qwen-task-unit",
            task_root="~/swe_codex_offload/task-unit",
        )

    cleanup_command = calls[0][-1]
    assert cleanup_command.startswith("set -eu;")
    assert "test ! -e" in cleanup_command
    assert "test ! -L" in cleanup_command
    assert "echo ok" not in cleanup_command


def _run_settings_script(
    runner: Any,
    *,
    action: str,
    path: Path,
) -> Any:
    return runner.subprocess.run(
        [
            sys.executable,
            "-c",
            runner._FIXED32_QWEN_SETTINGS_REMOTE_SCRIPT,
            action,
            str(path),
            runner.base64.b64encode(
                runner._FIXED32_QWEN_SETTINGS_BYTES
            ).decode("ascii"),
            runner._FIXED32_QWEN_SETTINGS_MODE,
            str(runner._FIXED32_QWEN_SETTINGS_UID),
            str(runner._FIXED32_QWEN_SETTINGS_GID),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "violation",
    ("swap", "symlink", "hardlink", "mode", "xattr"),
)
def test_fixed32_settings_reject_path_identity_and_metadata_tamper(
    tmp_path: Path,
    violation: str,
) -> None:
    runner = _load_runner()
    if (
        os.getuid() != runner._FIXED32_QWEN_SETTINGS_UID
        or os.getgid() != runner._FIXED32_QWEN_SETTINGS_GID
    ):
        pytest.skip("test process does not have the canonical remote owner")
    path = tmp_path / "qwen_system_settings.json"
    installed = _run_settings_script(
        runner,
        action="create",
        path=path,
    )
    assert installed.returncode == 0, installed.stderr
    before = runner._validate_fixed32_remote_settings_observation(
        json.loads(installed.stdout)
    )

    if violation == "swap":
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(runner._FIXED32_QWEN_SETTINGS_BYTES)
        replacement.chmod(0o444)
        os.replace(replacement, path)
    elif violation == "symlink":
        path.unlink()
        path.symlink_to(runner._FIXED32_QWEN_SETTINGS_PATH)
    elif violation == "hardlink":
        os.link(path, tmp_path / "settings-hardlink.json")
    elif violation == "mode":
        path.chmod(0o644)
    elif violation == "xattr":
        try:
            os.setxattr(path, b"user.fr13-test", b"x")
        except OSError:
            pytest.skip("test filesystem does not support user xattrs")

    verified = _run_settings_script(
        runner,
        action="verify",
        path=path,
    )
    if violation == "swap":
        assert verified.returncode == 0, verified.stderr
        after = runner._validate_fixed32_remote_settings_observation(
            json.loads(verified.stdout)
        )
        with pytest.raises(
            runner.Fixed32BoundaryError,
            match="file identity changed",
        ):
            runner._require_fixed32_remote_settings_stable(before, after)
    else:
        assert verified.returncode != 0


def test_fixed32_snapshot_path_is_content_addressed_and_promoted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[list[str]] = []
    observation = _fixed32_bundle_observation(runner)

    def fake_net_retry(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        return runner.subprocess.CompletedProcess(argv, 0, "", "")

    inspected_paths: list[str] = []

    def fake_inspect(_host: str, path: str) -> dict[str, Any]:
        inspected_paths.append(path)
        return copy.deepcopy(observation)

    monkeypatch.setattr(runner, "_net_retry", fake_net_retry)
    monkeypatch.setattr(
        runner,
        "_inspect_fixed32_qwen_bundle_remote_path",
        fake_inspect,
    )
    snapshot, observed = runner._create_fixed32_qwen_snapshot_remote(
        host="alienware",
        instance_id=TASK_A,
        task_root="~/lumo_proxy_offload/codex_work/task-unit",
    )

    assert snapshot.endswith(
        "/qwen_bundle-" + runner._FIXED32_QWEN_BUNDLE_TREE_SHA256
    )
    assert inspected_paths == [
        "~/lumo_proxy_offload/codex_work/task-unit/.qwen_bundle_snapshot",
        snapshot,
    ]
    assert observed == observation
    assert (
        "$HOME/" + runner._FIXED32_QWEN_BUNDLE_REMOTE_BASENAME
        in calls[0][-1]
    )
    assert "mv --" in calls[1][-1]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("bundle_tree", "write_probe_errno", 13),
        ("bundle_tree", "mount_mode", "rw"),
        ("system_settings", "write_probe_errno", 13),
        ("system_settings", "mode", "0644"),
        ("system_settings", "file_identity_sha256", "0" * 64),
    ),
)
def test_mounted_runtime_proof_tamper_fails_runner_and_gate(
    section: str,
    field: str,
    value: Any,
) -> None:
    runner = _load_runner()
    observation = _fixed32_bundle_observation(runner)
    proof = {
        "schema": runner._FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": observation,
        },
        "system_settings": {
            "container_path": runner._FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **runner._fixed32_expected_remote_settings_observation(),
            "file_identity_sha256": "1" * 64,
        },
    }
    if section == "system_settings" and field == "file_identity_sha256":
        value = "not-a-digest"
    proof[section][field] = value

    with pytest.raises(runner.Fixed32BoundaryError):
        runner._validate_fixed32_mounted_runtime_proof(
            proof,
            expected_bundle_observation=observation,
        )
    with pytest.raises(floor_gate.GateError):
        floor_gate._fixed32_mounted_runtime_proof(
            proof,
            label="proof",
        )


def test_fixed32_agent_placement_pins_alias_kernel_machine_and_daemon() -> None:
    runner = _load_runner()
    placement = runner._validate_fixed32_agent_placement_observation(
        copy.deepcopy(runner._FIXED32_AGENT_HOST_IDENTITY),
        measured_observation=copy.deepcopy(
            runner._FIXED32_MEASURED_HOST_IDENTITY
        ),
        remote_host="alienware",
    )

    assert placement["agent_host_identity"]["machine"] == "x86_64"
    assert placement["measured_host_identity"]["machine"] == "aarch64"
    assert floor_gate._fixed32_agent_placement(
        placement,
        label="placement",
    ) == runner._fixed32_canonical_json_sha256(placement)
    with pytest.raises(runner.Fixed32BoundaryError, match="exact canonical"):
        runner._validate_fixed32_agent_placement_observation(
            copy.deepcopy(runner._FIXED32_AGENT_HOST_IDENTITY),
            measured_observation=copy.deepcopy(
                runner._FIXED32_MEASURED_HOST_IDENTITY
            ),
            remote_host="mark-Alienware-Aurora-ACT1250",
        )
    tampered = copy.deepcopy(runner._FIXED32_AGENT_HOST_IDENTITY)
    tampered["docker_daemon_id_sha256"] = "0" * 64
    with pytest.raises(runner.Fixed32BoundaryError, match="Docker daemon"):
        runner._validate_fixed32_agent_placement_observation(
            tampered,
            measured_observation=copy.deepcopy(
                runner._FIXED32_MEASURED_HOST_IDENTITY
            ),
            remote_host="alienware",
        )
    tampered_measured = copy.deepcopy(
        runner._FIXED32_MEASURED_HOST_IDENTITY
    )
    tampered_measured["kernel"] = "changed"
    with pytest.raises(runner.Fixed32BoundaryError, match="measured host"):
        runner._validate_fixed32_agent_placement_observation(
            copy.deepcopy(runner._FIXED32_AGENT_HOST_IDENTITY),
            measured_observation=tampered_measured,
            remote_host="alienware",
        )


def test_fixed32_measured_host_identity_is_observed_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    observation = copy.deepcopy(runner._FIXED32_MEASURED_HOST_IDENTITY)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        return runner.subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(observation, sort_keys=True),
            "",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._inspect_fixed32_measured_host_local() == observation
    assert calls == [[sys.executable, "-c", runner._FIXED32_HOST_IDENTITY_SCRIPT]]

    observation["docker_daemon_id_sha256"] = "0" * 64
    with pytest.raises(runner.Fixed32BoundaryError, match="measured host"):
        runner._inspect_fixed32_measured_host_local()


def test_fixed32_runtime_manifest_includes_qwen_settings() -> None:
    path = "config/fr13_fixed32/qwen_system_settings.json"
    assert path in runtime_manifest.FIXED32_RUNTIME_DATA_AND_CONFIG
    manifest = runtime_manifest.build_manifest(
        REPO,
        profile="fixed32",
        sequence="scripts/fr13_fixed32_floor_timers_seq.sh",
    )
    records = manifest["closures"]["runtime_data_and_config"]
    record = next(item for item in records if item["path"] == path)
    assert record == {
        "path": path,
        "sha256": (
            "d1c7e744e9febaa96b341e01de24cc6e"
            "a07dd30bdf33352618b2c63de225ee9f"
        ),
        "size": 98,
    }


def _closure_cardinality_fixture_spec() -> Any:
    return runtime_manifest.ProfileSpec(
        host_script_source=("scripts/driver.sh",),
        python_package_source=(
            "src/fixture_pkg/__init__.py",
            "src/fixture_pkg/helper.py",
        ),
        runtime_data_and_config=(".secret.env", "config/runtime.json"),
        required_absence=("output/fallback/corpus_active.jsonl",),
        verdict_tools=("scripts/verdict.py",),
        package_dir="src/fixture_pkg",
        package_name="fixture_pkg",
        package_file_count=2,
    )


def test_runtime_closure_cardinality_matches_a_real_built_manifest(
    tmp_path: Path,
) -> None:
    """The gate's derivation must reproduce build_manifest's own summary.

    This is the cross-check that stops the two from drifting apart again.  It
    builds a REAL manifest from a fixture profile and compares, so a change to
    how build_manifest counts a closure fails here rather than silently making
    the per-pass gate unsatisfiable in the middle of a campaign.  It is
    deliberately hermetic -- a temp repo, not the checkout -- so it states the
    invariant even where the shipped closure is not fully materialised.
    """
    spec = _closure_cardinality_fixture_spec()
    sequence = "scripts/fixture_seq.sh"
    fixture_files = {
        "scripts/driver.sh": b"#!/usr/bin/env bash\nset -eu\n",
        sequence: b"run_variant tail fixed32 31 1\n",
        "scripts/verdict.py": b"print('verdict')\n",
        "src/fixture_pkg/__init__.py": b"from .helper import VALUE\n",
        "src/fixture_pkg/helper.py": b"VALUE = 32\n",
        ".secret.env": b"API_TOKEN=do-not-emit-this-value\n",
        "config/runtime.json": b'{"rows":32}\n',
    }
    for relative_path, content in fixture_files.items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = runtime_manifest.build_manifest(
        tmp_path,
        profile="fixed32",
        sequence=sequence,
        spec_override=spec,
    )
    summary = manifest["summary"]
    assert floor_gate.runtime_closure_cardinality(spec, sequence=sequence) == (
        summary["file_count"],
        summary["python_package_file_count"],
    )


def test_runtime_closure_cardinality_follows_the_shipped_fixed32_profile() -> None:
    """The shipped cardinality is derived from the profile, never transcribed.

    The literals this replaced (62 files / 25 Python package files) went stale
    twice -- the profile grew to 90/26 and then to 151/30 -- and because the gate
    demanded the stale numbers, every fixed32 pass exited rc=2 on a closure that
    was perfectly healthy.  Recompute independently here so a future closure
    change fails this test instead of a campaign.
    """
    spec = runtime_manifest.PROFILES[floor_gate.RUNTIME_MANIFEST_PROFILE]
    # build_manifest appends the sequence to host_script_source, and counts it.
    assert floor_gate.RUNTIME_MANIFEST_SEQUENCE not in spec.host_script_source
    expected_file_count = (
        len(spec.host_script_source)
        + 1
        + len(spec.python_package_source)
        + len(spec.runtime_data_and_config)
        + len(spec.verdict_tools)
    )
    assert len(spec.python_package_source) == spec.package_file_count
    assert floor_gate.runtime_closure_cardinality() == (
        expected_file_count,
        spec.package_file_count,
    )


def test_runtime_closure_cardinality_rejects_a_self_inconsistent_profile() -> None:
    """A profile contradicting its own package tuple fails closed.

    This is the exact contradiction the old literal encoded: the gate demanded 25
    Python package files while the profile declared 26, so no tree whatsoever
    could satisfy both halves.
    """
    spec = _closure_cardinality_fixture_spec()
    contradictory = dataclasses.replace(spec, package_file_count=3)
    with pytest.raises(floor_gate.GateError, match="self-inconsistent"):
        floor_gate.runtime_closure_cardinality(contradictory)


def test_fixed32_retry_policy_is_exactly_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setenv("SWE_EMPTY_PATCH_RETRIES", "0")
    runner._validate_fixed32_retry_policy()
    for invalid in ("1", "00", ""):
        monkeypatch.setenv("SWE_EMPTY_PATCH_RETRIES", invalid)
        with pytest.raises(runner.Fixed32BoundaryError, match="exactly"):
            runner._validate_fixed32_retry_policy()


def test_fixed32_agent_image_identity_is_independently_pinned() -> None:
    runner = _load_runner()
    expected = copy.deepcopy(
        floor_gate.FIXED32_AGENT_IMAGE_IDENTITIES[TASK_A]
    )

    assert floor_gate._fixed32_agent_image_identity(
        expected,
        task_id=TASK_A,
        label="image",
    ) == runner._fixed32_canonical_json_sha256(expected)
    expected["repo_digest"] = expected["repo_digest"][:-1] + "0"
    with pytest.raises(floor_gate.GateError, match="image identity differs"):
        floor_gate._fixed32_agent_image_identity(
            expected,
            task_id=TASK_A,
            label="image",
        )


def test_floor_gate_requires_v3_qwen_runtime_attestation() -> None:
    runner = _load_runner()
    attestation = runner._build_fixed32_qwen_runtime_attestation(
        bundle_observation=_fixed32_bundle_observation(runner),
        host_mode="remote",
    )

    assert floor_gate.FIXED32_REAL_TASK_PROVENANCE_SCHEMA == (
        "fr13-fixed32-real-task-provenance-v3"
    )
    assert floor_gate._fixed32_qwen_runtime_attestation(
        attestation,
        label="test attestation",
    ) == runner._fixed32_canonical_json_sha256(attestation)

    attestation["system_settings"]["mount_mode"] = "rw"
    with pytest.raises(floor_gate.GateError, match="system-settings evidence"):
        floor_gate._fixed32_qwen_runtime_attestation(
            attestation,
            label="test attestation",
        )


def test_fixed32_provenance_rejects_trace_version_or_postrun_digest(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_result_trace()
    events[0]["qwen_code_version"] = "0.19.6"
    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    task_key_id = "d" * 64
    meta = _fixed32_agent_meta(runner, tmp_path)

    with pytest.raises(
        runner.Fixed32BoundaryError,
        match="pinned Qwen 0.19.4",
    ):
        runner._fixed32_real_task_provenance(
            instance_id=TASK_A,
            trace_path=trace_path,
            agent_meta=meta,
            task_key_id=task_key_id,
            task_auth_before=_task_evidence(task_key_id, 0, 1),
            task_auth_after=_task_evidence(task_key_id, 13, 53),
        )

    events[0]["qwen_code_version"] = "0.19.4"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    meta["qwen_runtime_postrun_attestation_sha256"] = "0" * 64
    with pytest.raises(
        runner.Fixed32BoundaryError,
        match="pre/post runtime attestation digests differ",
    ):
        runner._fixed32_real_task_provenance(
            instance_id=TASK_A,
            trace_path=trace_path,
            agent_meta=meta,
            task_key_id=task_key_id,
            task_auth_before=_task_evidence(task_key_id, 0, 1),
            task_auth_after=_task_evidence(task_key_id, 13, 53),
        )


# ------------------------------------------------- no-patch terminal record
# A trajectory that ends without submitting a patch is legal traffic under
# temp-0.6 canonical sampling, and SWE-bench scores that instance unresolved.
# The eval worker never invokes the harness on an empty prediction, so it can
# report no harness exit code; the runner writes an explicit synthetic terminal
# instead and the traffic audit accepts exactly that record.

_NO_PATCH_MODEL_ID = "qwen3.8-27b-nvfp4-radixark::qwen-code-0.19.4::q38-a"
_NO_PATCH_EVAL_HOST = "mark-Alienware-Aurora-ACT1250"


def _no_patch_worker_report(task_id: str = TASK_A) -> dict[str, Any]:
    """The record the x86 eval worker writes when it skips the harness."""
    return {
        "instance_id": task_id,
        "verdict": "failed",
        "passed": False,
        "failure_mode": "patch_apply_failed",
        "error": "empty_patch",
        "arch": "x86_64",
        "eval_host": _NO_PATCH_EVAL_HOST,
        "eval_wall_clock_seconds": 0.0,
    }


def _synthesized_no_patch_terminal(task_id: str = TASK_A) -> dict[str, Any]:
    runner = _load_runner()
    terminal = runner._synthetic_no_patch_eval_report(
        _no_patch_worker_report(task_id),
        instance_id=task_id,
        dataset_name=floor_gate.SWE_VERIFIED_DATASET,
        model_name=_NO_PATCH_MODEL_ID,
        patch_text="",
    )
    assert terminal is not None
    return terminal


def _no_patch_task_dir(
    root: Path,
    task_id: str = TASK_A,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    task_dir = root / task_id
    (task_dir / "eval").mkdir(parents=True)
    (task_dir / "patch.diff").write_bytes(b"")
    (task_dir / "qwen_trace.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in _reasoning_only_final_qwen_result_trace()
        ),
        encoding="utf-8",
    )
    (task_dir / "eval" / "predictions.jsonl").write_text(
        json.dumps(
            {
                "instance_id": task_id,
                "model_name_or_path": _NO_PATCH_MODEL_ID,
                "model_patch": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    terminal = _synthesized_no_patch_terminal(task_id)
    (task_dir / "eval" / "eval_report.json").write_text(
        json.dumps(terminal, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "instance_id": task_id,
        "dataset_name": floor_gate.SWE_VERIFIED_DATASET,
        "patch_bytes": 0,
        "eval_report": terminal,
        "ended_at": "2026-08-10T20:46:28Z",
    }
    (task_dir / "runner_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return task_dir, metadata, terminal


def _accept_no_patch(
    task_dir: Path,
    metadata: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    floor_gate._fixed32_no_patch_eval_terminal(
        terminal,
        task_dir=task_dir,
        task_id=metadata["instance_id"],
        metadata=metadata,
        metadata_path=task_dir / "runner_metadata.json",
    )


def test_runner_rewrites_a_no_patch_worker_record_into_an_honest_terminal(
) -> None:
    terminal = _synthesized_no_patch_terminal()
    assert terminal == {
        "schema": floor_gate.FIXED32_SYNTHETIC_NO_PATCH_EVAL_SCHEMA,
        "track": "swe_bench",
        "instance_id": TASK_A,
        "model_id": _NO_PATCH_MODEL_ID,
        "dataset_name": floor_gate.SWE_VERIFIED_DATASET,
        "verdict": "failed",
        "passed": False,
        "failure_mode": "patch_apply_failed",
        "error": "empty_patch",
        "synthetic_no_patch": True,
        "harness_invoked": False,
        "harness_exit_code": None,
        "patch_bytes": 0,
        "eval_wall_clock_seconds": 0.0,
        "arch": "x86_64",
        "eval_host": _NO_PATCH_EVAL_HOST,
        "worker_report": _no_patch_worker_report(),
    }
    # The record never claims a harness ran, and it quotes the worker verbatim.
    assert terminal["harness_invoked"] is False
    assert terminal["harness_exit_code"] is None
    assert terminal["worker_report"] == _no_patch_worker_report()


@pytest.mark.parametrize(
    "patch_text,mutate",
    (
        # A submission that exists is evaluated by the harness, not synthesized.
        ("diff --git a b\n", lambda report: None),
        # The harness ran, so its own verdict stands.
        ("", lambda report: report.__setitem__("harness_exit_code", 0)),
        # A crashed eval must never become a no-patch failure.
        ("", lambda report: report.__setitem__("verdict", "crash")),
        ("", lambda report: report.__setitem__("failure_mode", "infra_error")),
        ("", lambda report: report.__setitem__("error", "patch_missing")),
        ("", lambda report: report.__setitem__("passed", True)),
        ("", lambda report: report.__setitem__("instance_id", "other__task-1")),
        # A nonzero eval wall clock means the worker did more than skip.
        ("", lambda report: report.__setitem__("eval_wall_clock_seconds", 12.5)),
        ("", lambda report: report.pop("arch")),
    ),
)
def test_runner_declines_to_synthesize_anything_but_an_empty_submission(
    patch_text: str,
    mutate: Any,
) -> None:
    runner = _load_runner()
    report = _no_patch_worker_report()
    mutate(report)
    assert (
        runner._synthetic_no_patch_eval_report(
            report,
            instance_id=TASK_A,
            dataset_name=floor_gate.SWE_VERIFIED_DATASET,
            model_name=_NO_PATCH_MODEL_ID,
            patch_text=patch_text,
        )
        is None
    )


def test_audit_accepts_the_honest_no_patch_terminal(tmp_path: Path) -> None:
    task_dir, metadata, terminal = _no_patch_task_dir(tmp_path)
    _accept_no_patch(task_dir, metadata, terminal)
    # The persisted eval artifact is the same record the metadata carries, so
    # the audit's byte-identity clause holds for the synthesized terminal too.
    assert (
        floor_gate.exact_json(
            task_dir / "eval" / "eval_report.json",
            label="eval_report",
        )
        == metadata["eval_report"]
    )


def test_audit_rejects_the_truncated_worker_record_the_gate_used_to_see(
    tmp_path: Path,
) -> None:
    """The pre-fix on-disk shape stays rejected -- the fix is not a loosening."""
    task_dir, metadata, _terminal = _no_patch_task_dir(tmp_path)
    truncated = _no_patch_worker_report()
    metadata["eval_report"] = truncated
    assert "schema" not in truncated
    assert "harness_exit_code" not in truncated
    assert "dataset_name" not in truncated
    with pytest.raises(floor_gate.GateError):
        _accept_no_patch(task_dir, metadata, truncated)


def _drop_result_event(task_dir: Path) -> None:
    """A killed trajectory never emits its terminal result event."""
    lines = (task_dir / "qwen_trace.jsonl").read_text().splitlines(True)
    (task_dir / "qwen_trace.jsonl").write_text("".join(lines[:-1]))


def _error_result_event(task_dir: Path) -> None:
    lines = (task_dir / "qwen_trace.jsonl").read_text().splitlines()
    final = json.loads(lines[-1])
    final["is_error"] = True
    final["subtype"] = "error_during_execution"
    lines[-1] = json.dumps(final)
    (task_dir / "qwen_trace.jsonl").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "corrupt",
    (
        # A crashed or aborted trajectory is not a task that declined to submit.
        lambda task_dir, metadata, terminal: _drop_result_event(task_dir),
        lambda task_dir, metadata, terminal: _error_result_event(task_dir),
        lambda task_dir, metadata, terminal: (
            task_dir / "orchestrator_crash.json"
        ).write_text("{}", encoding="utf-8"),
        lambda task_dir, metadata, terminal: (
            task_dir / "qwen_trace.jsonl"
        ).unlink(),
        # The submission must really have been empty.
        lambda task_dir, metadata, terminal: (
            task_dir / "patch.diff"
        ).write_text("diff --git a b\n", encoding="utf-8"),
        lambda task_dir, metadata, terminal: metadata.pop("patch_bytes"),
        lambda task_dir, metadata, terminal: metadata.__setitem__(
            "patch_bytes", 1872
        ),
        lambda task_dir, metadata, terminal: (
            task_dir / "eval" / "predictions.jsonl"
        ).write_text(
            json.dumps(
                {
                    "instance_id": TASK_A,
                    "model_name_or_path": _NO_PATCH_MODEL_ID,
                    "model_patch": "diff --git a b\n",
                }
            )
            + "\n",
            encoding="utf-8",
        ),
        lambda task_dir, metadata, terminal: (
            task_dir / "eval" / "predictions.jsonl"
        ).unlink(),
        # The worker's own record must show the skipped harness.
        lambda task_dir, metadata, terminal: terminal.pop("worker_report"),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "worker_report", dict(terminal["worker_report"], verdict="crash")
        ),
        lambda task_dir, metadata, terminal: terminal["worker_report"].pop(
            "error"
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "worker_report",
            dict(terminal["worker_report"], eval_host="somewhere-else"),
        ),
        # Nothing looser than the exact synthetic record is accepted.
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "harness_exit_code", 0
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "harness_invoked", True
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "verdict", "resolved"
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "passed", True
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "synthetic_no_patch", False
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "dataset_name", "princeton-nlp/SWE-bench"
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "instance_id", "other__task-1"
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "eval_wall_clock_seconds", 12.5
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__(
            "patch_bytes", 1872
        ),
        lambda task_dir, metadata, terminal: terminal.__setitem__("extra", 1),
        lambda task_dir, metadata, terminal: terminal.pop("track"),
    ),
)
def test_audit_no_patch_terminal_fails_closed(
    tmp_path: Path,
    corrupt: Any,
) -> None:
    task_dir, metadata, terminal = _no_patch_task_dir(tmp_path)
    corrupt(task_dir, metadata, terminal)
    with pytest.raises(floor_gate.GateError):
        _accept_no_patch(task_dir, metadata, terminal)


def test_traffic_audit_only_relaxes_the_terminal_for_the_synthetic_schema(
) -> None:
    source = (SCRIPTS / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert (
        'eval_report.get("schema")\n'
        "            == FIXED32_SYNTHETIC_NO_PATCH_EVAL_SCHEMA"
    ) in source
    # Every other report still has to carry a real harness exit code.
    assert (
        'or not isinstance(harness_exit_code, int)\n'
        "            ):\n"
        "                raise GateError(\n"
        '                    f"{metadata_path}: fixed32 task has no terminal '
        'SWE verdict"'
    ) in source


# --------------------------------------------------------------------------
# FR14 regression: the web_fetch side query.
#
# Every number below is measured, not invented. They are the real values of
# the first FR14 stock B1 serve of Qwen3.8-27B-NVFP4 (runroot
# output/fr14_b1_stock_20260816T200746Z, arm tail6_fixed32_b1stock) on
# astropy__astropy-13033 -- the task whose provenance validator killed the
# arm. The per-turn usage pairs are the proxy ingress ledger's own
# logical_complete records for that task's key (18 of them, seq 5..84); the
# metric deltas are vllm_metrics_post.txt minus vllm_metrics_pre.txt from the
# same task directory.
#
# The 9th ledger request -- 1123 prompt / 601 completion, wedged between the
# web_fetch turn and the next agent turn -- is the qwen-code 0.19.4
# WebFetchTool side query. It is served, billed, histogrammed at the ordinary
# 32768 max_tokens and recorded in our ingress ledger, but qwen-code emits no
# trace assistant record for it. Before the fix the algebra read 17 trace
# requests against 18 engine requests and failed closed.
# --------------------------------------------------------------------------

_WEB_FETCH_URL_13033 = (
    "https://raw.githubusercontent.com/astropy/astropy/main/"
    "astropy/timeseries/core.py"
)
_WEB_FETCH_PROMPT_13033 = (
    "Show the full source code of the _check_required_columns method and the "
    "_delay_required_column_checks contextmanager exactly as written. Include "
    "the exact error message strings."
)
_WEB_FETCH_TOOL_USE_ID_13033 = "chatcmpl-tool-a5f29906a5849a98"
# (input_tokens, output_tokens) per trace-visible turn, in order. Index 7 is
# the web_fetch turn; the hidden side query follows it.
_VISIBLE_TURN_USAGE_13033 = [
    (23_276, 108),
    (27_007, 84),
    (27_791, 484),
    (28_477, 137),
    (30_420, 91),
    (31_249, 1_886),
    (33_426, 74),
    (34_453, 141),
    (35_160, 1_409),
    (36_798, 2_166),
    (39_362, 212),
    (39_782, 161),
    (40_277, 308),
    (40_911, 297),
    (41_479, 227),
    (41_893, 288),
    (42_619, 743),
]
_WEB_FETCH_TURN_INDEX_13033 = 7
_HIDDEN_SIDE_QUERY_USAGE_13033 = (1_123, 601)
_ENGINE_COMPLETED_13033 = 18
_ENGINE_PROMPT_TOKENS_13033 = 595_503
_ENGINE_GENERATION_TOKENS_13033 = 9_417
_ENGINE_MAX_TOKENS_SUM_13033 = 589_824


def _qwen_web_fetch_trace_13033(
    *,
    tool_result_content: str | None = None,
    tool_result_is_error: bool = False,
    visible_turn_usage: list[tuple[int, int]] | None = None,
    web_fetch_input: dict[str, Any] | None = None,
    permission_denials: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild the real 13033 trajectory: 17 visible turns, one web_fetch."""
    session_id = contract.fixed32_trace_session_id(TASK_B)
    usage = (
        _VISIBLE_TURN_USAGE_13033
        if visible_turn_usage is None
        else visible_turn_usage
    )
    if tool_result_content is None:
        tool_result_content = (
            contract.QWEN_WEB_FETCH_SUCCESS_TEMPLATE.format(
                url=_WEB_FETCH_URL_13033
            )
        )
    events: list[dict[str, Any]] = [
        _context_event(
            event_type="system",
            event_id="system",
            session_id=session_id,
        )
    ]
    for index, (input_tokens, output_tokens) in enumerate(usage[:-1]):
        is_web_fetch = index == _WEB_FETCH_TURN_INDEX_13033
        tool_use_id = (
            _WEB_FETCH_TOOL_USE_ID_13033
            if is_web_fetch
            else f"chatcmpl-tool-{index:016x}"
        )
        turn = _assistant_event(
            response_id=f"turn-{index}",
            session_id=session_id,
            content=[
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": (
                        contract.QWEN_WEB_FETCH_TOOL_NAME
                        if is_web_fetch
                        else "run_shell_command"
                    ),
                    "input": (
                        (
                            {
                                "url": _WEB_FETCH_URL_13033,
                                "prompt": _WEB_FETCH_PROMPT_13033,
                            }
                            if web_fetch_input is None
                            else web_fetch_input
                        )
                        if is_web_fetch
                        else {}
                    ),
                }
            ],
            stop_reason="tool_use",
        )
        turn["message"]["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 0,
            "total_tokens": input_tokens + output_tokens,
        }
        events.append(turn)
        events.append(
            _user_event(
                event_id=f"tool-result-{index}",
                session_id=session_id,
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": (
                            tool_result_content
                            if is_web_fetch
                            else "tool result"
                        ),
                        "is_error": (
                            tool_result_is_error if is_web_fetch else False
                        ),
                    }
                ],
                parent_tool_use_id=None,
            )
        )
    final_input, final_output = usage[-1]
    final = _assistant_event(
        response_id="final-text",
        session_id=session_id,
        content=[{"type": "text", "text": "The fix is complete."}],
        stop_reason=None,
    )
    final["message"]["usage"] = {
        "input_tokens": final_input,
        "output_tokens": final_output,
        "cache_read_input_tokens": 0,
        "total_tokens": final_input + final_output,
    }
    events.append(final)
    visible_input = sum(pair[0] for pair in usage)
    visible_output = sum(pair[1] for pair in usage)
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "uuid": "result-uuid",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 407_429,
            "duration_api_ms": 389_862,
            "num_turns": len(usage),
            "result": "The fix is complete.",
            "usage": {
                "input_tokens": visible_input,
                "output_tokens": visible_output,
                "cache_read_input_tokens": 0,
                "total_tokens": visible_input + visible_output,
            },
            "permission_denials": (
                [] if permission_denials is None else permission_denials
            ),
        }
    )
    return events


def _add_hidden_side_query_usage(
    events: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Credit the side query to result.usage, exactly as qwen-code does.

    The real 13033 result record reports 595503/9417 -- the ledger's own
    totals, side query included -- even though no assistant record carries
    those 1123/601 tokens. That is what makes hidden_prompt_tokens a real
    measurement rather than a residue.
    """
    usage = events[-1]["usage"]
    usage["input_tokens"] += prompt_tokens
    usage["output_tokens"] += completion_tokens
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]


def test_qwen_web_fetch_side_query_reconciles_the_real_13033_arithmetic(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    events = _qwen_web_fetch_trace_13033()
    _add_hidden_side_query_usage(
        events,
        prompt_tokens=_HIDDEN_SIDE_QUERY_USAGE_13033[0],
        completion_tokens=_HIDDEN_SIDE_QUERY_USAGE_13033[1],
    )
    assert events[-1]["usage"]["input_tokens"] == _ENGINE_PROMPT_TOKENS_13033
    assert (
        events[-1]["usage"]["output_tokens"]
        == _ENGINE_GENERATION_TOKENS_13033
    )

    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=_ENGINE_COMPLETED_13033,
        compactions=0,
        normal_requests=_ENGINE_COMPLETED_13033,
        prompt_tokens=_ENGINE_PROMPT_TOKENS_13033,
        generation_tokens=_ENGINE_GENERATION_TOKENS_13033,
    )
    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_B),
        expected_completed_logical_model_requests=_ENGINE_COMPLETED_13033,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )

    assert trace_requests["hidden_web_fetch_model_requests"] == 1
    assert (
        trace_requests["completed_logical_model_requests"]
        == _ENGINE_COMPLETED_13033
    )
    assert len(trace_requests["model_request_ids"]) == _ENGINE_COMPLETED_13033
    web_fetch_ids = [
        request_id
        for request_id in trace_requests["model_request_ids"]
        if request_id.startswith("qwen-hidden-web-fetch-sha256:")
    ]
    assert len(web_fetch_ids) == 1
    evidence = trace_requests["qwen_compaction_metric_evidence"]
    # The engine served 18 requests at 32768 and nothing at 20000: the whole
    # gap was one untraced request, not a clamped max_tokens.
    assert evidence["normal_requests"] == _ENGINE_COMPLETED_13033
    assert evidence["total_compaction_requests"] == 0
    assert evidence["max_tokens_sum"] == _ENGINE_MAX_TOKENS_SUM_13033
    assert evidence["max_tokens_le_20000"] == 0
    assert (
        evidence["hidden_prompt_tokens"]
        == _HIDDEN_SIDE_QUERY_USAGE_13033[0]
    )
    assert (
        evidence["hidden_generation_tokens"]
        == _HIDDEN_SIDE_QUERY_USAGE_13033[1]
    )

    # Identity is deterministic across replays, like every other request class.
    assert trace_requests["model_request_ids"] == (
        contract.validate_fixed32_trace_model_requests(
            copy.deepcopy(events),
            expected_session_id=contract.fixed32_trace_session_id(TASK_B),
            expected_completed_logical_model_requests=(
                _ENGINE_COMPLETED_13033
            ),
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )["model_request_ids"]
    )

    # And the whole provenance gate that killed the arm now closes.
    trace_path = tmp_path / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    metrics_pre_path = tmp_path / "vllm_metrics_pre.txt"
    metrics_post_path = tmp_path / "vllm_metrics_post.txt"
    metrics_pre_path.write_bytes(metrics_pre)
    metrics_post_path.write_bytes(metrics_post)
    task_key_id = "c" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_B,
        trace_path=trace_path,
        agent_meta=_fixed32_agent_meta(runner, tmp_path, instance_id=TASK_B),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(task_key_id, 0, 1),
        task_auth_after=_task_evidence(
            task_key_id, _ENGINE_COMPLETED_13033, 73
        ),
        metrics_pre_path=metrics_pre_path,
        metrics_post_path=metrics_post_path,
    )
    assert (
        provenance["completed_logical_model_requests"]
        == _ENGINE_COMPLETED_13033
    )
    assert (
        provenance["trace_completed_logical_model_requests"]
        == _ENGINE_COMPLETED_13033
    )


def test_qwen_failed_web_fetch_claims_no_hidden_request() -> None:
    """The outer catch ran, so no completed engine request is owed."""
    events = _qwen_web_fetch_trace_13033(
        tool_result_content=(
            f"{contract.QWEN_WEB_FETCH_ERROR_PREFIX}Error during fetch for "
            f"{_WEB_FETCH_URL_13033}: Request failed with status code 404 "
            "Not Found"
        ),
        tool_result_is_error=True,
    )
    visible = len(_VISIBLE_TURN_USAGE_13033)
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=visible,
        compactions=0,
        normal_requests=visible,
        prompt_tokens=events[-1]["usage"]["input_tokens"],
        generation_tokens=events[-1]["usage"]["output_tokens"],
    )
    trace_requests = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=contract.fixed32_trace_session_id(TASK_B),
        expected_completed_logical_model_requests=visible,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
    )
    assert trace_requests["hidden_web_fetch_model_requests"] == 0
    assert trace_requests["completed_logical_model_requests"] == visible


def test_qwen_web_fetch_stays_fail_closed_on_unaccountable_traffic() -> None:
    """A credited side query still has to be earned by the exact display.

    Two failures the fix must NOT have introduced: a web_fetch whose result
    does not prove the side query ran cannot buy a request, and a successful
    web_fetch cannot cover an 19th engine request that nothing accounts for.
    """
    # 1. The display is not the one executeDirectFetch emits after the side
    #    query resolves -- so the trace cannot name the 18th request.
    forged = _qwen_web_fetch_trace_13033(
        tool_result_content=(
            "Content from https://example.invalid/other processed "
            "successfully."
        ),
    )
    with pytest.raises(contract.ContractError) as forged_error:
        contract.validate_fixed32_trace_model_requests(
            forged,
            expected_session_id=contract.fixed32_trace_session_id(TASK_B),
        )
    assert "web_fetch closure" in str(forged_error.value)

    # 2. One legitimate side query does not license a second unaccounted
    #    request: the algebra still names the shortfall and fails closed.
    events = _qwen_web_fetch_trace_13033()
    _add_hidden_side_query_usage(
        events,
        prompt_tokens=_HIDDEN_SIDE_QUERY_USAGE_13033[0],
        completion_tokens=_HIDDEN_SIDE_QUERY_USAGE_13033[1],
    )
    metrics_pre, metrics_post = _qwen_compaction_metrics(
        completed=_ENGINE_COMPLETED_13033 + 1,
        compactions=0,
        normal_requests=_ENGINE_COMPLETED_13033 + 1,
        prompt_tokens=_ENGINE_PROMPT_TOKENS_13033,
        generation_tokens=_ENGINE_GENERATION_TOKENS_13033,
    )
    with pytest.raises(contract.ContractError) as extra_error:
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=contract.fixed32_trace_session_id(TASK_B),
            expected_completed_logical_model_requests=(
                _ENGINE_COMPLETED_13033 + 1
            ),
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )
    message = str(extra_error.value)
    assert "max-token algebra does not reconcile" in message
    # The clause now names its numbers instead of making the next run guess.
    assert f"trace normal={_ENGINE_COMPLETED_13033}" in message
    assert f"completed={_ENGINE_COMPLETED_13033 + 1}" in message
    # The "one 32768 request short" read survives the era table -- it is now
    # named per deployed ceiling, so the reader sees WHICH era it would have
    # reconciled under if one request were accounted for.
    assert "shortfall per deployed ceiling" in message
    assert "32768: 32768" in message


# --------------------------------------------------------------------------
# FR14 validator shape-closure (2026-08-17). Three arms died in three serves to
# three shapes this validator did not model. Every value below is measured from
# the real traces named in each test, not invented.
#
# The fail-closed purpose is unchanged: the METER is still the engine plus the
# ingress ledger, and the trace is only allowed to NAME the requests they
# already counted. These tests pin that each newly-modelled shape accounts for
# the right NUMBER of requests, and that the unaccountable variant of the same
# shape still raises.
# --------------------------------------------------------------------------

# fr14_b1_stock_20260817T020534Z / astropy__astropy-13236, trace line 159:
# the model called web_fetch with a url and no prompt. qwen-code validates the
# call against its JSON schema BEFORE executing it, so this one never reached
# executeDirectFetch: it fetched nothing and issued no runSideQuery.
_SCHEMA_REJECTION_RESULT = "params must have required property 'prompt'"


def test_web_fetch_schema_rejection_hides_no_model_request() -> None:
    """A call the schema rejected owes zero completed engine requests.

    Killed the 2026-08-17T02:05Z arm as
    "fixed32 qwen web_fetch prompt is empty or invalid": the validator treated
    a malformed invocation as unaccountable traffic. It is the opposite -- the
    tool never ran, so there is nothing to account for. The ledger for this
    task therefore expects the VISIBLE turn count with NO hidden side query.
    """
    events = _qwen_web_fetch_trace_13033(
        web_fetch_input={"url": _WEB_FETCH_URL_13033},
        tool_result_content=_SCHEMA_REJECTION_RESULT,
        tool_result_is_error=True,
    )
    summary = contract.validate_fixed32_trace_model_requests(events)

    assert summary["hidden_web_fetch_model_requests"] == 0
    assert summary["completed_logical_model_requests"] == len(
        _VISIBLE_TURN_USAGE_13033
    )


def test_web_fetch_schema_rejection_still_raises_if_the_call_succeeded() -> None:
    """The fail-closed half: malformed input + a SUCCESS closure is impossible.

    If a call that the schema should have rejected comes back with the
    processed display, the trace is describing traffic that cannot have
    happened. That is exactly the unaccountable case the validator exists for,
    so it must still refuse -- otherwise the fix above would be a hole.
    """
    events = _qwen_web_fetch_trace_13033(
        web_fetch_input={"url": _WEB_FETCH_URL_13033},
        tool_result_content=contract.QWEN_WEB_FETCH_SUCCESS_TEMPLATE.format(
            url=_WEB_FETCH_URL_13033
        ),
        tool_result_is_error=False,
    )
    with pytest.raises(
        contract.ContractError, match="web_fetch prompt is empty or invalid"
    ):
        contract.validate_fixed32_trace_model_requests(events)


def test_valid_web_fetch_still_counts_its_hidden_side_query() -> None:
    """Guard against fixing the rejection by weakening the counting."""
    events = _qwen_web_fetch_trace_13033()
    summary = contract.validate_fixed32_trace_model_requests(events)

    assert summary["hidden_web_fetch_model_requests"] == 1
    assert summary["completed_logical_model_requests"] == (
        len(_VISIBLE_TURN_USAGE_13033) + 1
    )


# fr14_b1_stock_20260817T031507Z / astropy__astropy-13236 result record: under
# the no-net agent settings qwen-code enforces the web_fetch deny rule against
# equivalent shell commands, so the model's `curl https://...` came back
# "denied by permission rules" and was recorded here.
# The real record names tool_use chatcmpl-tool-a41779fbb2de9484, which that
# trace also contains (its tool_use is at line 260, its denial tool_result at
# 261). The fixture's own ids are synthetic, so the denial below points at the
# fixture's first tool_use -- the JOIN is the thing under test, not the string.
_REAL_PERMISSION_DENIAL = {
    "tool_name": "run_shell_command",
    "tool_use_id": "chatcmpl-tool-0000000000000000",
}


def test_permission_denials_do_not_break_request_accounting() -> None:
    """A denied tool call is normal evidence, and costs the ledger nothing.

    Killed the 2026-08-17T03:15Z arm as
    "fixed32 qwen result evidence is incomplete" because the validator demanded
    permission_denials == []. A denial hides no model request: the assistant's
    tool_use is already counted in its own group and the denial arrives as an
    ordinary paired tool_result. Unlike web_fetch, the denied tool never runs
    and never calls the model.

    Under the no-net settings denials are EXPECTED, so refusing them would have
    made the fail-closed validator fail on its own safety feature.
    """
    events = _qwen_web_fetch_trace_13033(
        permission_denials=[dict(_REAL_PERMISSION_DENIAL)]
    )
    summary = contract.validate_fixed32_trace_model_requests(events)

    assert summary["completed_logical_model_requests"] == (
        len(_VISIBLE_TURN_USAGE_13033) + 1
    )


@pytest.mark.parametrize(
    "denial",
    (
        {"tool_use_id": "chatcmpl-tool-a41779fbb2de9484"},
        {"tool_name": "run_shell_command"},
        {"tool_name": "", "tool_use_id": "chatcmpl-tool-a41779fbb2de9484"},
        {"tool_name": "run_shell_command", "tool_use_id": ""},
        "run_shell_command",
    ),
    ids=("no-name", "no-id", "empty-name", "empty-id", "not-a-record"),
)
def test_malformed_permission_denial_still_raises(denial: object) -> None:
    """Accepting denials must not mean accepting anything in that field."""
    events = _qwen_web_fetch_trace_13033(permission_denials=[denial])
    with pytest.raises(
        contract.ContractError, match="permission denial record is invalid"
    ):
        contract.validate_fixed32_trace_model_requests(events)


def test_permission_denials_must_be_a_list() -> None:
    events = _qwen_web_fetch_trace_13033()
    events[-1]["permission_denials"] = {"tool_name": "run_shell_command"}
    with pytest.raises(
        contract.ContractError, match="permission denials are invalid"
    ):
        contract.validate_fixed32_trace_model_requests(events)


def test_result_usage_must_still_be_a_mapping() -> None:
    """The half of the old predicate that was load-bearing stays."""
    events = _qwen_web_fetch_trace_13033()
    events[-1]["usage"] = None
    with pytest.raises(
        contract.ContractError, match="result evidence is incomplete"
    ):
        contract.validate_fixed32_trace_model_requests(events)


def test_permission_denial_naming_an_unknown_tool_use_raises() -> None:
    """A denial must belong to a call the trace actually contains.

    If it names a tool_use this trace has never seen, the trace is not a
    complete record of its own session -- and an independent request count off
    an incomplete record is worthless. Costs the ledger nothing, still refused.
    """
    events = _qwen_web_fetch_trace_13033(
        permission_denials=[
            {
                "tool_name": "run_shell_command",
                "tool_use_id": "chatcmpl-tool-neverseen",
            }
        ]
    )
    with pytest.raises(
        contract.ContractError, match="permission denial names an unknown tool use"
    ):
        contract.validate_fixed32_trace_model_requests(events)


# fr14_b1_probe_20260817T011303Z and T012523Z, trace line 2: qwen-code renders a
# client-side failure as an ordinary assistant text record with all-zero usage,
# then closes with subtype="success" / is_error=false / num_turns=1.
_API_ERROR_ENGINE = (
    "[API Error: EngineCore encountered an issue. See stack trace (above) "
    "for the root cause.]"
)
_API_ERROR_CONNECTION = "[API Error: Connection error. (cause: fetch failed)]"


@pytest.mark.parametrize(
    "banner", (_API_ERROR_ENGINE, _API_ERROR_CONNECTION),
    ids=("enginecore-died", "never-left-the-client"),
)
def test_api_error_banner_is_refused_not_counted_as_a_served_request(
    banner: str,
) -> None:
    """The silent-overcount class: counting traffic that never happened.

    Both probe traces close cleanly -- success, not-an-error, one turn -- around
    a single assistant record that is qwen-code narrating its own failure. The
    validator used to return completed_logical_model_requests = 1 for a request
    the engine served ZERO of. A fail-closed counter may under-claim and refuse;
    it must never invent a request, because that silently absolves a real gap
    somewhere else in the ledger.

    The trace cannot distinguish "EngineCore died after serving" from "the fetch
    never left the client", so the honest answer is refusal, not a guess.
    """
    events = _qwen_web_fetch_trace_13033()
    events[-2]["message"]["content"] = [{"type": "text", "text": banner}]
    with pytest.raises(
        contract.ContractError, match="client-side API error banner"
    ):
        contract.validate_fixed32_trace_model_requests(events)


def test_ordinary_text_that_merely_mentions_an_api_error_is_not_refused() -> None:
    """The refusal keys on the banner PREFIX, not on the words appearing anywhere.

    An agent discussing an API error in its final answer is an ordinary served
    turn and must still count.
    """
    events = _qwen_web_fetch_trace_13033()
    events[-2]["message"]["content"] = [
        {"type": "text", "text": "I fixed the handler that logged [API Error: ...] lines."}
    ]
    summary = contract.validate_fixed32_trace_model_requests(events)
    assert summary["completed_logical_model_requests"] == (
        len(_VISIBLE_TURN_USAGE_13033) + 1
    )
