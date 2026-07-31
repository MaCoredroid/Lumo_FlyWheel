from __future__ import annotations

import copy
import importlib.util
import json
import os
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
    del events[-2]


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
        _remove_final_text,
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
        b'{"memory":{"enableAutoSkill":false}}\n'
    )
    assert metadata == {
        "source": "config/fr13_fixed32/qwen_system_settings.json",
        "bytes": 37,
        "sha256": (
            "8a872a4f6f257f6d7a45f24f42500964"
            "f56e1500c5342218b71d02afe4d31fb6"
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
        "2643d1d64c03887654794d9bd00a88fb"
        "f9ced7362e034557cf196b8a37e744bc"
    )
    assert set(tree["entrypoints"]) == set(
        runner._FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS
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
    (package_root / "chunks/unpinned.js").write_text(
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
    return package_root / "chunks/unpinned.js"


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
    assert "$HOME/qwen_agent_bundle" in calls[0][-1]
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
            "8a872a4f6f257f6d7a45f24f42500964"
            "f56e1500c5342218b71d02afe4d31fb6"
        ),
        "size": 37,
    }


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
