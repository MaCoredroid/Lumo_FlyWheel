from __future__ import annotations

import copy
import importlib.util
import json
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
    return {
        "type": event_type,
        "uuid": event_id,
        "session_id": session_id,
        "parent_tool_use_id": parent_tool_use_id,
    }


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
        agent_meta={
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        },
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
        agent_meta={
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        },
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
        agent_meta={
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        },
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
