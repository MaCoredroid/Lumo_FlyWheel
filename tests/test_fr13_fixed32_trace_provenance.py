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
                    "name": "task",
                    "input": {},
                }
            ],
            stop_reason="tool_use",
        ),
        _context_event(
            event_type="user",
            event_id="parallel-dispatch-result",
            session_id=session_id,
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
        events.append(
            _context_event(
                event_type="user",
                event_id=f"nested-{group_index}-result",
                session_id=session_id,
                parent_tool_use_id="parallel-parent-tool",
            )
        )
    events.append(
        _context_event(
            event_type="user",
            event_id="parallel-top-resume",
            session_id=session_id,
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


def test_parallel_terminal_records_count_as_19_response_groups(
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

    assert trace_requests["completed_logical_model_requests"] == 19
    assert len(trace_requests["model_request_ids"]) == 19
    assert len(set(trace_requests["model_request_ids"])) == 19

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
        task_auth_after=_task_evidence(task_key_id, 19, 77),
    )
    assert provenance["trace_completed_logical_model_requests"] == 19
    assert provenance["completed_logical_model_requests"] == 19

    floor_trace = floor_gate._fixed32_trace_model_requests(
        trace_path,
        provenance=provenance,
    )
    assert floor_trace["completed_logical_model_requests"] == 19
    assert len(floor_trace["model_request_id_sha256s"]) == 19
    assert floor_trace["engine_id_joinable"] is False


def test_zero_work_nested_error_is_only_a_group_boundary() -> None:
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

    assert with_boundary["completed_logical_model_requests"] == 19
    assert with_boundary["model_request_ids"] == baseline["model_request_ids"]
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

    with pytest.raises(contract.ContractError, match="transition is invalid"):
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
