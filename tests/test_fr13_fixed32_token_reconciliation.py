"""Exact token reconciliation for fixed32 campaigns.

THE FAILURE THIS FILE EXISTS FOR. `validate_fixed32_qwen_campaign_metrics`
closed the campaign token identity with no slack against the wrong meter: the
sum of qwen-code's self-reported `result.usage`. That is a third-party agent's
own accounting, and it under-credits its own hidden requests three ways --

  1. a compaction the engine served and billed, whose summary the agent then
     rejected and discarded (control arm `astropy__astropy-14369`: exactly one
     uncredited 1,550-token request, a 361 s compaction at ~69.4k prompt),
  2. retried-and-discarded first turns (`astropy__astropy-13398`: five
     pre-first-turn requests, ~119k prompt),
  3. delegated sub-agent turns, which self-report `{"input_tokens": 0,
     "output_tokens": 0}` on every assistant record -- a mechanism this repo
     already documents at fr13_fixed32_contract.py:1579-1600.

Request-level diagnosis proved there is NO unattributed traffic: 639/639
engine requests belong to a ledger-attributed completion, zero retries, zero
preemptions, max-tokens algebra closing to the token. The gap is entirely the
agent's accounting. On the 2026-08-15 width-4 screen it was 189,780 prompt
(0.73%) and 5,654 generation (2.2%), concentrated in 3 of 32 task-instances --
so a 4-task gate reconciles by luck (P(clean) ~ 0.7 at n=4, ~ 0.2 at n=16) and
a 16-task arm cannot. Two arms and ~10 GPU-hours produced no verdict.

The fix is a different meter, not a tolerance: the proxy terminates every
completion, so it records the ENGINE's own per-request count on the ingress
ledger, inside `record_sha256`. The tests below pin the three mechanisms, the
compatibility path that keeps every historical artifact valid, and the guard
that makes a meter recording nothing fail the audit instead of falling back to
the meter it replaced -- the mistake `vllm_request_metrics.jsonl` got away with
for months, 0 bytes in every arm of every run, including the passing gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import fr13_fixed32_contract as contract  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402
from lumo_flywheel_serving.inference_proxy import (  # noqa: E402
    Fixed32DigestLedger,
    Fixed32IngressError,
    build_proxy_handler,
    derive_fixed32_task_bearer,
    verify_fixed32_ingress_ledger,
)


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign_fixtures = _load(
    "token_reconciliation_campaign_fixtures",
    "tests/test_fr13_fixed32_b4_campaign_provenance.py",
)
proxy_fixtures = _load(
    "token_reconciliation_proxy_fixtures",
    "tests/test_fr13_fixed32_ingress_proxy.py",
)

TASK_IDS = ("astropy__astropy-14369", "astropy__astropy-13398")

# ---------------------------------------------------------------------------
# the real numbers, from the 2026-08-15 width-4 screen control arm
# ---------------------------------------------------------------------------
# astropy__astropy-14369's four hidden requests. The first three are the task's
# exact hidden credit -- 144 + 443 + 125 = 712 -- and the fourth is the
# rejected compaction the engine billed and the agent discarded.
HIDDEN_PROMPT_TOKENS = (144, 443, 125)
HIDDEN_CREDIT_TOKENS = 712
DISCARDED_COMPACTION_PROMPT_TOKENS = 1_550
# The design pins the PROMPT side of each hidden request to the token. Their
# generation sides are not in the record, so the fixture picks them; what the
# tests assert is which meter carries them, not their size.
HIDDEN_GENERATION_TOKENS = (8, 8, 8)
DISCARDED_COMPACTION_GENERATION_TOKENS = 1_212
VISIBLE_TURNS = 25


def _assert_real_numbers_are_self_consistent() -> None:
    assert sum(HIDDEN_PROMPT_TOKENS) == HIDDEN_CREDIT_TOKENS


# ---------------------------------------------------------------------------
# ledger fixtures
# ---------------------------------------------------------------------------
def _new_schema_ledger(
    path: Path,
    usage: list[tuple[int | None, int | None]],
) -> bytes:
    """A ledger written by a proxy that meters token usage.

    One completed logical request per entry, written through the real writer so
    the chain and the digests are the production ones.
    """
    ledger = Fixed32DigestLedger(path, role="proxy")
    ledger.append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256="a" * 64,
    )
    for index, (prompt_tokens, completion_tokens) in enumerate(usage):
        digest = f"{index:064x}"
        ledger.append(
            phase="campaign",
            event="logical_begin",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            outcome="accepted",
        )
        ledger.append(
            phase="campaign",
            event="attempt_begin",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            wire_id_sha256=digest,
            engine_request_id_sha256=digest,
            outcome="dispatched",
            evidence_sha256=digest,
        )
        ledger.append(
            phase="campaign",
            event="attempt_result",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            wire_id_sha256=digest,
            engine_request_id_sha256=digest,
            status_code=200,
            outcome="response",
            evidence_sha256=digest,
        )
        ledger.append(
            phase="campaign",
            event="logical_complete",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            outcome="completed",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    ledger.append(
        phase="campaign",
        event="campaign_finalize",
        outcome="finalized",
        evidence_sha256="a" * 64,
    )
    ledger.close()
    return path.read_bytes()


def _old_schema_ledger(path: Path, completions: int) -> bytes:
    """A ledger from before the usage fields existed.

    Hand-built rather than written, because the production writer can no longer
    produce one: the whole point of the compatibility path is that artifacts
    already on disk keep validating.
    """
    rows: list[dict[str, Any]] = []
    head = "0" * 64

    def append(**fields: Any) -> None:
        nonlocal head
        row = {
            "schema": "fr13.fixed32.ingress-ledger-record.v1",
            "seq": len(rows),
            "role": "proxy",
            "phase": fields.pop("phase"),
            "event": fields.pop("event"),
            "route": fields.pop("route", None),
            "task_key_id": fields.pop("task_key_id", None),
            "logical_id_sha256": fields.pop("logical_id_sha256", None),
            "wire_id_sha256": fields.pop("wire_id_sha256", None),
            "engine_request_id_sha256": fields.pop(
                "engine_request_id_sha256", None
            ),
            "status_code": fields.pop("status_code", None),
            "outcome": fields.pop("outcome", None),
            "reason": fields.pop("reason", None),
            "evidence_sha256": fields.pop("evidence_sha256", None),
            "prev_sha256": head,
        }
        assert not fields
        head = hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        row["record_sha256"] = head
        rows.append(row)

    append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256="a" * 64,
    )
    for index in range(completions):
        digest = f"{index:064x}"
        append(
            phase="campaign",
            event="logical_begin",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            outcome="accepted",
        )
        append(
            phase="campaign",
            event="logical_complete",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256=digest,
            outcome="aborted",
            reason="no_completed_attempt",
        )
    append(
        phase="campaign",
        event="campaign_finalize",
        outcome="finalized",
        evidence_sha256="a" * 64,
    )
    raw = (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


# ---------------------------------------------------------------------------
# campaign fixtures
# ---------------------------------------------------------------------------
def _rejected_compaction_campaign() -> tuple[list[dict[str, Any]], bytes]:
    """astropy__astropy-14369's four hidden requests, one of them discarded.

    The agent's `result.usage` credits three of the four (712 tokens). The
    engine billed all four (712 + 1,550). The task-auth count knows there were
    four requests -- the token accounting is the only thing that disagrees.
    """
    tasks: list[dict[str, Any]] = []
    for index, instance_id in enumerate(TASK_IDS):
        carries_hidden = index == 0
        tasks.append(
            {
                "instance_id": instance_id,
                "expected_session_id": contract.fixed32_trace_session_id(
                    instance_id
                ),
                "expected_completed_logical_model_requests": (
                    VISIBLE_TURNS + (4 if carries_hidden else 0)
                ),
                "budget_capped": False,
                "events": campaign_fixtures._qwen_trace_with_request_count(
                    instance_id,
                    VISIBLE_TURNS,
                    hidden_input_tokens=(
                        HIDDEN_CREDIT_TOKENS if carries_hidden else 0
                    ),
                    hidden_output_tokens=(
                        sum(HIDDEN_GENERATION_TOKENS) if carries_hidden else 0
                    ),
                ),
            }
        )
    visible_requests = len(TASK_IDS) * VISIBLE_TURNS
    metrics_post = campaign_fixtures._metrics(
        visible_requests + 4,
        compactions=4,
        normal_requests=visible_requests,
        prompt_tokens=(
            visible_requests * 32
            + HIDDEN_CREDIT_TOKENS
            + DISCARDED_COMPACTION_PROMPT_TOKENS
        ),
        generation_tokens=(
            visible_requests * 8
            + sum(HIDDEN_GENERATION_TOKENS)
            + DISCARDED_COMPACTION_GENERATION_TOKENS
        ),
    )
    return tasks, metrics_post


def _rejected_compaction_ledger_usage() -> list[tuple[int, int]]:
    visible_requests = len(TASK_IDS) * VISIBLE_TURNS
    return (
        [(32, 8)] * visible_requests
        + list(zip(HIDDEN_PROMPT_TOKENS, HIDDEN_GENERATION_TOKENS, strict=True))
        + [
            (
                DISCARDED_COMPACTION_PROMPT_TOKENS,
                DISCARDED_COMPACTION_GENERATION_TOKENS,
            )
        ]
    )


DELEGATED_TOP_LEVEL_TURNS = 6
DELEGATED_SUBAGENT_TURNS = 3
# A delegated session also serves one terminal turn qwen-code records nowhere:
# the sub-agent's own final answer. The trace validator already accounts for it
# as a hidden terminal request; the engine billed it like any other.
DELEGATED_HIDDEN_TERMINAL_TURNS = 1
DELEGATED_PROMPT_TOKENS = 5_000
DELEGATED_GENERATION_TOKENS = 400


def _delegated_subagent_trace(instance_id: str) -> list[dict[str, Any]]:
    """A trace whose sub-agent turns self-report 0/0.

    The engine served every one of them. The agent reports zero for each and
    omits the sub-agent's closing turn entirely, so the trace meter cannot see
    tokens that certainly existed.
    """
    session_id = contract.fixed32_trace_session_id(instance_id)

    def assistant(
        uuid: str,
        parent: str | None,
        content: list[dict[str, Any]],
        stop_reason: str | None,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "type": "assistant",
            "uuid": uuid,
            "session_id": session_id,
            "parent_tool_use_id": parent,
            "message": {
                "id": uuid,
                "type": "message",
                "role": "assistant",
                "model": "qwen3.8-27b-nvfp4-radixark",
                "content": content,
                "stop_reason": stop_reason,
                "usage": usage,
            },
        }

    def tool_result(
        uuid: str,
        parent: str | None,
        tool_use_id: str,
        text: str,
    ) -> dict[str, Any]:
        return {
            "type": "user",
            "uuid": uuid,
            "session_id": session_id,
            "parent_tool_use_id": parent,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": text,
                        "is_error": False,
                    }
                ],
            },
        }

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
    for ordinal in range(DELEGATED_TOP_LEVEL_TURNS - 2):
        tool_id = f"tool-{ordinal}-{instance_id}"
        events.append(
            assistant(
                f"assistant-{ordinal}-{instance_id}",
                None,
                [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "read_file",
                        "input": {},
                    }
                ],
                "tool_use",
                {"input_tokens": 32, "output_tokens": 8},
            )
        )
        events.append(
            tool_result(
                f"tool-result-{ordinal}-{instance_id}", None, tool_id, "done"
            )
        )
    agent_tool_id = f"agent-tool-{instance_id}"
    prompt = "explore the failing test"
    events.append(
        assistant(
            f"assistant-agent-{instance_id}",
            None,
            [
                {
                    "type": "tool_use",
                    "id": agent_tool_id,
                    "name": "agent",
                    "input": {"description": "explore", "prompt": prompt},
                }
            ],
            "tool_use",
            {"input_tokens": 32, "output_tokens": 8},
        )
    )
    events.append(
        {
            "type": "user",
            "uuid": f"agent-prompt-{instance_id}",
            "session_id": session_id,
            "parent_tool_use_id": agent_tool_id,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        }
    )
    for ordinal in range(DELEGATED_SUBAGENT_TURNS):
        sub_tool_id = f"sub-tool-{ordinal}-{instance_id}"
        events.append(
            assistant(
                f"assistant-sub-{ordinal}-{instance_id}",
                agent_tool_id,
                [
                    {
                        "type": "tool_use",
                        "id": sub_tool_id,
                        "name": "read_file",
                        "input": {},
                    }
                ],
                "tool_use",
                # The mechanism, verbatim.
                {"input_tokens": 0, "output_tokens": 0},
            )
        )
        events.append(
            tool_result(
                f"sub-tool-result-{ordinal}-{instance_id}",
                agent_tool_id,
                sub_tool_id,
                "sub done",
            )
        )
    events.append(
        tool_result(
            f"agent-result-{instance_id}",
            None,
            agent_tool_id,
            "subagent report",
        )
    )
    events.append(
        assistant(
            f"final-{instance_id}",
            None,
            [{"type": "text", "text": "complete"}],
            None,
            {"input_tokens": 32, "output_tokens": 8},
        )
    )
    visible_input = DELEGATED_TOP_LEVEL_TURNS * 32
    visible_output = DELEGATED_TOP_LEVEL_TURNS * 8
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "uuid": f"result-{instance_id}",
            "session_id": session_id,
            "is_error": False,
            "duration_ms": 100,
            "duration_api_ms": 90,
            "num_turns": DELEGATED_TOP_LEVEL_TURNS,
            "result": "complete",
            "usage": {
                "input_tokens": visible_input,
                "output_tokens": visible_output,
                "total_tokens": visible_input + visible_output,
            },
            "permission_denials": [],
        }
    )
    return events


def _delegated_campaign() -> tuple[list[dict[str, Any]], bytes]:
    per_task = (
        DELEGATED_TOP_LEVEL_TURNS
        + DELEGATED_SUBAGENT_TURNS
        + DELEGATED_HIDDEN_TERMINAL_TURNS
    )
    tasks = [
        {
            "instance_id": instance_id,
            "expected_session_id": contract.fixed32_trace_session_id(
                instance_id
            ),
            "expected_completed_logical_model_requests": per_task,
            "budget_capped": False,
            "events": _delegated_subagent_trace(instance_id),
        }
        for instance_id in TASK_IDS
    ]
    tasks_count = len(TASK_IDS)
    unreported = DELEGATED_SUBAGENT_TURNS + DELEGATED_HIDDEN_TERMINAL_TURNS
    metrics_post = campaign_fixtures._metrics(
        tasks_count * per_task,
        compactions=0,
        normal_requests=tasks_count * per_task,
        prompt_tokens=(
            tasks_count * DELEGATED_TOP_LEVEL_TURNS * 32
            + tasks_count * unreported * DELEGATED_PROMPT_TOKENS
        ),
        generation_tokens=(
            tasks_count * DELEGATED_TOP_LEVEL_TURNS * 8
            + tasks_count * unreported * DELEGATED_GENERATION_TOKENS
        ),
    )
    return tasks, metrics_post


def _delegated_ledger_usage() -> list[tuple[int, int]]:
    unreported = DELEGATED_SUBAGENT_TURNS + DELEGATED_HIDDEN_TERMINAL_TURNS
    usage: list[tuple[int, int]] = []
    for _instance_id in TASK_IDS:
        usage += [(32, 8)] * DELEGATED_TOP_LEVEL_TURNS
        usage += [
            (DELEGATED_PROMPT_TOKENS, DELEGATED_GENERATION_TOKENS)
        ] * unreported
    return usage


# ---------------------------------------------------------------------------
# the regression: the ledger closes what the self-report cannot
# ---------------------------------------------------------------------------
def test_rejected_compaction_defeats_the_trace_meter() -> None:
    """astropy__astropy-14369, to the token."""
    _assert_real_numbers_are_self_consistent()
    tasks, metrics_post = _rejected_compaction_campaign()

    with pytest.raises(
        contract.ContractError,
        match="aggregate and vLLM token usage do not reconcile",
    ):
        contract.validate_fixed32_qwen_campaign_metrics(
            tasks,
            metrics_pre=campaign_fixtures._metrics(0),
            metrics_post=metrics_post,
        )


def test_rejected_compaction_reconciles_against_the_ledger(
    tmp_path: Path,
) -> None:
    _assert_real_numbers_are_self_consistent()
    tasks, metrics_post = _rejected_compaction_campaign()
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl", _rejected_compaction_ledger_usage()
    )

    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
        ingress_ledger=ledger,
    )

    token = reconciliation["token_reconciliation"]
    assert token["basis"] == contract.QWEN_TOKEN_BASIS_LEDGER
    assert token["ledger_token_usage_records"] == len(TASK_IDS) * VISIBLE_TURNS + 4
    # The gap is the discarded compaction, exactly, and it is published rather
    # than absorbed: a reader can see how much the agent failed to credit.
    assert (
        token["qwen_trace_prompt_token_gap"]
        == DISCARDED_COMPACTION_PROMPT_TOKENS
    )
    assert (
        token["qwen_trace_generation_token_gap"]
        == DISCARDED_COMPACTION_GENERATION_TOKENS
    )
    assert token["ledger_prompt_tokens"] == token["vllm_prompt_tokens"]
    assert token["ledger_generation_tokens"] == token["vllm_generation_tokens"]
    assert (
        reconciliation["metric_evidence"]["token_reconciliation"] == token
    )


def test_the_three_credited_hidden_requests_are_the_tasks_hidden_credit(
    tmp_path: Path,
) -> None:
    """144 + 443 + 125 = 712, and 1,550 is what the agent dropped."""
    tasks, metrics_post = _rejected_compaction_campaign()
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl", _rejected_compaction_ledger_usage()
    )
    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
        ingress_ledger=ledger,
    )
    evidence = reconciliation["metric_evidence"]
    assert evidence["hidden_prompt_tokens"] == HIDDEN_CREDIT_TOKENS
    assert (
        evidence["prompt_tokens"] - evidence["visible_prompt_tokens"]
        == HIDDEN_CREDIT_TOKENS + DISCARDED_COMPACTION_PROMPT_TOKENS
    )


def test_delegated_subagent_zero_zero_defeats_the_trace_meter() -> None:
    """The 0/0 self-report, the mechanism at contract.py:1579-1600."""
    tasks, metrics_post = _delegated_campaign()

    with pytest.raises(
        contract.ContractError,
        match="aggregate and vLLM token usage do not reconcile",
    ):
        contract.validate_fixed32_qwen_campaign_metrics(
            tasks,
            metrics_pre=campaign_fixtures._metrics(0),
            metrics_post=metrics_post,
        )


def test_delegated_subagent_zero_zero_reconciles_against_the_ledger(
    tmp_path: Path,
) -> None:
    tasks, metrics_post = _delegated_campaign()
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl", _delegated_ledger_usage()
    )

    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
        ingress_ledger=ledger,
    )

    token = reconciliation["token_reconciliation"]
    unreported = DELEGATED_SUBAGENT_TURNS + DELEGATED_HIDDEN_TERMINAL_TURNS
    assert token["basis"] == contract.QWEN_TOKEN_BASIS_LEDGER
    assert token["qwen_trace_prompt_token_gap"] == (
        len(TASK_IDS) * unreported * DELEGATED_PROMPT_TOKENS
    )
    assert token["ledger_prompt_tokens"] == token["vllm_prompt_tokens"]
    assert token["ledger_generation_tokens"] == token["vllm_generation_tokens"]


def test_ledger_sum_that_misses_a_request_still_fails_closed(
    tmp_path: Path,
) -> None:
    """The new meter is exact too: no slack was traded for coverage."""
    tasks, metrics_post = _rejected_compaction_campaign()
    usage = _rejected_compaction_ledger_usage()
    usage[-1] = (usage[-1][0] - 1, usage[-1][1])
    ledger = _new_schema_ledger(tmp_path / "proxy-ingress.jsonl", usage)

    with pytest.raises(
        contract.ContractError,
        match="ingress ledger and vLLM token usage do not reconcile",
    ):
        contract.validate_fixed32_qwen_campaign_metrics(
            tasks,
            metrics_pre=campaign_fixtures._metrics(0),
            metrics_post=metrics_post,
            ingress_ledger=ledger,
        )


# ---------------------------------------------------------------------------
# compatibility: historical artifacts keep validating, bit for bit
# ---------------------------------------------------------------------------
def _clean_four_task_campaign() -> tuple[list[dict[str, Any]], bytes]:
    """A B4 shape that already reconciles on the trace meter."""
    return campaign_fixtures._real_count_shape_campaign()


def test_old_schema_ledger_takes_the_compatibility_path(
    tmp_path: Path,
) -> None:
    tasks, metrics_post = _clean_four_task_campaign()
    ledger = _old_schema_ledger(tmp_path / "proxy-ingress.jsonl", 3)

    without = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
    )
    with_old = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
        ingress_ledger=ledger,
    )

    assert (
        with_old["token_reconciliation"]["basis"]
        == contract.QWEN_TOKEN_BASIS_TRACE
    )
    # Bit-identical, and that is the whole requirement: an old proof replayed
    # against its own old ledger must reproduce its own evidence digest.
    assert with_old["metric_evidence"] == without["metric_evidence"]
    assert (
        with_old["metric_evidence_sha256"] == without["metric_evidence_sha256"]
    )
    assert "token_reconciliation" not in with_old["metric_evidence"]


def test_the_branch_is_decided_by_field_presence_not_task_count(
    tmp_path: Path,
) -> None:
    """A 4-task arm on a metered ledger uses the ledger, like any other."""
    tasks, metrics_post = _clean_four_task_campaign()
    trace_prompt_total = sum(
        task["events"][-1]["usage"]["input_tokens"] for task in tasks
    )
    trace_generation_total = sum(
        task["events"][-1]["usage"]["output_tokens"] for task in tasks
    )
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl",
        [(trace_prompt_total, trace_generation_total)],
    )

    reconciliation = contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
        ingress_ledger=ledger,
    )

    assert len(tasks) == 4
    assert (
        reconciliation["token_reconciliation"]["basis"]
        == contract.QWEN_TOKEN_BASIS_LEDGER
    )


def test_the_real_historical_ledgers_read_as_old_schema() -> None:
    """The evidence on disk, not a fixture of it.

    Every ledger written before this change carries the usage keys on no row.
    If that ever reads as anything but `absent`, the compatibility branch has
    silently changed meaning for every artifact already sealed.
    """
    arm = (
        REPO
        / "output"
        / "fr13_gdn_single_launch_width4_screen_20260815T014426Z"
        / "pass_00"
        / "hydra27_fixed32_gdn_w4_single_launch_b4_20260815T014426Z"
        / "logs"
    )
    if not arm.is_dir():
        pytest.skip("screen arm evidence is not present in this checkout")
    for role in ("proxy", "engine"):
        path = arm / f"fr13_fixed32_{role}_ingress.jsonl"
        usage = contract.fixed32_ingress_ledger_token_usage(
            path.read_bytes(), role=role
        )
        assert usage["token_usage_schema"] == "absent"
        assert usage["completion_records"] > 0
        assert usage["token_usage_records"] == 0


# ---------------------------------------------------------------------------
# loud failure: a meter that records nothing fails the audit
# ---------------------------------------------------------------------------
def test_new_schema_ledger_with_no_data_fails_loudly(tmp_path: Path) -> None:
    """"Absent" and "empty" must not be indistinguishable from "fine"."""
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl", [(None, None)] * 3
    )

    with pytest.raises(
        contract.ContractError,
        match="records no token usage",
    ):
        contract.fixed32_ingress_ledger_token_usage(ledger, role="proxy")


def test_a_campaign_on_a_dead_meter_cannot_fall_back(tmp_path: Path) -> None:
    """The failure is raised even when the trace meter would have passed.

    This is the exact shape `vllm_request_metrics.jsonl` got away with: a
    campaign that reconciles by the old route while the new meter recorded
    nothing at all. Falling back here would reinstate the bug.
    """
    tasks, metrics_post = _clean_four_task_campaign()
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl", [(None, None)] * 3
    )

    contract.validate_fixed32_qwen_campaign_metrics(
        tasks,
        metrics_pre=campaign_fixtures._metrics(0),
        metrics_post=metrics_post,
    )
    with pytest.raises(
        contract.ContractError,
        match="records no token usage",
    ):
        contract.validate_fixed32_qwen_campaign_metrics(
            tasks,
            metrics_pre=campaign_fixtures._metrics(0),
            metrics_post=metrics_post,
            ingress_ledger=ledger,
        )


def test_a_partially_metered_ledger_fails_loudly(tmp_path: Path) -> None:
    ledger = _new_schema_ledger(
        tmp_path / "proxy-ingress.jsonl",
        [(32, 8), (None, None), (32, 8)],
    )

    with pytest.raises(
        contract.ContractError,
        match="token usage is incomplete",
    ):
        contract.fixed32_ingress_ledger_token_usage(ledger, role="proxy")


def test_a_half_recorded_row_fails_loudly(tmp_path: Path) -> None:
    ledger = _new_schema_ledger(tmp_path / "proxy-ingress.jsonl", [(32, 8)])
    rows = [json.loads(line) for line in ledger.decode().splitlines()]
    for row in rows:
        if row["event"] == "logical_complete":
            row["completion_tokens"] = None
            unsigned = {
                key: value
                for key, value in row.items()
                if key != "record_sha256"
            }
            row["record_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            break
    forged = (
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")

    with pytest.raises(
        contract.ContractError,
        match="half-recorded",
    ):
        contract.fixed32_ingress_ledger_token_usage(forged, role="proxy")


def test_a_spliced_mixed_schema_ledger_fails_loudly(tmp_path: Path) -> None:
    ledger = _new_schema_ledger(tmp_path / "proxy-ingress.jsonl", [(32, 8)])
    rows = [json.loads(line) for line in ledger.decode().splitlines()]
    del rows[-1]["prompt_tokens"]
    del rows[-1]["completion_tokens"]
    spliced = (
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")

    with pytest.raises(
        contract.ContractError,
        match="mixes token usage schemas",
    ):
        contract.fixed32_ingress_ledger_token_usage(spliced, role="proxy")


# ---------------------------------------------------------------------------
# the counts are tamper-evident, on the same chain as the identities
# ---------------------------------------------------------------------------
def test_editing_a_token_count_breaks_the_chain(tmp_path: Path) -> None:
    ledger_path = tmp_path / "proxy-ingress.jsonl"
    ledger = _new_schema_ledger(ledger_path, [(32, 8), (64, 16)])
    rows = [json.loads(line) for line in ledger.decode().splitlines()]
    edited = 0
    for row in rows:
        if row["event"] == "logical_complete":
            row["prompt_tokens"] = 1
            edited += 1
    assert edited == 2
    forged = (
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    ledger_path.write_bytes(forged)

    with pytest.raises(contract.ContractError, match="chain differs"):
        contract.fixed32_ingress_ledger_token_usage(forged, role="proxy")
    with pytest.raises(Fixed32IngressError, match="record digest mismatch"):
        verify_fixed32_ingress_ledger(ledger_path, expected_role="proxy")


def test_old_and_new_schema_ledgers_both_verify(tmp_path: Path) -> None:
    """Hash-chain compatibility, in both directions."""
    old_path = tmp_path / "old.jsonl"
    _old_schema_ledger(old_path, 2)
    new_path = tmp_path / "new.jsonl"
    _new_schema_ledger(new_path, [(32, 8), (64, 16)])

    old = verify_fixed32_ingress_ledger(old_path, expected_role="proxy")
    new = verify_fixed32_ingress_ledger(
        new_path, expected_role="proxy", require_finalized=True
    )
    assert old["active_requests"] == 0
    assert new["active_requests"] == 0

    for path in (old_path, new_path):
        rows, identity = floor_gate.load_fixed32_ingress_ledger(
            path,
            role="proxy",
            canonical_task_keys={"1" * 64},
            canonical_task_set_sha256="a" * 64,
        )
        assert identity["records"] == len(rows)


def test_the_writer_refuses_impossible_token_counts(tmp_path: Path) -> None:
    ledger = Fixed32DigestLedger(tmp_path / "writer.jsonl", role="proxy")
    ledger.append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256="a" * 64,
    )
    for kwargs in (
        {"prompt_tokens": -1},
        {"completion_tokens": -1},
        # bool is an int in Python; a True that means 1 token is a forgery.
        {"prompt_tokens": True},
        {"completion_tokens": False},
    ):
        with pytest.raises(Fixed32IngressError, match="token usage is invalid"):
            ledger.append(
                phase="campaign",
                event="logical_complete",
                route="chat",
                task_key_id="1" * 64,
                logical_id_sha256="2" * 64,
                outcome="completed",
                **kwargs,
            )
    with pytest.raises(
        Fixed32IngressError, match="token usage event is invalid"
    ):
        ledger.append(
            phase="campaign",
            event="logical_begin",
            route="chat",
            task_key_id="1" * 64,
            logical_id_sha256="2" * 64,
            outcome="accepted",
            prompt_tokens=32,
        )
    ledger.close()


def test_usage_on_a_non_completion_event_is_refused_by_readers(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "proxy-ingress.jsonl"
    ledger = _new_schema_ledger(ledger_path, [(32, 8)])
    rows = [json.loads(line) for line in ledger.decode().splitlines()]
    for row in rows:
        if row["event"] == "logical_begin":
            row["prompt_tokens"] = 32
            row["completion_tokens"] = 8
            unsigned = {
                key: value
                for key, value in row.items()
                if key != "record_sha256"
            }
            row["record_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            break
    # Re-chain from the edited row so only the legality rule can reject it.
    previous = "0" * 64
    for row in rows:
        row["prev_sha256"] = previous
        unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        previous = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        row["record_sha256"] = previous
    forged = (
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    ledger_path.write_bytes(forged)

    with pytest.raises(
        contract.ContractError, match="meters a non-completion event"
    ):
        contract.fixed32_ingress_ledger_token_usage(forged, role="proxy")
    with pytest.raises(
        Fixed32IngressError, match="token usage event is invalid"
    ):
        verify_fixed32_ingress_ledger(ledger_path, expected_role="proxy")
    with pytest.raises(
        floor_gate.GateError, match="token usage on a non-completion event"
    ):
        floor_gate.load_fixed32_ingress_ledger(
            ledger_path,
            role="proxy",
            canonical_task_keys={"1" * 64},
            canonical_task_set_sha256="a" * 64,
        )


# ---------------------------------------------------------------------------
# the proxy actually records it, on the route fixed32 traffic uses
# ---------------------------------------------------------------------------
def _chat_sse_body(prompt_tokens: int, completion_tokens: int) -> bytes:
    """A chat stream shaped like qwen-code's.

    qwen-code sends `stream: true` with `stream_options.include_usage`, so the
    engine's own count arrives on a trailing chunk whose `choices` is empty --
    which is why the usage sniffer cannot be tied to /v1/responses, the route
    this traffic never takes.
    """
    chunks = [
        {
            "id": "chatcmpl-upstream",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
            "usage": None,
        },
        {
            "id": "chatcmpl-upstream",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": None,
        },
        {
            "id": "chatcmpl-upstream",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    ]
    body = b""
    for chunk in chunks:
        body += b"data: " + json.dumps(chunk).encode("ascii") + b"\n\n"
    return body + b"data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("streaming", "prompt_tokens", "completion_tokens"),
    (
        (True, 69_431, DISCARDED_COMPACTION_GENERATION_TOKENS),
        (False, 144, 8),
    ),
)
def test_the_proxy_stamps_engine_usage_on_the_logical_complete_row(
    tmp_path: Path,
    streaming: bool,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    stream_body = _chat_sse_body(prompt_tokens, completion_tokens)
    json_body = json.dumps(
        {
            "id": "chatcmpl-upstream",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode("ascii")

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            payload = stream_body if streaming else json_body
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if streaming else "application/json",
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream_server, upstream_thread, upstream_url = (
        proxy_fixtures._start_server(Upstream)
    )
    ingress, secret_path, ledger_path = proxy_fixtures._proxy_ingress(tmp_path)
    ingress.begin(proxy_fixtures._begin_payload())
    task_bearer, _task_key_id = derive_fixed32_task_bearer(
        secret_path, proxy_fixtures.TASK_IDS[0]
    )
    proxy_server, proxy_thread, proxy_url = proxy_fixtures._start_server(
        build_proxy_handler(
            upstream_url,
            state_root=tmp_path / "state",
            fixed32_ingress=ingress,
        )
    )
    try:
        response = requests.post(
            f"{proxy_url}/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [],
                "stream": streaming,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {task_bearer}"},
            timeout=10,
        )
        assert response.status_code == 200
    finally:
        _stop = proxy_fixtures._stop_server
        _stop(proxy_server, proxy_thread)
        _stop(upstream_server, upstream_thread)
    ingress.finalize(proxy_fixtures._finalize_payload())
    ingress.ledger.close()

    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    completions = [row for row in rows if row["event"] == "logical_complete"]
    assert len(completions) == 1
    assert completions[0]["prompt_tokens"] == prompt_tokens
    assert completions[0]["completion_tokens"] == completion_tokens
    # Every other row carries the keys and no value: that is what distinguishes
    # this ledger from one written before the fields existed.
    assert all(
        row["prompt_tokens"] is None and row["completion_tokens"] is None
        for row in rows
        if row["event"] != "logical_complete"
    )
    verification = verify_fixed32_ingress_ledger(
        ledger_path, expected_role="proxy", require_finalized=True
    )
    assert verification["records"] == len(rows)
    usage = contract.fixed32_ingress_ledger_token_usage(
        ledger_path.read_bytes(), role="proxy"
    )
    assert usage["token_usage_schema"] == "present"
    assert usage["prompt_tokens"] == prompt_tokens
    assert usage["generation_tokens"] == completion_tokens


def test_the_proxy_ledger_never_leaks_usage_across_keepalive_requests(
    tmp_path: Path,
) -> None:
    """Two requests on one connection are two rows, not one doubled row."""
    bodies = [_chat_sse_body(144, 8), _chat_sse_body(443, 16)]
    served = threading.Semaphore(0)
    index = {"value": 0}

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            payload = bodies[min(index["value"], len(bodies) - 1)]
            index["value"] += 1
            served.release()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream_server, upstream_thread, upstream_url = (
        proxy_fixtures._start_server(Upstream)
    )
    ingress, secret_path, ledger_path = proxy_fixtures._proxy_ingress(tmp_path)
    ingress.begin(proxy_fixtures._begin_payload())
    task_bearer, _task_key_id = derive_fixed32_task_bearer(
        secret_path, proxy_fixtures.TASK_IDS[0]
    )
    proxy_server, proxy_thread, proxy_url = proxy_fixtures._start_server(
        build_proxy_handler(
            upstream_url,
            state_root=tmp_path / "state",
            fixed32_ingress=ingress,
        )
    )
    session = requests.Session()
    try:
        for _ in range(2):
            response = session.post(
                f"{proxy_url}/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": f"Bearer {task_bearer}"},
                timeout=10,
            )
            assert response.status_code == 200
    finally:
        session.close()
        _stop = proxy_fixtures._stop_server
        _stop(proxy_server, proxy_thread)
        _stop(upstream_server, upstream_thread)
    ingress.finalize(proxy_fixtures._finalize_payload())
    ingress.ledger.close()

    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    completions = [row for row in rows if row["event"] == "logical_complete"]
    assert [row["prompt_tokens"] for row in completions] == [144, 443]
    assert [row["completion_tokens"] for row in completions] == [8, 16]
