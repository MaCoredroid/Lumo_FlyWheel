"""Tests for the T3 schema-aware drafter decision core."""

from __future__ import annotations

import pytest

from lumo_flywheel_serving.schema_aware_drafter import (
    CODEX_ARGS_OPEN,
    CODEX_NAME_OPEN,
    CODEX_TOOL_CALL_CLOSE,
    CODEX_TOOL_CALL_OPEN,
    DraftProposal,
    propose,
)
from lumo_flywheel_serving.vllm_harness_oracle import HarnessOracleSnapshot


def _codex_snap(tool_name: str, schema: dict | None) -> HarnessOracleSnapshot:
    return HarnessOracleSnapshot(
        session_id="sess_x",
        turn_index=0,
        dialect="codex",
        expected_tool_call={"name": tool_name, "schema": schema},
    )


def _openai_snap(tool_name: str, schema: dict | None) -> HarnessOracleSnapshot:
    return HarnessOracleSnapshot(
        session_id="sess_x",
        turn_index=0,
        dialect="openai",
        expected_tool_call={"name": tool_name, "schema": schema},
    )


def test_propose_returns_none_for_empty_snapshot() -> None:
    assert propose(HarnessOracleSnapshot(), "anything") is None


def test_propose_returns_none_when_no_expected_tool_call() -> None:
    snap = HarnessOracleSnapshot(session_id="sess_x", dialect="codex")
    assert propose(snap, CODEX_TOOL_CALL_OPEN + CODEX_NAME_OPEN) is None


def test_codex_anchor1_emits_name_and_args_open() -> None:
    snap = _codex_snap("apply_patch", {"type": "object"})
    proposal = propose(snap, "blah blah " + CODEX_TOOL_CALL_OPEN + CODEX_NAME_OPEN)
    assert isinstance(proposal, DraftProposal)
    assert proposal.text == "apply_patch</name><arguments>{"
    assert proposal.confidence == pytest.approx(1.0)
    assert proposal.reason.startswith("codex_anchor_1")


def test_codex_no_anchor_when_already_inside_completed_call() -> None:
    snap = _codex_snap("apply_patch", {"type": "object"})
    text = (
        CODEX_TOOL_CALL_OPEN + "<name>apply_patch</name><arguments>{}</arguments>"
        + CODEX_TOOL_CALL_CLOSE
    )
    assert propose(snap, text) is None


def test_codex_anchor2_emits_first_required_property() -> None:
    schema = {
        "type": "object",
        "required": ["patch", "stage"],
        "properties": {
            "patch": {"type": "string"},
            "stage": {"type": "boolean"},
        },
    }
    snap = _codex_snap("apply_patch", schema)
    text = CODEX_TOOL_CALL_OPEN + "<name>apply_patch</name>" + CODEX_ARGS_OPEN + "{"
    proposal = propose(snap, text)
    assert proposal is not None
    assert proposal.text == '"patch":'
    assert proposal.reason.startswith("codex_anchor_2")


def test_codex_anchor3_emits_string_open_quote_after_property_marker() -> None:
    schema = {
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }
    snap = _codex_snap("read_file", schema)
    text = (
        CODEX_TOOL_CALL_OPEN + "<name>read_file</name>" + CODEX_ARGS_OPEN + '{"path":'
    )
    proposal = propose(snap, text)
    assert proposal is not None
    assert proposal.text == '"'
    assert proposal.reason.startswith("codex_anchor_3")


def test_codex_anchor3_skips_when_quote_already_present() -> None:
    schema = {
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }
    snap = _codex_snap("read_file", schema)
    text = (
        CODEX_TOOL_CALL_OPEN + "<name>read_file</name>" + CODEX_ARGS_OPEN + '{"path":"'
    )
    assert propose(snap, text) is None


def test_codex_anchor2_skips_non_required_first_property_priority() -> None:
    """Required list controls priority — properties insertion order is
    fallback only."""

    schema = {
        "type": "object",
        "required": ["second"],
        "properties": {
            "first": {"type": "string"},
            "second": {"type": "string"},
        },
    }
    snap = _codex_snap("tool", schema)
    text = CODEX_TOOL_CALL_OPEN + "<name>tool</name>" + CODEX_ARGS_OPEN + "{"
    proposal = propose(snap, text)
    assert proposal is not None
    assert proposal.text == '"second":'


def test_codex_falls_back_to_properties_order_when_no_required_list() -> None:
    schema = {
        "type": "object",
        "properties": {"alpha": {"type": "integer"}, "beta": {"type": "string"}},
    }
    snap = _codex_snap("tool", schema)
    text = CODEX_TOOL_CALL_OPEN + "<name>tool</name>" + CODEX_ARGS_OPEN + "{"
    proposal = propose(snap, text)
    assert proposal is not None
    assert proposal.text == '"alpha":'


def test_codex_returns_none_when_schema_has_no_properties() -> None:
    snap = _codex_snap("tool", {"type": "object"})
    text = CODEX_TOOL_CALL_OPEN + "<name>tool</name>" + CODEX_ARGS_OPEN + "{"
    assert propose(snap, text) is None


def test_openai_anchor1_emits_name_and_args_open() -> None:
    snap = _openai_snap("apply_patch", {"type": "object"})
    proposal = propose(snap, '{"name":"')
    assert proposal is not None
    assert proposal.text == 'apply_patch","arguments":{'
    assert proposal.confidence == pytest.approx(1.0)


def test_openai_anchor2_emits_first_required_property() -> None:
    schema = {
        "type": "object",
        "required": ["patch"],
        "properties": {"patch": {"type": "string"}},
    }
    snap = _openai_snap("apply_patch", schema)
    proposal = propose(snap, '"arguments":{')
    assert proposal is not None
    assert proposal.text == '"patch":"'  # string-typed -> includes opening quote
    assert proposal.confidence == pytest.approx(0.9)


def test_openai_anchor2_no_quote_for_non_string_property() -> None:
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
    }
    snap = _openai_snap("counter", schema)
    proposal = propose(snap, '"arguments":{')
    assert proposal is not None
    assert proposal.text == '"count":'  # no opening quote


def test_propose_returns_none_for_unknown_dialect() -> None:
    snap = HarnessOracleSnapshot(
        session_id="sess_x",
        turn_index=0,
        dialect="some-future-dialect",
        expected_tool_call={"name": "shell", "schema": {"type": "object"}},
    )
    assert propose(snap, '<tool_call><name>') is None
