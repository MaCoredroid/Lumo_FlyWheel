"""Tests for the vLLM-side harness oracle skeleton.

Round-trips every field the proxy can emit through
``parse_oracle_header`` -> ``HarnessOracleSnapshot`` -> ``encode_oracle_header``,
plus negative cases (missing header, malformed JSON, unknown fields)
to verify the drafter-coordinator's "absence is silence" contract holds.
"""

from __future__ import annotations

import json

from lumo_flywheel_serving.inference_proxy import (
    encode_oracle_snapshot_header,
    synthesize_oracle_snapshot,
)
from lumo_flywheel_serving.vllm_harness_oracle import (
    DEFAULT_SUFFIX_TREE_CAP_MB,
    ORACLE_SCHEMA,
    HarnessOracleSnapshot,
    encode_oracle_header,
    parse_oracle_header,
)


def test_parse_oracle_header_none_returns_empty_snapshot() -> None:
    snap = parse_oracle_header(None)
    assert snap.is_empty
    assert snap.session_id is None
    assert snap.suffix_tree_cap_mb == DEFAULT_SUFFIX_TREE_CAP_MB


def test_parse_oracle_header_malformed_json_returns_empty_snapshot() -> None:
    snap = parse_oracle_header("not-json")
    assert snap.is_empty


def test_parse_oracle_header_non_dict_payload_returns_empty_snapshot() -> None:
    snap = parse_oracle_header(json.dumps([1, 2, 3]))
    assert snap.is_empty


def test_parse_oracle_header_round_trips_proxy_synthesis() -> None:
    payload = {
        "model": "qwen3.5-27b",
        "input": [{"role": "user", "content": "fix it"}],
        "tools": [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {
                "type": "function",
                "name": "apply_patch",
                "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}},
            },
        ],
        "tool_choice": {"type": "function", "function": {"name": "apply_patch"}},
    }
    proxy_snap = synthesize_oracle_snapshot(payload)
    header = encode_oracle_snapshot_header(proxy_snap)
    parsed = parse_oracle_header(header)

    assert parsed.session_id == proxy_snap["session_id"]
    assert parsed.turn_index == 0
    assert parsed.is_session_open is True
    assert parsed.dialect == "codex"
    assert [s["name"] for s in parsed.tool_schemas] == ["shell", "apply_patch"]
    assert parsed.expected_tool_call == {
        "name": "apply_patch",
        "schema": {"type": "object", "properties": {"patch": {"type": "string"}}},
    }
    assert parsed.suffix_tree_cap_mb == 100
    assert not parsed.is_empty


def test_parse_oracle_header_tolerates_unknown_future_fields() -> None:
    raw = json.dumps(
        {
            "schema": "lumo.harness_oracle_snapshot.v9_unreleased",
            "session_id": "sess_abc",
            "turn_index": 4,
            "future_field_added_later": {"important": True},
        }
    )
    snap = parse_oracle_header(raw)
    assert snap.session_id == "sess_abc"
    assert snap.turn_index == 4


def test_parse_oracle_header_rejects_wrong_typed_known_fields() -> None:
    snap = parse_oracle_header(
        json.dumps(
            {
                "session_id": 123,  # wrong type
                "turn_index": "five",  # wrong type
                "tool_schemas": "should-be-a-list",
                "expected_tool_call": [1, 2],
                "is_session_open": "true-string",
            }
        )
    )
    assert snap.session_id is None
    assert snap.turn_index is None
    assert snap.tool_schemas == []
    assert snap.expected_tool_call is None
    assert snap.is_session_open is True


def test_encode_oracle_header_round_trips_a_full_snapshot() -> None:
    full = HarnessOracleSnapshot(
        session_id="sess_zzz",
        turn_index=7,
        is_session_open=False,
        is_session_close=True,
        dialect="codex",
        tool_schemas=[{"name": "shell", "parameters": {"type": "object"}}],
        expected_tool_call={"name": "apply_patch", "schema": {}},
        primed_texts=[{"text": "hello", "source_tag": "file:foo.py", "ttl_turns": 32, "max_chars": 4096}],
        plan_fingerprint={"structure_tokens": [1, 2, 3], "first_emission_turn": 2, "emission_count": 4},
        suffix_tree_cap_mb=200,
    )
    header = encode_oracle_header(full)
    again = parse_oracle_header(header)
    assert again == full


def test_encode_oracle_header_omits_empty_optional_fields() -> None:
    minimal = HarnessOracleSnapshot(session_id="sess_x", turn_index=0, is_session_open=True)
    header = encode_oracle_header(minimal)
    decoded = json.loads(header)
    assert decoded["session_id"] == "sess_x"
    assert "tool_schemas" not in decoded
    assert "expected_tool_call" not in decoded
    assert "primed_texts" not in decoded
    assert "plan_fingerprint" not in decoded


def test_is_empty_short_circuit_is_default_state() -> None:
    assert HarnessOracleSnapshot().is_empty
    assert not HarnessOracleSnapshot(session_id="sess_a").is_empty
    assert not HarnessOracleSnapshot(turn_index=0).is_empty
    assert not HarnessOracleSnapshot(is_session_open=True).is_empty


def test_module_can_load_in_isolation() -> None:
    """Ensure the module would survive being dropped into a vLLM
    site-packages tree by the prelaunch hook — i.e., no relative
    imports, no implicit dependencies on the rest of
    ``lumo_flywheel_serving``."""

    import importlib

    spec = importlib.util.find_spec("lumo_flywheel_serving.vllm_harness_oracle")
    assert spec is not None
    src = spec.loader.get_source(spec.name)
    assert "from .." not in src and "from . " not in src
    assert "from lumo_flywheel_serving" not in src
    assert src.startswith('"""Harness-oracle skeleton')
    assert "ORACLE_SCHEMA" in src
    assert "ORACLE_SCHEMA = " in src and ORACLE_SCHEMA in src
