from __future__ import annotations

from pathlib import Path

from gateway.adapter import normalize_events


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "visible_same_name_out_of_order.jsonl"
ORDINAL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "hidden_ordinal_trap.jsonl"


def test_normalize_events_keeps_call_ids_for_repeated_same_name_calls() -> None:
    items = normalize_events(FIXTURE)
    tool_calls = [item for item in items if item["kind"] == "tool_call"]
    tool_results = [item for item in items if item["kind"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-weather-sf", "call-weather-nyc"]
    assert [item["call_id"] for item in tool_results] == ["call-weather-nyc", "call-weather-sf"]
    assert [item["tool_name"] for item in tool_calls] == ["weather_lookup", "weather_lookup"]


def test_normalize_events_preserves_stream_call_ids_without_synthesizing_ordinals() -> None:
    items = normalize_events(ORDINAL_FIXTURE)
    tool_calls = [item for item in items if item["kind"] == "tool_call"]
    tool_results = [item for item in items if item["kind"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-17", "call-03"]
    assert [item["call_id"] for item in tool_results] == ["call-03", "call-17"]
