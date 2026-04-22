from __future__ import annotations

from pathlib import Path

from gateway.adapter import normalize_events


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "visible_same_name_out_of_order.jsonl"


def test_normalize_events_keeps_call_ids_for_repeated_same_name_calls() -> None:
    items = normalize_events(FIXTURE)
    tool_calls = [item for item in items if item["kind"] == "tool_call"]
    tool_results = [item for item in items if item["kind"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-weather-sf", "call-weather-nyc"]
    assert [item["call_id"] for item in tool_results] == ["call-weather-nyc", "call-weather-sf"]
    assert [item["tool_name"] for item in tool_calls] == ["weather_lookup", "weather_lookup"]



def test_normalize_events_retains_distinct_result_rows() -> None:
    items = normalize_events(FIXTURE)
    tool_results = [item for item in items if item["kind"] == "tool_result"]
    assert len(tool_results) == 2
    assert tool_results[0]["output"] != tool_results[1]["output"]
