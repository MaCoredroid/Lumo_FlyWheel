from __future__ import annotations

from pathlib import Path

from gateway.reducer import build_replay, render_replay


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "visible_same_name_out_of_order.jsonl"
THREE_CALL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "hidden_three_same_name.jsonl"
ORDINAL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "hidden_ordinal_trap.jsonl"
EXPECTED_RENDER = 'assistant: Need weather for SF and NYC before I answer.\ntool_call[call-weather-sf] weather_lookup {"city":"San Francisco"}\ntool_result[call-weather-sf] weather_lookup => San Francisco is 58F and foggy.\ntool_call[call-weather-nyc] weather_lookup {"city":"New York"}\ntool_result[call-weather-nyc] weather_lookup => New York is 71F and sunny.'


def test_build_replay_keeps_both_same_name_tool_invocations() -> None:
    transcript = build_replay(FIXTURE)
    tool_calls = [item for item in transcript if item["type"] == "tool_call"]
    tool_results = [item for item in transcript if item["type"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-weather-sf", "call-weather-nyc"]
    assert [item["call_id"] for item in tool_results] == ["call-weather-sf", "call-weather-nyc"]


def test_render_replay_matches_expected_summary() -> None:
    assert render_replay(FIXTURE) == EXPECTED_RENDER


def test_build_replay_keeps_all_same_name_calls_distinct() -> None:
    transcript = build_replay(THREE_CALL_FIXTURE)
    tool_calls = [item for item in transcript if item["type"] == "tool_call"]
    tool_results = [item for item in transcript if item["type"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == [
        "call-quote-aapl",
        "call-quote-msft",
        "call-quote-amzn",
    ]
    assert [item["call_id"] for item in tool_results] == [
        "call-quote-aapl",
        "call-quote-msft",
        "call-quote-amzn",
    ]


def test_build_replay_uses_stream_call_ids_without_synthesizing_ordinals() -> None:
    transcript = build_replay(ORDINAL_FIXTURE)
    tool_calls = [item for item in transcript if item["type"] == "tool_call"]

    assert [item["call_id"] for item in tool_calls] == ["call-17", "call-03"]
