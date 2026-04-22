from __future__ import annotations

from pathlib import Path

from gateway.reducer import build_replay, render_replay


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "visible_same_name_out_of_order.jsonl"
EXPECTED_RENDER = 'assistant: Need weather for SF and NYC before I answer.\ntool_call[call-weather-sf] weather_lookup {"city":"San Francisco"}\ntool_result[call-weather-sf] weather_lookup => San Francisco is 58F and foggy.\ntool_call[call-weather-nyc] weather_lookup {"city":"New York"}\ntool_result[call-weather-nyc] weather_lookup => New York is 71F and sunny.'


def test_build_replay_keeps_both_same_name_tool_invocations() -> None:
    transcript = build_replay(FIXTURE)
    tool_calls = [item for item in transcript if item["type"] == "tool_call"]
    tool_results = [item for item in transcript if item["type"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-weather-sf", "call-weather-nyc"]
    assert [item["call_id"] for item in tool_results] == ["call-weather-sf", "call-weather-nyc"]


def test_render_replay_matches_expected_summary() -> None:
    assert render_replay(FIXTURE) == EXPECTED_RENDER



def test_render_cli_summary_stays_stable() -> None:
    from gateway.reducer import render_cli_summary

    assert render_cli_summary(FIXTURE) == EXPECTED_RENDER
