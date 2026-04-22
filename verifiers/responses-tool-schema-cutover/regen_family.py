#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "responses-tool-schema-cutover"
VERIFIER_ROOT = REPO / "verifiers" / "responses-tool-schema-cutover"
VERIFIER_DATA_ROOT = REPO / "verifier_data" / "responses-tool-schema-cutover"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"

BASE_REQUIRED_DOC_PHRASES = [
    "response.output_item.added",
    "response.completed",
    "call_id",
    "tool result",
]
BASE_REQUIRED_CONFIG_TERMS = [
    'transport = "responses"',
    'tool_result_join_key = "call_id"',
    'replay_mode = "response_output_items"',
]
BASE_FORBIDDEN_CONFIG_TERMS = [
    "chat_completions",
    "tool_result_join_key = \"tool_name\"",
    "legacy_assistant_blob",
]

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "label": "clean baseline",
        "agent_note": "Visible baseline: two same-name tool calls, out-of-order tool results, no extra distractors.",
        "extra_files": {},
        "hidden_cases": [
            {
                "id": "same-name-hidden",
                "fixture": "fixtures/responses_stream/hidden_same_name.jsonl",
                "expected_render": "\n".join(
                    [
                        "assistant: Need quote and chart snapshots before I answer.",
                        'tool_call[call-quote-aapl] market_lookup {"field":"quote","ticker":"AAPL"}',
                        "tool_result[call-quote-aapl] market_lookup => AAPL 192.11",
                        'tool_call[call-chart-aapl] market_lookup {"field":"chart","ticker":"AAPL"}',
                        "tool_result[call-chart-aapl] market_lookup => 5d trend up",
                    ]
                ),
                "expected_call_ids": [
                    "call-quote-aapl",
                    "call-chart-aapl",
                ],
            }
        ],
    },
    {
        "id": "v2-noisy-distractor",
        "label": "noisy distractor",
        "agent_note": "A stale legacy fixture and operator note are present but must not drive the repair.",
        "extra_files": {
            "notes/stale_legacy_warning.md": "\n".join(
                [
                    "# Legacy warning",
                    "",
                    "Ignore the archived fixture under `fixtures/legacy_archive/` for live replay logic.",
                    "It predates Responses `call_id` routing and exists only for historical debugging.",
                    "",
                ]
            ),
            "fixtures/legacy_archive/chat_compat_snapshot.jsonl": json.dumps(
                {
                    "type": "legacy.chat_completion",
                    "message": {
                        "role": "assistant",
                        "function_call": {
                            "name": "market_lookup",
                            "arguments": "{\"ticker\":\"AAPL\"}",
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        },
        "hidden_cases": [
            {
                "id": "three-calls-same-name",
                "fixture": "fixtures/responses_stream/hidden_three_same_name.jsonl",
                "expected_render": "\n".join(
                    [
                        "assistant: Compare three tool responses without collapsing duplicate tool names.",
                        'tool_call[call-quote-aapl] market_lookup {"field":"quote","ticker":"AAPL"}',
                        "tool_result[call-quote-aapl] market_lookup => AAPL 192.11",
                        'tool_call[call-quote-msft] market_lookup {"field":"quote","ticker":"MSFT"}',
                        "tool_result[call-quote-msft] market_lookup => MSFT 418.33",
                        'tool_call[call-quote-amzn] market_lookup {"field":"quote","ticker":"AMZN"}',
                        "tool_result[call-quote-amzn] market_lookup => AMZN 184.02",
                    ]
                ),
                "expected_call_ids": [
                    "call-quote-aapl",
                    "call-quote-msft",
                    "call-quote-amzn",
                ],
            }
        ],
    },
    {
        "id": "v3-dirty-state",
        "label": "dirty state",
        "agent_note": "There is an abandoned ordinal-based scratch patch in-tree; finishing it is the wrong move.",
        "extra_files": {
            "gateway/_scratch_patch.py": "\n".join(
                [
                    '"""Abandoned attempt. Do not trust this ordinal-based join shim."""',
                    "",
                    "def attach_results_by_position(calls, results):",
                    "    pairs = []",
                    "    for index, call in enumerate(calls):",
                    "        result = results[index] if index < len(results) else None",
                    "        pairs.append((call.get(\"tool_name\"), result))",
                    "    return pairs",
                    "",
                ]
            ),
        },
        "hidden_cases": [
            {
                "id": "ordinal-trap",
                "fixture": "fixtures/responses_stream/hidden_ordinal_trap.jsonl",
                "expected_render": "\n".join(
                    [
                        "assistant: Preserve the original call ids from the stream, not synthetic ordinals.",
                        'tool_call[call-17] market_lookup {"field":"quote","ticker":"NVDA"}',
                        "tool_result[call-17] market_lookup => NVDA 914.02",
                        'tool_call[call-03] market_lookup {"field":"quote","ticker":"AMD"}',
                        "tool_result[call-03] market_lookup => AMD 182.44",
                    ]
                ),
                "expected_call_ids": ["call-17", "call-03"],
            }
        ],
    },
    {
        "id": "v4-multi-corpus-objective",
        "label": "multi corpus objective drift",
        "agent_note": "Release context makes operator-facing contract alignment mandatory, not optional cleanup.",
        "extra_files": {
            "release_context/cli_stability.md": "\n".join(
                [
                    "# CLI stability note",
                    "",
                    "The public summary format must remain `tool_call[...]` / `tool_result[...]` stable",
                    "during the Responses cutover so operator snapshots remain diff-friendly.",
                    "",
                ]
            ),
        },
        "hidden_cases": [
            {
                "id": "stable-summary",
                "fixture": "fixtures/responses_stream/hidden_stable_summary.jsonl",
                "expected_render": "\n".join(
                    [
                        "assistant: Keep the summary stable while repairing the join semantics.",
                        'tool_call[call-orders-page-1] orders_lookup {"page":1}',
                        "tool_result[call-orders-page-1] orders_lookup => 12 orders",
                        'tool_call[call-orders-page-2] orders_lookup {"page":2}',
                        "tool_result[call-orders-page-2] orders_lookup => 9 orders",
                    ]
                ),
                "expected_call_ids": [
                    "call-orders-page-1",
                    "call-orders-page-2",
                ],
            }
        ],
    },
    {
        "id": "v5-recovery-in-thread",
        "label": "recovery in thread",
        "agent_note": "Incident context shows an earlier chronology-blind fix regressed replay snapshots after rollback.",
        "extra_files": {
            "release_context/cli_stability.md": "\n".join(
                [
                    "# CLI stability note",
                    "",
                    "The public summary format must remain `tool_call[...]` / `tool_result[...]` stable",
                    "during the Responses cutover so operator snapshots remain diff-friendly.",
                    "",
                ]
            ),
            "incident_context/inc_204_ordinal_regression.md": "\n".join(
                [
                    "# INC-204",
                    "",
                    "An earlier repair grouped tool results by visible call ordinal instead of `call_id`.",
                    "The patch passed a narrow fixture and then rendered the wrong transcript when results",
                    "arrived out of order for repeated tool invocations.",
                    "",
                ]
            ),
        },
        "hidden_cases": [
            {
                "id": "incident-regression",
                "fixture": "fixtures/responses_stream/hidden_incident_regression.jsonl",
                "expected_render": "\n".join(
                    [
                        "assistant: Rebuild the replay after the chronology-blind rollback.",
                        'tool_call[call-route-primary] route_lookup {"route":"primary"}',
                        "tool_result[call-route-primary] route_lookup => primary ready",
                        'tool_call[call-route-fallback] route_lookup {"route":"fallback"}',
                        "tool_result[call-route-fallback] route_lookup => fallback warming",
                    ]
                ),
                "expected_call_ids": [
                    "call-route-primary",
                    "call-route-fallback",
                ],
            }
        ],
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{rel_path}\n".encode())
        else:
            h.update(f"F:{rel_path}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def jsonl(events: list[dict]) -> str:
    return "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)


def event_message(sequence: int, text: str) -> dict:
    return {
        "sequence": sequence,
        "type": "response.output_item.added",
        "item": {
            "type": "message",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def event_tool_call(sequence: int, call_id: str, tool_name: str, arguments: dict) -> dict:
    return {
        "sequence": sequence,
        "type": "response.output_item.added",
        "item": {
            "type": "tool_call",
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
        },
    }


def event_tool_result(sequence: int, call_id: str, tool_name: str, output: str) -> dict:
    return {
        "sequence": sequence,
        "type": "response.output_item.added",
        "item": {
            "type": "tool_result",
            "call_id": call_id,
            "tool_name": tool_name,
            "output": output,
        },
    }


def completed(sequence: int) -> dict:
    return {
        "sequence": sequence,
        "type": "response.completed",
        "response": {"id": "resp_demo"},
    }


VISIBLE_FIXTURE = jsonl(
    [
        event_message(1, "Need weather for SF and NYC before I answer."),
        event_tool_call(2, "call-weather-sf", "weather_lookup", {"city": "San Francisco"}),
        event_tool_call(3, "call-weather-nyc", "weather_lookup", {"city": "New York"}),
        event_tool_result(4, "call-weather-nyc", "weather_lookup", "New York is 71F and sunny."),
        event_tool_result(5, "call-weather-sf", "weather_lookup", "San Francisco is 58F and foggy."),
        completed(6),
    ]
)

VISIBLE_RENDER = "\n".join(
    [
        "assistant: Need weather for SF and NYC before I answer.",
        'tool_call[call-weather-sf] weather_lookup {"city":"San Francisco"}',
        "tool_result[call-weather-sf] weather_lookup => San Francisco is 58F and foggy.",
        'tool_call[call-weather-nyc] weather_lookup {"city":"New York"}',
        "tool_result[call-weather-nyc] weather_lookup => New York is 71F and sunny.",
    ]
)

HIDDEN_FIXTURES = {
    "hidden_same_name.jsonl": jsonl(
        [
            event_message(1, "Need quote and chart snapshots before I answer."),
            event_tool_call(2, "call-quote-aapl", "market_lookup", {"field": "quote", "ticker": "AAPL"}),
            event_tool_call(3, "call-chart-aapl", "market_lookup", {"field": "chart", "ticker": "AAPL"}),
            event_tool_result(4, "call-chart-aapl", "market_lookup", "5d trend up"),
            event_tool_result(5, "call-quote-aapl", "market_lookup", "AAPL 192.11"),
            completed(6),
        ]
    ),
    "hidden_three_same_name.jsonl": jsonl(
        [
            event_message(1, "Compare three tool responses without collapsing duplicate tool names."),
            event_tool_call(2, "call-quote-aapl", "market_lookup", {"field": "quote", "ticker": "AAPL"}),
            event_tool_call(3, "call-quote-msft", "market_lookup", {"field": "quote", "ticker": "MSFT"}),
            event_tool_call(4, "call-quote-amzn", "market_lookup", {"field": "quote", "ticker": "AMZN"}),
            event_tool_result(5, "call-quote-msft", "market_lookup", "MSFT 418.33"),
            event_tool_result(6, "call-quote-amzn", "market_lookup", "AMZN 184.02"),
            event_tool_result(7, "call-quote-aapl", "market_lookup", "AAPL 192.11"),
            completed(8),
        ]
    ),
    "hidden_ordinal_trap.jsonl": jsonl(
        [
            event_message(1, "Preserve the original call ids from the stream, not synthetic ordinals."),
            event_tool_call(2, "call-17", "market_lookup", {"field": "quote", "ticker": "NVDA"}),
            event_tool_call(3, "call-03", "market_lookup", {"field": "quote", "ticker": "AMD"}),
            event_tool_result(4, "call-03", "market_lookup", "AMD 182.44"),
            event_tool_result(5, "call-17", "market_lookup", "NVDA 914.02"),
            completed(6),
        ]
    ),
    "hidden_stable_summary.jsonl": jsonl(
        [
            event_message(1, "Keep the summary stable while repairing the join semantics."),
            event_tool_call(2, "call-orders-page-1", "orders_lookup", {"page": 1}),
            event_tool_call(3, "call-orders-page-2", "orders_lookup", {"page": 2}),
            event_tool_result(4, "call-orders-page-2", "orders_lookup", "9 orders"),
            event_tool_result(5, "call-orders-page-1", "orders_lookup", "12 orders"),
            completed(6),
        ]
    ),
    "hidden_incident_regression.jsonl": jsonl(
        [
            event_message(1, "Rebuild the replay after the chronology-blind rollback."),
            event_tool_call(2, "call-route-primary", "route_lookup", {"route": "primary"}),
            event_tool_call(3, "call-route-fallback", "route_lookup", {"route": "fallback"}),
            event_tool_result(4, "call-route-fallback", "route_lookup", "fallback warming"),
            event_tool_result(5, "call-route-primary", "route_lookup", "primary ready"),
            completed(6),
        ]
    ),
}


BROKEN_ADAPTER = """from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def normalize_events(path: str | Path) -> list[dict]:
    events = load_events(path)
    assistant_chunks: list[str] = []
    calls_by_tool: dict[str, dict] = {}
    results_by_tool: dict[str, str] = {}

    for sequence, event in enumerate(events, start=1):
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    assistant_chunks.append(part.get("text", ""))
        elif item_type == "tool_call":
            tool_name = item.get("tool_name")
            calls_by_tool[tool_name] = {
                "kind": "tool_call",
                "sequence": sequence,
                "tool_name": tool_name,
                "arguments": item.get("arguments", ""),
            }
        elif item_type == "tool_result":
            results_by_tool[item.get("tool_name")] = item.get("output", "")

    normalized: list[dict] = []
    if assistant_chunks:
        normalized.append(
            {
                "kind": "assistant_text",
                "sequence": 0,
                "text": " ".join(chunk for chunk in assistant_chunks if chunk).strip(),
            }
        )

    for tool_name, call in calls_by_tool.items():
        normalized.append(call)
        if tool_name in results_by_tool:
            normalized.append(
                {
                    "kind": "tool_result",
                    "sequence": call["sequence"] + 100,
                    "tool_name": tool_name,
                    "output": results_by_tool[tool_name],
                }
            )
    return normalized
"""


FIXED_ADAPTER = """from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def normalize_events(path: str | Path) -> list[dict]:
    normalized: list[dict] = []
    for fallback_sequence, event in enumerate(load_events(path), start=1):
        event_type = event.get("type")
        sequence = int(event.get("sequence", fallback_sequence))
        if event_type != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            parts = [
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            ]
            text = " ".join(part for part in parts if part).strip()
            if text:
                normalized.append(
                    {
                        "kind": "assistant_text",
                        "sequence": sequence,
                        "text": text,
                    }
                )
        elif item_type in {"tool_call", "function_call"}:
            normalized.append(
                {
                    "kind": "tool_call",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name") or item.get("name"),
                    "arguments": item.get("arguments", ""),
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            normalized.append(
                {
                    "kind": "tool_result",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name") or item.get("name"),
                    "output": item.get("output", ""),
                }
            )
    normalized.sort(key=lambda item: (item["sequence"], item["kind"]))
    return normalized
"""


BROKEN_REDUCER = """from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    pending_by_tool: dict[str, str] = {}
    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_result":
            pending_by_tool[item.get("tool_name")] = item.get("output", "")
            continue
        if kind == "tool_call":
            tool_name = item.get("tool_name")
            if any(row.get("tool_name") == tool_name for row in transcript):
                continue
            transcript.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
            if tool_name in pending_by_tool:
                transcript.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "output": pending_by_tool[tool_name],
                        "call_id": item.get("call_id"),
                    }
                )
    return transcript


def render_replay(path: str) -> str:
    lines: list[str] = []
    for item in build_replay(path):
        if item["type"] == "assistant":
            lines.append(f"assistant: {item['text']}")
        elif item["type"] == "tool_call":
            lines.append(f"tool_call[{item.get('call_id')}] {item['tool_name']} {item['arguments']}")
        elif item["type"] == "tool_result":
            lines.append(f"tool_result[{item.get('call_id')}] {item['tool_name']} => {item['output']}")
    return "\\n".join(lines)


def render_cli_summary(path: str) -> str:
    return render_replay(path)
"""


FIXED_REDUCER = """from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    call_positions: dict[str, int] = {}
    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_call":
            transcript.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
            call_positions[item.get("call_id")] = len(transcript) - 1
            continue
        if kind != "tool_result":
            continue

        call_id = item.get("call_id")
        tool_result = {
            "type": "tool_result",
            "tool_name": item.get("tool_name"),
            "output": item.get("output", ""),
            "call_id": call_id,
        }
        if call_id not in call_positions:
            transcript.append(tool_result)
            continue

        insert_at = call_positions[call_id] + 1
        while insert_at < len(transcript):
            row = transcript[insert_at]
            if row.get("type") != "tool_result" or row.get("call_id") != call_id:
                break
            insert_at += 1
        transcript.insert(insert_at, tool_result)
        for seen_call_id, position in list(call_positions.items()):
            if position >= insert_at:
                call_positions[seen_call_id] = position + 1
    return transcript


def render_replay(path: str) -> str:
    lines: list[str] = []
    for item in build_replay(path):
        if item["type"] == "assistant":
            lines.append(f"assistant: {item['text']}")
        elif item["type"] == "tool_call":
            lines.append(f"tool_call[{item['call_id']}] {item['tool_name']} {item['arguments']}")
        elif item["type"] == "tool_result":
            lines.append(f"tool_result[{item['call_id']}] {item['tool_name']} => {item['output']}")
    return "\\n".join(lines)


def render_cli_summary(path: str) -> str:
    return render_replay(path)
"""


BROKEN_CONFIG = """[runtime]
transport = "chat_completions"
tool_result_join_key = "tool_name"
replay_mode = "legacy_assistant_blob"
"""


FIXED_CONFIG = """[runtime]
transport = "responses"
tool_result_join_key = "call_id"
replay_mode = "response_output_items"
"""


BROKEN_DOC = """# Responses Cutover

The local adapter still mirrors the legacy Chat Completions contract.

- Inspect `function_call.arguments` to recover tool arguments.
- Treat repeated tool invocations as one logical tool step in snapshots.
- `response.completed` is only a footer marker; no per-item replay contract exists.
"""


FIXED_DOC = """# Responses Cutover

The gateway now replays Responses-native event items end to end.

- Read each `response.output_item.added` record as its own replay input.
- Preserve every repeated tool invocation as a distinct call keyed by `call_id`.
- Join tool results back to tool calls by `call_id`, not by tool name.
- `response.completed` closes the stream but does not replace the per-item replay contract.
"""


BASELINE_TEST_ADAPTER = f"""from __future__ import annotations

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
"""


ORACLE_TEST_ADAPTER = BASELINE_TEST_ADAPTER + """


def test_normalize_events_retains_distinct_result_rows() -> None:
    items = normalize_events(FIXTURE)
    tool_results = [item for item in items if item["kind"] == "tool_result"]
    assert len(tool_results) == 2
    assert tool_results[0]["output"] != tool_results[1]["output"]
"""


BASELINE_TEST_REDUCER = f"""from __future__ import annotations

from pathlib import Path

from gateway.reducer import build_replay, render_replay


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "responses_stream" / "visible_same_name_out_of_order.jsonl"
EXPECTED_RENDER = {VISIBLE_RENDER!r}


def test_build_replay_keeps_both_same_name_tool_invocations() -> None:
    transcript = build_replay(FIXTURE)
    tool_calls = [item for item in transcript if item["type"] == "tool_call"]
    tool_results = [item for item in transcript if item["type"] == "tool_result"]

    assert [item["call_id"] for item in tool_calls] == ["call-weather-sf", "call-weather-nyc"]
    assert [item["call_id"] for item in tool_results] == ["call-weather-sf", "call-weather-nyc"]


def test_render_replay_matches_expected_summary() -> None:
    assert render_replay(FIXTURE) == EXPECTED_RENDER
"""


ORACLE_TEST_REDUCER = BASELINE_TEST_REDUCER + """


def test_render_cli_summary_stays_stable() -> None:
    from gateway.reducer import render_cli_summary

    assert render_cli_summary(FIXTURE) == EXPECTED_RENDER
"""


def workspace_common(agent_note: str, variant_id: str) -> dict[str, str]:
    agents = "\n".join(
        [
            "# Responses Replay Repair",
            "",
            f"Variant: `{variant_id}`",
            "",
            agent_note,
            "",
            "Required solver behavior:",
            "- Inspect at least one replay fixture before patching.",
            "- Repair both `gateway/adapter.py` and `gateway/reducer.py` if needed.",
            "- Keep repeated same-name tool calls distinct by `call_id` all the way through render output.",
            "- Keep the public CLI summary format stable for successful runs.",
            "- Update `codex/config.toml` and `docs/migrations/responses-cutover.md` to the Responses-native contract.",
            "",
            "Anti-shortcuts:",
            "- Do not dedupe repeated tool calls by tool name.",
            "- Do not synthesize `call_id` from fixture position or visible ordinals.",
            "- Do not delete fixtures or relax tests to get green.",
            "",
        ]
    )
    return {
        ".scenario_variant": variant_id + "\n",
        "AGENTS.md": agents + "\n",
        "Dockerfile": "FROM python:3.11-slim\nWORKDIR /workspace\n",
        "gateway/__init__.py": "",
        "gateway/adapter.py": BROKEN_ADAPTER,
        "gateway/reducer.py": BROKEN_REDUCER,
        "codex/config.toml": BROKEN_CONFIG,
        "docs/migrations/responses-cutover.md": BROKEN_DOC,
        "tests/test_adapter_contract.py": BASELINE_TEST_ADAPTER,
        "tests/test_replay_render.py": BASELINE_TEST_REDUCER,
        "fixtures/responses_stream/visible_same_name_out_of_order.jsonl": VISIBLE_FIXTURE,
    }


def oracle_files() -> dict[str, str]:
    return {
        "gateway/adapter.py": FIXED_ADAPTER,
        "gateway/reducer.py": FIXED_REDUCER,
        "codex/config.toml": FIXED_CONFIG,
        "docs/migrations/responses-cutover.md": FIXED_DOC,
        "tests/test_adapter_contract.py": ORACLE_TEST_ADAPTER,
        "tests/test_replay_render.py": ORACLE_TEST_REDUCER,
    }


def hidden_test_text() -> str:
    return """from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AGENT_WS = Path(os.environ["AGENT_WS"]).resolve()
VERIFIER_DATA = Path(os.environ["VERIFIER_DATA"]).resolve()
VARIANT_ID = os.environ["VARIANT_ID"]
sys.path.insert(0, str(AGENT_WS))

from gateway.adapter import normalize_events  # noqa: E402
from gateway.reducer import render_replay  # noqa: E402

GOLD = json.loads((VERIFIER_DATA / VARIANT_ID / "gold_responses.json").read_text())


def test_hidden_replay_cases_match_gold() -> None:
    for case in GOLD["hidden_cases"]:
        fixture = AGENT_WS / case["fixture"]
        assert render_replay(fixture) == case["expected_render"]
        items = normalize_events(fixture)
        tool_calls = [item for item in items if item["kind"] == "tool_call"]
        assert [item["call_id"] for item in tool_calls] == case["expected_call_ids"]


def test_config_and_docs_match_cutover_contract() -> None:
    config_text = (AGENT_WS / "codex" / "config.toml").read_text()
    docs_text = (AGENT_WS / "docs" / "migrations" / "responses-cutover.md").read_text().lower()
    for needle in GOLD["required_config_terms"]:
        assert needle in config_text
    for needle in GOLD["forbidden_config_terms"]:
        assert needle not in config_text
    for needle in GOLD["required_doc_phrases"]:
        assert needle.lower() in docs_text
"""


def milestone_script(slot_id: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "from __future__ import annotations",
            "",
            "import json",
            "import os",
            "import sys",
            "",
            f"SLOT = {slot_id!r}",
            'result_file = os.environ.get("RESULT_FILE")',
            "if not result_file:",
            "    sys.exit(2)",
            "payload = json.loads(open(result_file).read())",
            'sys.exit(0 if payload.get("milestones", {}).get(SLOT) else 1)',
            "",
        ]
    )


def collect_manifest(root: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload[path.relative_to(root).as_posix()] = sha256_file(path)
    return payload


def workspace_lock_entry(root: Path) -> dict[str, object]:
    file_hashes = collect_manifest(root)
    tree_digest = hashlib.sha256()
    for rel, digest in sorted(file_hashes.items()):
        tree_digest.update(f"{rel}:{digest}\n".encode())
    return {
        "file_hashes": file_hashes,
        "tree_sha256": tree_digest.hexdigest(),
    }


def build_variant(spec: dict[str, object]) -> None:
    variant_id = spec["id"]
    workspace = WORKSPACE_ROOT / variant_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    files = workspace_common(spec["agent_note"], variant_id)
    for rel, text in files.items():
        write_text(workspace / rel, text)
    for rel, text in spec["extra_files"].items():
        write_text(workspace / rel, text)
    for name, text in HIDDEN_FIXTURES.items():
        write_text(workspace / "fixtures" / "responses_stream" / name, text)

    oracle_root = VERIFIER_DATA_ROOT / variant_id / "oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    for rel, text in oracle_files().items():
        write_text(oracle_root / rel, text)

    gold = {
        "variant_id": variant_id,
        "allowed_write_paths": [
            "gateway/adapter.py",
            "gateway/reducer.py",
            "tests/test_adapter_contract.py",
            "tests/test_replay_render.py",
            "codex/config.toml",
            "docs/migrations/responses-cutover.md",
        ],
        "core_paths": [
            "gateway/adapter.py",
            "gateway/reducer.py",
        ],
        "visible_fixture": "fixtures/responses_stream/visible_same_name_out_of_order.jsonl",
        "visible_render": VISIBLE_RENDER,
        "visible_call_ids": ["call-weather-sf", "call-weather-nyc"],
        "hidden_cases": spec["hidden_cases"],
        "required_doc_phrases": BASE_REQUIRED_DOC_PHRASES,
        "required_config_terms": BASE_REQUIRED_CONFIG_TERMS,
        "forbidden_config_terms": BASE_FORBIDDEN_CONFIG_TERMS,
        "required_test_markers": [
            "call_id",
            "render_replay",
            "visible_same_name_out_of_order",
        ],
        "partial_progress_points": 10,
    }
    dump_json(VERIFIER_DATA_ROOT / variant_id / "gold_responses.json", gold)
    dump_json(
        VERIFIER_DATA_ROOT / variant_id / "workspace_manifest.json",
        {
            "file_hashes": collect_manifest(workspace),
            "readonly_tree_hashes": {
                "fixtures": tree_hash(workspace, "fixtures"),
                "release_context": tree_hash(workspace, "release_context"),
                "incident_context": tree_hash(workspace, "incident_context"),
                "notes": tree_hash(workspace, "notes"),
                "gateway/_scratch_patch.py": tree_hash(workspace, "gateway/_scratch_patch.py"),
            },
        },
    )
    write_text(
        VERIFIER_DATA_ROOT / variant_id / "hidden_tests" / "test_responses_properties.py",
        hidden_test_text(),
    )
    milestones = VERIFIER_DATA_ROOT / variant_id / "milestones"
    milestones.mkdir(parents=True, exist_ok=True)
    shared = VERIFIER_DATA_ROOT / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for slot in [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]:
        script_name = {
            "M1_localization": "m1_localization.py",
            "M2_primary_fix": "m2_primary_fix.py",
            "M3_invariants": "m3_invariants.py",
            "M4_functional": "m4_functional.py",
            "M5_e2e": "m5_e2e.py",
        }[slot]
        shared_file = shared / script_name
        if not shared_file.exists():
            write_text(shared_file, milestone_script(slot))
            shared_file.chmod(0o755)
        link_path = milestones / script_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(Path("..") / ".." / "_milestones_shared" / script_name)


def write_manifest_lock() -> None:
    payload = {
        "family_id": "responses-tool-schema-cutover",
        "variants": {},
    }
    for spec in VARIANTS:
        variant_id = spec["id"]
        payload["variants"][variant_id] = workspace_lock_entry(WORKSPACE_ROOT / variant_id)
    dump_json(FAMILY_ROOT / "manifest.lock.json", payload)


def main() -> int:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in VARIANTS:
        build_variant(spec)
    write_manifest_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
