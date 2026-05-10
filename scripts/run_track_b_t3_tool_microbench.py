#!/usr/bin/env python3
"""T3 schema-aware tool drafter — tool-call-inclusive microbench.

Drives 5 sessions × 3 turns at ``tool_choice="auto"`` with a
non-trivial set of tool schemas, asking the model to invoke a
specific tool each turn. T3 fires on the structural-prefix tokens
of the function-call XML emission (``<function=`` + the function
name + ``><parameter`` etc.), which the schema-aware drafter
prebuilds with high confidence.

Goal: isolate T3's acceptance contribution. The 5×3 text-only
microbench has T3 path-reachable but never exercised because the
synthetic prompts don't force tool emission. With this driver
asking the model to call ``read_file`` / ``apply_patch`` / etc.,
T3's structural draft tokens enter the speculation budget.

The driver expects 5 distinct ``oracle_session_id``s (one per
first-user-message anchor) so the tool emissions accumulate per
session, allowing T1 + T2 + T3 to all contribute without
cross-session contamination.

Capture: ``/tmp/lumo-r3-t3-tool-microbench.json``.
"""
from __future__ import annotations

import json
import time

import requests

URL = "http://127.0.0.1:8022/v1/responses"
METRICS_URL = "http://127.0.0.1:9950/metrics"
HEADERS = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file by path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path"},
                "max_lines": {"type": "integer", "description": "Max lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "apply_patch",
        "description": "Apply a unified-diff patch to the repo.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text"},
                "dry_run": {"type": "boolean", "description": "Dry-run only"},
            },
            "required": ["patch"],
        },
    },
    {
        "type": "function",
        "name": "shell",
        "description": "Run a shell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "integer"},
            },
            "required": ["cmd"],
        },
    },
]

SESSIONS = [
    ("S1", "I'm auditing the auth middleware. Use the tools to read files and propose patches as needed."),
    ("S2", "Investigate the rate limiter in api/limiter.py. Use tools as needed."),
    ("S3", "The migration in db/0042 needs review. Use tools to read and analyze."),
    ("S4", "Trace the request flow in src/router.go end-to-end. Use tools as needed."),
    ("S5", "The compiler/parser.rs is dropping comments. Find the bug using tools."),
]

TURN_PROMPTS = [
    "Read src/widget.py first.",
    "Now read src/cache.py to compare.",
    "Run a shell command: ls -la src/",
]


def fetch_metrics() -> dict:
    try:
        r = requests.get(
            METRICS_URL, headers={"Authorization": "Bearer EMPTY"}, timeout=10
        )
        if r.status_code != 200:
            return {}
        out: dict = {}
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                key, val = line.rsplit(" ", 1)
                base = key.split("{", 1)[0].strip()
                if "spec_decode" in base:
                    out[base] = out.get(base, 0.0) + float(val)
            except ValueError:
                continue
        return out
    except requests.RequestException:
        return {}


def build_payload(anchor: str, history: list[dict], next_user: str) -> dict:
    inputs: list[dict] = [{"role": "user", "content": anchor}]
    inputs.extend(history)
    inputs.append({"role": "user", "content": next_user})
    return {
        "model": "qwen3.5-27b",
        "input": inputs,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "max_output_tokens": 256,
        "temperature": 0,
    }


def extract_tool_calls(body: dict) -> list[dict]:
    calls: list[dict] = []
    for o in body.get("output", []):
        if o.get("type") == "function_call":
            calls.append(
                {
                    "name": o.get("name"),
                    "args": o.get("arguments"),
                    "call_id": o.get("call_id"),
                }
            )
    return calls


def main() -> None:
    pre = fetch_metrics()
    bench_pre_acc = pre.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    bench_pre_drf = pre.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    results: list[dict] = []
    for sess_label, anchor in SESSIONS:
        history: list[dict] = []
        for turn_idx, user_text in enumerate(TURN_PROMPTS):
            payload = build_payload(anchor, history, user_text)
            t0 = time.perf_counter()
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
            elapsed = time.perf_counter() - t0
            body = r.json() if r.status_code == 200 else {}
            tcalls = extract_tool_calls(body)
            m_now = fetch_metrics()
            acc_delta = m_now.get(
                "vllm:spec_decode_num_accepted_tokens_total", 0.0
            ) - pre.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            drf_delta = m_now.get(
                "vllm:spec_decode_num_draft_tokens_total", 0.0
            ) - pre.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
            results.append(
                {
                    "session": sess_label,
                    "turn_index": turn_idx,
                    "status": r.status_code,
                    "elapsed_s": round(elapsed, 3),
                    "id": body.get("id"),
                    "tool_calls_emitted": len(tcalls),
                    "tool_call_names": [c["name"] for c in tcalls],
                    "draft_accepted_delta": acc_delta,
                    "draft_total_delta": drf_delta,
                    "acceptance_rate": (
                        acc_delta / drf_delta if drf_delta > 0 else 0.0
                    ),
                }
            )
            print(
                f"  {sess_label} turn{turn_idx} status={r.status_code} "
                f"elapsed={elapsed:.2f}s tools={[c['name'] for c in tcalls]} "
                f"accept={acc_delta:.0f}/{drf_delta:.0f} "
                f"rate={(acc_delta / drf_delta * 100) if drf_delta > 0 else 0.0:.1f}%"
            )
            pre = m_now
            # Re-feed assistant tool calls as history for next turn.
            for o in body.get("output", []):
                if o.get("type") == "function_call":
                    history.append(o)
                    history.append(
                        {
                            "type": "function_call_output",
                            "call_id": o.get("call_id"),
                            "output": (
                                "def widget(): pass\n" * 10
                            ),  # synthetic file blob
                        }
                    )
            history.append({"role": "user", "content": user_text})

    post = fetch_metrics()
    summary = {
        "schema": "lumo.track_b.t3_tool_microbench.v1",
        "results": results,
        "bench_aggregate": {
            "accepted_tokens": post.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            - bench_pre_acc,
            "draft_tokens": post.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
            - bench_pre_drf,
        },
        "ts": time.time(),
    }
    if summary["bench_aggregate"]["draft_tokens"] > 0:
        summary["bench_aggregate"]["acceptance_rate"] = (
            summary["bench_aggregate"]["accepted_tokens"]
            / summary["bench_aggregate"]["draft_tokens"]
        )
    out = "/tmp/lumo-r3-t3-tool-microbench.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    agg = summary["bench_aggregate"]
    print(
        f"\nBench delta: {agg['accepted_tokens']:.0f}/{agg['draft_tokens']:.0f} = "
        f"{agg.get('acceptance_rate', 0.0) * 100:.1f}%"
    )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
