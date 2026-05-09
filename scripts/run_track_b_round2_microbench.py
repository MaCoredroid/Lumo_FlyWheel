#!/usr/bin/env python3
"""Round 2 micro-benchmark — exercises T1 session scoping at scale.

Five distinct synthetic sessions, three turns each. Each session has
a unique first-user-message anchor (=> unique session_id). Within a
session, turn 1 and turn 2 share an in-conversation tool history;
turn 0 is cold.

Goal: validate that turn_index > 0 within a session consistently
shows higher spec_decode acceptance than turn_index == 0. With T1's
per-session suffix tree, we expect:
- turn 0: cold, acceptance reflects vanilla SuffixDecoding's
  prompt-only state (no prior responses in the per-session tree).
- turn 1+: warm, acceptance reflects the carried-over response
  tokens from turn 0 (session cache).
"""
import json
import time

import requests

URL = "http://127.0.0.1:8033/v1/responses"
HEADERS = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

# Five distinct first-user-messages -> five distinct session_ids.
SESSIONS = [
    ("S1", "review the implementation in src/widget.py and explain its caching strategy"),
    ("S2", "audit auth/middleware.go for race conditions in the session refresh path"),
    ("S3", "the migration in db/0042 needs a backfill -- what edge cases should we test"),
    ("S4", "trace the request flow from the API gateway through the auth service to the database"),
    ("S5", "the parser in compiler/lexer.rs is dropping comments -- find the bug"),
]

# Synthetic file content used as a function_call_output anchor on
# turns 1 and 2.
FILE_BLOB = "def widget_cache_init(self):\n    self._cache = {}\n    return self._cache\n" * 25

TOOLS = [
    {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "array"}}}},
    {"type": "function", "name": "apply_patch", "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}}},
]


def build_payload(user_message: str, history_turns: int) -> dict:
    inputs = [{"role": "user", "content": user_message}]
    for i in range(history_turns):
        inputs.append({
            "type": "function_call",
            "call_id": f"c{i}",
            "name": "shell",
            "arguments": json.dumps({"cmd": ["cat", f"src/file{i}.py"]}),
        })
        inputs.append({
            "type": "function_call_output",
            "call_id": f"c{i}",
            "output": FILE_BLOB,
        })
    return {
        "model": "qwen3.5-27b",
        "input": inputs,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_output_tokens": 64,
        "temperature": 0,
    }


def main() -> None:
    results = []
    for sess_label, anchor_msg in SESSIONS:
        for turn_idx in range(3):
            payload = build_payload(anchor_msg, history_turns=turn_idx)
            t0 = time.perf_counter()
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=180)
            elapsed = time.perf_counter() - t0
            body = r.json() if r.status_code == 200 else {}
            results.append({
                "session": sess_label,
                "turn_index": turn_idx,
                "status": r.status_code,
                "id": body.get("id"),
                "elapsed_s": round(elapsed, 3),
            })
            print(f"  {sess_label} turn{turn_idx} status={r.status_code} elapsed={elapsed:.2f}s id={body.get('id')}")

    out = "/tmp/lumo-r2-microbench-results.json"
    with open(out, "w") as fh:
        json.dump({"results": results, "ts": time.time()}, fh, indent=2)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
