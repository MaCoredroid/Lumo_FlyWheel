#!/usr/bin/env python3
"""Round 3 — driven 4-point Track 2 ablation.

Runs four ablation points by toggling
``/tmp/lumo_track_b_runtime_flags.json`` inside the live vLLM
container (no relaunches between points). Each point uses the same
5×3 microbench workload so the per-technique acceptance contribution
is comparable.

Points:

- A (T1 only): T2/T3/T4 disabled
- B (+T2): T3/T4 disabled
- C (+T2+T3): T4 disabled
- D (all on): no disables

Pre-flight: confirms the ``_lumo_track_b_disabled`` patch is present
in the container's site-packages suffix_decoding.py — without it,
the file-based flags do nothing.

Capture: ``/tmp/lumo-r3-ablation.json``.
"""
from __future__ import annotations

import json
import subprocess
import time

import requests

CONTAINER = "lumo-vllm-track-b-suffix"
URL = "http://127.0.0.1:8022/v1/responses"
METRICS_URL = "http://127.0.0.1:9950/metrics"
HEADERS = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
FLAGS_PATH = "/tmp/lumo_track_b_runtime_flags.json"


SESSIONS = [
    ("S1", "review the implementation in src/widget.py and explain its caching strategy"),
    ("S2", "audit auth/middleware.go for race conditions in the session refresh path"),
    ("S3", "the migration in db/0042 needs a backfill -- what edge cases should we test"),
    ("S4", "trace the request flow from the API gateway through the auth service to the database"),
    ("S5", "the parser in compiler/lexer.rs is dropping comments -- find the bug"),
]
FILE_BLOB = "def widget_cache_init(self):\n    self._cache = {}\n    return self._cache\n" * 25
TOOLS = [
    {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "array"}}}},
    {"type": "function", "name": "apply_patch", "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}}},
]


def write_flags(payload: dict) -> None:
    """Write the runtime-disable flags into the container."""

    body = json.dumps(payload)
    subprocess.run(
        [
            "docker", "exec", CONTAINER, "bash", "-lc",
            f"printf '%s' {body!r} > {FLAGS_PATH}",
        ],
        check=True,
        capture_output=True,
    )


def confirm_patch_present() -> bool:
    rc = subprocess.run(
        [
            "docker", "exec", CONTAINER, "bash", "-lc",
            (
                "grep -c 'def _lumo_track_b_disabled' "
                "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py"
            ),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(rc.stdout.strip()) > 0
    except (ValueError, AttributeError):
        return False


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


def reset_prefix_cache() -> None:
    try:
        requests.post(
            "http://127.0.0.1:9950/reset_prefix_cache",
            headers={"Authorization": "Bearer EMPTY"},
            timeout=10,
        )
    except requests.RequestException:
        pass


def run_microbench(label: str) -> dict:
    reset_prefix_cache()
    pre = fetch_metrics()
    bench_pre_acc = pre.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    bench_pre_drf = pre.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    rows: list[dict] = []
    for sess_label, anchor in SESSIONS:
        for turn_idx in range(3):
            payload = build_payload(anchor, turn_idx)
            t0 = time.perf_counter()
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=180)
            elapsed = time.perf_counter() - t0
            body = r.json() if r.status_code == 200 else {}
            rows.append({
                "session": sess_label,
                "turn_index": turn_idx,
                "status": r.status_code,
                "elapsed_s": round(elapsed, 3),
                "id": body.get("id"),
            })
    post = fetch_metrics()
    acc = post.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - bench_pre_acc
    drf = post.get("vllm:spec_decode_num_draft_tokens_total", 0.0) - bench_pre_drf
    rate = (acc / drf) if drf > 0 else 0.0
    print(
        f"  [{label}] accepted={acc:.0f} drafted={drf:.0f} "
        f"rate={rate*100:.1f}%"
    )
    return {
        "label": label,
        "rows": rows,
        "accepted_tokens": acc,
        "draft_tokens": drf,
        "acceptance_rate": rate,
    }


def main() -> None:
    if not confirm_patch_present():
        print(
            "FATAL: _lumo_track_b_disabled not present in container suffix_decoding.py — "
            "relaunch first to land the prelaunch update."
        )
        raise SystemExit(2)

    points = [
        ("A_T1_only", {"T2": True, "T3": True, "T4": True}),
        ("B_T1_T2", {"T2": False, "T3": True, "T4": True}),
        ("C_T1_T2_T3", {"T2": False, "T3": False, "T4": True}),
        ("D_all_on", {"T2": False, "T3": False, "T4": False}),
    ]
    results: list[dict] = []
    for label, flags in points:
        print(f"\n=== Ablation {label}: flags={flags} ===")
        write_flags(flags)
        time.sleep(1)
        res = run_microbench(label)
        res["flags"] = flags
        results.append(res)

    out = "/tmp/lumo-r3-ablation.json"
    with open(out, "w") as fh:
        json.dump(
            {
                "schema": "lumo.track_b.round3_ablation.v1",
                "container": CONTAINER,
                "points": results,
                "ts": time.time(),
            },
            fh,
            indent=2,
        )
    print()
    print(f"{'point':>15}  rate     accepted   drafted")
    for r in results:
        print(
            f"{r['label']:>15}  {r['acceptance_rate']*100:>5.1f}%  "
            f"{int(r['accepted_tokens']):>9}  {int(r['draft_tokens']):>8}"
        )
    print(f"\nResults: {out}")
    # Best-effort: clear flags so we don't leave leftover state.
    write_flags({"T2": False, "T3": False, "T4": False})


if __name__ == "__main__":
    main()
