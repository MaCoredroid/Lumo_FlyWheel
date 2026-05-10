#!/usr/bin/env python3
"""T5 lifecycle — cross-session contamination test.

Two distinct first-user-message anchors → two distinct
``oracle_session_id``s. Each session sends the same prompt with a
*content-distinctive* substring that, if leaked into a sibling
session's per-session suffix tree, would manifest as elevated
acceptance on that substring's tokens.

Test logic:

1. **Session A**: send 3 turns whose responses contain a unique
   anchor phrase ``ANCHOR_A`` (long, unlikely to recur naturally).
2. **Session B**: send 3 turns whose first-user prompt anchor is
   different. Within session B, observe whether ``ANCHOR_A`` ever
   shows up at unexpectedly-high acceptance — it shouldn't, since
   the per-session suffix tree is supposed to be scoped to A.
3. **Session A again**: re-issue. Acceptance for ``ANCHOR_A``
   should be *high* on session A re-emission (T1 working) but
   shouldn't have polluted session B above (T5 + T1 isolation
   working).

This is an indirect test (we can't peek inside the per-session
``SuffixDecodingCache``), but acceptance-rate comparison is a
sufficient signal for the failure mode the spec calls out:
"Drafter state leaks across sessions".

Capture: ``/tmp/lumo-r3-t5-contamination.json``.
"""
from __future__ import annotations

import json
import time

import requests

URL = "http://127.0.0.1:8022/v1/responses"
METRICS_URL = "http://127.0.0.1:9950/metrics"
HEADERS = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}


SESSION_A_ANCHOR = (
    "I'm working on the photometric calibration pipeline in "
    "telescope/photometry/calibration.py. The flat-field correction "
    "uses zenith_attenuation_lambda_557nm = 0.183 as the reference "
    "value. Each turn, restate the file path AND the constant value "
    "verbatim in your reply, then add one sentence of analysis."
)

SESSION_B_ANCHOR = (
    "I'm investigating a bug in the rate limiter at api/limiter.py. "
    "The token bucket is leaking under burst load. Each turn, "
    "restate the file path verbatim and add one sentence of analysis. "
    "Do not reference any photometry, calibration, or telescope code."
)

SESSION_A_TURNS = [
    "First analysis turn. Remember to restate path + constant.",
    "Second analysis turn. Same restating rule.",
    "Third analysis turn. Continue restating.",
]

SESSION_B_TURNS = [
    "First analysis turn. Remember to restate path.",
    "Second analysis turn. Same rule.",
    "Third analysis turn.",
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


def extract_text(body: dict) -> str:
    parts: list[str] = []
    for o in body.get("output", []):
        for c in o.get("content", []):
            if c.get("type") in ("output_text", "text", "reasoning_text"):
                parts.append(c.get("text", ""))
    return "\n".join(p for p in parts if p)


def run_turn(anchor: str, history: list[dict], next_user: str) -> dict:
    inputs: list[dict] = [{"role": "user", "content": anchor}]
    inputs.extend(history)
    inputs.append({"role": "user", "content": next_user})
    payload = {
        "model": "qwen3.5-27b",
        "input": inputs,
        "tool_choice": "auto",
        "max_output_tokens": 512,
        "temperature": 0,
    }
    pre = fetch_metrics()
    t0 = time.perf_counter()
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
    elapsed = time.perf_counter() - t0
    body = r.json() if r.status_code == 200 else {}
    text = extract_text(body)
    post = fetch_metrics()
    acc = post.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - pre.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0.0
    )
    drf = post.get("vllm:spec_decode_num_draft_tokens_total", 0.0) - pre.get(
        "vllm:spec_decode_num_draft_tokens_total", 0.0
    )
    return {
        "status": r.status_code,
        "elapsed_s": round(elapsed, 3),
        "acc": acc,
        "drf": drf,
        "rate": (acc / drf) if drf > 0 else 0.0,
        "text": text,
    }


def run_session(label: str, anchor: str, prompts: list[str]) -> tuple[list[dict], list[dict]]:
    history: list[dict] = []
    rows: list[dict] = []
    for i, p in enumerate(prompts):
        res = run_turn(anchor, history, p)
        rows.append(
            {
                "session": label,
                "turn_index": i,
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
                "acc": res["acc"],
                "drf": res["drf"],
                "rate": res["rate"],
                "mentions_calibration": (
                    "calibration" in res["text"].lower()
                    or "photometry" in res["text"].lower()
                ),
                "mentions_limiter": "limiter" in res["text"].lower(),
                "text_first_120": res["text"][:120],
            }
        )
        print(
            f"  {label} turn{i} status={res['status']} elapsed={res['elapsed_s']}s "
            f"accept={res['acc']:.0f}/{res['drf']:.0f} rate={res['rate']*100:.1f}% "
            f"calib={rows[-1]['mentions_calibration']} limiter={rows[-1]['mentions_limiter']}"
        )
        if res["status"] == 200 and res["text"]:
            history.append({"role": "user", "content": p})
            history.append({"role": "assistant", "content": res["text"]})
    return rows, history


def main() -> None:
    print("=== Session A (calibration) — first run ===")
    a1, _ = run_session("A1", SESSION_A_ANCHOR, SESSION_A_TURNS)
    print("\n=== Session B (limiter) — should NOT inherit A's tree ===")
    b, _ = run_session("B", SESSION_B_ANCHOR, SESSION_B_TURNS)
    print("\n=== Session A again — should still benefit from its own warm tree ===")
    a2, _ = run_session("A2", SESSION_A_ANCHOR, SESSION_A_TURNS)

    # Aggregate rates
    def agg(rows: list[dict]) -> tuple[float, float, float]:
        a = sum(r["acc"] for r in rows)
        d = sum(r["drf"] for r in rows)
        return a, d, (a / d) if d > 0 else 0.0

    a1_acc, a1_drf, a1_rate = agg(a1)
    b_acc, b_drf, b_rate = agg(b)
    a2_acc, a2_drf, a2_rate = agg(a2)

    print()
    print(f"A1 (cold A):       {a1_acc:.0f}/{a1_drf:.0f} = {a1_rate*100:.1f}%")
    print(f"B  (cold B):       {b_acc:.0f}/{b_drf:.0f} = {b_rate*100:.1f}%")
    print(f"A2 (warm A reuse): {a2_acc:.0f}/{a2_drf:.0f} = {a2_rate*100:.1f}%")

    # Acceptance-rate isolation gate: if A's per-session suffix tree
    # had leaked into session B, B's cold acceptance rate would be
    # comparable to A2 (warm A reuse), not to A1 (cold A). The
    # spec failure mode is "drafter state leaks across sessions";
    # an unleaked B should be roughly comparable to A1 (both cold)
    # and meaningfully lower than A2.
    a2_minus_b = a2_rate - b_rate
    isolation_pass = a2_minus_b > 0.05  # at least 5 pp gap A2 > B
    print(
        f"\nIsolation gate: A2_warm({a2_rate*100:.1f}%) - "
        f"B_cold({b_rate*100:.1f}%) = "
        f"{a2_minus_b*100:+.1f}pp → {'PASS' if isolation_pass else 'FAIL'} "
        f"(B not inheriting A's warm tree)"
    )

    out = "/tmp/lumo-r3-t5-contamination.json"
    with open(out, "w") as fh:
        json.dump(
            {
                "schema": "lumo.track_b.t5_contamination.v1",
                "session_A1": a1,
                "session_B": b,
                "session_A2": a2,
                "aggregates": {
                    "A1": {"acc": a1_acc, "drf": a1_drf, "rate": a1_rate},
                    "B": {"acc": b_acc, "drf": b_drf, "rate": b_rate},
                    "A2": {"acc": a2_acc, "drf": a2_drf, "rate": a2_rate},
                },
                "isolation_pass": isolation_pass,
                "ts": time.time(),
            },
            fh,
            indent=2,
        )
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
