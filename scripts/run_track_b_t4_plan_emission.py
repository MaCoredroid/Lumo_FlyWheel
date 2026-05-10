#!/usr/bin/env python3
"""T4 plan-structure pre-drafter — multi-emission validation driver.

Drives a single session through 4 turns where the model is asked to
emit a numbered plan, with each turn asking for a "revised plan" that
matches the prior structure. Goal: confirm the in-loop observation
hook populates the per-session ``PlanRegistry`` and that, by the
third emission, ``best_activated_fingerprint`` has a winner so
``propose()`` can fire on the fourth turn.

The drafter is silent on activation by design (no patched-in metric
yet), so this driver infers activation from a **comparison**: turn 0
is cold (no prior emissions), turn 3 is post-activation (3 prior
emissions). If T4 is doing anything, turn 3 should show a meaningfully
larger jump in vLLM's spec_decode acceptance than what T1's session
scoping would produce alone (since T1 only learns from prior response
content, not from the new structural draft).

Usage: ``python3 scripts/run_track_b_t4_plan_emission.py``

Writes a JSON summary to ``/tmp/lumo-r2-t4-plan-emission.json``.
"""
from __future__ import annotations

import json
import time

import requests

URL = "http://127.0.0.1:8022/v1/responses"
METRICS_URL = "http://127.0.0.1:9950/metrics"
HEADERS = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

ANCHOR = (
    "/no_think\n"
    "I'm working on a multi-step refactor of the auth middleware. "
    "Each turn I'll tell you what I just finished and ask for the "
    "updated plan. Always reply with a numbered plan in the same "
    "shape: '## Plan\\n1. ...\\n2. ...\\n3. ...\\n4. ...'. "
    "Do NOT think out loud — emit the plan directly."
)

TURN_PROMPTS = [
    "/no_think Initial plan please. The four high-level steps for the refactor. Reply with the numbered plan only.",
    "/no_think I just finished step 1. Updated plan please. Reply with the numbered plan only.",
    "/no_think Just finished step 2. Updated plan please. Reply with the numbered plan only.",
    "/no_think Just finished step 3. Updated plan please. Reply with the numbered plan only.",
]


def fetch_metrics() -> dict:
    try:
        r = requests.get(METRICS_URL, headers={"Authorization": "Bearer EMPTY"}, timeout=10)
        if r.status_code != 200:
            return {}
        out: dict = {}
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                key, val = line.rsplit(" ", 1)
                # Strip Prometheus labels: foo_name{label="x"} -> foo_name
                base = key.split("{", 1)[0].strip()
                if "spec_decode" in base:
                    out[base] = out.get(base, 0.0) + float(val)
            except ValueError:
                continue
        return out
    except requests.RequestException:
        return {}


def build_payload(history: list[dict], next_user: str) -> dict:
    inputs = [{"role": "user", "content": ANCHOR}]
    inputs.extend(history)
    inputs.append({"role": "user", "content": next_user})
    return {
        "model": "qwen3.5-27b",
        "input": inputs,
        "tool_choice": "auto",
        "max_output_tokens": 1024,
        "temperature": 0,
    }


def extract_text(body: dict) -> str:
    """Capture text from both message and reasoning blocks so we
    can detect plan structure regardless of which channel the
    model uses."""

    parts: list[str] = []
    for o in body.get("output", []):
        for c in o.get("content", []):
            t = c.get("type")
            if t in ("output_text", "text", "reasoning_text"):
                parts.append(c.get("text", ""))
    return "\n".join(p for p in parts if p)


def main() -> None:
    history: list[dict] = []
    metrics_pre = fetch_metrics()
    results: list[dict] = []
    for turn_idx, user_text in enumerate(TURN_PROMPTS):
        payload = build_payload(history, user_text)
        t0 = time.perf_counter()
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
        elapsed = time.perf_counter() - t0
        body = r.json() if r.status_code == 200 else {}
        text = extract_text(body)
        m_now = fetch_metrics()
        accepted = m_now.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0.0
        ) - metrics_pre.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0.0
        )
        drafted = m_now.get(
            "vllm:spec_decode_num_draft_tokens_total", 0.0
        ) - metrics_pre.get(
            "vllm:spec_decode_num_draft_tokens_total", 0.0
        )
        results.append({
            "turn_index": turn_idx,
            "status": r.status_code,
            "elapsed_s": round(elapsed, 3),
            "id": body.get("id"),
            "draft_accepted_delta": accepted,
            "draft_total_delta": drafted,
            "acceptance_rate": (accepted / drafted) if drafted > 0 else 0.0,
            "model_text_len": len(text),
            "model_text_first_200": text[:200],
            "looks_like_numbered_plan": (
                "1." in text and "2." in text and "3." in text
            ),
        })
        print(
            f"  turn{turn_idx} status={r.status_code} elapsed={elapsed:.2f}s "
            f"accept={accepted:.0f}/{drafted:.0f} "
            f"plan_emitted={'yes' if results[-1]['looks_like_numbered_plan'] else 'no'}"
        )
        metrics_pre = m_now
        if r.status_code == 200 and text:
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": text})

    out_path = "/tmp/lumo-r2-t4-plan-emission.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "schema": "lumo.track_b.t4_plan_emission.v1",
                "results": results,
                "ts": time.time(),
            },
            fh,
            indent=2,
        )
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
