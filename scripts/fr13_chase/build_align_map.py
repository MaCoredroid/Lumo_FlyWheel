#!/usr/bin/env python3
"""Build a global_event_idx -> capture_call alignment map for a native window.

The FR13_FINAL_LOGIT_CAPTURE counter saves every 6-row forward, which
includes trailing request-boundary forwards that have no spec-trace event
(fully clipped at max_tokens).  Within a request captures are consecutive;
extras appear only at request boundaries.  Per request we search a small
base window and map consecutively.

Scoring per candidate base: fraction of the request's events whose capture
row-0 fp32 argmax equals the event's first emitted token.  Greedy windows
must score 1.0 at the chosen base (served tokens are computed from these
logits); sampled (t06) windows score statistically (argmax usually sampled)
and the best base must dominate alternatives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from fr13_disc_lib import (  # noqa: E402
    load_jsonl,
    load_probe,
    probe_records,
    walk_native_events,
)


def row0_argmax(win: Path, prefix: str, call: int, cache: dict) -> int | None:
    if call in cache:
        return cache[call]
    p = win / "logs" / f"{prefix}.call{call}.pt"
    if not p.exists():
        cache[call] = None
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    cache[call] = int(d["logits"][0].float().argmax())
    return cache[call]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--probe-name", required=True)
    ap.add_argument("--capture-prefix", default="native_final_logits")
    ap.add_argument("--native-num-spec", type=int, default=5)
    ap.add_argument("--first-call", type=int, default=None,
                    help="lowest capture call index in this window (default: min on disk)")
    ap.add_argument("--max-extra", type=int, default=6)
    ap.add_argument("--strict-greedy", action="store_true",
                    help="require score 1.0 per request")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    win = Path(args.window)
    probe = load_probe(win / args.probe_name)
    records = probe_records(probe)
    events, diag = walk_native_events(
        load_jsonl(win / "logs/per_req_spec_trace.jsonl"),
        load_jsonl(win / "logs/fr10_mtp_draft_trace.jsonl"),
        records,
        num_spec=args.native_num_spec,
    )
    calls_on_disk = sorted(
        int(p.name.split(".call")[1].split(".pt")[0])
        for p in (win / "logs").glob(f"{args.capture_prefix}.call*.pt")
    )
    first_call = args.first_call if args.first_call is not None else calls_on_disk[0]

    by_prompt: dict[int, list[dict]] = {}
    for ev in events:
        by_prompt.setdefault(ev["prompt_id"], []).append(ev)
    for evs in by_prompt.values():
        evs.sort(key=lambda e: e["event_idx_in_prompt"])

    cache: dict[int, int | None] = {}
    mapping: dict[int, int] = {}
    report = []
    ptr = first_call
    for pid in sorted(by_prompt):
        evs = by_prompt[pid]
        best = None
        for extra in range(args.max_extra + 1):
            base = ptr + extra
            ok = tot = 0
            for ev in evs:
                am = row0_argmax(win, args.capture_prefix, base + ev["event_idx_in_prompt"], cache)
                if am is None:
                    continue
                tot += 1
                if am == int(ev["emitted"][0]):
                    ok += 1
            score = ok / tot if tot else 0.0
            if best is None or score > best[1]:
                best = (base, score, tot)
            if score == 1.0:
                break
        base, score, tot = best
        if args.strict_greedy and score < 1.0:
            raise AssertionError(f"prompt {pid}: best base {base} score {score} < 1.0 (greedy)")
        for ev in evs:
            mapping[ev["global_event_idx"]] = base + ev["event_idx_in_prompt"]
        report.append({"prompt_id": pid, "base_call": base, "skipped_before": base - ptr,
                       "score": score, "events": len(evs), "scored": tot})
        ptr = base + len(evs)
    out = {
        "window": str(win),
        "first_call": first_call,
        "calls_on_disk": [calls_on_disk[0], calls_on_disk[-1]],
        "n_calls": len(calls_on_disk),
        "events": len(events),
        "walk_diag": diag,
        "per_request": report,
        "map": {str(k): v for k, v in sorted(mapping.items())},
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "map"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
