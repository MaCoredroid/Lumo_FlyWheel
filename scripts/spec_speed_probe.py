#!/usr/bin/env python3
"""B=1 spec-decode speed probe — clean accept-vs-cost decomposition for comparing
speculative configs (E linear chain vs F branching tree) on the *currently
serving* vLLM, without the scheduler-interleave / agent-wall noise of the B=4
SWE run.

Why this exists: decode_tps = committed_tokens_per_event / wall_per_event. E and F
can differ on EITHER lever (acceptance OR event cost), so a single TPS number
can't explain *why* one is faster. This probe sends N fixed prompts strictly
sequentially (concurrency 1 -> per-request spec-trace rows are contiguous and
their ts-deltas are clean per-event latency) and reports both levers per config.

It is config-agnostic: relaunch a config, run the probe with a --label, repeat
for the other config, then diff the JSON outputs. Hits vLLM directly on :9950
(bypasses the proxy so temperature/top_p are controlled here).

Token accounting: per speculative event the target verifies `draft` proposed
tokens, accepts `acc` of them, and always samples 1 more (the bonus token), so
committed_tokens_per_event = acc + 1. decode_tps = sum(acc + 1) / decode_span.

Metrics (per request + aggregate):
  mean_acc_per_event      tokens accepted per event   (E3~2.25, F canary~1.07)
  mean_draft_per_event    nodes verified per event    (E3=3, F=6)
  accepted_per_node       acc/draft = verify efficiency (E3~0.75, F~0.18)
  wasted_nodes_per_event  draft-acc
  mean_event_ms           per-event wall latency (clean at B=1)
  decode_tps              committed/decode_span (single-stream)
  acc_dist                acceptance depth profile

NOTE: this is a SINGLE-STREAM (B=1) decode rate; it is meant for apples-to-apples
E-vs-F comparison, NOT to equal the B=4 dgx_steptrace numbers from the SWE sweep.
A no-spec config (OFF) produces no spec-trace events -> reported as events=0.
"""
from __future__ import annotations
import argparse, collections, json, os, subprocess, time, urllib.request
from pathlib import Path

TRACE = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl"
DEFAULT_FB_CONTROL = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fb_control.json"
DEFAULT_HOTPLUG_REF_N8 = "output/spec_speed_probe/E8_depth_search_256.json"
PORT = 9950
CONTAINER = "lumo-vllm-track-b-suffix"

# Fixed, deterministic prompt set (coding/reasoning, representative of the
# workload class). Keep stable across runs so E and F see identical inputs.
PROMPTS = [
    "Write a Python function to reverse a singly linked list, then explain it.",
    "Explain what a hash table is and how collisions are handled, in detail.",
    "Implement binary search over a sorted list in Python with comments.",
    "What is the time complexity of merge sort and why? Walk through it.",
    "Write a Python function to check if a string is a palindrome, with tests.",
    "Describe the difference between a process and a thread, with examples.",
    "Implement a Python decorator that times function execution and logs it.",
    "Explain how a B-tree differs from a binary search tree and when to use each.",
    "Write a SQL query to find the second highest salary and explain it.",
    "Implement quicksort in Python and explain the pivot-choice tradeoffs.",
]


def get_api_key() -> str:
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-c",
         'tr "\\0" "\\n" < /proc/$(pgrep -f "vllm serve"|head -1)/environ | sed -n "s/^VLLM_API_KEY=//p"'],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def trace_count() -> int:
    p = Path(TRACE)
    if not p.exists():
        return 0
    with p.open() as f:
        return sum(1 for _ in f)


def read_trace(pre: int, post: int) -> list[dict]:
    rows = []
    if not Path(TRACE).exists():
        return rows
    with open(TRACE) as f:
        for i, line in enumerate(f, 1):
            if pre < i <= post:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    return rows


def write_fb_control(path: Path, depth: int, k: int, assert_depth: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": int(depth),
        "k": int(k),
        "updated_at": round(time.time(), 6),
    }
    if assert_depth is not None:
        payload["assert_depth"] = int(assert_depth)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tmp.replace(path)


def validate_widths(rows: list[dict], active_depth: int | None) -> dict:
    mismatches = []
    for idx, row in enumerate(rows):
        proposal_width = int(row.get("proposal_width", row.get("draft", -1)))
        verify_width = int(row.get("verify_width", row.get("draft", -1)))
        expected = int(active_depth) if active_depth is not None else verify_width
        if proposal_width != verify_width or verify_width != expected:
            mismatches.append({
                "event_index": idx,
                "rid": row.get("rid"),
                "proposal_width": proposal_width,
                "verify_width": verify_width,
                "expected_width": expected,
            })
            if len(mismatches) >= 20:
                break
    return {
        "valid_width": not mismatches,
        "proposal_equals_verify": not mismatches,
        "mismatch_count_sampled": len(mismatches),
        "mismatches": mismatches,
    }


def hotplug_reference_delta(path: Path, agg: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        ref = json.loads(path.read_text())
        ref_agg = ref.get("aggregate", {})
        return {
            "path": str(path),
            "reference_acc_per_event": ref_agg.get("mean_acc_per_event"),
            "reference_decode_tps": ref_agg.get("decode_tps"),
            "delta_acc_per_event": (
                round(agg.get("mean_acc_per_event") - ref_agg.get("mean_acc_per_event"), 3)
                if agg.get("mean_acc_per_event") is not None and ref_agg.get("mean_acc_per_event") is not None
                else None
            ),
            "delta_decode_tps": (
                round(agg.get("decode_tps") - ref_agg.get("decode_tps"), 3)
                if agg.get("decode_tps") is not None and ref_agg.get("decode_tps") is not None
                else None
            ),
        }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def chat(prompt: str, key: str, max_tokens: int, temp: float, top_p: float):
    body = json.dumps({
        "model": "qwen3.6-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def summarize(rows: list[dict]) -> dict:
    """Per-request OR aggregate summary from a flat list of spec-event rows.
    For aggregate use, decode_tps/span are recomputed by the caller across
    requests to avoid counting inter-request gaps; here span is within-list."""
    n = len(rows)
    if n == 0:
        return {"events": 0}
    drafts = [r["draft"] for r in rows]
    accs = [r["acc"] for r in rows]
    ts = [r["ts"] for r in rows]
    sum_acc, sum_draft = sum(accs), sum(drafts)
    committed = sum_acc + n  # accepted drafts + 1 bonus target token per event
    span = (ts[-1] - ts[0]) if n > 1 else 0.0
    return {
        "events": n,
        "mean_acc_per_event": round(sum_acc / n, 3),
        "mean_draft_per_event": round(sum_draft / n, 3),
        "accepted_per_node": round(sum_acc / sum_draft, 4) if sum_draft else None,
        "wasted_nodes_per_event": round((sum_draft - sum_acc) / n, 3),
        "committed_tokens": committed,
        "decode_span_s": round(span, 4),
        "mean_event_ms": round(span / (n - 1) * 1000, 2) if n > 1 else None,
        "decode_tps": round(committed / span, 3) if span > 0 else None,
        "acc_dist": dict(sorted(collections.Counter(accs).items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="config label, e.g. E3 or Fa")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.6, help="match the sweep (0.6)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fb-control-depth", type=int, default=None)
    ap.add_argument("--fb-control-k", type=int, default=None)
    ap.add_argument("--fb-assert-depth", type=int, default=None,
                    help="positive-control only: expected actual width for the runtime assert")
    ap.add_argument("--fb-control-file", default=DEFAULT_FB_CONTROL)
    ap.add_argument("--launch-n-max", type=int, default=None)
    ap.add_argument("--certifiable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hotplug-reference", default=None)
    args = ap.parse_args()

    if (args.fb_control_depth is None) != (args.fb_control_k is None):
        raise SystemExit("--fb-control-depth and --fb-control-k must be supplied together")

    hotplug = args.fb_control_depth is not None
    certifiable = bool(args.certifiable and not hotplug)
    if hotplug:
        write_fb_control(Path(args.fb_control_file), args.fb_control_depth,
                         args.fb_control_k, assert_depth=args.fb_assert_depth)
        extra = f" assert_depth={args.fb_assert_depth}" if args.fb_assert_depth is not None else ""
        print(f"poked F_b control: depth={args.fb_control_depth} k={args.fb_control_k}{extra} -> {args.fb_control_file}")

    key = get_api_key()
    if not key:
        raise SystemExit("could not read VLLM_API_KEY from container (is it serving?)")

    per_req, all_rows = [], []
    for i, prompt in enumerate(PROMPTS[: args.n_prompts]):
        pre = trace_count()
        d, wall = chat(prompt, key, args.max_tokens, args.temp, args.top_p)
        time.sleep(0.3)  # let the line-buffered trace flush
        rows = read_trace(pre, trace_count())
        rec = summarize(rows)
        usage = d.get("usage", {})
        rec["prompt_idx"] = i
        rec["api_wall_s"] = round(wall, 3)
        rec["completion_tokens"] = usage.get("completion_tokens")
        per_req.append(rec)
        all_rows += rows
        print(f"  [{i}] events={rec['events']} acc/ev={rec.get('mean_acc_per_event')} "
              f"draft/ev={rec.get('mean_draft_per_event')} tps={rec.get('decode_tps')}")

    # Aggregate: pool events for per-event means/dist, but sum per-request spans
    # for decode_tps so inter-request gaps + prefill don't pollute it.
    agg = summarize(all_rows)
    tot_committed = sum(r.get("committed_tokens", 0) for r in per_req)
    tot_span = sum(r.get("decode_span_s", 0) or 0 for r in per_req)
    tot_gaps = sum((r["events"] - 1) for r in per_req if r["events"] > 1)
    agg["decode_tps"] = round(tot_committed / tot_span, 3) if tot_span > 0 else None
    agg["mean_event_ms"] = round(tot_span / tot_gaps * 1000, 2) if tot_gaps > 0 else None
    agg["decode_span_s"] = round(tot_span, 4)
    agg["committed_tokens"] = tot_committed
    width_validation = validate_widths(all_rows, args.fb_control_depth)

    ref_path = Path(args.hotplug_reference) if args.hotplug_reference else None
    if hotplug and ref_path is None and args.fb_control_depth == 8 and Path(DEFAULT_HOTPLUG_REF_N8).exists():
        ref_path = Path(DEFAULT_HOTPLUG_REF_N8)
    ref_delta = hotplug_reference_delta(ref_path, agg) if ref_path else None

    result = {
        "label": args.label, "temp": args.temp, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "n_prompts": len(per_req),
        "thinking_enabled": "template_default_true",
        "decode_token_scope": "thinking_plus_answer",
        "launch_n_max": args.launch_n_max,
        "active_depth": args.fb_control_depth,
        "active_k": args.fb_control_k,
        "certifiable": certifiable,
        "fb_control_file": args.fb_control_file if hotplug else None,
        "width_validation": width_validation,
        "hotplug_reference": ref_delta,
        "aggregate": agg, "per_request": per_req,
    }
    out = Path(args.out or f"output/spec_speed_probe/{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    a = agg
    print(f"\n=== spec_speed_probe [{args.label}] (B=1, temp={args.temp}) ===")
    print(f"  events={a['events']}  acc/event={a.get('mean_acc_per_event')}  "
          f"draft/event={a.get('mean_draft_per_event')}  accepted/node={a.get('accepted_per_node')}")
    print(f"  wasted_nodes/event={a.get('wasted_nodes_per_event')}  mean_event_ms={a.get('mean_event_ms')}  "
          f"decode_tps={a.get('decode_tps')}")
    print(f"  acc_dist={a.get('acc_dist')}")
    print(f"  provenance: launch_n_max={args.launch_n_max} active_depth={args.fb_control_depth} "
          f"active_k={args.fb_control_k} certifiable={certifiable}")
    print(f"  width_validation={width_validation['valid_width']}")
    if ref_delta:
        print(f"  hotplug_reference={ref_delta}")
    print(f"  -> {out}")
    if not width_validation["valid_width"]:
        print("ERROR: proposal_width != verify_width or verify_width != active_depth; probe invalid")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
