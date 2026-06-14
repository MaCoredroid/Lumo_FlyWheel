#!/usr/bin/env python3
"""FR13_GDN_SUBOP_MAB capture driver (same-boot).

Drives the 4 pinned SWE prompts at the canonical gold-gate rider battery
(/v1/completions RAW text, max_tokens=128, temp 0.0, top_p 1.0, seed 1313,
return_token_ids, fr10_decode_mode=tree_mtp). The in-process FR13_GDN_SUBOP_MAB
hook (default-OFF; armed here via the launcher env) fires on each L0 tree-verify
forward and re-runs the GDN sub-op sequence at M=10/M=5/M=1, writing one JSONL
record per captured event.

Two passes of the 4 prompts to assert WITHIN-BOOT DETERMINISM (class-8 gate):
pass-1 served streams must be byte-identical to pass-0 (per-prompt [T/F]).
Records tok/draft (spec_decode accepted/draft from /metrics) for the engagement
gate. No re-implementation of the request path: mirrors fr13_node5_ladder_drive.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


def post(ep, path, payload, timeout=900):
    req = urllib.request.Request(
        ep.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read().decode()
    return json.loads(b) if b else None


def get_text(ep, path, timeout=15):
    with urllib.request.urlopen(ep.rstrip("/") + path, timeout=timeout) as r:
        return r.read().decode()


def wait_health(ep, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            get_text(ep, "/health", timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("health timeout")


def one_completion(ep, model, prompt, max_tokens, seed):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": 1,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": "tree_mtp"},
    }
    d = post(ep, "/v1/completions", payload)
    ch = d["choices"][0]
    return {
        "served_token_ids": ch.get("token_ids") or [],
        "finish_reason": ch.get("finish_reason"),
    }


def metric_vals(text):
    out = {}
    for name in ("spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens"):
        m = re.findall(rf"^vllm:{name}(?:_total)?\{{[^}}]*\}}\s+([0-9.eE+-]+)$", text, re.M)
        if not m:
            m = re.findall(rf"^vllm:{name}(?:_total)?\s+([0-9.eE+-]+)$", text, re.M)
        out[name] = sum(float(x) for x in m) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:9950")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--prompts", default="output/fr13_acceptance_ladder/prompts_swe4.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1313)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="output/fr13_gdn_subop_mab/drive_result.json")
    args = ap.parse_args()

    ep = args.endpoint
    wait_health(ep, 600)
    prompts = json.loads(Path(args.prompts).read_text())
    result = {"ts": time.time(), "endpoint": ep, "model": args.model,
              "seed": args.seed, "max_tokens": args.max_tokens,
              "num_prompts": len(prompts)}

    reps = []
    metrics_snaps = []
    for rep in range(args.reps):
        try:
            post(ep, "/reset_prefix_cache", {}, 30)
        except Exception:
            pass
        try:
            m_before = metric_vals(get_text(ep, "/metrics", timeout=15))
        except Exception:
            m_before = {}
        streams = []
        for pid, prompt in enumerate(prompts):
            rec = one_completion(ep, args.model, prompt, args.max_tokens, args.seed)
            streams.append(rec["served_token_ids"])
            print(f"[rep{rep}] pid={pid} served_len={len(rec['served_token_ids'])}", flush=True)
        try:
            m_after = metric_vals(get_text(ep, "/metrics", timeout=15))
        except Exception:
            m_after = {}
        reps.append(streams)
        metrics_snaps.append({"before": m_before, "after": m_after})

    # within-boot determinism: per-prompt byte-identity rep1 vs rep0
    base = reps[0]
    det = []
    for pid in range(len(prompts)):
        ok = all(reps[r][pid] == base[pid] for r in range(1, args.reps))
        det.append(bool(ok))
    result["within_boot_det"] = det
    result["served_lens_rep0"] = [len(s) for s in base]
    result["served_first8_rep0"] = [s[:8] for s in base]

    # tok/draft from the metric delta across rep0 (engagement)
    m0 = metrics_snaps[0]
    acc = (m0["after"].get("spec_decode_num_accepted_tokens") or 0) - \
          (m0["before"].get("spec_decode_num_accepted_tokens") or 0)
    drf = (m0["after"].get("spec_decode_num_draft_tokens") or 0) - \
          (m0["before"].get("spec_decode_num_draft_tokens") or 0)
    result["spec_accepted_delta_rep0"] = acc
    result["spec_draft_delta_rep0"] = drf
    # tok/draft engagement: num_speculative_tokens (=9 for cat9) drafted per step.
    # accept/event = accepted/(draft/num_spec)+? — we report accepted+1 / steps via ratio.
    result["metrics_snaps"] = metrics_snaps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out, flush=True)
    print("within_boot_det", det, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
