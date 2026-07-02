#!/usr/bin/env python3
"""fr13_carrier_locator_probe.py — single-cell GENERATION-LEVEL carrier probe (fast de-confounder).

Replays banked tool-call-boundary prompts (qwen-code chatreq_*.json dumps) N times against ONE live
vLLM kernel config, SINGLE-TURN, temp 0.6, and measures the MALFORMED-MARKUP emission rate directly on
the raw completions. No agent loop, no nudge net, no eval, no offload tunnel -> removes the four confounds
(kernel/num_spec/topology coupling is broken by the grid; nudge + trajectory + infra removed by going
single-turn direct-to-vLLM). Run once per {FLASH,TREE}x{ns5,ns8} cell (fr13_carrier_locator_grid.sh) on
the SAME prompt corpus to localize which kernel knob carries the structured-output corruption.

The discriminator is exactly the validated one: vLLM's qwen3_xml parser lifts a WELL-FORMED tool call into
message.tool_calls (=clean); a MALFORMED one fails to parse and its raw markup falls through to
message.content, where classify_text (imported from the carrier gate) flags malformed_markup.

Optionally banks top_logprobs for the temp-0.6 q-vs-p TV follow-on.
Read-only against the server. GPU work is the server's; this is a thin client.
"""
import argparse, glob, json, os, sys, time, urllib.request, urllib.error
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_structured_output_carrier_gate import classify_text  # the validated, FP-fixed classifier


def load_prompt(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    return {"messages": msgs, "tools": d.get("tools")}


def post(endpoint, body, timeout=240):
    req = urllib.request.Request(
        endpoint, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spec_counters(metrics_endpoint):
    """Snapshot vLLM spec-decode counters (for a clean same-prompt accept-rate per cell).
    Retries (the /metrics endpoint can be slow while the server is mid-probe) and surfaces the
    error rather than returning silent None, so a transient failure is visible not invisible."""
    keys = ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_drafts_total", "vllm:generation_tokens_total")
    last_err = "unknown"
    for _attempt in range(5):
        out = dict.fromkeys(keys, 0.0)
        try:
            with urllib.request.urlopen(metrics_endpoint, timeout=20) as r:  # /metrics is unauthenticated
                body = r.read().decode("utf-8", "ignore")
            found = False
            for ln in body.splitlines():
                for k in keys:
                    if ln.startswith(k) and (" " in ln):
                        try:
                            out[k] = float(ln.split()[-1]); found = True
                        except Exception:
                            pass
            if found:
                return out
            last_err = "spec-counters-not-found-in-metrics"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
    return {"_error": last_err}   # non-None so spec_delta surfaces the cause


def spec_delta(before, after):
    if not before or not after or "_error" in before or "_error" in after:
        return {"error": (before or {}).get("_error") or (after or {}).get("_error") or "no-counters"}
    d = {k: after.get(k, 0) - before.get(k, 0) for k in before}
    dt = d["vllm:spec_decode_num_draft_tokens_total"]
    ac = d["vllm:spec_decode_num_accepted_tokens_total"]
    dr = d["vllm:spec_decode_num_drafts_total"]
    gt = d["vllm:generation_tokens_total"]
    return {
        "drafts": dr, "draft_tokens": dt, "accepted_tokens": ac, "generation_tokens": gt,
        "accept_rate": round(ac / dt, 4) if dt else None,
        "accepted_per_step": round(ac / dr, 3) if dr else None,
        "draft_width": round(dt / dr, 2) if dr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:9950/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True, help="chatreq_*.json files or dir(s)")
    ap.add_argument("--prompt-limit", type=int, default=0, help="0=all; else first N prompts")
    ap.add_argument("--samples", type=int, default=20, help="completions per prompt (temp 0.6 => a rate)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)   # deployment temp; NEVER 0
    ap.add_argument("--cell", required=True, help="cell label, e.g. D_TREE_ns5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logprobs", action="store_true", help="bank top_logprobs for the TV follow-on")
    a = ap.parse_args()

    files = []
    for p in a.prompts:
        files += sorted(glob.glob(os.path.join(p, "chatreq_*.json"))) if os.path.isdir(p) else [p]
    if a.prompt_limit:
        files = files[: a.prompt_limit]

    labels = Counter()
    rows = []
    n_ok = n_err = 0
    t0 = time.time()
    metrics_endpoint = a.endpoint.rsplit("/v1/", 1)[0] + "/metrics"
    spec_before = spec_counters(metrics_endpoint)   # clean same-prompt spec-accept per cell
    for pf in files:
        pr = load_prompt(pf)
        if not pr:
            continue
        for s in range(a.samples):
            body = {"model": a.model, "messages": pr["messages"], "temperature": a.temperature,
                    "max_tokens": a.max_tokens, "stream": False}
            if pr["tools"]:
                body["tools"] = pr["tools"]
            if a.logprobs:
                body["logprobs"] = True
                body["top_logprobs"] = 20
            try:
                resp = post(a.endpoint, body)
            except Exception as e:
                n_err += 1
                rows.append({"prompt": os.path.basename(pf), "sample": s, "label": "REQ_ERR", "err": str(e)[:120]})
                continue
            n_ok += 1
            ch = (resp.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            if tcs:                                   # parser lifted a well-formed call => clean
                lab = "clean_toolcall"
            else:
                lab = classify_text(text) or "clean_notoolcall"
            labels[lab] += 1
            row = {"prompt": os.path.basename(pf), "sample": s, "label": lab,
                   "n_tool_calls": len(tcs or []), "finish": ch.get("finish_reason"), "text_head": text[:200]}
            if a.logprobs:
                row["logprobs"] = ch.get("logprobs")
            rows.append(row)

    spec_after = spec_counters(metrics_endpoint)
    total = n_ok
    malformed = labels.get("malformed_markup", 0)
    summary = {
        "cell": a.cell, "endpoint": a.endpoint, "model": a.model,
        "n_prompts": len(files), "samples": a.samples, "n_ok": n_ok, "n_err": n_err,
        "elapsed_s": round(time.time() - t0, 1),
        "spec": spec_delta(spec_before, spec_after),   # accept_rate/accepted_per_step on THESE prompts
        "labels": dict(labels),
        "malformed_markup": malformed,
        "malformed_rate": round(malformed / total, 4) if total else None,
        "degenerate_loop": labels.get("degenerate_loop", 0),
        "off_topic_derail": labels.get("off_topic_derail", 0),
        "clean_toolcall": labels.get("clean_toolcall", 0),
        "clean_toolcall_rate": round(labels.get("clean_toolcall", 0) / total, 4) if total else None,
        "clean_notoolcall": labels.get("clean_notoolcall", 0),   # emitted prose, no call = nudge-confound zone
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
