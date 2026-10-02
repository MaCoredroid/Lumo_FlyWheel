#!/usr/bin/env python3
"""Pooled decode rate (paper Eq. 5) from server /metrics snapshots bracketing a SWE arm:
D = (dGenTokens - dRequests) / (dE2E_sum - dTTFT_sum). Works for vLLM and SGLang metric names.
Usage: swe_pooled.py BEFORE.txt AFTER.txt"""
import re, sys
NAMES = {
    "gen": ["vllm:generation_tokens_total", "sglang:generation_tokens_total"],
    "req": ["vllm:request_success_total", "sglang:num_requests_total"],
    "e2e": ["vllm:e2e_request_latency_seconds_sum", "sglang:e2e_request_latency_seconds_sum"],
    "ttft": ["vllm:time_to_first_token_seconds_sum", "sglang:time_to_first_token_seconds_sum"],
}
def parse(p):
    out = {}
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)', line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out
b, a = parse(sys.argv[1]), parse(sys.argv[2])
vals = {}
for k, names in NAMES.items():
    n = next((x for x in names if x in a), None)
    vals[k] = (a[n] - b.get(n, 0.0)) if n else None
    vals[k + "_metric"] = n
if None in (vals["gen"], vals["req"], vals["e2e"], vals["ttft"]):
    print({"error": "missing metric", **vals}); sys.exit(1)
rate = (vals["gen"] - vals["req"]) / (vals["e2e"] - vals["ttft"])
print({"pooled_decode_tok_s": round(rate, 2), "gen_tokens": int(vals["gen"]), "requests": int(vals["req"]),
       "e2e_s": round(vals["e2e"], 1), "ttft_s": round(vals["ttft"], 1)})
