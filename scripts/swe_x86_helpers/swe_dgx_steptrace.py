#!/usr/bin/env python3
"""High-frequency vLLM iteration/step + GPU sampler for the concurrency probe.

Writes one JSON line per sample (~1.5s) with cumulative counters so analysis can
delta them per arm-window:
  - gen, prompt: vllm:generation_tokens_total / prompt_tokens_total
  - iter_sum, iter_cnt: vllm:iteration_tokens_total _sum/_count
       (_cnt delta = engine steps in window; iter_sum delta = tokens across steps;
        per-step latency = window_seconds / steps; tokens/step = iter_sum/iter_cnt)
  - running, waiting: vllm:num_requests_running / num_requests_waiting (batch gauge)
  - acc, draft, drafts: spec-decode totals
  - dec_sum, pre_sum: vllm:request_decode/prefill_time_seconds_sum (global histogram sums)
  - gpu_util, mem_util, power_w, temp_c: nvidia-smi (mem_util is ~0 on GB10, kept for record)
"""
import json, os, subprocess, time, urllib.request

METRICS = os.environ.get("LUMO_VLLM_METRICS_URL", "http://127.0.0.1:9950/metrics")
OUT = os.environ.get("LUMO_SWE_DGX_STEPTRACE", "/tmp/swe_dgx_steptrace.jsonl")
WANT = {
    "vllm:generation_tokens_total": "gen",
    "vllm:prompt_tokens_total": "prompt",
    "vllm:iteration_tokens_total_sum": "iter_sum",
    "vllm:iteration_tokens_total_count": "iter_cnt",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:spec_decode_num_accepted_tokens_total": "acc",
    "vllm:spec_decode_num_draft_tokens_total": "draft",
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:request_decode_time_seconds_sum": "dec_sum",
    "vllm:request_prefill_time_seconds_sum": "pre_sum",
}


def scrape():
    rec = {v: 0.0 for v in WANT.values()}
    try:
        with urllib.request.urlopen(METRICS, timeout=3) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = WANT.get(name)
        if key is None:
            continue
        try:
            rec[key] += float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return rec


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        a = [x.strip() for x in out.split(",")]
        return {"gpu_util": float(a[0]), "mem_util": float(a[1]),
                "power_w": float(a[2]), "temp_c": float(a[3])}
    except Exception:
        return {"gpu_util": None, "mem_util": None, "power_w": None, "temp_c": None}


def main():
    with open(OUT, "a", buffering=1) as fh:
        while True:
            rec = scrape()
            if rec is not None:
                rec["ts"] = time.time()
                rec.update(gpu())
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            time.sleep(1.5)


if __name__ == "__main__":
    main()
