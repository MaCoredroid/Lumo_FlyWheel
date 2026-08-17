#!/usr/bin/env python3
"""Reduce the FR14 arm-B K64-vs-K0 ablation to accept / TPS / step wall.

Same reduction arm A used for leg 3 (ablation_a_leg3.json), so the two are
directly comparable: client-side numbers from sglang bench_serving's summary,
engine-side numbers from a /metrics bracket taken immediately before and after
the bench run.

    accept_per_event    = d(spec_decode_num_accepted_tokens) / d(spec_decode_num_drafts)
    committed_per_event = accept_per_event + 1        (the verified root token)
    decode_only_tps     = d(generation_tokens) / d(request_decode_time_seconds_sum)
    step_wall_ms        = d(request_decode_time_seconds_sum) / d(spec_decode_num_drafts) * 1000

Usage: python3 armb_k64_ablation_reduce.py <ablation_out_dir> <output.json>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
)

CLIENT_FIELDS = {
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
    "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
    "output_token_throughput_tps": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "total_token_throughput_tps": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "mean_e2e_latency_ms": r"Mean E2E Latency \(ms\):\s+([\d.]+)",
    "median_e2e_latency_ms": r"Median E2E Latency \(ms\):\s+([\d.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([\d.]+)",
}


def scrape_counters(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            continue
        for name in COUNTERS:
            if line.startswith(name):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return out


def scrape_client(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    text = path.read_text(errors="replace")
    for key, pattern in CLIENT_FIELDS.items():
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1))
    return out


def reduce_arm(armdir: Path) -> dict[str, object]:
    pre = scrape_counters(armdir / "metrics_pre.txt")
    post = scrape_counters(armdir / "metrics_post.txt")
    delta = {k: post.get(k, 0.0) - pre.get(k, 0.0) for k in COUNTERS}

    drafts = delta["vllm:spec_decode_num_drafts_total"]
    draft_tokens = delta["vllm:spec_decode_num_draft_tokens_total"]
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    gen = delta["vllm:generation_tokens_total"]
    decode_s = delta["vllm:request_decode_time_seconds_sum"]
    prefill_s = delta["vllm:request_prefill_time_seconds_sum"]

    floor_txt = (armdir / "floor.txt")
    floor: dict[str, object] = {}
    if floor_txt.is_file():
        parts = floor_txt.read_text().split()
        if len(parts) == 2:
            floor = {
                "mandatory_weight_bytes": int(parts[0]),
                "weight_floor_ms": float(parts[1]),
            }

    bracket: dict[str, object] = {
        "generation_tokens": gen,
        "spec_decode_num_drafts": drafts,
        "spec_decode_num_draft_tokens": draft_tokens,
        "spec_decode_num_accepted_tokens": accepted,
        "request_decode_time_seconds_sum": round(decode_s, 4),
        "request_prefill_time_seconds_sum": round(prefill_s, 4),
    }
    if drafts:
        bracket["tok_per_draft"] = round(draft_tokens / drafts, 3)
        bracket["accept_per_event"] = round(accepted / drafts, 4)
        bracket["committed_per_event"] = round(accepted / drafts + 1, 4)
        bracket["step_wall_ms"] = round(decode_s / drafts * 1000, 3)
    if decode_s:
        bracket["decode_only_tps"] = round(gen / decode_s, 3)
    if decode_s + prefill_s:
        bracket["prefill_frac"] = round(prefill_s / (decode_s + prefill_s), 4)
    if floor and drafts and decode_s:
        bracket["floor_ratio"] = round(
            (decode_s / drafts * 1000) / float(floor["weight_floor_ms"]), 4
        )

    # BOTH logs. boot_container.log is captured at health, which is BEFORE the
    # first propose -- and _fr13_dvk_prepare (and therefore the Phase-1 dequant
    # banner) fires on the first propose, i.e. only run_container.log has it.
    # Reading just the boot log would report "no dequant" for an arm that did
    # dequantise, and would make the K0 arm's inertness claim vacuous.
    boot_text = ""
    for name in ("boot_container.log", "run_container.log"):
        path = armdir / name
        if path.is_file():
            boot_text += path.read_text(errors="replace")
    return {
        "client": scrape_client(armdir / "bench.log"),
        "engine_bracket": bracket,
        "floor": floor,
        "dvk": {
            "shim_built_lines": [
                l.strip() for l in boot_text.splitlines() if "[FR13_DRAFT_VOCAB]" in l
            ][:2],
            "dequant_lines": [
                l.strip() for l in boot_text.splitlines() if "FR14_DVK_DEQUANT" in l
            ][:2],
            "root_engaged_lines": [
                l.strip()
                for l in boot_text.splitlines()
                if "[FR13_DRAFT_VOCAB_ROOT] engaged" in l
            ][:2],
        },
        "lmhead_route_lines": [
            l.strip() for l in boot_text.splitlines() if "FR14_LMHEAD_QUANT_ROUTE" in l
        ][:2],
        "tracebacks": [
            l.strip()
            for l in boot_text.splitlines()
            if "Traceback (most recent call last)" in l
        ][:2],
    }


def main() -> int:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    arms = {name: reduce_arm(root / name) for name in ("k64", "k0")}

    k64 = arms["k64"]["engine_bracket"]
    k0 = arms["k0"]["engine_bracket"]
    comparison: dict[str, object] = {}
    for field in (
        "accept_per_event",
        "committed_per_event",
        "step_wall_ms",
        "decode_only_tps",
        "floor_ratio",
    ):
        if field in k64 and field in k0:
            comparison[field] = {
                "k64": k64[field],
                "k0": k0[field],
                "delta_k0_minus_k64": round(float(k0[field]) - float(k64[field]), 4),
            }
    for field in ("output_token_throughput_tps", "mean_tpot_ms", "median_itl_ms"):
        a = arms["k64"]["client"].get(field)
        b = arms["k0"]["client"].get(field)
        if a is not None and b is not None:
            comparison[field] = {
                "k64": a,
                "k0": b,
                "delta_k0_minus_k64": round(b - a, 4),
            }

    payload = {
        "schema": "fr14.armb.k64_ablation.v1",
        "lane": "DIAGNOSTIC -- calibration-grade, NON-CITABLE",
        "question": (
            "Under the NVFP4 lm_head, full-vocab drafting costs only +0.807 ms "
            "of FLOOR over K64 (93.152 vs 92.345) against +34.3 ms in the fp8 "
            "era. Bytes say +-0; does the WALL agree? K64 also bought DFWD "
            "compute (65k vs 248k rows/GEMV) and pays a subset-miss acceptance "
            "penalty on out-of-corpus content."
        ),
        "workload": (
            "sglang bench_serving --backend vllm, random 1024/1024, "
            "8 prompts, concurrency 1, seed 1, temp 0.6/top_p 0.95/top_k 20 -- "
            "identical content in both arms"
        ),
        "arms": arms,
        "comparison": comparison,
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
