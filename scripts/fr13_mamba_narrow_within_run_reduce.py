#!/usr/bin/env python3
"""Reduce the FR13_MAMBA_SPEC_BLOCKS_CDIV within-run pair to one verdict artifact.

Consumes only evidence the two arms recorded themselves:

  <arm>/metrics_before_swe.txt, <arm>/metrics_after_swe.txt
      vllm:prefix_cache_{queries,hits}_total -> campaign-window APC hit rate.
      This is the PRIMARY claim under test and the same basis the
      20260809T064230Z salvage note used, so the numbers are comparable to it.
  <arm>/docker_full.log
      "Maximum concurrency", "GPU KV cache size", and the loggers.py timeline
      (Running: N, GPU KV cache usage, Prefix cache hit rate). The reservation
      evidence is boot-time and therefore NOT agent-trajectory dependent, which
      is what makes it the corroborating channel for the APC move.
  <arm>/deploy_speed_fullwall.json
      per-request / aggregate decode rates and events per step, work-census
      gated by fr13_measure.py itself.
  <arm>/logs/fr13_fixed32_{engine,proxy}_ingress.jsonl
      request counts, so a wall-clock difference can be attributed to
      trajectory length rather than read as a speed effect.

FAILS CLOSED: every arm must supply the metrics bracket and the boot log, and
the APC denominators must be non-zero. A missing bracket is reported as an
error, never as a zero.

CPU-only, pure stdlib.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

RE_METRIC = re.compile(
    r"^vllm:prefix_cache_(queries|hits)_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M
)
RE_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")
RE_KVSIZE = re.compile(r"GPU KV cache size:\s*([\d,]+) tokens")
RE_RUNNING = re.compile(
    r"Running:\s*(\d+)\s*reqs.*?GPU KV cache usage:\s*([\d.]+)%.*?"
    r"Prefix cache hit rate:\s*([\d.]+)%"
)


def _counters(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise SystemExit(f"missing metrics bracket: {path}")
    out: dict[str, float] = {}
    for kind, value in RE_METRIC.findall(path.read_text(errors="replace")):
        out[kind] = float(value)
    for kind in ("queries", "hits"):
        if kind not in out:
            raise SystemExit(f"{path} has no prefix_cache_{kind}_total")
    return out


def _lines(text: str) -> int:
    return len(text.splitlines())


def read_arm(runroot: Path, arm: str) -> dict[str, Any]:
    d = runroot / arm
    before = _counters(d / "metrics_before_swe.txt")
    after = _counters(d / "metrics_after_swe.txt")
    dq = after["queries"] - before["queries"]
    dh = after["hits"] - before["hits"]
    if dq <= 0:
        raise SystemExit(f"{arm}: APC denominator is {dq}; the campaign window is empty")

    log = d / "docker_full.log"
    if not log.is_file():
        raise SystemExit(f"{arm}: missing boot/serve log {log}")
    text = log.read_text(errors="replace")
    conc = RE_CONC.search(text)
    kv = RE_KVSIZE.search(text)
    if not conc or not kv:
        raise SystemExit(f"{arm}: boot log lacks the kv_cache_utils reservation lines")
    rows = RE_RUNNING.findall(text)
    if not rows:
        raise SystemExit(f"{arm}: boot log has no loggers.py Running/usage timeline")
    usage = [float(u) for _, u, _ in rows]
    hits_timeline = [float(h) for _, _, h in rows]
    run1 = [float(u) for n, u, _ in rows if int(n) == 1]

    speed: dict[str, Any] = {}
    sp = d / "deploy_speed_fullwall.json"
    if sp.is_file():
        raw = json.loads(sp.read_text(encoding="utf-8"))
        for k in (
            "per_request_decode_tps",
            "aggregate_decode_tps",
            "effective_concurrency",
            "accept_per_event",
            "committed_per_event",
            "s_per_fwd",
            "s_per_fwd_gpu",
            "derived_tps",
            "derived_tps_gpu",
            "prefill_frac",
            "n_tasks",
        ):
            if k in raw:
                speed[k] = raw[k]

    def count(rel: str) -> int | None:
        p = d / rel
        return _lines(p.read_text(errors="replace")) if p.is_file() else None

    return {
        "arm": arm,
        "apc": {
            "queries": dq,
            "hits": dh,
            "hit_rate": dh / dq,
            "queries_before": before["queries"],
            "hits_before": before["hits"],
        },
        "reservation": {
            "gpu_kv_cache_tokens": int(kv.group(1).replace(",", "")),
            "max_concurrency_x": float(conc.group(2)),
            "tokens_per_request": int(conc.group(1).replace(",", "")),
            "kv_usage_min_pct": min(usage),
            "kv_usage_peak_pct": max(usage),
            "kv_usage_floor_running1_pct": (min(run1) if run1 else None),
            "n_logger_lines": len(rows),
            "n_running1_lines": len(run1),
            "hit_rate_timeline_last_pct": hits_timeline[-1],
            "hit_rate_timeline_max_pct": max(hits_timeline),
        },
        "speed": speed,
        "requests": {
            "engine_ingress_rows": count("logs/fr13_fixed32_engine_ingress.jsonl"),
            "proxy_ingress_rows": count("logs/fr13_fixed32_proxy_ingress.jsonl"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runroot", required=True, type=Path)
    ap.add_argument("--off-arm", required=True)
    ap.add_argument("--on-arm", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    off = read_arm(args.runroot, args.off_arm)
    on = read_arm(args.runroot, args.on_arm)

    delta = {
        "apc_hit_rate_off": off["apc"]["hit_rate"],
        "apc_hit_rate_on": on["apc"]["hit_rate"],
        "apc_hit_rate_delta_pp": (on["apc"]["hit_rate"] - off["apc"]["hit_rate"]) * 100.0,
        "max_concurrency_off_x": off["reservation"]["max_concurrency_x"],
        "max_concurrency_on_x": on["reservation"]["max_concurrency_x"],
        "max_concurrency_ratio": (
            on["reservation"]["max_concurrency_x"] / off["reservation"]["max_concurrency_x"]
        ),
        "kv_peak_off_pct": off["reservation"]["kv_usage_peak_pct"],
        "kv_peak_on_pct": on["reservation"]["kv_usage_peak_pct"],
        "kv_floor_running1_off_pct": off["reservation"]["kv_usage_floor_running1_pct"],
        "kv_floor_running1_on_pct": on["reservation"]["kv_usage_floor_running1_pct"],
        "same_pool": (
            off["reservation"]["gpu_kv_cache_tokens"]
            == on["reservation"]["gpu_kv_cache_tokens"]
        ),
    }
    for key in ("per_request_decode_tps", "aggregate_decode_tps", "effective_concurrency",
                "accept_per_event", "committed_per_event"):
        a, b = off["speed"].get(key), on["speed"].get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            delta[f"{key}_off"] = a
            delta[f"{key}_on"] = b
            delta[f"{key}_delta_frac"] = (b - a) / a

    out = {
        "schema": "fr13.mamba_narrow.within_run_pair.v1",
        "classification": "diagnostic_within_run_lever_pair",
        "citable_cutlass_timing": False,
        "formal_floor_acceptance_eligible": False,
        "lever": "FR13_MAMBA_SPEC_BLOCKS_CDIV",
        "only_arm_delta": "FR13_MAMBA_SPEC_BLOCKS_CDIV_0_to_1",
        "both_arms_cutlass": "stock",
        "primary_claim": (
            "the mamba per-request speculative page reservation, not attention-KV "
            "capacity, is the binding constraint on exact4 B4 APC recovery"
        ),
        "off": off,
        "on": on,
        "delta": delta,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"APC hit rate   OFF {off['apc']['hit_rate']*100:6.2f}%  "
          f"ON {on['apc']['hit_rate']*100:6.2f}%  "
          f"delta {delta['apc_hit_rate_delta_pp']:+.2f}pp")
    print(f"max concurrency OFF {delta['max_concurrency_off_x']:.2f}x  "
          f"ON {delta['max_concurrency_on_x']:.2f}x  "
          f"ratio {delta['max_concurrency_ratio']:.3f}")
    print(f"kv peak         OFF {delta['kv_peak_off_pct']:.1f}%   "
          f"ON {delta['kv_peak_on_pct']:.1f}%")
    print(f"kv floor Run:1  OFF {delta['kv_floor_running1_off_pct']}%  "
          f"ON {delta['kv_floor_running1_on_pct']}%")
    if "per_request_decode_tps_off" in delta:
        print(f"per-request tps OFF {delta['per_request_decode_tps_off']:.3f}  "
              f"ON {delta['per_request_decode_tps_on']:.3f}  "
              f"({delta['per_request_decode_tps_delta_frac']*100:+.2f}%)")
    print(f"same pool: {delta['same_pool']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
