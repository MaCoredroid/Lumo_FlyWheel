#!/usr/bin/env python3
"""FR14 promotion A/B: the fused-draft-top-k verdict, read from the dfwd span.

WHY THIS FILE AND NOT deploy_speed. fused_draft_topk.md 8.2 pre-registers the
instrument: "A stack-level dfwd delta of 0.3078 ms is small against a 49-53 ms
span, so the serve A/B needs the SPAN TIMER, not the step total." The lever
replaces argmax(248320)+topk(248320,3) with one launch on each of the five head
reads, and the bracket that contains those calls is the drafter split's `lmhead`
term. So the verdict is:

    lmhead_seconds / n_lmhead      top-k ON  vs  top-k OFF

read out of /logs/fr13_dfwd_split.json (FR13_DFWD_SPLIT=1, both arms), with the
whole-drafter span (FR13_DFWD_GPU_TIMER) reported beside it as the containing
bracket and step_wall reported beside that as the thing the doctrine forbids
using as the verdict.

Acceptance is NOT a reading for this lever: the selection is byte-exact (6 840
configurations, zero raw-byte mismatches, plus a 24-replay CUDA-graph gate), so
it cannot move acceptance and any accept delta here is trajectory divergence.
It is printed only so that claim stays falsifiable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _one(paths: list[Path], label: str) -> dict:
    """Sidecars are written as <name>.<producer-pid>; there must be exactly one."""
    if len(paths) != 1:
        raise SystemExit(
            f"expected exactly one {label} sidecar, found {len(paths)}: {paths}"
        )
    return json.loads(paths[0].read_text())


def arm(runroot: Path, armname: str) -> dict:
    logs = runroot / armname / "logs"
    sidecars = Path("output/fr13_sfwd_sidecar")
    split = _one(sorted(logs.glob("fr13_dfwd_split.json*")), "dfwd_split")
    dfwd = _one(sorted(sidecars.glob(f"{armname}_dfwd.json*")), "dfwd span")
    cfwd = _one(sorted(sidecars.glob(f"{armname}_cfwd.json*")), "cfwd span")
    sfwd = _one(sorted(sidecars.glob(f"{armname}.json")) or
                [p for p in sorted(sidecars.glob(f"{armname}.json.*"))
                 if "samples" not in p.name], "sfwd span")
    lfwd_paths = [p for p in sorted(sidecars.glob(f"{armname}_lfwd.json*"))]
    out = {
        "arm": armname,
        "runroot": str(runroot),
        "dfwd_split": split,
        "dfwd_span": dfwd,
        "cfwd_span": cfwd,
        "sfwd_span": sfwd,
        "lfwd_span": json.loads(lfwd_paths[0].read_text()) if lfwd_paths else None,
    }
    # Per-step millisecond derivations, each on its OWN denominator -- the split
    # terms count head reads, the span timers count steps, and mixing them is
    # the basis-mismatch this campaign has already been bitten by once.
    d = {}
    if split.get("n_lmhead"):
        d["lmhead_ms_per_unit"] = split["lmhead_seconds"] / split["n_lmhead"] * 1e3
    if split.get("n_model"):
        d["model_ms_per_unit"] = split["model_seconds"] / split["n_model"] * 1e3
    if split.get("n_sample"):
        d["sample_ms_per_unit"] = split["sample_seconds"] / split["n_sample"] * 1e3
    if dfwd.get("n_spans"):
        d["dfwd_ms_per_step"] = dfwd["gpu_seconds"] / dfwd["n_spans"] * 1e3
    if cfwd.get("n_spans"):
        d["cfwd_ms_per_step"] = cfwd["gpu_seconds"] / cfwd["n_spans"] * 1e3
    if isinstance(sfwd, dict) and sfwd.get("n_spans"):
        d["sfwd_ms_per_step"] = sfwd["gpu_seconds"] / sfwd["n_spans"] * 1e3
    if out["lfwd_span"] and out["lfwd_span"].get("n_spans"):
        d["lfwd_ms_per_step"] = (
            out["lfwd_span"]["gpu_seconds"] / out["lfwd_span"]["n_spans"] * 1e3
        )
    out["derived_ms"] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-runroot", required=True)
    ap.add_argument("--on-arm", required=True)
    ap.add_argument("--off-runroot", required=True)
    ap.add_argument("--off-arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    on = arm(Path(a.on_runroot), a.on_arm)
    off = arm(Path(a.off_runroot), a.off_arm)

    deltas = {}
    for k in sorted(set(on["derived_ms"]) | set(off["derived_ms"])):
        o, f = on["derived_ms"].get(k), off["derived_ms"].get(k)
        if o is None or f is None:
            continue
        deltas[k] = {
            "topk_on": o,
            "topk_off": f,
            "delta_on_minus_off_ms": o - f,
            "delta_pct": (o - f) / f * 100.0 if f else float("nan"),
        }

    out = {
        "schema": "fr14.promotion_ab.dfwd_span_verdict.v1",
        "instrument": "dfwd_split.lmhead (pre-registered in fused_draft_topk.md 8.2)",
        "predicted_ms_per_step": -0.3078,
        "predicted_note": (
            "fused_draft_topk.md 6: -0.308 ms/step measured offline (8.6x on the "
            "selection stage), NOT the briefed -4.5; the surface was mislabeled "
            "and >=4.1 ms of it lives in the head GEMM projection."
        ),
        "acceptance_is_not_a_reading": (
            "byte-exact selection, 6840 configs / 0 raw-byte mismatches; any "
            "accept delta here is trajectory divergence, printed to keep the "
            "byte-exactness claim falsifiable."
        ),
        "topk_on": on,
        "topk_off": off,
        "deltas": deltas,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"{'term':22s} {'top-k ON':>13s} {'top-k OFF':>13s} {'delta ms':>11s} {'delta %':>9s}")
    for k, v in deltas.items():
        print(f"{k:22s} {v['topk_on']:13.5f} {v['topk_off']:13.5f} "
              f"{v['delta_on_minus_off_ms']:11.5f} {v['delta_pct']:8.2f}%")
    print()
    print("raw counts (denominators differ by design -- split terms count head "
          "reads, span timers count steps):")
    for label, rec in (("ON ", on), ("OFF", off)):
        s = rec["dfwd_split"]
        print(f"  {label} n_lmhead={s.get('n_lmhead')} n_model={s.get('n_model')} "
              f"dfwd_spans={rec['dfwd_span'].get('n_spans')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
