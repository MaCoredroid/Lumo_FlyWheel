#!/usr/bin/env python3
"""Per-technique applicability analyzer for Round 2.

Reads JSONL rows produced by ``lumo_flywheel_serving.inference_proxy``'s
Track B per-request capture (schema
``lumo.track_b.vllm_request_metrics.v1``) and answers, for each of the
five Round 2 harness-coupled techniques: what fraction of turns would
the technique fire on, and what theoretical maximum wallclock
reduction does that imply on this corpus?

Theoretical-ceiling-only by design — the script does not pretend to
predict the actual speedup of any technique, which depends on per-
technique acceptance rates that only land after the techniques are
implemented (Steps 4-9). Instead it answers the gating question:
"is the technique even reachable on real Codex traffic?" — the
prerequisite for ROI conversations.

Inputs (one of):
- A directory tree containing ``vllm_request_metrics.jsonl`` files
  (the v2 Round 0 layout).
- An explicit list of jsonl paths via ``--input``.

Output JSON (schema ``lumo.track_b.round2_applicability.v1``) groups
metrics by:
- corpus totals (turns, prompt tokens, prefill_sum_s, decode_sum_s).
- per-technique application rates (turns covered, decode/prefill in
  covered turns, theoretical max reduction).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "lumo.track_b.round2_applicability.v1"
DEFAULT_GLOB = "vllm_request_metrics.jsonl"


# Per-technique decode-side speedup target — taken straight from the
# v2 spec recalibration. Treat as a ceiling: applicability * speedup
# is the very best the technique could do; real numbers will be
# fractions of this once acceptance rates land.
TECHNIQUE_DECODE_SPEEDUP_TARGET = {
    "T1_cross_turn_ngram": 1.5,        # On top of SuffixDecoding baseline.
    "T2_read_file_priming": 2.0,       # Confined to reasoning regime.
    "T3_schema_aware_tool_drafter": 3.0,  # Tool-call regime; biggest target.
    "T4_plan_structure_predrafting": 2.0,  # Plan-emission turns only.
    "T5_lifecycle": 1.0,               # Bookkeeping; no direct decode delta.
}


def _iter_jsonl_files(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        if not root.is_dir():
            continue
        out.extend(sorted(root.rglob(DEFAULT_GLOB)))
    return out


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _technique_applies(row: dict[str, Any]) -> dict[str, bool]:
    """Decide which techniques would fire on a given turn.

    Decisions are grounded in fields available in the v1 capture
    schema. Where a signal isn't recoverable from the capture row
    (e.g., ``primed_texts``), we use a conservative proxy
    (e.g., reasoning-regime turns) and document the inference.
    """

    regime = row.get("regime")
    return {
        # T1 fires on every turn — SuffixDecoding already runs everywhere
        # and the cross-turn extension just changes how the suffix tree
        # is partitioned.
        "T1_cross_turn_ngram": True,
        # T2 (read_file priming) is most useful on reasoning turns where
        # the model is reading file contents. Tool-call turns also see
        # primed text but are dominated by name/argument decoding which
        # T3 covers. Use reasoning regime as the conservative proxy.
        "T2_read_file_priming": regime in {"reasoning", "summary"},
        # T3 fires when this turn is a tool-call regime turn — the
        # forced/auto tool_choice produces a structured emission the
        # schema-aware drafter can pre-fill. ``tool_call_observed`` is
        # equivalent on completed turns; use it for robustness.
        "T3_schema_aware_tool_drafter": (
            regime == "tool-call" or bool(row.get("tool_call_observed"))
        ),
        # T4 fires on plan-emission turns. Capture rows don't tag plan
        # emissions today (would need oracle_plan_fingerprint, which the
        # current proxy doesn't synthesise). Mark False; revisit when
        # the harness emits plan_fingerprint.
        "T4_plan_structure_predrafting": False,
        # T5 (lifecycle) is bookkeeping — fires on every turn but its
        # decode contribution is zero. Mark True so totals are well-
        # defined; speedup target is 1.0x so nothing else moves.
        "T5_lifecycle": True,
    }


def _accumulate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "turns": 0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "prefill_sum_s": 0.0,
        "decode_sum_s": 0.0,
        "wallclock_s": 0.0,
    }
    regimes: dict[str, dict[str, float]] = defaultdict(
        lambda: dict(turns=0.0, prefill_sum_s=0.0, decode_sum_s=0.0, wallclock_s=0.0)
    )
    technique_acc: dict[str, dict[str, float]] = defaultdict(
        lambda: dict(turns=0.0, prefill_sum_s=0.0, decode_sum_s=0.0, wallclock_s=0.0)
    )
    for row in rows:
        if row.get("schema") != "lumo.track_b.vllm_request_metrics.v1":
            continue
        if row.get("upstream_status") != 200:
            continue
        if not row.get("saw_response_completed"):
            continue
        prefill = _safe_float(row.get("prefill_sum_s")) or 0.0
        decode = _safe_float(row.get("decode_sum_s")) or 0.0
        wallclock = _safe_float(row.get("wallclock_s")) or 0.0
        prompt_t = _safe_float(row.get("prompt_tokens")) or 0.0
        completion_t = _safe_float(row.get("completion_tokens")) or 0.0
        totals["turns"] += 1
        totals["prompt_tokens"] += prompt_t
        totals["completion_tokens"] += completion_t
        totals["prefill_sum_s"] += prefill
        totals["decode_sum_s"] += decode
        totals["wallclock_s"] += wallclock

        regime = row.get("regime") or "unknown"
        regimes[regime]["turns"] += 1
        regimes[regime]["prefill_sum_s"] += prefill
        regimes[regime]["decode_sum_s"] += decode
        regimes[regime]["wallclock_s"] += wallclock

        for technique, fires in _technique_applies(row).items():
            if not fires:
                continue
            technique_acc[technique]["turns"] += 1
            technique_acc[technique]["prefill_sum_s"] += prefill
            technique_acc[technique]["decode_sum_s"] += decode
            technique_acc[technique]["wallclock_s"] += wallclock

    return {"totals": totals, "regimes": dict(regimes), "techniques": dict(technique_acc)}


def _theoretical_decode_reduction(
    technique_decode_s: float, total_decode_s: float, speedup: float
) -> dict[str, float | None]:
    """Compute the maximum decode-time reduction the technique could
    deliver on this corpus.

    Reduction = decode_in_covered_turns - (decode_in_covered_turns /
    speedup). Capped at the technique's covered share of the total —
    the ratio between this and ``total_decode_s`` is the corpus-wide
    decode improvement ceiling.
    """

    if speedup <= 1.0:
        reduction = 0.0
    else:
        reduction = technique_decode_s * (1.0 - 1.0 / speedup)
    fraction_of_decode = (
        reduction / total_decode_s if total_decode_s > 0 else None
    )
    return {
        "decode_reduction_ceiling_s": reduction,
        "decode_reduction_ceiling_fraction_of_corpus": fraction_of_decode,
    }


def build_report(jsonl_paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files_read: list[str] = []
    for path in jsonl_paths:
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            files_read.append(str(path))
        except OSError:
            continue

    accumulated = _accumulate(rows)
    totals = accumulated["totals"]
    techniques_out: dict[str, dict[str, Any]] = {}
    for technique, acc in accumulated["techniques"].items():
        speedup = TECHNIQUE_DECODE_SPEEDUP_TARGET.get(technique, 1.0)
        ceiling = _theoretical_decode_reduction(
            acc["decode_sum_s"], totals["decode_sum_s"], speedup
        )
        techniques_out[technique] = {
            "turns_covered": int(acc["turns"]),
            "turns_covered_fraction": (
                acc["turns"] / totals["turns"] if totals["turns"] else None
            ),
            "prefill_sum_s_covered": acc["prefill_sum_s"],
            "decode_sum_s_covered": acc["decode_sum_s"],
            "decode_sum_s_fraction_of_corpus": (
                acc["decode_sum_s"] / totals["decode_sum_s"]
                if totals["decode_sum_s"] > 0
                else None
            ),
            "decode_speedup_target_x": speedup,
            **ceiling,
        }

    regimes_out = {
        regime: {
            "turns": int(stats["turns"]),
            "turns_fraction": (
                stats["turns"] / totals["turns"] if totals["turns"] else None
            ),
            "prefill_sum_s": stats["prefill_sum_s"],
            "prefill_sum_s_fraction": (
                stats["prefill_sum_s"] / totals["prefill_sum_s"]
                if totals["prefill_sum_s"] > 0
                else None
            ),
            "decode_sum_s": stats["decode_sum_s"],
            "decode_sum_s_fraction": (
                stats["decode_sum_s"] / totals["decode_sum_s"]
                if totals["decode_sum_s"] > 0
                else None
            ),
        }
        for regime, stats in accumulated["regimes"].items()
    }

    return {
        "schema": SCHEMA,
        "producer": "build_track_b_round2_applicability",
        "files_read": files_read,
        "files_read_count": len(files_read),
        "rows_seen": len(rows),
        "totals": {
            "turns": int(totals["turns"]),
            "prompt_tokens_total": int(totals["prompt_tokens"]),
            "completion_tokens_total": int(totals["completion_tokens"]),
            "prefill_sum_s_total": totals["prefill_sum_s"],
            "decode_sum_s_total": totals["decode_sum_s"],
            "wallclock_s_total": totals["wallclock_s"],
        },
        "regimes": regimes_out,
        "techniques": techniques_out,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-technique Round 2 applicability + theoretical ceiling "
            "from Track B per-request capture rows."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSONL file path or directory tree to scan recursively (repeatable).",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the JSON summary."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="After writing, print the per-technique table to stdout.",
    )
    return parser.parse_args(argv)


def _print_table(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print(f"# Track B Round 2 applicability  ({report['files_read_count']} files, {totals['turns']} turns)")
    print(
        f"  total prefill={totals['prefill_sum_s_total']:.1f}s  "
        f"decode={totals['decode_sum_s_total']:.1f}s  "
        f"wallclock={totals['wallclock_s_total']:.1f}s"
    )
    print()
    print("## Regimes")
    print(f"  {'regime':<12} {'turns':>6} {'frac':>6} {'prefill_s':>10} {'decode_s':>10}")
    for regime, stats in sorted(report["regimes"].items()):
        print(
            f"  {regime:<12} {stats['turns']:>6} "
            f"{stats['turns_fraction']:>6.1%} "
            f"{stats['prefill_sum_s']:>10.1f} {stats['decode_sum_s']:>10.1f}"
        )
    print()
    print("## Techniques (theoretical ceilings)")
    print(
        f"  {'technique':<32} {'fires':>6} {'frac':>6} {'dec_cov':>8} "
        f"{'speedup':>8} {'reduce_s':>10} {'frac':>6}"
    )
    for technique, stats in sorted(report["techniques"].items()):
        frac = stats.get("turns_covered_fraction") or 0
        dec_frac = stats.get("decode_sum_s_fraction_of_corpus") or 0
        red = stats.get("decode_reduction_ceiling_s") or 0
        red_frac = stats.get("decode_reduction_ceiling_fraction_of_corpus") or 0
        print(
            f"  {technique:<32} {stats['turns_covered']:>6} "
            f"{frac:>6.1%} {dec_frac:>8.1%} "
            f"{stats['decode_speedup_target_x']:>7.2f}x "
            f"{red:>10.1f} {red_frac:>6.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [Path(p) for p in args.input]
    paths = _iter_jsonl_files(inputs)
    if not paths:
        print(f"No JSONL files found under: {inputs}", file=sys.stderr)
        return 1
    report = build_report(paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.print:
        _print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
