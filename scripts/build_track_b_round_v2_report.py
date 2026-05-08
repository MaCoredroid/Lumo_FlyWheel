#!/usr/bin/env python3
"""Compose the final Round 0 v2 report.

Reads the v2 round summary + proxy capture rows + (optionally) the v1
reduced-contract summary at output/track_b_e2e/round_0/round_summary.json
and emits:

- A merged JSON artifact at the v2 round directory documenting the live
  config, per-task wallclock, per-regime acceptance, and the wallclock
  delta vs the v1 reduced-contract baseline.
- A short markdown report with the same content for human review.

Designed to run once after the v2 sweep completes. No external dependencies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _collect_runner_metadata(round_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(round_dir.glob("*/run_*/runner_metadata.json")):
        data = _safe_load_json(path)
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _per_attempt_summary(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in metadata_rows:
        task_id = f"{row.get('family')}/{row.get('variant')}"
        by_task.setdefault(task_id, []).append(row)
    out: dict[str, Any] = {}
    for task_id, runs in by_task.items():
        runs_sorted = sorted(runs, key=lambda r: r.get("attempt", 0))
        # Discard the cold attempt (run_01) per the round driver's convention.
        measured = [r for r in runs_sorted if r.get("attempt", 0) > 1]
        elapsed_values = [r.get("elapsed_s") for r in measured if isinstance(r.get("elapsed_s"), (int, float))]
        retry_counts = [r.get("zero_token_retry_count", 0) for r in measured]
        out[task_id] = {
            "attempts": len(runs_sorted),
            "measured_attempts": len(measured),
            "median_elapsed_s": statistics.median(elapsed_values) if elapsed_values else None,
            "p90_elapsed_s": (
                statistics.quantiles(elapsed_values, n=10, method="inclusive")[-1]
                if len(elapsed_values) >= 2
                else (elapsed_values[0] if elapsed_values else None)
            ),
            "min_elapsed_s": min(elapsed_values) if elapsed_values else None,
            "max_elapsed_s": max(elapsed_values) if elapsed_values else None,
            "zero_token_retry_count_total": sum(retry_counts),
            "zero_token_retry_cohort_count": sum(1 for c in retry_counts if c > 0),
            "non_zero_exit_count": sum(1 for r in measured if r.get("codex_exit_code") not in (0, None)),
        }
    return out


def _per_regime_summary(capture_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from build_track_b_per_regime_acceptance import aggregate  # type: ignore

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    return aggregate(capture_rows)


def _load_capture_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _markdown_report(
    *,
    v2_summary: dict[str, Any] | None,
    v1_summary: dict[str, Any] | None,
    per_attempt: dict[str, Any],
    per_regime: dict[str, Any],
    runtime_config_hash: str,
) -> str:
    lines: list[str] = ["# Track B E2E Round 0 v2 Report", ""]
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("## Summary")
    if v2_summary:
        lines.append(
            f"- v2 tasks_completed: **{v2_summary.get('tasks_completed')}**, "
            f"trusted_task_count: **{v2_summary.get('trusted_task_count')}**, "
            f"median_wallclock_s: **{v2_summary.get('median_wallclock_s')}**, "
            f"aggregate_wallclock_s: **{v2_summary.get('aggregate_wallclock_s')}**"
        )
        lines.append(f"- v2 runtime_config_hash: `{v2_summary.get('runtime_config_hash')}`")
    else:
        lines.append("- v2 round_summary.json: NOT FOUND (sweep may still be in progress)")
    if v1_summary:
        lines.append(
            f"- v1 (reduced-contract) tasks_completed: {v1_summary.get('tasks_completed')}, "
            f"median_wallclock_s: {v1_summary.get('median_wallclock_s')}, "
            f"aggregate_wallclock_s: {v1_summary.get('aggregate_wallclock_s')}"
        )
    if v2_summary and v1_summary:
        v2_med = v2_summary.get("median_wallclock_s")
        v1_med = v1_summary.get("median_wallclock_s")
        if isinstance(v2_med, (int, float)) and isinstance(v1_med, (int, float)) and v1_med > 0:
            delta_pct = 100 * (v2_med - v1_med) / v1_med
            lines.append(f"- v2 median wallclock vs v1: **{v2_med:.2f}s vs {v1_med:.2f}s ({delta_pct:+.1f}%)**")

    lines.append("")
    lines.append("## Per-task wallclock (measured attempts only; cold run_01 discarded)")
    lines.append("")
    lines.append("| task | attempts | measured | median elapsed | p90 elapsed | retries | non-zero exit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for task_id, stats in sorted(per_attempt.items()):
        lines.append(
            f"| {task_id} | {stats['attempts']} | {stats['measured_attempts']} | "
            f"{stats['median_elapsed_s']:.2f}s | {stats['p90_elapsed_s']:.2f}s | "
            f"{stats['zero_token_retry_count_total']} | {stats['non_zero_exit_count']} |"
        )

    lines.append("")
    lines.append("## Per-regime acceptance (proxy-capture aggregation)")
    lines.append("")
    lines.append(f"Total captured rows: **{per_regime.get('row_count')}**")
    lines.append(f"Aggregate accepted/draft: **{per_regime.get('aggregate_accepted_per_draft_token')}**")
    lines.append("")
    lines.append("| regime | rows | agg accept | p50 accept | p90 accept | p50 decode_tps | tokens out |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for regime, stats in sorted(per_regime.get("regimes", {}).items()):
        agg = stats.get("accepted_per_draft_token_aggregate")
        p50 = stats.get("accepted_per_draft_token_p50")
        p90 = stats.get("accepted_per_draft_token_p90")
        tps = stats.get("decode_tps_p50")
        lines.append(
            f"| {regime} | {stats['row_count']} | "
            f"{agg if agg is None else f'{agg:.3f}'} | "
            f"{p50 if p50 is None else f'{p50:.3f}'} | "
            f"{p90 if p90 is None else f'{p90:.3f}'} | "
            f"{tps if tps is None else f'{tps:.2f}'} | "
            f"{stats['completion_tokens_total']} |"
        )

    lines.append("")
    lines.append("## §6.5 diagnosis (regime-level)")
    diagnoses = []
    regimes = per_regime.get("regimes", {})
    for regime, stats in sorted(regimes.items()):
        agg = stats.get("accepted_per_draft_token_aggregate")
        if not isinstance(agg, (int, float)):
            diagnoses.append(f"- `{regime}`: no spec-decode counters captured")
            continue
        if agg < 0.20:
            diagnoses.append(
                f"- `{regime}`: **low-acceptance** (agg accepted/draft = {agg:.3f} < 0.20); "
                "biggest headroom — Techniques 2/4 should target this regime first"
            )
        elif agg < 0.50:
            diagnoses.append(
                f"- `{regime}`: moderate (agg = {agg:.3f}); SuffixDecoding has some traction "
                "but is not yet pulling its weight — Techniques 1/3 are reasonable next bets"
            )
        else:
            diagnoses.append(
                f"- `{regime}`: strong (agg = {agg:.3f}); SuffixDecoding carrying its weight here"
            )
    if not diagnoses:
        diagnoses.append("- (no regimes captured)")
    lines.extend(diagnoses)

    lines.append("")
    lines.append(f"## Runtime config")
    lines.append("")
    lines.append(f"- runtime_config_hash: `{runtime_config_hash}`")
    lines.append("- live spec_decode: `method=suffix, num_speculative_tokens=12, suffix_decoding_max_tree_depth=32`")
    lines.append("- arctic-inference==0.1.2 installed via prelaunch hook")
    lines.append("- proxy capture path: /tmp/track_b_e2e_proxy_capture/request_metrics.jsonl")
    lines.append(f"- proxy capture writer: TrackBRequestMetricsCapture (lumo.track_b.vllm_request_metrics.v1)")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose the final Track B Round 0 v2 report.")
    parser.add_argument(
        "--v2-round-dir",
        type=Path,
        default=REPO_ROOT / "output" / "track_b_e2e_v2" / "round_0",
    )
    parser.add_argument(
        "--v1-round-summary",
        type=Path,
        default=REPO_ROOT / "output" / "track_b_e2e" / "round_0" / "round_summary.json",
    )
    parser.add_argument(
        "--proxy-capture",
        type=Path,
        default=Path("/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"),
    )
    parser.add_argument("--runtime-config-hash", default="")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: <v2_round_dir>/round_v2_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Default: docs/reports/auto_research/track-b-e2e-round0-v2-report-<date>.md",
    )
    args = parser.parse_args()

    v2_round_dir: Path = args.v2_round_dir
    v2_round_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or (v2_round_dir / "round_v2_report.json")
    out_md = args.out_md or (
        REPO_ROOT
        / "docs"
        / "reports"
        / "auto_research"
        / f"track-b-e2e-round0-v2-report-{datetime.now(UTC).strftime('%Y%m%d')}.md"
    )

    v2_summary = _safe_load_json(v2_round_dir / "round_summary.json")
    v1_summary = _safe_load_json(args.v1_round_summary)
    metadata_rows = _collect_runner_metadata(v2_round_dir)
    per_attempt = _per_attempt_summary(metadata_rows)

    capture_rows = _load_capture_rows(args.proxy_capture)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_track_b_per_regime_acceptance import aggregate  # type: ignore

    per_regime = aggregate(capture_rows)
    runtime_config_hash = args.runtime_config_hash or (
        (v2_summary or {}).get("runtime_config_hash") or ""
    )
    if not runtime_config_hash and metadata_rows:
        for row in metadata_rows:
            candidate = row.get("runtime_config_hash")
            if candidate:
                runtime_config_hash = candidate
                break

    payload = {
        "schema": "lumo.track_b.round_v2_report.v1",
        "generated_at": _now(),
        "runtime_config_hash": runtime_config_hash,
        "v2_round_summary": v2_summary,
        "v1_round_summary": v1_summary,
        "per_attempt": per_attempt,
        "per_regime": per_regime,
        "metadata_row_count": len(metadata_rows),
        "proxy_capture_row_count": len(capture_rows),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = _markdown_report(
        v2_summary=v2_summary,
        v1_summary=v1_summary,
        per_attempt=per_attempt,
        per_regime=per_regime,
        runtime_config_hash=runtime_config_hash,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    print(f"json  -> {out_json}")
    print(f"md    -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
