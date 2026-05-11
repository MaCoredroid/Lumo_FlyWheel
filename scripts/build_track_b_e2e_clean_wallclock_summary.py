#!/usr/bin/env python3
"""Round 4a-and-later: produce a sibling clean-wallclock summary alongside the
operational round_summary.json.

Two numbers, each answering a different question:

  operational_wallclock = sum/median of every measured attempt's elapsed_s
                          (matches existing round_summary.json semantics)
  clean_wallclock       = sum/median over only attempts where Codex 0.128.0's
                          zero-token quirk did NOT fire (zero_token_retry_count
                          == 0). For task-level median, computed across that
                          task's clean attempts; for sample-level median,
                          computed across per-task clean medians (parallel
                          to operational median's task-then-sample structure).
                          Tasks with zero clean measured attempts are listed
                          but excluded from the sample median.

The clean number is the right anchor for technique comparison (Round 4b+
drafter work) because it doesn't bake in cross-round variance from a known
upstream Codex bug. The operational number is the right anchor for
deployability decisions because it's what a user actually experiences.

Caveat documented inline: "clean" still includes vLLM-side prefill cost paid
on the failed-retry codex calls (vLLM has no signal Codex disconnected mid-
SSE). So clean isn't purely "no quirk overhead"; it's "no quirk overhead
visible in the per-attempt wallclock the runner records." This is the right
honest framing because (a) the runner's elapsed_s IS user-visible cost, and
(b) the failed-retry's vLLM prefill is a server-side artifact a user wouldn't
attribute to the successful attempt's wallclock.

Schema: lumo.track_b.e2e_round_summary_clean.v1
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "lumo.track_b.e2e_round_summary_clean.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _attempt_produced_output(run_dir: Path) -> bool:
    """Authoritative 'did codex emit any tokens' check via codex_stdout.log.

    Independent of retry_count, which is unreliable when --zero-token-retries
    was 0 (v3 case): the runner never tried again, but codex still emitted a
    turn.completed{output_tokens:0}. We need to know whether the attempt
    actually produced anything.
    """

    stdout_path = run_dir / "codex_stdout.log"
    if not stdout_path.is_file():
        return False
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("type") == "turn.completed":
            usage = row.get("usage") or {}
            if int(usage.get("output_tokens") or 0) > 0:
                return True
    return False


def _measured_attempts(round_dir: Path) -> dict[str, list[dict[str, Any]]]:
    per_task: dict[str, list[dict[str, Any]]] = {}
    for meta_path in sorted(round_dir.glob("*__*/run_*/runner_metadata.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        family = m.get("family")
        variant = m.get("variant")
        attempt = int(m.get("attempt", 0))
        if not family or not variant or attempt < 2:
            # Skip: attempt 1 is cold-discarded by --discard-cold-attempt-exit policy.
            continue
        task_id = f"{family}/{variant}"
        per_task.setdefault(task_id, []).append({
            "attempt": attempt,
            "elapsed_s": float(m.get("elapsed_s") or 0.0),
            "zero_token_retry_count": int(m.get("zero_token_retry_count") or 0),
            "codex_exit_code": m.get("codex_exit_code"),
            "produced_output": _attempt_produced_output(meta_path.parent),
            "meta_path": str(meta_path.relative_to(round_dir)),
        })
    return per_task


def _classify(attempt: dict[str, Any], max_retries: int) -> str:
    """Cohort classifier. Authoritative 'did real work happen' signal is
    `produced_output` (parsed from codex_stdout.log). zero_token_retry_count
    refines whether the success/failure happened on the first try."""
    produced = attempt["produced_output"]
    rc = attempt["zero_token_retry_count"]
    if produced and rc == 0:
        return "clean"
    if produced and 0 < rc <= max_retries:
        return "retry_recovered"
    if not produced and rc >= max_retries:
        return "retry_exhausted"
    if not produced and rc == 0:
        # v3 case: --zero-token-retries was 0, runner never retried, codex
        # emitted turn.completed{output_tokens:0}. Treat as zero-token failure.
        return "zero_token_no_retry"
    # Defensive: not produced, 0 < rc < max_retries
    # Should not happen — runner only stops retrying on success or max-out.
    return "anomaly"


def build_summary(round_dir: Path, max_retries: int = 3) -> dict[str, Any]:
    per_task = _measured_attempts(round_dir)
    per_task_rows: list[dict[str, Any]] = []
    cohort_counts = {
        "clean": 0,
        "retry_recovered": 0,
        "retry_exhausted": 0,
        "zero_token_no_retry": 0,
        "anomaly": 0,
    }
    operational_per_task_medians: list[float] = []
    clean_per_task_medians: list[float] = []
    operational_attempts_sum = 0.0  # sum of EVERY measured attempt's elapsed
    clean_attempts_sum = 0.0        # sum of every CLEAN measured attempt's elapsed
    tasks_with_no_clean: list[str] = []

    for task_id, attempts in sorted(per_task.items()):
        for a in attempts:
            cohort_counts[_classify(a, max_retries)] += 1
        op_elapsed = [a["elapsed_s"] for a in attempts]
        op_med = statistics.median(op_elapsed) if op_elapsed else None
        operational_attempts_sum += sum(op_elapsed)
        if op_med is not None:
            operational_per_task_medians.append(op_med)

        clean_attempts = [a for a in attempts if _classify(a, max_retries) == "clean"]
        clean_elapsed = [a["elapsed_s"] for a in clean_attempts]
        clean_med = statistics.median(clean_elapsed) if clean_elapsed else None
        clean_attempts_sum += sum(clean_elapsed)
        if clean_med is not None:
            clean_per_task_medians.append(clean_med)
        else:
            tasks_with_no_clean.append(task_id)

        per_task_cohorts = [_classify(a, max_retries) for a in attempts]
        per_task_rows.append({
            "task_id": task_id,
            "measured_attempt_count": len(attempts),
            "clean_attempt_count": per_task_cohorts.count("clean"),
            "retry_recovered_attempt_count": per_task_cohorts.count("retry_recovered"),
            "retry_exhausted_attempt_count": per_task_cohorts.count("retry_exhausted"),
            "zero_token_no_retry_attempt_count": per_task_cohorts.count("zero_token_no_retry"),
            "operational_median_wallclock_s": round(op_med, 3) if op_med is not None else None,
            "clean_median_wallclock_s": round(clean_med, 3) if clean_med is not None else None,
            "operational_attempt_elapsed_s": [round(e, 3) for e in op_elapsed],
            "clean_attempt_elapsed_s": [round(e, 3) for e in clean_elapsed],
        })

    operational_sample_median = (
        statistics.median(operational_per_task_medians) if operational_per_task_medians else None
    )
    clean_sample_median = (
        statistics.median(clean_per_task_medians) if clean_per_task_medians else None
    )
    # Aggregate semantics matched to existing round_summary.json — sum of
    # per-task medians (proxies "total wallclock for one run-through of the
    # sample at the per-task representative cost"). Also expose
    # sum_of_attempts for users who want raw total compute.
    operational_aggregate_of_medians = sum(operational_per_task_medians)
    clean_aggregate_of_medians = sum(clean_per_task_medians)

    # For quirk_overhead: we want apples-to-apples (same task set on both
    # sides of the subtraction). Sum operational medians restricted to the
    # tasks that have a clean median.
    clean_eligible_task_ids = {row["task_id"] for row in per_task_rows if row["clean_median_wallclock_s"] is not None}
    operational_aggregate_clean_eligible = sum(
        row["operational_median_wallclock_s"] or 0.0
        for row in per_task_rows
        if row["task_id"] in clean_eligible_task_ids
    )

    quirk_overhead_aggregate_s = (
        operational_aggregate_clean_eligible - clean_aggregate_of_medians
        if clean_per_task_medians else None
    )
    quirk_overhead_pct_of_operational = (
        100.0 * quirk_overhead_aggregate_s / operational_aggregate_clean_eligible
        if quirk_overhead_aggregate_s is not None and operational_aggregate_clean_eligible > 0
        else None
    )

    return {
        "schema": SCHEMA,
        "computed_at": _now(),
        "round_dir": str(round_dir),
        "max_retries_assumed": max_retries,
        "task_count": len(per_task),
        "tasks_with_no_clean_attempt": tasks_with_no_clean,
        "tasks_with_clean_attempt_count": len(per_task) - len(tasks_with_no_clean),
        "cohort_counts": cohort_counts,
        "cohort_pct_of_measured_attempts": {
            k: round(100.0 * v / max(sum(cohort_counts.values()), 1), 1)
            for k, v in cohort_counts.items()
        },
        "operational": {
            "sample_median_wallclock_s": round(operational_sample_median, 3) if operational_sample_median is not None else None,
            "aggregate_wallclock_s": round(operational_aggregate_of_medians, 3),
            "sum_of_all_measured_attempts_s": round(operational_attempts_sum, 3),
            "definition": (
                "sample_median = median across per-task medians (matches existing "
                "round_summary.json semantics). aggregate = sum of per-task medians "
                "(proxies sample wallclock for one pass through 13 tasks at per-task "
                "representative cost). sum_of_all_measured_attempts = total runner "
                "subprocess seconds (3 attempts/task × 13 tasks)."
            ),
        },
        "clean": {
            "sample_median_wallclock_s": round(clean_sample_median, 3) if clean_sample_median is not None else None,
            "aggregate_wallclock_s": round(clean_aggregate_of_medians, 3),
            "sum_of_all_clean_attempts_s": round(clean_attempts_sum, 3),
            "definition": (
                "Same aggregation as operational, but per-task medians are computed "
                "only over attempts where zero_token_retry_count==0. Tasks with zero "
                "clean attempts are excluded from sample_median and from aggregate. "
                "Anchors technique comparison without quirk-incidence variance."
            ),
            "caveat": (
                "'Clean' means 'no quirk overhead visible in this attempt's runner "
                "elapsed_s'. The vLLM server still pays prefill cost on failed-retry "
                "codex calls in OTHER attempts; that server-side cost is not "
                "attributed back to clean attempts."
            ),
            "tasks_excluded_from_sample_median": tasks_with_no_clean,
        },
        "quirk_overhead": {
            "aggregate_s": round(quirk_overhead_aggregate_s, 3) if quirk_overhead_aggregate_s is not None else None,
            "pct_of_operational_aggregate": round(quirk_overhead_pct_of_operational, 2) if quirk_overhead_pct_of_operational is not None else None,
            "operational_aggregate_clean_eligible_s": round(operational_aggregate_clean_eligible, 3),
            "definition": (
                "Apples-to-apples: sum operational per-task medians restricted "
                "to the same task set as clean (i.e., excluding tasks with zero "
                "clean attempts), then subtract clean.aggregate_wallclock_s. "
                "Approximates user-visible cost of the Codex 0.128.0 zero-token "
                "quirk on a per-task representative basis."
            ),
        },
        "per_task": per_task_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--round-dir", required=True, help="e.g. output/track_b_e2e_v4a/round_0")
    p.add_argument("--out", default="", help="Output path (default: <round-dir>/round_summary_clean.json)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="--zero-token-retries default the runner used; classification of retry_exhausted depends on this")
    args = p.parse_args()
    round_dir = Path(args.round_dir).resolve()
    if not round_dir.is_dir():
        raise SystemExit(f"--round-dir not found: {round_dir}")
    summary = build_summary(round_dir, max_retries=args.max_retries)
    out = Path(args.out) if args.out else round_dir / "round_summary_clean.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Compact stdout summary
    print(json.dumps({
        "out": str(out),
        "task_count": summary["task_count"],
        "tasks_with_no_clean": len(summary["tasks_with_no_clean_attempt"]),
        "cohort_counts": summary["cohort_counts"],
        "operational_sample_median_s": summary["operational"]["sample_median_wallclock_s"],
        "clean_sample_median_s": summary["clean"]["sample_median_wallclock_s"],
        "operational_aggregate_all_tasks_s": summary["operational"]["aggregate_wallclock_s"],
        "operational_aggregate_clean_eligible_s": summary["quirk_overhead"]["operational_aggregate_clean_eligible_s"],
        "clean_aggregate_s": summary["clean"]["aggregate_wallclock_s"],
        "quirk_overhead_aggregate_s": summary["quirk_overhead"]["aggregate_s"],
        "quirk_overhead_pct": summary["quirk_overhead"]["pct_of_operational_aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
