#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/nightly-regression-watch"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
SCORER = REPO / "verifiers/nightly-regression-watch/score_ranking.py"
REPORT_DIR = FAMILY / "report"
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Repair the nightly regression watch in place. "
    "You must run `pytest -q ops_digest/tests` and regenerate "
    "`ops_digest/fixtures/inbox/generated_digest.md` from code before finishing. "
    "Do not create sibling automation files and do not modify any file outside the five allowed repair surfaces."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_codex(workspace: Path, log_path: Path, last_message_path: Path, timeout_seconds: int) -> tuple[int, float]:
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(workspace),
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--model",
        "gpt-5.4",
        "-c",
        'model_reasoning_effort="high"',
        "--ephemeral",
        "--color",
        "never",
        "--output-last-message",
        str(last_message_path),
        PROMPT,
    ]
    started = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_message_seen_at: float | None = None
        while True:
            rc = proc.poll()
            elapsed = time.time() - started
            if rc is not None:
                return rc, elapsed
            if elapsed > timeout_seconds:
                proc.kill()
                log_file.write(f"\n[probe] timeout after {timeout_seconds}s\n")
                return 124, elapsed
            if last_message_path.exists():
                mtime = last_message_path.stat().st_mtime
                if last_message_seen_at is None or mtime > last_message_seen_at:
                    last_message_seen_at = mtime
                elif time.time() - last_message_seen_at >= 8:
                    log_file.write("\n[probe] last message stabilized; terminating codex exec\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                        return 0, time.time() - started
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        return 0, time.time() - started
            time.sleep(2)


def score_workspace(workspace: Path, variant_id: str, result_file: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VARIANT_ID": variant_id,
            "RESULT_FILE": str(result_file),
        }
    )
    subprocess.run(
        [sys.executable, str(SCORER)],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return json.loads(result_file.read_text())


def monotonic_with_tolerance(means: list[float], tolerance: float = 3.0) -> bool:
    for left, right in zip(means, means[1:]):
        if left + tolerance < right:
            return False
    return True


def write_report(
    attempt_id: str,
    jsonl_path: Path,
    report_path: Path,
    command_path: Path,
    records: list[dict],
    variants: list[str],
) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_variant[record["variant"]].append(record)

    per_variant = []
    mean_scores = []
    all_scores = []
    all_m_training = []
    for variant in variants:
        runs = by_variant[variant]
        scores = [run["score"] for run in runs]
        m_values = [run["M_training"] for run in runs]
        all_scores.extend(scores)
        all_m_training.extend(m_values)
        mean = statistics.mean(scores)
        mean_scores.append(mean)
        per_variant.append(
            {
                "variant": variant,
                "scores": scores,
                "mean": mean,
                "min": min(scores),
                "max": max(scores),
                "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "m_training_mean": statistics.mean(m_values),
            }
        )

    family_mean = statistics.mean(all_scores)
    max_variant_mean = max(item["mean"] for item in per_variant)
    min_variant_mean = min(item["mean"] for item in per_variant)
    monotonic_ok = monotonic_with_tolerance(mean_scores, tolerance=3.0)
    acceptance = (
        15.0 <= family_mean <= 25.0
        and max_variant_mean <= 40.0
        and min_variant_mean <= 10.0
        and monotonic_ok
    )
    overall_m_stdev = statistics.pstdev(all_m_training) if len(all_m_training) > 1 else 0.0

    lines = [
        f"# Live probe report — {attempt_id}",
        "",
        f"Run command: `{command_path.read_text().strip()}`",
        f"Probe records: `{jsonl_path.name}`",
        "",
        "## Per-variant results",
        "",
        "| variant | scores | mean | min | max | stdev | mean M_training |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_variant:
        score_list = ", ".join(str(score) for score in item["scores"])
        lines.append(
            f"| {item['variant']} | {score_list} | {item['mean']:.2f} | {item['min']} | {item['max']} | {item['stdev']:.2f} | {item['m_training_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Layer A gate values",
            "",
            f"- family_mean = {family_mean:.2f}",
            f"- max_variant_mean = {max_variant_mean:.2f}",
            f"- min_variant_mean = {min_variant_mean:.2f}",
            f"- monotonic_within_plus_3 = {monotonic_ok}",
            f"- acceptance_judgment = {'green' if acceptance else 'red'}",
            f"- current_observed_stdev_M_training = {overall_m_stdev:.4f}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    summary = {
        "attempt_id": attempt_id,
        "family_mean": round(family_mean, 2),
        "max_variant_mean": round(max_variant_mean, 2),
        "min_variant_mean": round(min_variant_mean, 2),
        "monotonic_within_plus_3": monotonic_ok,
        "acceptance": acceptance,
        "current_observed_stdev_M_training": round(overall_m_stdev, 4),
        "variants": variants,
        "per_variant": per_variant,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", default=f"attempt_live_{utc_now()}")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = REPORT_DIR / "probe_run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = REPORT_DIR / f"{args.attempt_id}_probe_runs.jsonl"
    report_path = REPORT_DIR / f"{args.attempt_id}_probe_report.txt"
    summary_path = REPORT_DIR / f"{args.attempt_id}_summary.json"
    command_path = REPORT_DIR / f"{args.attempt_id}_command.txt"
    command_path.write_text(
        "python3 benchmark_blueprints/families/nightly-regression-watch/tools/probe_family_live.py "
        f"--attempt-id {args.attempt_id} --n {args.n} --timeout-seconds {args.timeout_seconds}"
    )

    records: list[dict] = []
    for variant in args.variants:
        for run_index in range(1, args.n + 1):
            run_tag = f"{args.attempt_id}-{variant}-run{run_index}"
            with tempfile.TemporaryDirectory(prefix=f"{run_tag}_") as tmp:
                work_root = Path(tmp)
                workspace = work_root / "workspace"
                shutil.copytree(WORKSPACE_BUNDLE / variant, workspace)
                result_file = work_root / "verify_result.json"
                last_message_path = logs_dir / f"{run_tag}.last_message.txt"
                log_path = logs_dir / f"{run_tag}.log"
                codex_exit, codex_seconds = run_codex(workspace, log_path, last_message_path, args.timeout_seconds)
                score = score_workspace(workspace, variant, result_file)
                record = {
                    "attempt_id": args.attempt_id,
                    "variant": variant,
                    "run_index": run_index,
                    "codex_exit": codex_exit,
                    "codex_seconds": round(codex_seconds, 2),
                    "score": score["score"],
                    "P_benchmark": score["P_benchmark"],
                    "M_training": score["M_training"],
                    "pass": score["pass"],
                    "integrity_flag": score["integrity_flag"],
                    "ceilings_applied": score["ceilings_applied"],
                    "errors": score["errors"],
                    "milestones": score["milestones"],
                }
                records.append(record)
                with jsonl_path.open("a") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")

    summary = write_report(args.attempt_id, jsonl_path, report_path, command_path, records, list(args.variants))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
