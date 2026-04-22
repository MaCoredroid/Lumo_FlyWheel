#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/candidate-worktree-shootout"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/candidate-worktree-shootout"
SCORER = REPO / "verifiers/candidate-worktree-shootout/score_shootout.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROBE_PROMPT = """Read AGENTS.md and complete the benchmark task in this workspace.
Use two isolated directories under artifacts/comparison/worktrees/ for Candidate A and Candidate B.
Produce the required files under artifacts/comparison/.
Land one coherent final patch in the main workspace and do not modify immutable surfaces.
Return when the workspace satisfies the task."""


def run_once(variant: str, run_idx: int, attempt_dir: Path, timeout_s: int) -> dict:
    run_dir = attempt_dir / variant / f"run_{run_idx:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    workspace = run_dir / "workspace"
    shutil.copytree(WS_BUNDLE / variant, workspace)

    events_file = run_dir / "codex_events.jsonl"
    stderr_file = run_dir / "codex_stderr.log"
    last_message_file = run_dir / "last_message.txt"
    result_file = run_dir / "verify_result.json"
    command_file = run_dir / "command.txt"

    cmd = [
        "codex",
        "exec",
        "--full-auto",
        "--ephemeral",
        "-m",
        "gpt-5.4",
        "-c",
        'reasoning_effort="high"',
        "--cd",
        str(workspace),
        "--output-last-message",
        str(last_message_file),
        "--json",
        PROBE_PROMPT,
    ]
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(" ".join(cmd) + "\n")

    started = time.time()
    timed_out = False
    with events_file.open("w") as stdout_fh, stderr_file.open("w") as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            stderr_fh.write(f"[run_probe] timed out after {timeout_s} seconds\n")
    elapsed = round(time.time() - started, 2)

    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": variant,
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
    verify_result = json.loads(result_file.read_text())
    return {
        "variant": variant,
        "run_idx": run_idx,
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "events_file": events_file.relative_to(FAMILY).as_posix(),
        "stderr_file": stderr_file.relative_to(FAMILY).as_posix(),
        "last_message_file": last_message_file.relative_to(FAMILY).as_posix(),
        "verify_result_file": result_file.relative_to(FAMILY).as_posix(),
        "score": verify_result["P_benchmark"],
        "M_training": verify_result["M_training"],
        "integrity_flag": verify_result["integrity_flag"],
        "ceilings_applied": verify_result["ceilings_applied"],
        "pass": verify_result["pass"],
    }


def monotonic_with_tolerance(values: list[float], tolerance: float = 3.0) -> bool:
    return all(values[i] + tolerance >= values[i + 1] for i in range(len(values) - 1))


def format_table(rows: list[dict]) -> str:
    lines = [
        "| Variant | n | mean | stdev | min | max | scores | ceilings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        ceilings = ", ".join(row["unique_ceilings"]) if row["unique_ceilings"] else "—"
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['mean']:.2f} | {row['stdev']:.2f} | "
            f"{row['min']} | {row['max']} | {row['scores']} | {ceilings} |"
        )
    return "\n".join(lines)


def build_report(attempt_name: str, model: str, results: list[dict], summary_rows: list[dict], gates: dict) -> str:
    return "\n".join(
        [
            f"# {attempt_name} probe report",
            "",
            f"Model: `{model}`",
            f"Runs: `{len(results)}` total (`{summary_rows[0]['n']}` per variant)",
            "",
            format_table(summary_rows),
            "",
            "## Layer A gates",
            "",
            f"- family_mean: `{gates['family_mean']:.2f}`",
            f"- max_variant_mean: `{gates['max_variant_mean']:.2f}`",
            f"- min_variant_mean: `{gates['min_variant_mean']:.2f}`",
            f"- monotonic_within_tolerance: `{gates['monotonic_within_tolerance']}`",
            f"- acceptance_judgment: `{gates['acceptance_judgment']}`",
        ]
    ) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-variant", type=int, default=3)
    ap.add_argument("--attempt-name", default="attempt_02")
    ap.add_argument("--timeout-seconds", type=int, default=1800)
    args = ap.parse_args()

    report_dir = FAMILY / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = report_dir / args.attempt_name
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for variant in VARIANTS:
        for run_idx in range(1, args.runs_per_variant + 1):
            result = run_once(variant, run_idx, attempt_dir, args.timeout_seconds)
            all_results.append(result)

    summary_rows = []
    variant_means = []
    m_training_stdevs = []
    for variant in VARIANTS:
        rows = [row for row in all_results if row["variant"] == variant]
        scores = [row["score"] for row in rows]
        m_scores = [row["M_training"] for row in rows]
        ceilings = sorted({ceiling for row in rows for ceiling in row["ceilings_applied"]})
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        m_stdev = statistics.pstdev(m_scores) if len(m_scores) > 1 else 0.0
        summary_rows.append(
            {
                "variant": variant,
                "n": len(rows),
                "mean": mean,
                "stdev": stdev,
                "min": min(scores),
                "max": max(scores),
                "scores": scores,
                "unique_ceilings": ceilings,
            }
        )
        variant_means.append(mean)
        m_training_stdevs.append(m_stdev)

    gates = {
        "family_mean": statistics.mean(variant_means),
        "max_variant_mean": max(variant_means),
        "min_variant_mean": min(variant_means),
        "monotonic_within_tolerance": monotonic_with_tolerance(variant_means, tolerance=3.0),
    }
    gates["acceptance_judgment"] = (
        "green"
        if (
            15.0 <= gates["family_mean"] <= 25.0
            and gates["max_variant_mean"] <= 40.0
            and gates["min_variant_mean"] <= 10.0
            and gates["monotonic_within_tolerance"]
        )
        else "red"
    )

    payload = {
        "attempt_name": args.attempt_name,
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "runs_per_variant": args.runs_per_variant,
        "prompt": PROBE_PROMPT,
        "exact_command_template": [
            "codex",
            "exec",
            "--full-auto",
            "--ephemeral",
            "-m",
            "gpt-5.4",
            "-c",
            'reasoning_effort="high"',
            "--cd",
            "<workspace>",
            "--output-last-message",
            "<last_message_file>",
            "--json",
            PROBE_PROMPT,
        ],
        "results": all_results,
        "summary": summary_rows,
        "layer_a_gates": gates,
        "current_observed_stdev_M_training": max(m_training_stdevs) if m_training_stdevs else 0.0,
        "generated_at_unix": time.time(),
    }

    results_path = report_dir / f"{args.attempt_name}_probe_results.json"
    report_path = report_dir / f"{args.attempt_name}_probe_report.txt"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path.write_text(build_report(args.attempt_name, "gpt-5.4", all_results, summary_rows, gates))
    print(report_path)
    print(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
