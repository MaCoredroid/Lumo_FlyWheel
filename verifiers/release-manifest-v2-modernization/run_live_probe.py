#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "release-manifest-v2-modernization"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_release_modernization.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Repair the release-path modernization in place, run the visible checks, "
    "and write artifacts/release_smoke_report.json by running "
    "`python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json`. "
    "Do not modify read-only surfaces or anything outside the allowed repair paths."
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def monotonicity(values: list[float], tolerance: int = 3) -> bool:
    return all(values[i] + tolerance >= values[i + 1] for i in range(len(values) - 1))


def summarize_variant(results: list[dict]) -> dict:
    scores = [int(item["score"]) for item in results]
    m_scores = [float(item["M_training"]) for item in results]
    return {
        "n": len(scores),
        "scores": scores,
        "m_training_scores": m_scores,
        "mean": round(statistics.mean(scores), 2),
        "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "mean_M_training": round(statistics.mean(m_scores), 4),
    }


def summarize_family(by_variant: dict[str, list[dict]]) -> dict:
    variant_order = list(by_variant.keys())
    variant_summaries = {variant: summarize_variant(rows) for variant, rows in by_variant.items()}
    means = [variant_summaries[variant]["mean"] for variant in variant_order]
    return {
        "variant_order": variant_order,
        "variants": variant_summaries,
        "family_mean": round(statistics.mean(means), 2),
        "max_variant_mean": max(means),
        "min_variant_mean": min(means),
        "monotonicity_tolerance_3": monotonicity(means, tolerance=3),
        "family_mean_window": [15, 25],
        "max_variant_cap": 40,
        "min_variant_floor": 10,
    }


def assess(summary: dict) -> dict:
    family_mean_ok = 15 <= summary["family_mean"] <= 25
    max_variant_ok = summary["max_variant_mean"] <= 40
    min_variant_ok = summary["min_variant_mean"] <= 10
    monotonic_ok = bool(summary["monotonicity_tolerance_3"])
    return {
        "family_mean_ok": family_mean_ok,
        "max_variant_ok": max_variant_ok,
        "min_variant_ok": min_variant_ok,
        "monotonic_ok": monotonic_ok,
        "all_pass": family_mean_ok and max_variant_ok and min_variant_ok and monotonic_ok,
    }


def write_probe_report(path: Path, command: str, summary: dict, assessment: dict) -> None:
    lines = [
        f"CNB-55 live probe report — {FAMILY_ID}",
        "",
        f"command: {command}",
        "",
        f"family_mean = {summary['family_mean']:.2f}   (window {summary['family_mean_window'][0]}-{summary['family_mean_window'][1]})",
        f"max_variant_mean = {summary['max_variant_mean']:.2f}   (cap {summary['max_variant_cap']})",
        f"min_variant_mean = {summary['min_variant_mean']:.2f}   (must have at least one <= {summary['min_variant_floor']})",
        f"monotonicity within +/-3 = {summary['monotonicity_tolerance_3']}",
        "",
        f"{'variant':<32} {'n':>3} {'mean':>7} {'stdev':>7} {'min':>5} {'max':>5}  scores",
        "-" * 74,
    ]
    for variant in summary["variant_order"]:
        item = summary["variants"][variant]
        lines.append(
            f"{variant:<32} {item['n']:>3} {item['mean']:>7.2f} {item['stdev']:>7.2f} "
            f"{item['min']:>5} {item['max']:>5}  {item['scores']}"
        )
    lines.extend(
        [
            "",
            "Acceptance checks:",
            f"  family_mean in window: {assessment['family_mean_ok']}",
            f"  max variant <= cap: {assessment['max_variant_ok']}",
            f"  at least one variant <= hard floor: {assessment['min_variant_ok']}",
            f"  monotonic V1>=V2>=V3>=V4>=V5 +/-3: {assessment['monotonic_ok']}",
            "",
            f"overall: {'ALL PASS' if assessment['all_pass'] else 'HARDEN NEEDED'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_markdown(attempt_dir: Path, command: str, rows: list[dict], summary: dict, assessment: dict) -> None:
    lines = [
        f"# {attempt_dir.name} live probe",
        "",
        f"- command: `{command}`",
        f"- family mean: `{summary['family_mean']}`",
        f"- max variant mean: `{summary['max_variant_mean']}`",
        f"- min variant mean: `{summary['min_variant_mean']}`",
        f"- monotonicity within +/-3: `{summary['monotonicity_tolerance_3']}`",
        f"- Layer A judgment: `{'passed' if assessment['all_pass'] else 'not yet passed'}`",
        "",
        "| variant | run | codex_exit | seconds | score | M_training | pass | integrity | ceilings | errors |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in rows:
        ceilings = ",".join(item["ceilings"]) or "—"
        errors = ",".join(item["errors"]) or "—"
        lines.append(
            f"| {item['variant']} | {item['run_index']} | {item['codex_exit']} | {item['seconds']} | "
            f"{item['score']} | {item['M_training']:.2f} | {item['pass']} | {item['integrity_flag']} | "
            f"{ceilings} | {errors} |"
        )
    lines.extend(
        [
            "",
            "## Per-Variant Means",
            "",
            "| variant | mean | stdev | min | max | scores |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for variant in summary["variant_order"]:
        item = summary["variants"][variant]
        lines.append(
            f"| {variant} | {item['mean']:.2f} | {item['stdev']:.2f} | {item['min']} | {item['max']} | {item['scores']} |"
        )
    (attempt_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_variant(variant: str, run_index: int, timeout_seconds: int, attempt_dir: Path, model: str, reasoning_effort: str) -> dict:
    logs_dir = attempt_dir / "codex_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{variant}-run{run_index:02d}.log"
    work_root = Path(tempfile.mkdtemp(prefix=f"{FAMILY_ID}_{variant}_run{run_index:02d}_"))
    ws = work_root / "workspace"
    shutil.copytree(WORKSPACE_BUNDLE / variant, ws)
    result_path = work_root / "verify_result.json"

    start = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            [
                "timeout",
                str(timeout_seconds),
                "codex",
                "exec",
                "--cd",
                str(ws),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--ephemeral",
                "--model",
                model,
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                PROMPT,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = round(time.time() - start, 2)

    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result_path),
            "VARIANT_ID": variant,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run([os.environ.get("PYTHON", "python3"), str(SCORER)], env=env, check=True)
    result = load_json(result_path)

    return {
        "variant": variant,
        "run_index": run_index,
        "codex_exit": proc.returncode,
        "seconds": elapsed,
        "score": int(result["score"]),
        "P_benchmark": int(result["P_benchmark"]),
        "M_training": float(result["M_training"]),
        "pass": bool(result["pass"]),
        "integrity_flag": int(result["integrity_flag"]),
        "ceilings": list(result.get("ceilings_applied", [])),
        "errors": list(result.get("errors", [])),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    attempt_dir = FAMILY / "report" / args.attempt
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    by_variant: dict[str, list[dict]] = {variant: [] for variant in args.variants}
    for variant in args.variants:
        for run_index in range(1, args.runs + 1):
            row = run_variant(
                variant=variant,
                run_index=run_index,
                timeout_seconds=args.timeout_seconds,
                attempt_dir=attempt_dir,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            rows.append(row)
            by_variant[variant].append(row)

    command = (
        f"python3 verifiers/{FAMILY_ID}/run_live_probe.py --attempt {args.attempt} "
        f"--timeout-seconds {args.timeout_seconds} --runs {args.runs} "
        f"--model {args.model} --reasoning-effort {args.reasoning_effort} "
        f"--variants {' '.join(args.variants)}"
    )

    probe_runs_path = attempt_dir / "probe_runs.jsonl"
    with probe_runs_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize_family(by_variant)
    assessment = assess(summary)
    payload = {
        "family_id": FAMILY_ID,
        "attempt": args.attempt,
        "command": command,
        "prompt": PROMPT,
        "rows": rows,
        "summary": summary,
        "assessment": assessment,
    }
    (attempt_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(attempt_dir, command, rows, summary, assessment)
    write_probe_report(FAMILY / "report" / f"{args.attempt}_probe_report.txt", command, summary, assessment)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
