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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY = "pr-intent-regression-review"
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
BUNDLE_DIR = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY / "workspace_bundle"
REPORT_DIR = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY / "report"
SCORER = REPO_ROOT / "verifiers" / FAMILY / "score_review.py"
VERIFIER_DATA = REPO_ROOT / "verifier_data" / FAMILY
DEFAULT_TIMEOUT = 1800
TOLERANCE = 3.0


def probe_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def codex_prompt() -> str:
    return (
        "Read AGENTS.md in this directory and follow it exactly. "
        "Inspect the PR description, unified diff, flattened diff view, CI snapshot, and the changed repo surfaces. "
        "Author review_input.json at the workspace root, validate it with ./bin/cnb55-pr-review validate review_input.json, "
        "then run ./bin/cnb55-pr-review submit review_input.json. "
        "Do not edit repo/, review/, release_context/, incident_context/, AGENTS.md, Dockerfile, or bin/. "
        "Only write review_input.json and artifacts/*. "
        "Do not use the network."
    )


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    per_variant = {variant: [] for variant in VARIANTS}
    for row in rows:
        per_variant[row["variant"]].append(row)

    summary = {}
    for variant, records in per_variant.items():
        scores = [record["score"] for record in records]
        m_scores = [record["M_training"] for record in records]
        ceiling_hits: dict[str, int] = {}
        for record in records:
            for ceiling in record.get("ceilings_applied", []):
                ceiling_hits[ceiling] = ceiling_hits.get(ceiling, 0) + 1
        summary[variant] = {
            "n": len(records),
            "scores": scores,
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "mean_M_training": statistics.mean(m_scores),
            "stdev_M_training": statistics.stdev(m_scores) if len(m_scores) > 1 else 0.0,
            "ceiling_hits": ceiling_hits,
            "shortcut_runs": sum(1 for record in records if record.get("shortcut_detected")),
        }

    means = [summary[variant]["mean"] for variant in VARIANTS]
    family_mean = statistics.mean(means)
    max_variant_mean = max(means)
    min_variant_mean = min(means)
    monotonic_breaks = []
    for idx in range(len(VARIANTS) - 1):
        left = VARIANTS[idx]
        right = VARIANTS[idx + 1]
        if summary[left]["mean"] + TOLERANCE < summary[right]["mean"]:
            monotonic_breaks.append(
                f"{left} ({summary[left]['mean']:.2f}) < {right} ({summary[right]['mean']:.2f}) beyond +/-{TOLERANCE}"
            )
    gate = {
        "family_mean": family_mean,
        "family_mean_ok": 15.0 <= family_mean <= 25.0,
        "max_variant_mean": max_variant_mean,
        "max_variant_ok": max_variant_mean <= 40.0,
        "min_variant_mean": min_variant_mean,
        "hard_variant_ok": min_variant_mean <= 10.0,
        "monotonic_ok": len(monotonic_breaks) == 0,
        "monotonic_breaks": monotonic_breaks,
    }
    gate["all_pass"] = all(
        [gate["family_mean_ok"], gate["max_variant_ok"], gate["hard_variant_ok"], gate["monotonic_ok"]]
    )
    return {"variants": summary, "gate": gate}


def write_report_text(summary: dict, out_path: Path, *, run_id: str, attempt: str, n: int) -> None:
    lines = [
        f"probe_run_id={run_id}",
        f"family={FAMILY}",
        f"attempt={attempt}",
        f"n={n}",
        "",
        f"{'variant':<32} {'n':>3} {'mean':>7} {'stdev':>7} {'min':>5} {'max':>5}  scores  ceilings",
        "-" * 110,
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        ceilings = ",".join(f"{key}x{value}" for key, value in sorted(row["ceiling_hits"].items())) or "-"
        score_text = ",".join(str(score) for score in row["scores"])
        lines.append(
            f"{variant:<32} {row['n']:>3} {row['mean']:>7.2f} {row['stdev']:>7.2f} "
            f"{row['min']:>5} {row['max']:>5}  [{score_text}]  {ceilings}"
        )

    gate = summary["gate"]
    lines.extend(
        [
            "",
            f"family_mean = {gate['family_mean']:.2f} (target 15-25)",
            f"max_variant_mean = {gate['max_variant_mean']:.2f} (cap 40)",
            f"min_variant_mean = {gate['min_variant_mean']:.2f} (need <= 10)",
            f"monotonic_ok = {gate['monotonic_ok']}",
            f"monotonic_breaks = {gate['monotonic_breaks'] or '[]'}",
            "",
            f"family_mean_ok = {gate['family_mean_ok']}",
            f"max_variant_ok = {gate['max_variant_ok']}",
            f"hard_variant_ok = {gate['hard_variant_ok']}",
            f"all_pass = {gate['all_pass']}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def run_one(
    *,
    variant: str,
    run_index: int,
    run_id: str,
    timeout_seconds: int,
    work_root: Path,
    log_dir: Path,
    attempt_dir: Path,
    runs_jsonl: Path,
    model: str,
    reasoning_effort: str,
) -> dict:
    run_tag = f"{run_id}-{variant}-run{run_index}"
    staged_root = work_root / run_tag
    workspace = staged_root / "workspace"
    results_dir = staged_root / "results"
    result_file = results_dir / "verify_result.json"
    log_file = log_dir / f"{run_tag}.log"
    artifact_dir = attempt_dir / variant / f"run_{run_index:02d}"

    if staged_root.exists():
        shutil.rmtree(staged_root)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUNDLE_DIR / variant, workspace)

    cmd = [
        "codex",
        "exec",
        "--cd",
        str(workspace),
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f'reasoning_effort="{reasoning_effort}"',
        "--ephemeral",
        codex_prompt(),
    ]

    start = time.time()
    codex_exit = 0
    codex_env = os.environ.copy()
    codex_env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_file.open("w") as log_handle:
        try:
            proc = subprocess.run(
                cmd,
                env=codex_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                text=True,
            )
            codex_exit = proc.returncode
        except subprocess.TimeoutExpired:
            codex_exit = 124
            log_handle.write(f"\n[probe-runner] timeout after {timeout_seconds}s\n")
    elapsed = int(time.time() - start)

    score_env = os.environ.copy()
    score_env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result_file),
            "VARIANT_ID": variant,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run(
        [sys.executable, str(SCORER)],
        env=score_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if not result_file.exists():
        result = {
            "score": 0,
            "P_benchmark": 0,
            "M_training": 0.0,
            "raw_score_pre_ceiling": 0,
            "pass": False,
            "shortcut_detected": False,
            "integrity_flag": 0,
            "integrity_rules_fired": [],
            "ceilings_applied": ["missing_verify_result"],
            "milestones": {},
            "breakdown": {},
            "errors": ["scorer did not produce verify_result.json"],
        }
    else:
        result = json.loads(result_file.read_text())

    shutil.copy2(log_file, artifact_dir / "codex.log")
    shutil.copy2(result_file, artifact_dir / "verify_result.json")
    shutil.copytree(workspace / "artifacts", artifact_dir / "artifacts", dirs_exist_ok=True)
    review_input = workspace / "review_input.json"
    if review_input.exists():
        shutil.copy2(review_input, artifact_dir / "review_input.json")

    record = {
        "probe_run_id": run_id,
        "variant": variant,
        "run_index": run_index,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "command": " ".join(cmd),
        "prompt": codex_prompt(),
        "codex_exit": codex_exit,
        "codex_seconds": elapsed,
        "workspace_path": str(workspace),
        "score": int(result.get("score", 0)),
        "P_benchmark": int(result.get("P_benchmark", result.get("score", 0))),
        "M_training": float(result.get("M_training", 0.0)),
        "raw_score_pre_ceiling": int(result.get("raw_score_pre_ceiling", 0)),
        "pass": bool(result.get("pass", False)),
        "shortcut_detected": bool(result.get("shortcut_detected", False)),
        "integrity_flag": int(result.get("integrity_flag", 0)),
        "integrity_rules_fired": list(result.get("integrity_rules_fired", [])),
        "ceilings_applied": list(result.get("ceilings_applied", [])),
        "milestones": dict(result.get("milestones", {})),
        "breakdown": dict(result.get("breakdown", {})),
        "errors": list(result.get("errors", [])),
        "artifact_dir": str(artifact_dir.resolve()),
        "codex_log_path": str((artifact_dir / "codex.log").resolve()),
        "verify_result_path": str((artifact_dir / "verify_result.json").resolve()),
    }
    append_jsonl(runs_jsonl, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--probe-run-id", default=None)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()

    run_id = args.probe_run_id or probe_run_id()
    attempt_dir = REPORT_DIR / args.attempt
    log_dir = attempt_dir / "live_probe_logs"
    runs_jsonl = attempt_dir / "probe_runs.jsonl"
    summary_json = attempt_dir / "probe_summary.json"
    summary_txt = attempt_dir / "probe_report.txt"

    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.keep_work:
        work_root = attempt_dir / "probe_work"
        work_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"{FAMILY}_{run_id}_")
        work_root = Path(temp_ctx.name)

    records: list[dict] = []
    try:
        print(
            f"probe_run_id={run_id} family={FAMILY} attempt={args.attempt} n={args.n} "
            f"variants={args.variants} report={runs_jsonl}"
        )
        for variant in args.variants:
            if not (BUNDLE_DIR / variant).exists():
                print(f"missing variant bundle: {variant}", file=sys.stderr)
                return 2
            for run_index in range(1, args.n + 1):
                rec = run_one(
                    variant=variant,
                    run_index=run_index,
                    run_id=run_id,
                    timeout_seconds=args.timeout,
                    work_root=work_root,
                    log_dir=log_dir,
                    attempt_dir=attempt_dir,
                    runs_jsonl=runs_jsonl,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                )
                records.append(rec)
                print(
                    f"{variant} run {run_index}/{args.n}: "
                    f"score={rec['score']} raw={rec['raw_score_pre_ceiling']} "
                    f"M={rec['M_training']:.4f} pass={rec['pass']} ceilings={rec['ceilings_applied']}"
                )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    summary = summarize(records)
    meta = {
        "probe_run_id": run_id,
        "family": FAMILY,
        "attempt": args.attempt,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "n": args.n,
        "variants": args.variants,
        "runs_jsonl": str(runs_jsonl.resolve()),
        "summary": summary,
    }
    summary_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    write_report_text(summary, summary_txt, run_id=run_id, attempt=args.attempt, n=args.n)
    print(f"done: {runs_jsonl}")
    print(f"summary: {summary_json}")
    print(f"report: {summary_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
