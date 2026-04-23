#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "sqlalchemy-2-session-modernization"
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "sqlalchemy-2-session-modernization"
SCORER = REPO / "verifiers" / "sqlalchemy-2-session-modernization" / "score_sqlalchemy_session_modernization.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Inspect the current code and docs before editing. "
    "Fix the SQLAlchemy 2 modernization task in this workspace. "
    "Keep edits within the allowed files from AGENTS.md. "
    "Run `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py` before you finish. "
    "Do not modify tests, seed data, or contextual note files."
)


def run_subprocess(
    command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def score_workspace(workspace: Path, variant: str, result_path: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result_path),
            "VARIANT_ID": variant,
        }
    )
    scorer = subprocess.run(
        [sys.executable, str(SCORER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if scorer.returncode != 0:
        raise RuntimeError(
            f"scorer failed for {variant}: stdout={scorer.stdout}\nstderr={scorer.stderr}"
        )
    return json.loads(result_path.read_text())


def summarize_attempt(records: list[dict], ordered_variants: list[str]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(record)

    rows = []
    for variant in ordered_variants:
        samples = by_variant.get(variant, [])
        p_scores = [int(sample["P_benchmark"]) for sample in samples]
        m_scores = [float(sample["M_training"]) for sample in samples]
        mean = float(statistics.mean(p_scores)) if p_scores else 0.0
        stdev = float(statistics.pstdev(p_scores)) if len(p_scores) > 1 else 0.0
        mean_m = float(statistics.mean(m_scores)) if m_scores else 0.0
        stdev_m = float(statistics.pstdev(m_scores)) if len(m_scores) > 1 else 0.0
        rows.append(
            {
                "variant": variant,
                "n": len(samples),
                "mean": round(mean, 2),
                "stdev": round(stdev, 2),
                "mean_M_training": round(mean_m, 4),
                "stdev_M_training": round(stdev_m, 4),
                "min": min(p_scores) if p_scores else 0,
                "max": max(p_scores) if p_scores else 0,
                "scores": p_scores,
                "m_scores": [round(score, 4) for score in m_scores],
                "ceilings": sorted(
                    {
                        ceiling
                        for sample in samples
                        for ceiling in sample.get("ceilings_applied", [])
                    }
                ),
                "integrity_hits": sum(
                    int(sample.get("integrity_flag", 0)) for sample in samples
                ),
            }
        )

    means = [row["mean"] for row in rows]
    all_m_scores = [float(record["M_training"]) for record in records]
    family_mean = round(sum(means) / len(means), 2) if means else 0.0
    family_m_mean = round(float(statistics.mean(all_m_scores)), 4) if all_m_scores else 0.0
    family_m_stdev = (
        round(float(statistics.pstdev(all_m_scores)), 4) if len(all_m_scores) > 1 else 0.0
    )

    tolerance = 3.0
    monotonic_ok = True
    for left, right in zip(rows, rows[1:]):
        if left["mean"] + tolerance < right["mean"]:
            monotonic_ok = False
            break

    return {
        "rows": rows,
        "family_mean": family_mean,
        "family_mean_M_training": family_m_mean,
        "current_observed_stdev_M_training": family_m_stdev,
        "max_variant_mean": max((row["mean"] for row in rows), default=0.0),
        "min_variant_mean": min((row["mean"] for row in rows), default=0.0),
        "acceptance": {
            "family_mean_window": 15.0 <= family_mean <= 25.0,
            "max_variant_le_40": max((row["mean"] for row in rows), default=0.0) <= 40.0,
            "at_least_one_variant_le_10": min((row["mean"] for row in rows), default=0.0) <= 10.0,
            "monotonic_with_tolerance_3": monotonic_ok,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()

    attempt_dir = FAMILY_ROOT / "report" / args.attempt
    logs_dir = attempt_dir / "logs"
    artifacts_dir = attempt_dir / "artifacts"
    workdirs_root = VERIFIER_DATA / "_probe_workdirs" / args.attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    workdirs_root.mkdir(parents=True, exist_ok=True)

    runs_jsonl = attempt_dir / "probe_runs.jsonl"
    metadata = {
        "attempt": args.attempt,
        "created_at_epoch": int(time.time()),
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "n": args.n,
        "variants": args.variants,
        "prompt": PROMPT,
        "timeout_seconds": args.timeout_seconds,
        "codex_command": [
            "timeout",
            str(args.timeout_seconds),
            "codex",
            "exec",
            "--cd",
            "<workspace>",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--color",
            "never",
            "--ephemeral",
            "--model",
            "gpt-5.4",
            "-c",
            'model_reasoning_effort="high"',
            "-o",
            "<last_message_file>",
            PROMPT,
        ],
        "scorer_command": [
            sys.executable,
            str(SCORER),
        ],
    }
    (attempt_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    records: list[dict] = []
    for variant in args.variants:
        for run_index in range(1, args.n + 1):
            run_tag = f"{variant}-run{run_index}"
            run_root = workdirs_root / run_tag
            workspace = run_root / "workspace"
            last_message_file = run_root / "last_message.txt"
            if run_root.exists():
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(WORKSPACE_BUNDLE / variant, workspace)

            started = time.time()
            codex = run_subprocess(
                [
                    "timeout",
                    str(args.timeout_seconds),
                    "codex",
                    "exec",
                    "--cd",
                    str(workspace),
                    "--skip-git-repo-check",
                    "--sandbox",
                    "workspace-write",
                    "--color",
                    "never",
                    "--ephemeral",
                    "--model",
                    "gpt-5.4",
                    "-c",
                    'model_reasoning_effort="high"',
                    "-o",
                    str(last_message_file),
                    PROMPT,
                ]
            )
            duration = round(time.time() - started, 2)

            log_path = logs_dir / f"{run_tag}.log"
            log_path.write_text(
                codex.stdout + "\n--- STDERR ---\n" + codex.stderr,
                encoding="utf-8",
            )

            result_path = run_root / "verify_result.json"
            result = score_workspace(workspace, variant, result_path)
            shutil.copy(result_path, artifacts_dir / f"{run_tag}_verify_result.json")
            if last_message_file.exists():
                shutil.copy(
                    last_message_file,
                    artifacts_dir / f"{run_tag}_last_message.txt",
                )

            saved_files: dict[str, str] = {}
            for rel in (
                "app/repository.py",
                "app/worker.py",
                "app/admin_cli.py",
                "app/api.py",
                "docs/deploy/sqlalchemy2-cutover.md",
            ):
                path = workspace / rel
                if path.exists():
                    safe_name = rel.replace("/", "__")
                    out = artifacts_dir / f"{run_tag}_{safe_name}"
                    shutil.copy(path, out)
                    saved_files[rel] = str(out.relative_to(attempt_dir))

            record = {
                "attempt": args.attempt,
                "variant": variant,
                "run_index": run_index,
                "codex_exit": codex.returncode,
                "codex_seconds": duration,
                "score": int(result.get("score", 0)),
                "raw_score_pre_ceiling": int(result.get("raw_score_pre_ceiling", 0)),
                "P_benchmark": int(result.get("P_benchmark", 0)),
                "M_training": float(result.get("M_training", 0.0)),
                "pass": bool(result.get("pass", False)),
                "integrity_flag": int(result.get("integrity_flag", 0)),
                "shortcut_detected": bool(result.get("shortcut_detected", False)),
                "ceilings_applied": list(result.get("ceilings_applied", [])),
                "errors": list(result.get("errors", [])),
                "log_file": str(log_path.relative_to(attempt_dir)),
                "verify_result_file": f"artifacts/{run_tag}_verify_result.json",
                "last_message_file": (
                    f"artifacts/{run_tag}_last_message.txt"
                    if last_message_file.exists()
                    else None
                ),
                "saved_files": saved_files,
            }
            with runs_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)

    summary = summarize_attempt(records, args.variants)
    (attempt_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(attempt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
