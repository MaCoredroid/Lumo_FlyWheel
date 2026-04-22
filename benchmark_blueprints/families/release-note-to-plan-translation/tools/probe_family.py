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

REPO = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "release-note-to-plan-translation"
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "release-note-to-plan-translation"
SCORER = REPO / "verifiers" / "release-note-to-plan-translation" / "score_ranking.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Author brief_input.json at the workspace root and run "
    "./bin/cnb55-brief submit brief_input.json to produce brief/manager_brief.json. "
    "Do not modify any file outside brief/."
)


def run_subprocess(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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
            "CNB55_SEED": "42",
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
        raise RuntimeError(f"scorer failed for {variant}: {scorer.stderr}")
    return json.loads(result_path.read_text())


def summarize_attempt(records: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(record)

    summary_rows = []
    for variant in DEFAULT_VARIANTS:
        rows = by_variant.get(variant, [])
        scores = [row["score"] for row in rows]
        m_scores = [row["M_training"] for row in rows]
        means = float(statistics.mean(scores)) if scores else 0.0
        stdev = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
        m_mean = float(statistics.mean(m_scores)) if m_scores else 0.0
        m_stdev = float(statistics.pstdev(m_scores)) if len(m_scores) > 1 else 0.0
        summary_rows.append(
            {
                "variant": variant,
                "n": len(rows),
                "mean": round(means, 2),
                "stdev": round(stdev, 2),
                "mean_M_training": round(m_mean, 4),
                "stdev_M_training": round(m_stdev, 4),
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "scores": scores,
                "m_scores": [round(score, 4) for score in m_scores],
                "ceilings": sorted({ceil for row in rows for ceil in row.get("ceilings_applied", [])}),
                "integrity_hits": sum(int(row.get("integrity_flag", 0)) for row in rows),
            }
        )

    means = [row["mean"] for row in summary_rows]
    all_m_scores = [row["M_training"] for row in records]
    family_mean = round(sum(means) / len(means), 2) if means else 0.0
    family_m_mean = round(float(statistics.mean(all_m_scores)), 4) if all_m_scores else 0.0
    family_m_stdev = round(float(statistics.pstdev(all_m_scores)), 4) if len(all_m_scores) > 1 else 0.0
    tolerance = 3.0
    monotonic_ok = True
    for left, right in zip(summary_rows, summary_rows[1:]):
        if left["mean"] + tolerance < right["mean"]:
            monotonic_ok = False
            break

    return {
        "rows": summary_rows,
        "family_mean": family_mean,
        "family_mean_M_training": family_m_mean,
        "current_observed_stdev_M_training": family_m_stdev,
        "max_variant_mean": max((row["mean"] for row in summary_rows), default=0.0),
        "min_variant_mean": min((row["mean"] for row in summary_rows), default=0.0),
        "acceptance": {
            "family_mean_window": 15.0 <= family_mean <= 25.0,
            "max_variant_le_40": max((row["mean"] for row in summary_rows), default=0.0) <= 40.0,
            "at_least_one_variant_le_10": min((row["mean"] for row in summary_rows), default=0.0) <= 10.0,
            "monotonic_with_tolerance_3": monotonic_ok,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    attempt_dir = FAMILY_ROOT / "report" / args.attempt
    logs_dir = attempt_dir / "logs"
    artifacts_dir = attempt_dir / "artifacts"
    probe_root = VERIFIER_DATA / "_probe_workdirs" / args.attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    probe_root.mkdir(parents=True, exist_ok=True)

    runs_jsonl = attempt_dir / "probe_runs.jsonl"
    metadata = {
        "attempt": args.attempt,
        "created_at_epoch": int(time.time()),
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "n": args.n,
        "variants": args.variants,
        "prompt": PROMPT,
        "codex_command": [
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
            PROMPT,
        ],
    }
    (attempt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    records: list[dict] = []
    for variant in args.variants:
        for run_index in range(1, args.n + 1):
            run_tag = f"{variant}-run{run_index}"
            workdir = probe_root / run_tag
            workspace = workdir / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
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
                    PROMPT,
                ]
            )
            duration = round(time.time() - started, 2)

            (logs_dir / f"{run_tag}.log").write_text(
                codex.stdout + "\n--- STDERR ---\n" + codex.stderr,
                encoding="utf-8",
            )

            result_path = workdir / "verify_result.json"
            result = score_workspace(workspace, variant, result_path)
            shutil.copy(result_path, artifacts_dir / f"{run_tag}_verify_result.json")
            brief_json = workspace / "brief" / "manager_brief.json"
            brief_md = workspace / "brief" / "manager_brief.md"
            if brief_json.exists():
                shutil.copy(brief_json, artifacts_dir / f"{run_tag}_manager_brief.json")
            if brief_md.exists():
                shutil.copy(brief_md, artifacts_dir / f"{run_tag}_manager_brief.md")

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
                "log_file": f"logs/{run_tag}.log",
                "verify_result_file": f"artifacts/{run_tag}_verify_result.json",
                "brief_json_file": f"artifacts/{run_tag}_manager_brief.json" if brief_json.exists() else None,
                "brief_md_file": f"artifacts/{run_tag}_manager_brief.md" if brief_md.exists() else None,
            }
            records.append(record)
            with runs_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

            shutil.rmtree(workdir, ignore_errors=True)

    summary = summarize_attempt(records)
    (attempt_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"attempt_dir": str(attempt_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
