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
FAMILY_ID = "sandbox-policy-ci-drift"
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_repair.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and complete the benchmark task in this workspace only. "
    "Repair the repo so make ci passes and the parser, preview, workflow, config, and release note "
    "all align on the same sandbox and approval-policy contract. "
    "Do not modify any immutable evidence or tests."
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

    rows = []
    for variant in DEFAULT_VARIANTS:
        entries = by_variant.get(variant, [])
        scores = [entry["score"] for entry in entries]
        m_scores = [entry["M_training"] for entry in entries]
        rows.append(
            {
                "variant": variant,
                "n": len(entries),
                "mean": round(float(statistics.mean(scores)), 2) if scores else 0.0,
                "stdev": round(float(statistics.pstdev(scores)), 2) if len(scores) > 1 else 0.0,
                "mean_M_training": round(float(statistics.mean(m_scores)), 4) if m_scores else 0.0,
                "stdev_M_training": round(float(statistics.pstdev(m_scores)), 4) if len(m_scores) > 1 else 0.0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "scores": scores,
                "m_scores": [round(score, 4) for score in m_scores],
                "ceilings": sorted({ceiling for entry in entries for ceiling in entry.get("ceilings_applied", [])}),
                "integrity_hits": sum(int(entry.get("integrity_flag", 0)) for entry in entries),
            }
        )

    means = [row["mean"] for row in rows]
    monotonic_ok = True
    for left, right in zip(rows, rows[1:]):
        if left["mean"] + 3.0 < right["mean"]:
            monotonic_ok = False
            break

    all_m_scores = [entry["M_training"] for entry in records]
    family_mean = round(sum(means) / len(means), 2) if means else 0.0
    return {
        "rows": rows,
        "family_mean": family_mean,
        "family_mean_M_training": round(float(statistics.mean(all_m_scores)), 4) if all_m_scores else 0.0,
        "current_observed_stdev_M_training": round(float(statistics.pstdev(all_m_scores)), 4) if len(all_m_scores) > 1 else 0.0,
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
    probe_root = attempt_dir / "workdirs"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
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
        "codex_command_template": [
            "codex",
            "exec",
            "--full-auto",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--cd",
            "<workspace>",
            "--output-last-message",
            "<last_message_file>",
            "-m",
            "gpt-5.4",
            "-c",
            'reasoning_effort="high"',
            PROMPT,
        ],
    }
    (attempt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records: list[dict] = []
    for variant in args.variants:
        for run_index in range(1, args.n + 1):
            run_tag = f"{variant}-run{run_index}"
            workdir = probe_root / run_tag
            workspace = workdir / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(WORKSPACE_BUNDLE / variant, workspace)
            last_message_path = workdir / "last_message.txt"
            command_txt = workdir / "command.txt"
            command = [
                "timeout",
                str(args.timeout_seconds),
                "codex",
                "exec",
                "--full-auto",
                "--ephemeral",
                "--skip-git-repo-check",
                "--json",
                "--cd",
                str(workspace),
                "--output-last-message",
                str(last_message_path),
                "-m",
                "gpt-5.4",
                "-c",
                'reasoning_effort="high"',
                PROMPT,
            ]
            command_txt.write_text(" ".join(command) + "\n", encoding="utf-8")

            started = time.time()
            codex = run_subprocess(command)
            duration = round(time.time() - started, 2)

            (logs_dir / f"{run_tag}.stdout.jsonl").write_text(codex.stdout, encoding="utf-8")
            (logs_dir / f"{run_tag}.stderr.log").write_text(codex.stderr, encoding="utf-8")

            result_path = workdir / "verify_result.json"
            result = score_workspace(workspace, variant, result_path)
            shutil.copy(result_path, artifacts_dir / f"{run_tag}_verify_result.json")

            for rel, suffix in [
                ("brief/manager_brief.json", "_manager_brief.json"),
                ("brief/manager_brief.md", "_manager_brief.md"),
                ("brief_input.json", "_brief_input.json"),
            ]:
                src = workspace / rel
                if src.exists():
                    shutil.copy(src, artifacts_dir / f"{run_tag}{suffix}")

            record = {
                "attempt": args.attempt,
                "variant": variant,
                "run_index": run_index,
                "codex_exit": codex.returncode,
                "codex_seconds": duration,
                "command": " ".join(command),
                "score": int(result.get("score", 0)),
                "raw_score_pre_ceiling": int(result.get("raw_score_pre_ceiling", 0)),
                "P_benchmark": int(result.get("P_benchmark", 0)),
                "M_training": float(result.get("M_training", 0.0)),
                "pass": bool(result.get("pass", False)),
                "integrity_flag": int(result.get("integrity_flag", 0)),
                "shortcut_detected": bool(result.get("shortcut_detected", False)),
                "ceilings_applied": list(result.get("ceilings_applied", [])),
                "errors": list(result.get("errors", [])),
                "log_file": f"logs/{run_tag}.stdout.jsonl",
                "stderr_file": f"logs/{run_tag}.stderr.log",
                "last_message_file": str(last_message_path.relative_to(attempt_dir)),
                "verify_result_file": f"artifacts/{run_tag}_verify_result.json",
                "brief_input_file": f"artifacts/{run_tag}_brief_input.json" if (workspace / "brief_input.json").exists() else None,
                "brief_json_file": f"artifacts/{run_tag}_manager_brief.json" if (workspace / "brief" / "manager_brief.json").exists() else None,
                "brief_md_file": f"artifacts/{run_tag}_manager_brief.md" if (workspace / "brief" / "manager_brief.md").exists() else None,
            }
            records.append(record)
            with runs_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = summarize_attempt(records)
    (attempt_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"attempt_dir": str(attempt_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
