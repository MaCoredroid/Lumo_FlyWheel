#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from shutil import which


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ID = "workspace-approval-state-threading"
WORKSPACE_ROOT = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY_ID / "workspace_bundle"
VERIFIER_ROOT = REPO_ROOT / "verifier_data" / FAMILY_ID
SCORER = REPO_ROOT / "verifiers" / FAMILY_ID / "score_workspace_approval.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md and complete the task fully in this workspace only. "
    "Run bin/run-visible-tests before finishing. Update code, tests, docs, preview artifact, "
    "and rollout note. Do not modify read-only context files."
)


def run_cmd(cmd: list[str], *, cwd: Path, stdout_path: Path | None = None, stderr_path: Path | None = None, timeout_seconds: int | None = None) -> tuple[int, float]:
    start = time.time()
    stdout_handle = stdout_path.open("w") if stdout_path else subprocess.DEVNULL
    stderr_handle = stderr_path.open("w") if stderr_path else subprocess.DEVNULL
    try:
        completed = subprocess.run(cmd, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, timeout=timeout_seconds)
        rc = completed.returncode
    finally:
        if stdout_path:
            stdout_handle.close()
        if stderr_path:
            stderr_handle.close()
    return rc, round(time.time() - start, 3)


def score_workspace(workspace: Path, variant: str, result_path: Path) -> dict:
    grader_python = which("python3.11") or sys.executable
    env = {
        **dict(Path.home().env if False else {}),
        **{
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_ROOT),
            "RESULT_FILE": str(result_path),
            "VARIANT_ID": variant,
        },
    }
    subprocess.run([grader_python, str(SCORER)], cwd=REPO_ROOT, check=True, env=env)
    return json.loads(result_path.read_text())


def compute_gates(ordered_scores: list[int]) -> dict:
    family_mean = round(sum(ordered_scores) / len(ordered_scores), 2)
    max_score = max(ordered_scores)
    min_score = min(ordered_scores)
    monotonic_ok = True
    for left, right in zip(ordered_scores, ordered_scores[1:]):
        if left + 3 < right:
            monotonic_ok = False
            break
    return {
        "family_mean": family_mean,
        "max_variant_mean": max_score,
        "min_variant_mean": min_score,
        "family_mean_gate": 15 <= family_mean <= 25,
        "max_gate": max_score <= 40,
        "min_gate": min_score <= 10,
        "monotonic_gate": monotonic_ok,
    }


def main() -> None:
    attempt_id = sys.argv[1] if len(sys.argv) > 1 else "attempt_02"
    selected_variants = sys.argv[2:] if len(sys.argv) > 2 else VARIANTS
    probe_root = VERIFIER_ROOT / "live_probes" / attempt_id
    probe_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for variant in selected_variants:
        variant_root = probe_root / variant
        workspace = variant_root / "workspace"
        if variant_root.exists():
            shutil.rmtree(variant_root)
        variant_root.mkdir(parents=True)
        shutil.copytree(WORKSPACE_ROOT / variant, workspace)

        stdout_path = variant_root / "codex_stdout.log"
        stderr_path = variant_root / "codex_stderr.log"
        last_message_path = variant_root / "codex_last_message.txt"

        cmd = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(workspace),
            "-m",
            "gpt-5.4",
            "-o",
            str(last_message_path),
            PROMPT,
        ]
        rc, duration = run_cmd(cmd, cwd=REPO_ROOT, stdout_path=stdout_path, stderr_path=stderr_path, timeout_seconds=600)

        diff_stat_path = variant_root / "diff_stat.txt"
        run_cmd(
            ["git", "diff", "--no-index", "--stat", str(WORKSPACE_ROOT / variant), str(workspace)],
            cwd=REPO_ROOT,
            stdout_path=diff_stat_path,
            stderr_path=variant_root / "diff_stderr.log",
        )

        verify_result_path = variant_root / "verify_result.json"
        scored = score_workspace(workspace, variant, verify_result_path)

        (variant_root / "probe_meta.json").write_text(
            json.dumps(
                {
                    "variant_id": variant,
                    "codex_command": " ".join(cmd),
                    "codex_exit_code": rc,
                    "duration_seconds": duration,
                    "artifact_root": str(variant_root),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{variant}: score={scored['score']} P={scored['P_benchmark']} M={scored['M_training']} "
            f"pass={scored['pass']} exit={rc} duration={duration}",
            flush=True,
        )

    summary_rows = []
    for variant in VARIANTS:
        variant_root = probe_root / variant
        meta_path = variant_root / "probe_meta.json"
        result_path = variant_root / "verify_result.json"
        if not meta_path.exists() or not result_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        result = json.loads(result_path.read_text())
        summary_rows.append(
            {
                "variant_id": variant,
                "codex_command": meta["codex_command"],
                "codex_exit_code": meta["codex_exit_code"],
                "duration_seconds": meta["duration_seconds"],
                "score": result["score"],
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "pass": result["pass"],
                "visible_tests_ok": result["visible_tests_ok"],
                "ceilings_applied": result["ceilings_applied"],
                "integrity_flag": result["integrity_flag"],
                "milestones": result["milestones"],
                "artifact_root": meta["artifact_root"],
            }
        )

    ordered_scores = [row["score"] for row in summary_rows]
    gates = compute_gates(ordered_scores)
    summary = {
        "attempt_id": attempt_id,
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "variants": summary_rows,
        "layer_a_gates": gates,
    }
    summary_path = probe_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        f"# {attempt_id} probe report",
        "",
        f"Model: `gpt-5.4`",
        "Reasoning effort: `high`",
        "",
        "| variant | exit | duration_s | score | P_benchmark | M_training | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['variant_id']} | {row['codex_exit_code']} | {row['duration_seconds']} | "
            f"{row['score']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['pass']} | "
            f"{', '.join(row['ceilings_applied']) or '—'} |"
        )
    report_lines.extend(
        [
            "",
            "## Layer A gates",
            "",
            f"- `family_mean`: {gates['family_mean']} (`pass={gates['family_mean_gate']}`)",
            f"- `max_variant_mean`: {gates['max_variant_mean']} (`pass={gates['max_gate']}`)",
            f"- `min_variant_mean`: {gates['min_variant_mean']} (`pass={gates['min_gate']}`)",
            f"- `monotonic V1>=V2>=V3>=V4>=V5 +/-3`: {gates['monotonic_gate']}",
            "",
            f"Summary artifact: `{summary_path}`",
        ]
    )
    report_path = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY_ID / "report" / f"{attempt_id}_probe_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
