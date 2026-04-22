#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[2]
FAMILY_ID = "transcript-merge-regression"
VARIANT_ORDER = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_transcript_merge.py"
SHARED_PROBE_REPORT = REPO / "scripts" / "probe_report.py"


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, stdout=None, stderr=None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
        check=check,
    )


def stage_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def codex_prompt() -> str:
    return (
        "Read AGENTS.md and follow it exactly. "
        "Start by inspecting replay/merge.py, replay/render.py, replay/incident_summary.py, "
        "artifacts/sessions/visible_collision_part1.jsonl, artifacts/sessions/visible_collision_part2.jsonl, "
        "artifacts/sessions/debug_after_completion.jsonl, and reports/incidents/transcript-merge.md. "
        "Then repair the reducer invariant in replay/merge.py, keep replay/render.py honest, "
        "treat debug-only post-completion noise differently from legitimate deferred tool output after completion, "
        "update replay/incident_summary.py so summary semantics stay tied to merged events, "
        "and update reports/incidents/transcript-merge.md with the real bug boundary. "
        "run python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary, "
        "and stop after the workspace is in its best final state. "
        "Do not modify artifacts/sessions, tests/locked, or any file outside the intended task files."
    )


def score_workspace(workspace: Path, variant: str, result_file: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
        }
    )
    run_cmd([sys.executable, str(SCORER)], env=env, check=False)
    return json.loads(result_file.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-id", default="attempt_02_live_probe")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=1800)
    ap.add_argument("--probe-root", default="/tmp/transcript_merge_probe")
    ap.add_argument("--variants", nargs="*", default=VARIANT_ORDER)
    args = ap.parse_args()

    probe_run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_dir = FAMILY / "report"
    attempt_dir = report_dir / args.attempt_id
    logs_dir = attempt_dir / "probe_run_logs"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = attempt_dir / "probe_runs.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    print(f"probe_run_id={probe_run_id}")
    print(f"attempt_dir={attempt_dir}")
    print(f"variants={args.variants}")
    print(f"n={args.n}")

    for variant in args.variants:
        src = WORKSPACE_BUNDLE / variant
        if not src.is_dir():
            raise SystemExit(f"missing workspace bundle: {src}")
        for run_index in range(1, args.n + 1):
            run_tag = f"{probe_run_id}-{variant}-run{run_index}"
            work_root = Path(args.probe_root) / run_tag
            workspace = work_root / "workspace"
            results = work_root / "results"
            stage_workspace(src, workspace)
            results.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{run_tag}.log"
            last_message = logs_dir / f"{run_tag}.final.txt"
            started = time.time()
            cmd = [
                "codex",
                "exec",
                "--model",
                "gpt-5.4",
                "-c",
                'model_reasoning_effort="high"',
                "--cd",
                str(workspace),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--ephemeral",
                "--output-last-message",
                str(last_message),
                codex_prompt(),
            ]
            print(f"=== {variant} run {run_index}/{args.n} ({run_tag}) ===")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with log_path.open("w") as log_file:
                proc = subprocess.run(
                    ["timeout", str(args.timeout_seconds), *cmd],
                    cwd=workspace,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = int(time.time() - started)
            result_file = results / "verify_result.json"
            scored = score_workspace(workspace, variant, result_file)
            rec = {
                "probe_run_id": probe_run_id,
                "variant": variant,
                "run_index": run_index,
                "codex_exit": proc.returncode,
                "codex_seconds": elapsed,
                "workspace_path": str(workspace),
                "score": int(scored.get("score", 0)),
                "raw_score_pre_ceiling": int(scored.get("raw_score_pre_ceiling", 0)),
                "pass": bool(scored.get("pass", False)),
                "shortcut_detected": bool(scored.get("shortcut_detected", False)),
                "ceilings_applied": list(scored.get("ceilings_applied", [])),
                "milestones": dict(scored.get("milestones", {})),
                "breakdown": dict(scored.get("breakdown", {})),
                "errors": list(scored.get("errors", [])),
                "result_file": str(result_file),
                "log_file": str(log_path),
                "last_message_file": str(last_message),
            }
            with jsonl_path.open("a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            print(
                f"  codex_exit={proc.returncode} seconds={elapsed} "
                f"score={rec['score']} raw={rec['raw_score_pre_ceiling']} "
                f"pass={rec['pass']} ceilings={rec['ceilings_applied']}"
            )

    report_path = attempt_dir / "probe_report.txt"
    report_json_path = attempt_dir / "probe_report.json"
    with report_path.open("w") as fh:
        run_cmd(
            [
                sys.executable,
                str(SHARED_PROBE_REPORT),
                str(jsonl_path),
                "--probe-run-id",
                probe_run_id,
                "--emit-json",
            ],
            stdout=fh,
        )
    report_text = report_path.read_text()
    if "JSON_SUMMARY:\n" in report_text:
        _, json_block = report_text.split("JSON_SUMMARY:\n", 1)
        report_json_path.write_text(json_block)
    print(f"jsonl={jsonl_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
