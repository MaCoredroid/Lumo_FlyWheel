#!/usr/bin/env python3
"""Run family graders against the §18 validation workspaces.

For each active v4a task with a grader, invoke the task's scoring script
with the workspace from `/tmp/qwen_proxy_validation/<task>/workspace/`.
Captures the verify_result.json output and tallies pass/fail + scores.

The grader interface (per `verify.sh` files):
  AGENT_WS:        path to the agent's modified workspace
  VERIFIER_DATA:   path to reference / oracle data for this task
  RESULT_FILE:     output JSON path
  VARIANT_ID:      e.g. v1-clean-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map task -> grader script (relative to verifiers/<task>/)
TASK_GRADERS = {
    "dead-flag-reachability-audit": "score_reachability.py",
    "fanout-fullstack-release-blocker": "score_release_blocker.py",
    "policy-aware-request-resolution": "score_ranking.py",
    "release-note-to-plan-translation": "score_ranking.py",
    "responses-sdk-adapter-cutover": "score_responses_cutover.py",
    "security-audit-hotfix-remediation": "score_hotfix.py",
    "sqlalchemy-2-session-modernization": "score_sqlalchemy_session_modernization.py",
    "transcript-merge-regression": "score_transcript_merge.py",
}


def run_grader(task: str, workspace: Path, out_dir: Path) -> dict:
    grader = REPO_ROOT / "verifiers" / task / TASK_GRADERS[task]
    if not grader.is_file():
        return {"task": task, "status": "GRADER_MISSING", "grader": str(grader)}
    verifier_data = REPO_ROOT / "verifier_data" / task
    if not verifier_data.is_dir():
        return {"task": task, "status": "VERIFIER_DATA_MISSING", "verifier_data": str(verifier_data)}
    if not workspace.is_dir():
        return {"task": task, "status": "WORKSPACE_MISSING", "workspace": str(workspace)}

    result_file = out_dir / f"{task}.verify_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "AGENT_WS": str(workspace),
        "VERIFIER_DATA": str(verifier_data),
        "RESULT_FILE": str(result_file),
        "VARIANT_ID": "v1-clean-baseline",
        "CNB55_SEED": "42",
    }
    try:
        proc = subprocess.run(
            ["/usr/bin/python3", str(grader)],
            env=env,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"task": task, "status": "GRADER_TIMEOUT"}

    info = {
        "task": task,
        "grader_rc": proc.returncode,
        "grader_stdout_tail": (proc.stdout or "")[-500:],
        "grader_stderr_tail": (proc.stderr or "")[-500:],
    }
    if result_file.is_file():
        try:
            payload = json.loads(result_file.read_text())
            info["result"] = payload
            # Common fields
            info["final_score"] = payload.get("final_score") or payload.get("score") or payload.get("raw_score")
            info["pass"] = payload.get("pass") or payload.get("passed") or (
                isinstance(info["final_score"], (int, float)) and info["final_score"] >= 65
            )
            info["breakdown"] = payload.get("breakdown")
            info["errors"] = payload.get("errors") or []
            info["integrity_flag"] = payload.get("integrity_flag")
            info["status"] = "GRADED"
        except json.JSONDecodeError as exc:
            info["status"] = "RESULT_PARSE_FAIL"
            info["parse_error"] = str(exc)
    else:
        info["status"] = "NO_RESULT_FILE"
    return info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--validation-root",
        default="/tmp/qwen_proxy_validation",
        help="Root containing <task>/workspace/ from validation sweep",
    )
    p.add_argument(
        "--out-dir",
        default="/tmp/qwen_proxy_grading",
        help="Where to write grader output JSONs + summary",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of tasks",
    )
    args = p.parse_args()

    val_root = Path(args.validation_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(TASK_GRADERS.keys()) if not args.only else [
        t for t in TASK_GRADERS if t in args.only.split(",")
    ]
    results: list[dict] = []
    print(f"Running {len(tasks)} graders ...\n", flush=True)
    for t in tasks:
        ws = val_root / t / "workspace"
        print(f"[grading] {t} ws={ws}", flush=True)
        r = run_grader(t, ws, out_dir)
        results.append(r)
        score = r.get("final_score")
        passed = r.get("pass")
        status = r.get("status")
        print(
            f"  -> status={status} grader_rc={r.get('grader_rc')} "
            f"score={score} pass={passed}",
            flush=True,
        )

    summary = out_dir / "summary.json"
    summary.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== SUMMARY ===")
    print(f"{'task':<42} {'rc':>4} {'score':>6} {'pass':>5} {'status':<22}")
    graded = passed = 0
    for r in results:
        score = r.get("final_score")
        score_str = "n/a" if score is None else str(score)
        print(
            f"{r['task']:<42} {str(r.get('grader_rc','?')):>4} "
            f"{score_str:>6} {str(r.get('pass','?')):>5} {r.get('status',''):<22}"
        )
        if r.get("status") == "GRADED":
            graded += 1
            if r.get("pass"):
                passed += 1
    print(f"\nGraded: {graded}/{len(results)}  |  Passed: {passed}/{graded}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
