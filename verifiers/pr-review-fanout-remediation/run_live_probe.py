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

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY = "pr-review-fanout-remediation"
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
BUNDLE_DIR = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY / "workspace_bundle"
REPORT_DIR = REPO_ROOT / "benchmark_blueprints" / "families" / FAMILY / "report"
SCORER = REPO_ROOT / "verifiers" / FAMILY / "score_review_fanout.py"
VERIFIER_DATA = REPO_ROOT / "verifier_data" / FAMILY
DEFAULT_TIMEOUT = 1800


def probe_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def codex_prompt() -> str:
    return (
        "Read AGENTS.md in this directory and follow it exactly. "
        "Classify review threads before editing code. "
        "You must update src/policy/approval_router.py, src/policy/preview.py, "
        "tests/test_preview.py, and docs/approval_policy.md only as needed, then "
        "write artifacts/review/thread_triage.json, artifacts/review/reply_draft.md, "
        "and artifacts/review/remediation_report.md. "
        "Use exact thread ids from .github/review_export/pr_214_threads.json. "
        "Do not modify .github/review_export/, artifacts/review/thread_digest.md, "
        "tests/test_approval_router.py, AGENTS.md, Dockerfile, .scenario_variant, "
        "release_context/, or incident_context/. "
        "Do not add legacy_preview_hint. "
        "Do not use the network."
    )


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def run_one(
    variant: str,
    run_index: int,
    run_id: str,
    timeout_seconds: int,
    work_root: Path,
    log_dir: Path,
    runs_jsonl: Path,
) -> dict:
    run_tag = f"{run_id}-{variant}-run{run_index}"
    staged_root = work_root / run_tag
    workspace = staged_root / "workspace"
    results_dir = staged_root / "results"
    result_file = results_dir / "verify_result.json"
    log_file = log_dir / f"{run_tag}.log"

    if staged_root.exists():
        shutil.rmtree(staged_root)
    results_dir.mkdir(parents=True, exist_ok=True)
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
        "gpt-5.4",
        "-c",
        'model_reasoning_effort="high"',
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
            "raw_score_pre_ceiling": 0,
            "pass": False,
            "shortcut_detected": False,
            "ceilings_applied": ["missing_verify_result"],
            "milestones": {},
            "breakdown": {},
            "errors": ["scorer did not produce verify_result.json"],
        }
    else:
        result = json.loads(result_file.read_text())

    record = {
        "probe_run_id": run_id,
        "variant": variant,
        "run_index": run_index,
        "codex_exit": codex_exit,
        "codex_seconds": elapsed,
        "workspace_path": str(workspace),
        "score": int(result.get("score", 0)),
        "raw_score_pre_ceiling": int(result.get("raw_score_pre_ceiling", 0)),
        "pass": bool(result.get("pass", False)),
        "shortcut_detected": bool(result.get("shortcut_detected", False)),
        "integrity_flag": int(result.get("integrity_flag", 0)),
        "integrity_rules_fired": list(result.get("integrity_rules_fired", [])),
        "ceilings_applied": list(result.get("ceilings_applied", [])),
        "milestones": dict(result.get("milestones", {})),
        "breakdown": dict(result.get("breakdown", {})),
        "errors": list(result.get("errors", [])),
    }
    append_jsonl(runs_jsonl, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--probe-run-id", default=None)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    run_id = args.probe_run_id or probe_run_id()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = REPORT_DIR / "live_probe_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runs_jsonl = REPORT_DIR / "probe_runs.jsonl"

    if args.keep_work:
        work_root = REPORT_DIR / "probe_work" / run_id
        work_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"{FAMILY}_{run_id}_")
        work_root = Path(temp_ctx.name)

    try:
        print(
            f"probe_run_id={run_id} family={FAMILY} n={args.n} "
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
                    runs_jsonl=runs_jsonl,
                )
                print(
                    f"{variant} run {run_index}/{args.n}: "
                    f"score={rec['score']} raw={rec['raw_score_pre_ceiling']} "
                    f"pass={rec['pass']} ceilings={rec['ceilings_applied']}"
                )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    print(f"done: {runs_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
