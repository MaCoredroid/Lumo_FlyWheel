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

REPO = Path(__file__).resolve().parent.parent.parent
FAMILY = REPO / "benchmark_blueprints/families/parallel-explorer-root-cause-map"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/parallel-explorer-root-cause-map"
SCORER = REPO / "verifiers/parallel-explorer-root-cause-map/score_ranking.py"
REPORT_DIR = FAMILY / "report"
RUNS_JSONL = REPORT_DIR / "probe_runs.jsonl"
LOG_DIR = REPORT_DIR / "live_probe_logs"
VARIANT_ORDER = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


def run_once(probe_run_id: str, variant: str, run_index: int, timeout_seconds: int) -> dict:
    tag = f"{probe_run_id}-{variant}-run{run_index}"
    work = Path(tempfile.mkdtemp(prefix=f"probe_{variant}_"))
    ws = work / "workspace"
    result_file = work / "verify_result.json"
    log_file = LOG_DIR / f"{tag}.log"
    shutil.copytree(WS_BUNDLE / variant, ws)
    prompt = (
        "Read AGENTS.md in this directory and follow it exactly. "
        "Author brief_input.json at the workspace root, run ./bin/cnb55-brief validate brief_input.json, "
        "then run ./bin/cnb55-brief submit brief_input.json. Do not modify any file outside brief_input.json and brief/."
    )
    started = time.time()
    completed = subprocess.run(
        [
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
            prompt,
        ],
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = int(time.time() - started)
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VER_DATA),
            "RESULT_FILE": str(result_file),
            "VARIANT_ID": variant,
        }
    )
    subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
    scored = json.loads(result_file.read_text())
    rec = {
        "probe_run_id": probe_run_id,
        "variant": variant,
        "run_index": run_index,
        "codex_exit": completed.returncode,
        "codex_seconds": elapsed,
        "score": scored["score"],
        "raw_score_pre_ceiling": scored["raw_score_pre_ceiling"],
        "pass": scored["pass"],
        "shortcut_detected": scored["shortcut_detected"],
        "ceilings_applied": scored.get("ceilings_applied", []),
        "errors": scored.get("errors", []),
        "log_file": str(log_file),
        "workspace_path": str(ws),
    }
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--probe-run-id", default=None)
    args = parser.parse_args()

    variants = args.variants or list(VARIANT_ORDER)
    probe_run_id = args.probe_run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_JSONL.open("a") as out:
        for variant in variants:
            for run_index in range(1, args.runs + 1):
                rec = run_once(probe_run_id, variant, run_index, args.timeout)
                out.write(json.dumps(rec, sort_keys=True) + "\n")
                print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
