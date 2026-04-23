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

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "codex-skill-runtime-v2-split"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_skill_runtime.py"
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Refactor the runtime bundle into one canonical structured handoff path. "
    "Run pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py "
    "and the handoff smoke command before finishing. "
    "Do not modify tests, fixtures, the legacy monolith, the legacy prompt, release_context, incident_context, "
    "or unrelated dirty-state skill files."
)


def run_variant(variant: str, timeout_sec: int, model: str, reasoning: str, log_dir: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}-{variant}-") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        log_file = log_dir / f"{variant}.log"
        started = time.time()
        with log_file.open("w") as handle:
            try:
                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
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
                        "--model",
                        model,
                        "-c",
                        f'model_reasoning_effort="{reasoning}"',
                        PROMPT,
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_sec,
                    check=False,
                    env=env,
                )
                codex_exit = int(completed.returncode)
                timed_out = False
            except subprocess.TimeoutExpired:
                codex_exit = 124
                timed_out = True
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        verify = json.loads(result_file.read_text())
        return {
            "variant": variant,
            "codex_exit": codex_exit,
            "timed_out": timed_out,
            "wall_clock_seconds": round(time.time() - started, 2),
            "score": verify["score"],
            "M_training": verify["M_training"],
            "pass": verify["pass"],
            "integrity_flag": verify["integrity_flag"],
            "ceilings": verify["ceilings_applied"],
            "log_file": str(log_file),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args()

    report_dir = FAMILY / "probe_runs" / args.run_id
    logs_dir = report_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "probe_runs.jsonl"

    rows = []
    for variant in args.variants:
        row = run_variant(variant, args.timeout_sec, args.model, args.reasoning, logs_dir)
        rows.append(row)
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))

    scores = [float(row["score"]) for row in rows]
    m_scores = [float(row["M_training"]) for row in rows]
    summary = {
        "run_id": args.run_id,
        "variants": args.variants,
        "family_mean": round(sum(scores) / len(scores), 2),
        "max_variant_mean": max(scores),
        "min_variant_mean": min(scores),
        "all_scores": scores,
        "current_observed_stdev_M_training": round(statistics.pstdev(m_scores), 4) if len(m_scores) > 1 else 0.0,
        "jsonl_path": str(jsonl_path),
    }
    summary_path = report_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
