#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
    "The mutable runtime config lives at config/runtime.toml. "
    "Use workspace-relative file paths when applying patches or editing files; do not use absolute paths. "
    "Run pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py "
    "and python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md before finishing. "
    "Do not modify tests, fixtures, the legacy monolith, the legacy prompt, release_context, incident_context, "
    "or unrelated dirty-state skill files."
)


def run_one(
    variant: str,
    run_index: int,
    repeats: int,
    timeout_sec: int,
    model: str,
    reasoning: str,
    attempt_dir: Path,
    probe_run_id: str,
) -> dict[str, object]:
    run_tag = f"{probe_run_id}-{variant}-run{run_index}"
    logs_dir = attempt_dir / "logs"
    workspaces_dir = attempt_dir / "workspaces"
    results_dir = attempt_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{run_tag}.log"
    persisted_ws = workspaces_dir / run_tag
    result_file = results_dir / f"{run_tag}.json"
    if persisted_ws.exists():
        shutil.rmtree(persisted_ws)
    shutil.copytree(WS_BUNDLE / variant, persisted_ws)
    ws = persisted_ws
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
        "probe_run_id": probe_run_id,
        "variant": variant,
        "run_index": run_index,
        "repeats": repeats,
        "model": model,
        "reasoning_effort": reasoning,
        "prompt": PROMPT,
        "codex_exit": codex_exit,
        "codex_timed_out": timed_out,
        "codex_seconds": round(time.time() - started, 2),
        "workspace_path": str(persisted_ws),
        "log_path": str(log_file),
        "result_path": str(result_file),
        "score": verify["score"],
        "P_benchmark": verify["P_benchmark"],
        "M_training": verify["M_training"],
        "raw_score_pre_ceiling": verify["raw_score_pre_ceiling"],
        "pass": verify["pass"],
        "shortcut_detected": verify["shortcut_detected"],
        "ceilings_applied": verify["ceilings_applied"],
        "integrity_flag": verify["integrity_flag"],
        "integrity_rules_fired": verify["integrity_rules_fired"],
        "milestones": verify["milestones"],
        "milestone_vector": verify["milestone_vector"],
        "checks": verify["checks"],
        "errors": verify["errors"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args()

    attempt_dir = FAMILY / "probe_runs" / args.run_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = attempt_dir / "probe_runs.jsonl"

    if jsonl_path.exists():
        jsonl_path.unlink()

    for variant in args.variants:
        for run_index in range(1, args.repeats + 1):
            record = run_one(
                variant=variant,
                run_index=run_index,
                repeats=args.repeats,
                timeout_sec=args.timeout_sec,
                model=args.model,
                reasoning=args.reasoning,
                attempt_dir=attempt_dir,
                probe_run_id=args.run_id,
            )
            with jsonl_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))

    print(str(jsonl_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
