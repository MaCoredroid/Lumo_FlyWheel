#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/backlog-decomposition-scheduler"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/backlog-decomposition-scheduler"
SCORER = REPO / "verifiers/backlog-decomposition-scheduler/score_schedule.py"
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


def probe_variant(variant: str, model: str, reasoning: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"bds_probe_{variant}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        prompt = "Read AGENTS.md, inspect the workspace evidence, and solve the task completely."
        cmd = [
            "codex",
            "-a",
            "never",
            "-s",
            "danger-full-access",
            "exec",
            "--skip-git-repo-check",
            "--json",
            "-m",
            model,
            "-c",
            f'reasoning_effort="{reasoning}"',
            "-c",
            f'model_reasoning_effort="{reasoning}"',
            "-C",
            str(ws),
            prompt,
        ]
        solve = subprocess.run(cmd, check=False, capture_output=True, text=True)
        result_path = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VER_DATA),
                "RESULT_FILE": str(result_path),
                "VARIANT_ID": variant,
            }
        )
        subprocess.run([sys.executable, str(SCORER)], check=True, env=env)
        scored = json.loads(result_path.read_text())
        scored["solve_returncode"] = solve.returncode
        scored["solve_stdout_tail"] = solve.stdout[-1000:]
        scored["solve_stderr_tail"] = solve.stderr[-1000:]
        return scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--out", default=str(FAMILY / "probe_latest.json"))
    args = ap.parse_args()

    variants = [args.variant] if args.variant else VARIANTS
    results = {}
    for variant in variants:
        results[variant] = probe_variant(variant, args.model, args.reasoning)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
