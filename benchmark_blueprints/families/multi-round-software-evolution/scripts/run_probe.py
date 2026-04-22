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

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/multi-round-software-evolution"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/multi-round-software-evolution"
SCORER = REPO / "verifiers/multi-round-software-evolution/score_round_plan.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Author brief_input.json at the workspace root and run "
    "./bin/cnb55-evolution submit brief_input.json to produce brief/round_plan.json. "
    "Do not modify any file outside brief/."
)


def score_workspace(ws: Path, variant: str) -> dict:
    result_file = ws / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "CNB55_SEED": "42",
        }
    )
    subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
    return json.loads(result_file.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", choices=VARIANTS)
    ap.add_argument("--model", default="gpt-5.4")
    args = ap.parse_args()

    variants = args.variant or VARIANTS
    rows = []
    for variant in variants:
        with tempfile.TemporaryDirectory(prefix=f"mrse_probe_{variant}_") as tmp:
            ws = Path(tmp) / "workspace"
            shutil.copytree(WS_BUNDLE / variant, ws)
            cmd = [
                "codex",
                "exec",
                "--model",
                args.model,
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                PROMPT,
            ]
            run = subprocess.run(cmd, cwd=ws, check=False, capture_output=True, text=True)
            result = score_workspace(ws, variant)
            rows.append(
                {
                    "variant": variant,
                    "codex_exit": run.returncode,
                    "codex_stderr_tail": (run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""),
                    "P_benchmark": result["P_benchmark"],
                    "M_training": result["M_training"],
                    "pass": result["pass"],
                    "ceilings_applied": result["ceilings_applied"],
                }
            )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
