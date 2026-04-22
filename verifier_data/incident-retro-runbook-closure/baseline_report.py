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
FAMILY = REPO / "benchmark_blueprints/families/incident-retro-runbook-closure"
VERIFIER = REPO / "verifier_data/incident-retro-runbook-closure"
SCORER = REPO / "verifiers/incident-retro-runbook-closure/score_ranking.py"

if str(VERIFIER) not in sys.path:
    sys.path.insert(0, str(VERIFIER))

import run_verification_matrix as rvm  # noqa: E402

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

BASELINES = [
    ("oracle", rvm.copy_oracle, "Oracle"),
    ("empty", rvm.empty, "Empty"),
    ("schedule_drift_shortcut", rvm.schedule_drift, "Schedule-drift shortcut"),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="incident_baseline_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
        builder(ws, variant)
        result_file = Path(tmp) / "result.json"
        env = os.environ.copy()
        env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def render(rows: list[dict]) -> str:
    lines = ["# Baseline report", ""]
    lines.append("| variant | case | P_benchmark | M_training | pass | integrity | ceilings |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for row in rows:
        ceilings = ",".join(row["ceilings"]) or "—"
        lines.append(
            f"| {row['variant']} | {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['pass']} | {row['integrity_flag']} | {ceilings} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for variant in VARIANTS:
        for case_id, builder, label in BASELINES:
            result = score_variant(variant, builder)
            rows.append(
                {
                    "variant": variant,
                    "case_id": case_id,
                    "label": label,
                    "P_benchmark": result["P_benchmark"],
                    "M_training": result["M_training"],
                    "pass": result["pass"],
                    "integrity_flag": result["integrity_flag"],
                    "ceilings": result["ceilings_applied"],
                }
            )
    text = render(rows)
    args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
