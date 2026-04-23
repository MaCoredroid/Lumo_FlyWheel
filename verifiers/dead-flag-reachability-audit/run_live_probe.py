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
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints" / "families" / "dead-flag-reachability-audit"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "dead-flag-reachability-audit"
SCORER = REPO / "verifiers" / "dead-flag-reachability-audit" / "score_reachability.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return round(var ** 0.5, 2)


def run_one(variant: str, run_index: int, prompt: str, work_root: Path) -> dict:
    ws = work_root / f"{variant}-run{run_index}" / "workspace"
    ws.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WS_BUNDLE / variant, ws)
    result_file = ws.parent / "verify_result.json"
    log_file = ws.parent / "codex.log"
    start = time.time()
    with log_file.open("w") as handle:
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--cd",
                str(ws),
                prompt,
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = int(time.time() - start)
    env = os.environ.copy()
    env.update({
        "AGENT_WS": str(ws),
        "VERIFIER_DATA": str(VERIFIER_DATA),
        "VARIANT_ID": variant,
        "RESULT_FILE": str(result_file),
        "PYTHONPATH": str(ws / "src"),
    })
    subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
    result = json.loads(result_file.read_text())
    return {
        "variant": variant,
        "run_index": run_index,
        "codex_exit": proc.returncode,
        "codex_seconds": elapsed,
        "score": result["P_benchmark"],
        "M_training": result["M_training"],
        "integrity_flag": result["integrity_flag"],
        "pass": result["pass"],
        "ceilings_applied": result["ceilings_applied"],
        "result": result,
        "workspace_path": str(ws),
        "log_path": str(log_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-jsonl", default=str(FAMILY / "report" / "probe_runs.jsonl"))
    parser.add_argument("--out-summary", default=str(FAMILY / "report" / "probe_summary_latest.json"))
    args = parser.parse_args()

    prompt = (
        "Read AGENTS.md, inspect the workspace, author brief_input.json, "
        "run ./bin/cnb55-flag-audit validate brief_input.json, then run "
        "./bin/cnb55-flag-audit submit brief_input.json. Do not modify files "
        "outside brief/ or artifacts/."
    )

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    work_root = out_jsonl.parent / "live_probe_workspaces"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in VARIANTS:
        for run_index in range(1, args.repeats + 1):
            row = run_one(variant, run_index, prompt, work_root)
            rows.append(row)
            with out_jsonl.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(int(row["score"]))

    summary = {
        "family_id": "dead-flag-reachability-audit",
        "repeats": args.repeats,
        "variant_stats": {
            variant: {
                "n": len(grouped[variant]),
                "mean": mean(grouped[variant]),
                "stdev": stdev(grouped[variant]),
                "scores": grouped[variant],
            }
            for variant in VARIANTS
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_jsonl}")
    print(f"wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
