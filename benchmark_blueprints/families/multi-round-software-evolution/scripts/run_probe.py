#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
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


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def monotonic_with_tolerance(means: list[float], tolerance: float) -> bool:
    return all(means[i] + tolerance >= means[i + 1] for i in range(len(means) - 1))


def family_summary(rows: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    ordered = []
    for variant in VARIANTS:
        runs = by_variant.get(variant, [])
        scores = [r["P_benchmark"] for r in runs]
        m_scores = [r["M_training"] for r in runs]
        ordered.append(
            {
                "variant": variant,
                "runs": len(runs),
                "mean_P_benchmark": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "min_P_benchmark": min(scores) if scores else 0,
                "max_P_benchmark": max(scores) if scores else 0,
                "mean_M_training": round(sum(m_scores) / len(m_scores), 4) if m_scores else 0.0,
                "selected_focuses": [r.get("selected_focus_id") for r in runs],
            }
        )
    means = [row["mean_P_benchmark"] for row in ordered]
    all_m = [r["M_training"] for r in rows]
    return {
        "variants": ordered,
        "family_mean_P_benchmark": round(sum(means) / len(means), 2) if means else 0.0,
        "max_variant_mean": max(means) if means else 0.0,
        "min_variant_mean": min(means) if means else 0.0,
        "monotonic_with_tolerance_3": monotonic_with_tolerance(means, 3.0),
        "oracle_gate_reference": "oracle>=90 empty=0 shortcut<=30 checked separately by regen_family.py",
        "observed_stdev_M_training": round(statistics.pstdev(all_m), 4) if len(all_m) >= 2 else 0.0,
        "layer_a_gate": {
            "family_mean_in_range_15_25": 15 <= (sum(means) / len(means) if means else 0.0) <= 25,
            "max_variant_mean_le_40": max(means) <= 40 if means else False,
            "min_variant_mean_le_10": min(means) <= 10 if means else False,
            "monotonic_with_tolerance_3": monotonic_with_tolerance(means, 3.0),
        },
    }


def write_report(path: Path, runs: int, model: str, summary: dict, rows: list[dict]) -> None:
    gate = summary["layer_a_gate"]
    lines = [
        "# Multi-Round Software Evolution Probe Report",
        "",
        f"Model: `{model}`",
        f"Runs per variant: `{runs}`",
        "",
        "## Per-Variant Means",
        "",
        "| Variant | Mean P | Min | Max | Mean M | Selected Focuses |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| {row['variant']} | {row['mean_P_benchmark']:.2f} | {row['min_P_benchmark']} | {row['max_P_benchmark']} | {row['mean_M_training']:.4f} | {', '.join(row['selected_focuses'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer A Gate",
            "",
            f"- family mean = `{summary['family_mean_P_benchmark']}` -> {'PASS' if gate['family_mean_in_range_15_25'] else 'FAIL'} for `[15,25]`",
            f"- max variant mean = `{summary['max_variant_mean']}` -> {'PASS' if gate['max_variant_mean_le_40'] else 'FAIL'} for `<= 40`",
            f"- min variant mean = `{summary['min_variant_mean']}` -> {'PASS' if gate['min_variant_mean_le_10'] else 'FAIL'} for `<= 10`",
            f"- monotonic within +/-3 = `{summary['monotonic_with_tolerance_3']}` -> {'PASS' if gate['monotonic_with_tolerance_3'] else 'FAIL'}",
            "",
            "## Raw Runs",
            "",
            "| Run | Variant | Exit | P | M | Pass | Focus | Ceilings |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_index']} | {row['variant']} | {row['codex_exit']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['pass']} | {row.get('selected_focus_id', '')} | {', '.join(row['ceilings_applied']) or '—'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", choices=VARIANTS)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout-seconds", type=int, default=180)
    ap.add_argument("--json-out")
    ap.add_argument("--report-out")
    args = ap.parse_args()

    variants = args.variant or VARIANTS
    rows = []
    for run_index in range(1, args.runs + 1):
        for variant in variants:
            with tempfile.TemporaryDirectory(prefix=f"mrse_probe_{variant}_r{run_index}_") as tmp:
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
                try:
                    run = subprocess.run(
                        cmd,
                        cwd=ws,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout_seconds,
                    )
                    timed_out = False
                except subprocess.TimeoutExpired as exc:
                    run = subprocess.CompletedProcess(
                        cmd,
                        124,
                        stdout=normalize_text(exc.stdout),
                        stderr=normalize_text(exc.stderr) or "timed out",
                    )
                    timed_out = True
                result = score_workspace(ws, variant)
                brief_path = ws / "brief" / "round_plan.json"
                selected_focus_id = ""
                if brief_path.exists():
                    try:
                        selected_focus_id = json.loads(brief_path.read_text()).get("selected_focus", {}).get("focus_id", "")
                    except json.JSONDecodeError:
                        selected_focus_id = ""
                rows.append(
                    {
                        "run_index": run_index,
                        "variant": variant,
                        "timed_out": timed_out,
                        "codex_exit": run.returncode,
                        "codex_stdout_tail": normalize_text(run.stdout).strip()[-400:],
                        "codex_stderr_tail": normalize_text(run.stderr).strip()[-400:],
                        "selected_focus_id": selected_focus_id,
                        "P_benchmark": result["P_benchmark"],
                        "M_training": result["M_training"],
                        "pass": result["pass"],
                        "ceilings_applied": result["ceilings_applied"],
                    }
                )
    payload = {"runs": rows, "summary": family_summary(rows)}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.report_out:
        write_report(Path(args.report_out), args.runs, args.model, payload["summary"], rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
