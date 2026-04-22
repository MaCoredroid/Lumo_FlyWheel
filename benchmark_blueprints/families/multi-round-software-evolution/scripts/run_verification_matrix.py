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


def oracle_payload(variant: str) -> dict:
    return json.loads((VER_DATA / variant / "oracle" / "round_plan.json").read_text())


def make_oracle(ws: Path, variant: str) -> None:
    dst = ws / "brief"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(VER_DATA / variant / "oracle" / "round_plan.json", dst / "round_plan.json")
    shutil.copy(VER_DATA / variant / "oracle" / "round_plan.md", dst / "round_plan.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_vague_metric(ws: Path, variant: str) -> None:
    data = oracle_payload(variant)
    data["partial_progress_metric"]["baseline"] = "Improve the current state."
    data["partial_progress_metric"]["target"] = "Make it materially better."
    data["partial_progress_metric"]["guardrail"] = "Stay safe."
    data["partial_progress_metric"]["measurement_plan"] = ["Look at it", "Ship it"]
    dst = ws / "brief"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "round_plan.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_shortcut(ws: Path, variant: str) -> None:
    data = oracle_payload(variant)
    if variant == "v4-multi-corpus-objective":
        data["selected_focus"]["focus_id"] = "schema-snapshot-alignment"
        data["selected_focus"]["title"] = "Schema Snapshot Alignment"
    elif variant == "v5-recovery-in-thread":
        data["selected_focus"]["focus_id"] = "streaming-watermark-alignment"
        data["selected_focus"]["title"] = "Streaming Watermark Alignment"
    elif variant == "v3-dirty-state":
        data["selected_focus"]["focus_id"] = "cold-start-cache-tuning"
        data["selected_focus"]["title"] = "Cold-Start Cache Tuning"
    else:
        data["selected_focus"]["focus_id"] = "contained-pager-cleanup"
        data["selected_focus"]["title"] = "Contained Pager Cleanup"
    dst = ws / "brief"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "round_plan.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_boundary_missing(ws: Path, variant: str) -> None:
    data = oracle_payload(variant)
    data["do_not_touch"] = data["do_not_touch"][:1]
    if variant in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        data["do_not_touch"] = []
    dst = ws / "brief"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "round_plan.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    test_file = next((ws / "tests").rglob("*.py"))
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle (full pass)"),
    ("02_empty", make_empty, "Empty (no brief)"),
    ("03_vague_metric", make_vague_metric, "Vague partial-progress metric"),
    ("04_shortcut", make_shortcut, "Shortcut focus"),
    ("05_boundary_missing", make_boundary_missing, "Boundary missing"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder, tag: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"mrse_vmx_{tag}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
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
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_R(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m - 0.6 * h, 4)


def derive_S_TTC(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_G(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-clean-baseline")
    ap.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = ap.parse_args()

    rows = []
    out = Path(args.out)
    for tag, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder, tag)
        rows.append(
            {
                "label": label,
                "P_benchmark": result.get("P_benchmark", 0),
                "M_training": result.get("M_training", 0.0),
                "G": derive_G(result),
                "R": derive_R(result),
                "S_TTC": derive_S_TTC(result),
                "integrity_flag": result.get("integrity_flag", 0),
                "pass": result.get("pass", False),
                "ceilings_applied": result.get("ceilings_applied", []),
                "integrity_rules_fired": result.get("integrity_rules_fired", []),
            }
        )

    with out.open("w") as fh:
        fh.write(f"# §5 verification matrix — {args.variant}\n\n")
        fh.write(f"Generated by `benchmark_blueprints/families/multi-round-software-evolution/scripts/run_verification_matrix.py` against `{args.variant}` with CNB55_SEED=42.\n\n")
        fh.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings_applied"]) or "—"
            if row["integrity_rules_fired"]:
                ceilings = f"H={','.join(row['integrity_rules_fired'])}"
            fh.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {ceilings} |\n"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
