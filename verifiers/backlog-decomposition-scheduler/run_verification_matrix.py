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


def load_oracle(variant: str) -> dict:
    return json.loads((VER_DATA / variant / "oracle" / "schedule_brief.json").read_text())


def write_brief(ws: Path, brief: dict) -> None:
    brief_dir = ws / "brief"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "schedule_brief.json").write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")


def make_oracle(ws: Path, variant: str) -> None:
    write_brief(ws, load_oracle(variant))
    md = VER_DATA / variant / "oracle" / "schedule_brief.md"
    if md.exists():
        shutil.copy(md, ws / "brief" / "schedule_brief.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    brief = load_oracle(variant)
    for entry in brief["schedule"]:
        entry["citations"] = []
    write_brief(ws, brief)


def make_shortcut(ws: Path, variant: str) -> None:
    gold = json.loads((VER_DATA / variant / "gold_schedule.json").read_text())
    brief = load_oracle(variant)
    risky = gold["risky_item_id"]
    for entry in brief["schedule"]:
        if entry["item_id"] == risky:
            entry["slot"] = 2
        elif entry["slot"] == 2 and entry["item_id"] != risky:
            entry["slot"] = 3
    write_brief(ws, brief)


def make_near_miss(ws: Path, variant: str) -> None:
    brief = load_oracle(variant)
    slots = {entry["item_id"]: entry for entry in brief["schedule"]}
    if "B2" in slots and "B4" in slots:
        slots["B2"]["slot"], slots["B4"]["slot"] = slots["B4"]["slot"], slots["B2"]["slot"]
    write_brief(ws, brief)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    test_file = ws / "tests" / "test_schedule_brief.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("Oracle (full pass)", make_oracle),
    ("Empty (no brief)", make_empty),
    ("RAWR grounding_stripped", make_grounding_stripped),
    ("Shortcut risky rollout", make_shortcut),
    ("Near-miss slot swap", make_near_miss),
    ("Delete-tests adversarial", make_delete_tests),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="bds_vmx_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        builder(ws, variant)
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
        return json.loads(result_path.read_text())


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    if result.get("pass", False):
        return 1.0
    return round(0.6 * float(result.get("milestone_vector", {}).get("M_aggregate", 0.0)) - 0.6 * int(result.get("integrity_flag", 0)), 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    return int(round(1000 * p + 100 * m - 100 * h + 10 * derive_g(result)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-clean-baseline")
    ap.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = ap.parse_args()

    rows = []
    for label, builder in TRAJECTORIES:
        result = score_variant(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity_flag": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings_applied": result["ceilings_applied"],
                "integrity_rules_fired": result["integrity_rules_fired"],
            }
        )

    out = Path(args.out)
    out.write_text("")
    with out.open("w") as f:
        f.write(f"# §5 verification matrix — {args.variant}\n\n")
        f.write(f"Generated by `verifiers/backlog-decomposition-scheduler/run_verification_matrix.py` against `{args.variant}`.\n\n")
        f.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings_applied"]) or "—"
            if row["integrity_rules_fired"]:
                ceilings = "H=" + ",".join(row["integrity_rules_fired"])
            f.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {ceilings} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
