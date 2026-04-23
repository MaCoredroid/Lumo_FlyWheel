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

REPO = Path(__file__).resolve().parent.parent.parent
FAMILY = REPO / "benchmark_blueprints/families/parallel-explorer-root-cause-map"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/parallel-explorer-root-cause-map"
SCORER = REPO / "verifiers/parallel-explorer-root-cause-map/score_ranking.py"


def load_oracle(variant: str) -> dict:
    return json.loads((VER_DATA / variant / "oracle" / "manager_brief.json").read_text())


def write_brief(ws: Path, brief: dict) -> None:
    dst = ws / "brief" / "manager_brief.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")


def make_oracle(ws: Path, variant: str) -> None:
    oracle_dir = VER_DATA / variant / "oracle"
    shutil.copy(oracle_dir / "manager_brief.json", ws / "brief" / "manager_brief.json")
    shutil.copy(oracle_dir / "manager_brief.md", ws / "brief" / "manager_brief.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    brief = load_oracle(variant)
    for entry in brief["ranking"]:
        entry["evidence_paths"] = []
    for row in brief["evidence_table"]:
        row["test_or_artifact"] = ""
    write_brief(ws, brief)


def make_pick_renderer(ws: Path, variant: str) -> None:
    brief = load_oracle(variant)
    entries = {entry["suspect_id"]: entry for entry in brief["ranking"]}
    new_order = [
        entries["S3-renderer-duplicate-headings"],
        entries["S1-fs-alias-normalization"],
        entries["S2-aggregation-grouping"],
        entries["S4-env-watchlist-parser"],
    ]
    for idx, entry in enumerate(new_order, start=1):
        entry["rank"] = idx
        if entry["suspect_id"] == "S3-renderer-duplicate-headings":
            entry["role"] = "primary"
    brief["accepted"] = "S3-renderer-duplicate-headings"
    brief["ranking"] = new_order
    brief["remediation_plan"]["patch_target_file"] = "src/release_readiness/renderers/markdown_renderer.py"
    brief["remediation_plan"]["patch_target_symbol"] = "render_blocked_owner_section"
    write_brief(ws, brief)


def make_top1_wrong(ws: Path, variant: str) -> None:
    brief = load_oracle(variant)
    entries = {entry["suspect_id"]: entry for entry in brief["ranking"]}
    new_order = [
        entries["S2-aggregation-grouping"],
        entries["S1-fs-alias-normalization"],
        entries["S3-renderer-duplicate-headings"],
        entries["S4-env-watchlist-parser"],
    ]
    for idx, entry in enumerate(new_order, start=1):
        entry["rank"] = idx
    new_order[0]["role"] = "primary"
    brief["accepted"] = "S2-aggregation-grouping"
    brief["ranking"] = new_order
    brief["remediation_plan"]["patch_target_file"] = "src/release_readiness/core/aggregation.py"
    brief["remediation_plan"]["patch_target_symbol"] = "merge_blocked_owner_rows"
    write_brief(ws, brief)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    target = ws / "tests" / "test_root_cause_map.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_renderer", make_pick_renderer, "Pick renderer"),
    ("05_top1_wrong", make_top1_wrong, "Top1 wrong"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="root_cause_vmx_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
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
        return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_R(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m_agg - 0.6 * h, 4)


def derive_S_TTC(result: dict) -> int:
    P = 1 if result.get("pass", False) else 0
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    H = int(result.get("integrity_flag", 0))
    G = derive_G(result)
    return int(round(1000 * P + 100 * m_agg - 100 * H + 10 * G))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for tag, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_G(result),
                "R": derive_R(result),
                "S_TTC": derive_S_TTC(result),
                "integrity_flag": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": result.get("ceilings_applied", []),
                "integrity_rules_fired": result.get("integrity_rules_fired", []),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Verification Matrix — {args.variant}\n\n")
        f.write(f"Generated by `verifiers/parallel-explorer-root-cause-map/run_verification_matrix.py`.\n\n")
        f.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings"]) or "—"
            if row["integrity_rules_fired"]:
                ceilings = "H=" + ",".join(row["integrity_rules_fired"])
            f.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {ceilings} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
