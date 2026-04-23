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
FAMILY = REPO / "benchmark_blueprints" / "families" / "dead-flag-reachability-audit"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "dead-flag-reachability-audit"
SCORER = REPO / "verifiers" / "dead-flag-reachability-audit" / "score_reachability.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def copy_oracle(ws: Path, variant: str) -> None:
    oracle_dir = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle_dir / "flag_audit.json", ws / "brief" / "flag_audit.json")
    shutil.copy(oracle_dir / "flag_audit.md", ws / "artifacts" / "flag_audit.md")
    shutil.copy(oracle_dir / "reachability_matrix.json", ws / "artifacts" / "reachability_matrix.json")
    shutil.copy(oracle_dir / "cleanup.patchplan.md", ws / "artifacts" / "cleanup.patchplan.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    doc = load_json(VERIFIER_DATA / variant / "oracle" / "flag_audit.json")
    for row in doc["flags"]:
        row["evidence"] = [
            "docs/preview_rollout_runbook.md",
            "docs/false_positive_notes.md",
        ]
    out = ws / "brief_input.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    subprocess.run([sys.executable, str(ws / "bin/cnb55-flag-audit"), "submit", str(out)], cwd=ws, check=True)


def make_alias_collapse(ws: Path, variant: str) -> None:
    doc = load_json(VERIFIER_DATA / variant / "oracle" / "flag_audit.json")
    for row in doc["flags"]:
        if row["flag"] == "ENABLE_PREVIEW_V2":
            row["status"] = "live"
            row["alias_of"] = None
            row["rationale"] = "Parsed envs and docs prove the flag is a standalone live runtime toggle."
    out = ws / "brief_input.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    subprocess.run([sys.executable, str(ws / "bin/cnb55-flag-audit"), "submit", str(out)], cwd=ws, check=True)


def make_force_legacy_live(ws: Path, variant: str) -> None:
    doc = load_json(VERIFIER_DATA / variant / "oracle" / "flag_audit.json")
    for row in doc["flags"]:
        if row["flag"] == "PREVIEW_FORCE_LEGACY":
            row["status"] = "partial"
            row["runtime_branch_symbol"] = "preview_runtime_branch:legacy_force_override"
            row["rationale"] = "The parser, helper tests, and unfinished patch together prove the flag still has a runtime path."
    out = ws / "brief_input.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    subprocess.run([sys.executable, str(ws / "bin/cnb55-flag-audit"), "submit", str(out)], cwd=ws, check=True)


def make_delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "tests" / "test_shadow_preview_live.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_alias_collapse", make_alias_collapse, "Alias collapse"),
    ("05_force_legacy_live", make_force_legacy_live, "Force-legacy as live"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="dead-flag-matrix-") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update({
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "PYTHONPATH": str(ws / "src"),
        })
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
        return load_json(result_file)


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m - 0.6 * h, 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder)
        rows.append({
            "label": label,
            "P_benchmark": result["P_benchmark"],
            "M_training": result["M_training"],
            "G": derive_g(result),
            "R": derive_r(result),
            "S_TTC": derive_sttc(result),
            "integrity": result["integrity_flag"],
            "pass": result["pass"],
            "ceilings": result["integrity_rules_fired"] or result["ceilings_applied"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write(f"# §5 verification matrix — {args.variant}\n\n")
        handle.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings"]) if row["ceilings"] else "—"
            handle.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
                f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | "
                f"{row['pass']} | {ceilings} |\n"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
