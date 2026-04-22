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
FAMILY = REPO / "benchmark_blueprints" / "families" / "delegation-merge-salvage"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "delegation-merge-salvage"
SCORER = REPO / "verifiers" / "delegation-merge-salvage" / "score_ranking.py"


def copy_oracle(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle_workspace"
    if oracle.exists():
        for path in oracle.rglob("*"):
            if path.is_file():
                rel = path.relative_to(oracle)
                target = ws / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "salvage_postmortem.md", ws / "artifacts" / "delegation" / "salvage_postmortem.md")
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "verification.md", ws / "artifacts" / "delegation" / "verification.md")
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "reviewer_note.md", ws / "artifacts" / "delegation" / "reviewer_note.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_worker_a_wholesale(ws: Path, variant: str) -> None:
    patch_path = ws / "artifacts" / "delegation" / "worker_a.patch"
    subprocess.run(["patch", "-p1", "-i", str(patch_path)], cwd=ws, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_worker_b_wholesale(ws: Path, variant: str) -> None:
    patch_path = ws / "artifacts" / "delegation" / "worker_b.patch"
    subprocess.run(["patch", "-p1", "-i", str(patch_path)], cwd=ws, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_generic_salvage(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    (ws / "artifacts" / "delegation" / "salvage_postmortem.md").write_text(
        "# Salvage Postmortem\n\n"
        "## Kept from Worker A\n\n- kept the good markdown parts\n\n"
        "## Rejected from Worker A\n\n- rejected the bad contract changes\n\n"
        "## Kept from Worker B\n\n- kept the docs\n\n"
        "## Rejected from Worker B\n\n- rejected the fixture churn\n"
    )


def make_delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "tests" / "test_cli.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle (full pass)"),
    ("02_empty", make_empty, "Empty (no salvage)"),
    ("03_worker_a_wholesale", make_worker_a_wholesale, "Worker A wholesale"),
    ("04_worker_b_wholesale", make_worker_b_wholesale, "Worker B wholesale"),
    ("05_generic_salvage", make_generic_salvage, "Generic salvage prose"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


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


def score_trajectory(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="delegation-merge-salvage-") as tmp:
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
        })
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
        return json.loads(result_file.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for tag, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder)
        rows.append({
            "label": label,
            "P_benchmark": result["P_benchmark"],
            "M_training": result["M_training"],
            "G": derive_g(result),
            "R": derive_r(result),
            "S_TTC": derive_sttc(result),
            "integrity_flag": result["integrity_flag"],
            "pass": result["pass"],
            "ceilings": result["integrity_rules_fired"] or result["ceilings_applied"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write(f"# §5 verification matrix — {args.variant}\n\n")
        handle.write(
            "Generated by `verifiers/delegation-merge-salvage/run_verification_matrix.py` "
            f"against `{args.variant}`.\n\n"
        )
        handle.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings"]) if row["ceilings"] else "—"
            handle.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
                f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | "
                f"{row['pass']} | {ceilings} |\n"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
