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


REPO_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO_ROOT / "benchmark_blueprints/families/sqlalchemy-2-session-modernization"
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO_ROOT / "verifier_data/sqlalchemy-2-session-modernization"
SCORER = REPO_ROOT / "verifiers/sqlalchemy-2-session-modernization/score_sqlalchemy_session_modernization.py"


def apply_oracle(workspace: Path, variant: str) -> None:
    oracle_root = VERIFIER_DATA / variant / "oracle" / "files"
    for source in sorted(oracle_root.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(oracle_root)
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)


def make_empty(workspace: Path, variant: str) -> None:
    return


def make_grounding_stripped(workspace: Path, variant: str) -> None:
    apply_oracle(workspace, variant)
    (workspace / "app/worker.py").write_text((WORKSPACE_BUNDLE / variant / "app/worker.py").read_text())
    (workspace / "app/admin_cli.py").write_text((WORKSPACE_BUNDLE / variant / "app/admin_cli.py").read_text())


def make_helper_commit_shortcut(workspace: Path, variant: str) -> None:
    apply_oracle(workspace, variant)
    repo = workspace / "app/repository.py"
    repo.write_text(repo.read_text().replace("session.flush()", "session.commit()"))


def make_visible_only(workspace: Path, variant: str) -> None:
    apply_oracle(workspace, variant)
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        baseline_worker = (WORKSPACE_BUNDLE / variant / "app/worker.py").read_text()
        oracle_worker = (workspace / "app/worker.py").read_text()
        baseline_batch = "def settle_batch" + baseline_worker.split("def settle_batch", 1)[1]
        oracle_single = oracle_worker.split("def settle_batch", 1)[0].rstrip()
        (workspace / "app/worker.py").write_text(oracle_single + "\n\n" + baseline_batch)

        baseline_admin = (WORKSPACE_BUNDLE / variant / "app/admin_cli.py").read_text()
        oracle_admin = (workspace / "app/admin_cli.py").read_text()
        baseline_batch_admin = "def reconcile_batch" + baseline_admin.split("def reconcile_batch", 1)[1]
        oracle_single_admin = oracle_admin.split("def reconcile_batch", 1)[0].rstrip()
        (workspace / "app/admin_cli.py").write_text(oracle_single_admin + "\n\n" + baseline_batch_admin)


def make_delete_tests(workspace: Path, variant: str) -> None:
    apply_oracle(workspace, variant)
    target = workspace / "tests/test_api.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("oracle", apply_oracle, "Oracle"),
    ("empty", make_empty, "Empty"),
    ("grounding_stripped", make_grounding_stripped, "Query-only rewrite"),
    ("helper_commit_shortcut", make_helper_commit_shortcut, "Helper commit shortcut"),
    ("visible_only", make_visible_only, "Visible-only repair"),
    ("delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_workspace(workspace: Path, variant: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"sqla_vm_{variant}_") as tmpdir:
        result_file = Path(tmpdir) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(workspace),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], cwd=str(REPO_ROOT), env=env, check=True)
        return json.loads(result_file.read_text())


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m_agg - 0.6 * h, 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m_agg - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY_ROOT / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for tag, builder, label in TRAJECTORIES:
        with tempfile.TemporaryDirectory(prefix=f"sqla_matrix_{tag}_") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            shutil.copytree(WORKSPACE_BUNDLE / args.variant, workspace)
            builder(workspace, args.variant)
            result = score_workspace(workspace, args.variant)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity_flag": result["integrity_flag"],
                "ceilings": ",".join(result.get("ceilings_applied", [])) or "-",
            }
        )

    lines = [
        f"# Verification Matrix — {args.variant}",
        "",
        "| Trajectory | P_benchmark | M_training | G | R | S_TTC | Integrity | Ceilings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
            f"{row['G']:.4f} | {row['R']:.4f} | {row['S_TTC']} | "
            f"{row['integrity_flag']} | {row['ceilings']} |"
        )
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
