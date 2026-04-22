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
FAMILY = REPO / "benchmark_blueprints/families/nightly-regression-watch"
SCORER = REPO / "verifiers/nightly-regression-watch/score_ranking.py"
VERIFIER = REPO / "verifier_data/nightly-regression-watch"


def copy_oracle(ws: Path, variant: str) -> None:
    src = VERIFIER / variant / "oracle"
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)


def empty(_: Path, __: str) -> None:
    return


def docs_only(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    # revert code only
    bundle = FAMILY / "workspace_bundle" / variant
    shutil.copy(bundle / "ops_digest/src/schema.py", ws / "ops_digest/src/schema.py")
    shutil.copy(bundle / "ops_digest/src/digest_builder.py", ws / "ops_digest/src/digest_builder.py")


def latest_wrong(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "ops_digest/src/digest_builder.py"
    text = target.read_text()
    target.write_text(
        text.replace("latest = {}", "latest = {}\n    # mutated by matrix").replace(
            "if current is None or run[\"completed_at\"] > current[\"completed_at\"]:",
            "if current is None:",
        )
    )
    out = subprocess.run(
        [sys.executable, "-m", "ops_digest.src.digest_builder", "--fixtures", "ops_digest/fixtures/runs", "--out", "ops_digest/fixtures/inbox/generated_digest.md"],
        cwd=ws,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError("failed to regenerate latest_wrong digest")


def second_automation(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    sibling = ws / "ops_digest/automation/nightly_regression_watch_v2.toml"
    sibling.write_text((ws / "ops_digest/automation/nightly_regression_watch.toml").read_text())


def delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "ops_digest/tests/test_schema.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", empty, "Empty"),
    ("03_docs_only", docs_only, "Docs-only repair"),
    ("04_latest_wrong", latest_wrong, "Latest-of-day wrong"),
    ("05_second_automation", second_automation, "Second automation sibling"),
    ("06_delete_tests", delete_tests, "Delete-tests adversarial"),
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


def derive_s_ttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def score(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="nightly_matrix_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
        builder(ws, variant)
        result_file = Path(tmp) / "result.json"
        env = os.environ.copy()
        env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score(args.variant, builder)
        rows.append({
            "label": label,
            "P_benchmark": result["P_benchmark"],
            "M_training": result["M_training"],
            "G": derive_g(result),
            "R": derive_r(result),
            "S_TTC": derive_s_ttc(result),
            "integrity": result["integrity_flag"],
            "pass": result["pass"],
            "ceilings": ",".join(result["ceilings_applied"]) or "—",
        })

    out = Path(args.out)
    out.write_text(f"# Verification matrix — {args.variant}\n\n")
    with out.open("a") as fh:
        fh.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            fh.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
