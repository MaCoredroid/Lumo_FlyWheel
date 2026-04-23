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
FAMILY = REPO / "benchmark_blueprints/families/runbook-code-reconciliation"
VERIFIER = REPO / "verifier_data/runbook-code-reconciliation"
SCORER = REPO / "verifiers/runbook-code-reconciliation/score_reconciliation.py"


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


def legacy_primary(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    gold = json.loads((VERIFIER / variant / "gold_repair.json").read_text())
    runbook = ws / "docs/runbooks/release_preview.md"
    runbook.write_text(
        runbook.read_text()
        .replace(gold["expected_current_command"], gold["legacy_primary_command"])
        .replace(gold["current_env_var"], gold["deprecated_env_var"])
        .replace(gold["current_flag"], gold["deprecated_flag"])
        .replace("compatibility-only", "primary")
    )
    facts = ws / "artifacts/reconciliation_facts.json"
    payload = json.loads(facts.read_text())
    payload["primary_entrypoint"] = gold["legacy_alias"]
    payload["current_flag"] = gold["deprecated_flag"]
    payload["current_env_var"] = gold["deprecated_env_var"]
    facts.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def facts_unvalidated(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    notes = ws / "artifacts/verification_notes.md"
    notes.write_text(
        "# Verification Notes\n\n"
        "## Checked directly\n"
        "- Reviewed README fragments for the closest command.\n\n"
        "## Inferred from code\n"
        "- The code appears to prefer the current naming.\n\n"
        "## Remaining caveats\n"
        "- Did not run live help commands.\n"
    )
    facts = ws / "artifacts/reconciliation_facts.json"
    payload = json.loads(facts.read_text())
    payload["evidence_commands"] = ["rg -n release_preview README_fragments docs src scripts"]
    facts.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    test_file = ws / "tests/test_release_preview_cli.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", empty, "Empty"),
    ("03_facts_unvalidated", facts_unvalidated, "RAWR docs-only harmonization"),
    ("04_legacy_primary", legacy_primary, "Legacy primary shortcut"),
    ("05_delete_tests", delete_tests, "Delete-tests adversarial"),
]


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


def score(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="runbook_matrix_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
        builder(ws, variant)
        result_file = Path(tmp) / "result.json"
        env = os.environ.copy()
        env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-clean-baseline")
    ap.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = ap.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score(args.variant, builder)
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
                "ceilings": ",".join(result["ceilings_applied"]) or "—",
            }
        )

    out = Path(args.out)
    lines = [
        f"# Verification matrix — {args.variant}",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {row['ceilings']} |"
        )
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
