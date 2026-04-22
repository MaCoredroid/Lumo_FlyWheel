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
FAMILY = REPO / "benchmark_blueprints" / "families" / "release-note-to-plan-translation"
VERIFIER_DATA = REPO / "verifier_data" / "release-note-to-plan-translation"
SCORER = REPO / "verifiers" / "release-note-to-plan-translation" / "score_ranking.py"


def load_gold(variant: str) -> dict:
    return json.loads((VERIFIER_DATA / variant / "gold_ranking.json").read_text())


def make_oracle(ws: Path, variant: str) -> None:
    (ws / "brief").mkdir(parents=True, exist_ok=True)
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "manager_brief.json", ws / "brief" / "manager_brief.json")
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "manager_brief.md", ws / "brief" / "manager_brief.md")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    brief = json.loads((VERIFIER_DATA / variant / "oracle" / "manager_brief.json").read_text())
    for step in brief["ordered_steps"]:
        step["evidence"] = []
    brief["primary_risk"]["evidence"] = []
    out = ws / "brief"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manager_brief.json").write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    (out / "manager_brief.md").write_text("grounding stripped\n")


def make_pick_ceiling(ws: Path, variant: str) -> None:
    gold = load_gold(variant)
    brief = json.loads((VERIFIER_DATA / variant / "oracle" / "manager_brief.json").read_text())
    bad = gold["matrix_pick_ceiling_id"]
    ordered = [step for step in brief["ordered_steps"] if step["step_id"] == bad] + [
        step for step in brief["ordered_steps"] if step["step_id"] != bad
    ]
    for rank, step in enumerate(ordered, start=1):
        step["rank"] = rank
    brief["first_milestone_id"] = bad
    brief["ordered_steps"] = ordered
    out = ws / "brief"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manager_brief.json").write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    (out / "manager_brief.md").write_text("pick ceiling\n")


def make_top1_wrong(ws: Path, variant: str) -> None:
    brief = json.loads((VERIFIER_DATA / variant / "oracle" / "manager_brief.json").read_text())
    ordered = list(brief["ordered_steps"])
    ordered[0], ordered[1] = ordered[1], ordered[0]
    for rank, step in enumerate(ordered, start=1):
        step["rank"] = rank
    brief["ordered_steps"] = ordered
    brief["first_milestone_id"] = ordered[0]["step_id"]
    out = ws / "brief"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manager_brief.json").write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    (out / "manager_brief.md").write_text("top1 wrong\n")


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    test_file = ws / "tests" / "test_plan_brief.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_rawr_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_ceiling", make_pick_ceiling, "Pick ceiling"),
    ("05_top1_wrong", make_top1_wrong, "Top1 wrong"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vmx_{variant}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result_file),
                "VARIANT_ID": variant,
                "CNB55_SEED": "42",
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
    parser.add_argument("--out")
    args = parser.parse_args()

    out = Path(args.out) if args.out else FAMILY / ("verification_matrix.md" if args.variant == "v1-clean-baseline" else f"verification_matrix_{args.variant}.md")
    rows = []
    for tag, builder, label in TRAJECTORIES:
        result = score_variant(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_G(result),
                "R": derive_R(result),
                "S_TTC": derive_S_TTC(result),
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": ",".join(result.get("ceilings_applied", [])) or "—",
            }
        )
    lines = [
        f"# Verification Matrix — {args.variant}",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |"
        )
    lines.extend(
        [
            "",
            "Expected shape:",
            "- Oracle should clear the pass bar and stay near-or-above 0.90 M_training.",
            "- Empty should score 0.",
            "- RAWR grounding_stripped should trigger `plan_without_grounding`.",
            "- Pick ceiling should trip the family-specific ceiling for the variant.",
            "- Delete-tests should set `integrity_flag = 1`.",
        ]
    )
    out.write_text("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
