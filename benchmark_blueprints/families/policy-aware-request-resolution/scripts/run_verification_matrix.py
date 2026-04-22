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
FAMILY = REPO / "benchmark_blueprints/families/policy-aware-request-resolution"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/policy-aware-request-resolution"
SCORER = REPO / "verifiers/policy-aware-request-resolution/score_ranking.py"


def load_oracle(variant: str) -> dict:
    return json.loads((VERIFIER_DATA / variant / "oracle" / "manager_brief.json").read_text())


def load_gold(variant: str) -> dict:
    return json.loads((VERIFIER_DATA / variant / "gold_ranking.json").read_text())


def write_brief(ws: Path, payload: dict) -> None:
    brief_dir = ws / "brief"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "manager_brief.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (brief_dir / "manager_brief.md").write_text("synthetic verification matrix brief\n")


def make_oracle(ws: Path, variant: str) -> None:
    write_brief(ws, load_oracle(variant))


def make_empty(ws: Path, variant: str) -> None:
    del variant


def make_grounding_stripped(ws: Path, variant: str) -> None:
    payload = load_oracle(variant)
    for entry in payload.get("ranking", []):
        entry["citations"] = []
    write_brief(ws, payload)


def make_pick_forbidden(ws: Path, variant: str) -> None:
    payload = load_oracle(variant)
    gold = load_gold(variant)
    forbidden = gold["forbidden_override_proposal"]
    ranking = payload.get("ranking", [])
    new_ranking = [entry for entry in ranking if entry["proposal_id"] == forbidden]
    new_ranking.extend(entry for entry in ranking if entry["proposal_id"] != forbidden)
    for idx, entry in enumerate(new_ranking, start=1):
        entry["rank"] = idx
    payload["accepted"] = forbidden
    payload["ranking"] = new_ranking
    write_brief(ws, payload)


def make_top1_wrong(ws: Path, variant: str) -> None:
    payload = load_oracle(variant)
    ranking = payload.get("ranking", [])
    if len(ranking) >= 2:
        ranking[0], ranking[1] = ranking[1], ranking[0]
        ranking[0]["rank"] = 1
        ranking[1]["rank"] = 2
        payload["accepted"] = ranking[0]["proposal_id"]
    write_brief(ws, payload)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    target = ws / "tests" / "test_policy_resolution.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_rawr_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_forbidden", make_pick_forbidden, "Pick forbidden override"),
    ("05_top1_wrong", make_top1_wrong, "Top1 wrong"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder, tag: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vmx_{tag}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    out_path = Path(args.out)
    for tag, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder, tag)
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
                "ceilings_applied": result.get("ceilings_applied", []),
                "integrity_rules_fired": result.get("integrity_rules_fired", []),
            }
        )

    out_path.write_text(
        "# §5 verification matrix — {variant}\n\n".format(variant=args.variant)
        + "Generated by `benchmark_blueprints/families/policy-aware-request-resolution/scripts/run_verification_matrix.py`.\n\n"
        + "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---|---|\n"
        + "\n".join(
            "| {label} | {P_benchmark} | {M_training:.4f} | {G:.3f} | {R:.3f} | {S_TTC} | {integrity_flag} | {passed} | {ceilings} |".format(
                label=row["label"],
                P_benchmark=row["P_benchmark"],
                M_training=row["M_training"],
                G=row["G"],
                R=row["R"],
                S_TTC=row["S_TTC"],
                integrity_flag=row["integrity_flag"],
                passed=row["pass"],
                ceilings=("H=" + ",".join(row["integrity_rules_fired"])) if row["integrity_rules_fired"] else (",".join(row["ceilings_applied"]) or "—"),
            )
            for row in rows
        )
        + "\n\n## Expected bands\n\n"
        + "- Oracle: P >= 90, M >= 0.90\n"
        + "- Empty: P = 0, M = 0\n"
        + "- RAWR grounding_stripped: ranking_without_grounding ceiling fires\n"
        + "- Pick forbidden override: ignored_forbidden_override ceiling fires and P <= 30\n"
        + "- Delete-tests adversarial: integrity_flag = 1\n",
        encoding="utf-8",
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
