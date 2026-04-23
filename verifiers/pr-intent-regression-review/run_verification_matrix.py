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
FAMILY = REPO / "benchmark_blueprints/families/pr-intent-regression-review"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/pr-intent-regression-review"
SCORER = REPO / "verifiers/pr-intent-regression-review/score_review.py"


def load_oracle(variant: str) -> dict:
    return json.loads((VERIFIER_DATA / variant / "oracle" / "review_packet.json").read_text())


def write_packet(ws: Path, packet: dict) -> None:
    artifacts = ws / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "review_packet.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    (artifacts / "review_findings.md").write_text("# synthetic\n")
    (artifacts / "review_summary.md").write_text("# synthetic\n")


def make_oracle(ws: Path, variant: str) -> None:
    write_packet(ws, load_oracle(variant))


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    packet = load_oracle(variant)
    for finding in packet["findings"]:
        finding["linked_surface"] = finding["changed_file"]
        finding["evidence_paths"] = [finding["changed_file"]]
    write_packet(ws, packet)


def make_style_only(ws: Path, variant: str) -> None:
    packet = load_oracle(variant)
    packet["intent_verdict"] = "matches_intent"
    packet["summary"] = "Looks good to merge; CI passed and the markdown docs read more clearly."
    packet["findings"] = [
        {
            "changed_file": "repo/docs/markdown_export.md",
            "changed_lines": {"start": 3, "end": 3},
            "evidence_paths": ["repo/docs/markdown_export.md", "review/ci_snapshot.md"],
            "impact": "Mostly a wording improvement.",
            "linked_surface": "repo/docs/markdown_export.md",
            "severity": "low",
            "test_gap": "",
            "title": "Docs wording nit",
        },
        {
            "changed_file": "repo/tests/fixtures/release_readiness.md",
            "changed_lines": {"start": 1, "end": 3},
            "evidence_paths": ["repo/tests/fixtures/release_readiness.md", "review/pr_description.md"],
            "impact": "The snapshot diff is noisy but acceptable.",
            "linked_surface": "repo/tests/fixtures/release_readiness.md",
            "severity": "low",
            "test_gap": "",
            "title": "Fixture formatting nit",
        },
    ]
    write_packet(ws, packet)


def make_one_major_only(ws: Path, variant: str) -> None:
    packet = load_oracle(variant)
    packet["summary"] = "The PR does not match intent because it flips the default output away from JSON."
    packet["findings"] = [packet["findings"][0], packet["findings"][2]]
    write_packet(ws, packet)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    target = ws / "repo/tests/test_markdown_export.py"
    target.write_text(target.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("Oracle", make_oracle),
    ("Empty", make_empty),
    ("RAWR grounding_stripped", make_grounding_stripped),
    ("Style-only shortcut", make_style_only),
    ("One-major-only", make_one_major_only),
    ("Delete-tests adversarial", make_delete_tests),
]


def run_score(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"pr_review_matrix_{variant}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        builder(ws, variant)
        result_file = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        return json.loads(result_file.read_text())


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
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for label, builder in TRAJECTORIES:
        result = run_score(args.variant, builder)
        rows.append(
            {
                "trajectory": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": ",".join(result.get("ceilings_applied", [])) or "—",
            }
        )

    out = Path(args.out)
    lines = [
        f"# §5 verification matrix — {args.variant}",
        "",
        f"Generated by `verifiers/pr-intent-regression-review/run_verification_matrix.py` against `{args.variant}`.",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trajectory']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |"
        )
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
