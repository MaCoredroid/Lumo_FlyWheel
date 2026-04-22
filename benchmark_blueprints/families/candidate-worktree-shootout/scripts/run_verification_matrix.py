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
FAMILY = REPO / "benchmark_blueprints/families/candidate-worktree-shootout"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/candidate-worktree-shootout"
SCORER = REPO / "verifiers/candidate-worktree-shootout/score_shootout.py"


def overlay_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def make_oracle(ws: Path, variant: str) -> None:
    overlay_tree(VERIFIER_DATA / variant / "oracle_overlay", ws)


def make_grounding_stripped(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    stripped = """# Candidate A Evaluation\n- Worktree Path: `/tmp/a`\n\n## Commands Run\n- `python -m pytest -q tests/test_cli.py`\n"""
    (ws / "artifacts/comparison/candidate_a_eval.md").write_text(stripped)
    (ws / "artifacts/comparison/candidate_b_eval.md").write_text(stripped.replace("A", "B", 1).replace("/tmp/a", "/tmp/b"))
    (ws / "artifacts/comparison/comparison_summary.md").write_text(
        "# Comparison Summary\n\n- Chosen Strategy: `candidate_b_service_layer`\n- Rejected Strategy: `candidate_a_cli_local`\n"
    )
    (ws / "artifacts/comparison/verification_note.md").write_text(
        "# Verification Note\n\n- Candidate A Worktree: `/tmp/a`\n- Candidate B Worktree: `/tmp/b`\n"
    )


def make_pick_ceiling(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    cli = ws / "src/report_filters/cli.py"
    cli.write_text(
        """from __future__ import annotations\n\nfrom .normalization import normalize_label\nfrom .service import build_filter_query\n\n\ndef render_filters(raw: str) -> str:\n    parts = [normalize_label(piece) for piece in raw.split(\",\")]\n    return build_filter_query(parts)\n"""
    )
    service = ws / "src/report_filters/service.py"
    service.write_text(
        """from __future__ import annotations\n\n\ndef compile_filters(labels: list[str]) -> list[str]:\n    compiled: list[str] = []\n    for label in labels:\n        cleaned = label.strip().lower()\n        if cleaned:\n            compiled.append(cleaned)\n    return compiled\n\n\ndef build_filter_query(labels: list[str]) -> str:\n    return \",\".join(compile_filters(labels))\n"""
    )
    (ws / "tests/test_service.py").write_text(
        "from report_filters.service import compile_filters\n\n\ndef test_compile_filters_handles_basic_whitespace() -> None:\n    assert compile_filters([' Already Clean ']) == ['already clean']\n"
    )
    (ws / "artifacts/comparison/comparison_summary.md").write_text(
        "# Comparison Summary\n\n- Chosen Strategy: `candidate_a_cli_local`\n- Rejected Strategy: `candidate_b_service_layer`\n\nCLI-only looked faster.\n"
    )


def make_blended_patch(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    cli = ws / "src/report_filters/cli.py"
    cli.write_text(
        """from __future__ import annotations\n\nfrom .normalization import normalize_label\nfrom .service import build_filter_query\n\n\ndef render_filters(raw: str) -> str:\n    parts = [normalize_label(piece) for piece in raw.split(\",\")]\n    return build_filter_query(parts)\n"""
    )


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    test_file = ws / "tests/test_cli.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


def make_empty(ws: Path, variant: str) -> None:
    return


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle (full pass)"),
    ("02_empty", make_empty, "Empty (no repair)"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_ceiling", make_pick_ceiling, "Pick ceiling (CLI-local)"),
    ("05_blended_patch", make_blended_patch, "Blended ownership"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_one(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"cws_matrix_{variant}_") as tmp:
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
        subprocess.run(
            [sys.executable, str(SCORER)],
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(result_file.read_text())


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    if result.get("pass", False):
        return 1.0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = float(result.get("integrity_flag", 0))
    return round(0.6 * m - 0.6 * h, 4)


def derive_s_ttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-clean-baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score_one(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_s_ttc(result),
                "integrity_flag": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": result["ceilings_applied"] or result["integrity_rules_fired"] or ["—"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(f"# §5 verification matrix — {args.variant}\n\n")
        fh.write(
            "Generated by `benchmark_blueprints/families/candidate-worktree-shootout/scripts/run_verification_matrix.py` "
            f"against `{args.variant}`.\n\n"
        )
        fh.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            fh.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | "
                f"{row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | "
                f"{','.join(row['ceilings'])} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
