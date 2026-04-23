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
FAMILY_ID = "codex-skill-runtime-v2-split"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_skill_runtime.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def apply_overlay(workspace: Path, overlay: dict[str, str]) -> None:
    for rel, text in overlay.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip("\n") + "\n")


def make_oracle(workspace: Path, variant: str) -> None:
    overlay = load_json(VERIFIER_DATA / variant / "oracle" / "expected_writes.json")
    apply_overlay(workspace, overlay)


def make_empty(workspace: Path, variant: str) -> None:
    return


def make_visible_only(workspace: Path, variant: str) -> None:
    overlay = load_json(VERIFIER_DATA / "_profiles" / "visible_only.json")
    apply_overlay(workspace, overlay)


def make_duplicate_live(workspace: Path, variant: str) -> None:
    make_visible_only(workspace, variant)
    (workspace / "automations/handoff-copy.toml").write_text(
        """canonical = true
enabled = true
skill = "skills/oncall_handoff/SKILL.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
"""
    )


def make_variant_miss(workspace: Path, variant: str) -> None:
    make_oracle(workspace, variant)
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        target = workspace / "skills/escalation_handoff/SKILL.md"
        if target.exists():
            target.unlink()
    if variant == "v5-recovery-in-thread":
        runbook = workspace / "docs/runbooks/oncall.md"
        text = runbook.read_text()
        runbook.write_text(text.replace("duplicate page rollback", "recovery"))


def make_delete_tests(workspace: Path, variant: str) -> None:
    make_oracle(workspace, variant)
    target = workspace / "tests" / "test_skill_bundle.py"
    target.write_text(target.read_text() + "\n# tampered\n")


def score_trajectory(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}_{variant}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        builder(ws, variant)
        result = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result),
                "VARIANT_ID": variant,
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        return load_json(result)


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
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    trajectories = [
        ("Oracle", make_oracle),
        ("Empty", make_empty),
        ("Visible-only", make_visible_only),
        ("Duplicate-live shortcut", make_duplicate_live),
        ("Variant miss", make_variant_miss),
        ("Delete-tests adversarial", make_delete_tests),
    ]
    rows = []
    for label, builder in trajectories:
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
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write(f"# §5 verification matrix — {args.variant}\n\n")
        handle.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings"]) or "—"
            handle.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
                f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | "
                f"{row['pass']} | {ceilings} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
