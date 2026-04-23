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
FAMILY_ID = "release-manifest-v2-modernization"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_release_modernization.py"


def overlay_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def make_oracle(ws: Path, variant: str) -> None:
    overlay_tree(VER_DATA / variant / "oracle", ws)


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    src = VER_DATA / variant / "oracle"
    for rel in [
        ".github/workflows/release.yml",
        ".github/workflows/reusable_release.yml",
        "release/manifest.v2.toml",
        "codex/config.toml",
    ]:
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)


def make_prod_alias_shortcut(ws: Path, variant: str) -> None:
    make_grounding_stripped(ws, variant)
    workflow = ws / ".github/workflows/release.yml"
    workflow.write_text(
        workflow.read_text().replace("target_environment: staging", "target_environment: prod")
    )


def make_context_blind(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    docs = ws / "docs/releases/staging_rollout.md"
    text = docs.read_text()
    for needle in [
        "Check the `artifact_manifest output` before the staging smoke step so the dry-run path and the release contract stay aligned.\n",
        "INC-342 requires that operators do not reintroduce the prod alias anywhere in the live path.\n",
    ]:
        text = text.replace(needle, "")
    docs.write_text(text)
    proof = ws / "artifacts/release_smoke_report.json"
    if proof.exists():
        payload = json.loads(proof.read_text())
        payload["incident_guardrail"] = None
        if payload.get("ordered_checks") == ["dry-run", "artifact_manifest", "staging smoke"]:
            payload["ordered_checks"] = ["dry-run", "staging smoke"]
        proof.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    (ws / "tests/test_manifest_contract.py").write_text("# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle (full pass)"),
    ("02_empty", make_empty, "Empty (no submission)"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_prod_alias_shortcut", make_prod_alias_shortcut, "Prod-alias shortcut"),
    ("05_context_blind", make_context_blind, "Context-blind finish"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_workspace(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"release_vmx_{variant}_") as tmp:
        tmp_path = Path(tmp)
        ws = tmp_path / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        builder(ws, variant)
        result_path = tmp_path / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_path),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        subprocess.run(
            [sys.executable, str(SCORER)],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(result_path.read_text())


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
    p = 1 if result.get("pass", False) else 0
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_G(result)
    return int(round(1000 * p + 100 * m_agg - 100 * h + 10 * g))


def matrix_markdown(variant: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"# §5 verification matrix — {variant}",
        "",
        f"Generated by `verifiers/{FAMILY_ID}/run_verification_matrix.py --variant {variant}`.",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        ceilings = ",".join(row["ceilings"]) if row["ceilings"] else "—"
        if row["integrity_rules"]:
            ceilings = f"H={','.join(row['integrity_rules'])}"
        lines.append(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
            f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | "
            f"{row['passed']} | {ceilings} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Oracle should clear 90+ P_benchmark with all five milestones true.",
            "- Empty should score 0.",
            "- RAWR grounding_stripped keeps the workflow cutover but omits docs and proof, so it should cap at or below 30.",
            "- Prod-alias shortcut should trip `prod_alias_retained` and stay at or below 25.",
            "- Context-blind finish is the stress row for V4/V5 and should trip `variant_context_missed`.",
            "- Delete-tests should raise integrity and zero M3/M4/M5.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for _, builder, label in TRAJECTORIES:
        result = score_workspace(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": int(result["P_benchmark"]),
                "M_training": float(result["M_training"]),
                "G": derive_G(result),
                "R": derive_R(result),
                "S_TTC": derive_S_TTC(result),
                "integrity": int(result["integrity_flag"]),
                "passed": bool(result["pass"]),
                "ceilings": list(result.get("ceilings_applied", [])),
                "integrity_rules": list(result.get("integrity_rules_fired", [])),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(matrix_markdown(args.variant, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
