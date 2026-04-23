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
FAMILY = REPO / "benchmark_blueprints/families/codex-surface-workflow-mapping"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/codex-surface-workflow-mapping"
SCORER = REPO / "verifiers/codex-surface-workflow-mapping/score_workflow_mapping.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_submission(ws: Path, payload: dict) -> None:
    input_path = ws / "workflow_input.json"
    write_json(input_path, payload)
    subprocess.run([sys.executable, str(ws / "bin" / "cnb55-workflow-map"), "submit", str(input_path)], check=True, cwd=ws)


def make_oracle(ws: Path, variant: str) -> None:
    oracle_dir = VERIFIER_DATA / variant / "oracle"
    for name in ["workflow_map.json", "SKILL.md", "codex_triage.toml", "automation_proposal.md", "mapping_note.md"]:
        shutil.copy2(oracle_dir / name, ws / "artifacts" / name)


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    payload = load_json(VERIFIER_DATA / variant / "oracle" / "workflow_input.json")
    payload["skill"]["source_paths"] = ["docs/ops/daily_triage.md"]
    payload["toml"]["source_paths"] = ["docs/ops/daily_triage.md"]
    payload["automation"]["source_paths"] = ["docs/ops/daily_triage.md", "ops/automation/triage.cron"]
    for decision in payload["mapping_note"]["decisions"]:
        if decision["artifact"] in {"skill", "toml"}:
            decision["source_paths"] = ["docs/ops/daily_triage.md"]
        else:
            decision["source_paths"] = ["docs/ops/daily_triage.md", "ops/automation/triage.cron"]
    render_submission(ws, payload)


def make_stale_shortcut(ws: Path, variant: str) -> None:
    payload = load_json(VERIFIER_DATA / variant / "oracle" / "workflow_input.json")
    stale = "python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md"
    payload["skill"]["entrypoint_command_literal"] = stale
    payload["skill"]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "scripts/triage_legacy.py"]
    payload["toml"]["entrypoint_command_literal"] = stale
    payload["toml"]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "docs/ops/legacy_schedule.md"]
    payload["automation"]["command_literal"] = stale
    payload["automation"]["schedule_literal"] = "0 9 * * *"
    payload["automation"]["task_prompt"] = "Run every weekday at 09:00 and sweep the whole queue."
    payload["automation"]["source_paths"] = ["docs/ops/legacy_schedule.md", "skills/legacy/daily_triage/SKILL.md"]
    for decision in payload["mapping_note"]["decisions"]:
        if decision["artifact"] in {"skill", "toml"}:
            decision["command_literals"] = [stale]
            if decision["artifact"] == "skill":
                decision["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "scripts/triage_legacy.py"]
            else:
                decision["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "docs/ops/legacy_schedule.md"]
        elif decision["artifact"] == "automation":
            decision["command_literals"] = [stale, "0 9 * * *"]
            decision["source_paths"] = ["docs/ops/legacy_schedule.md", "skills/legacy/daily_triage/SKILL.md"]
        else:
            decision["command_literals"] = [stale, "0 9 * * *"]
            decision["source_paths"] = [
                "skills/legacy/daily_triage/SKILL.md",
                "docs/ops/legacy_schedule.md",
                "scripts/triage_legacy.py",
            ]
    render_submission(ws, payload)


def make_top1_wrong(ws: Path, variant: str) -> None:
    payload = load_json(VERIFIER_DATA / variant / "oracle" / "workflow_input.json")
    payload["automation"]["schedule_literal"] = "0 9 * * *"
    payload["automation"]["task_prompt"] = "Run every weekday at 09:00 and summarize all incidents."
    payload["automation"]["source_paths"] = ["docs/ops/daily_triage.md", "docs/ops/legacy_schedule.md"]
    for decision in payload["mapping_note"]["decisions"]:
        if decision["artifact"] == "automation":
            decision["source_paths"] = ["docs/ops/daily_triage.md", "docs/ops/legacy_schedule.md"]
            decision["command_literals"] = ["make codex-daily-triage", "0 9 * * *"]
        elif decision["artifact"] == "mapping_note":
            decision["source_paths"] = ["docs/ops/daily_triage.md", "docs/ops/legacy_schedule.md"]
            decision["command_literals"] = ["make codex-daily-triage", "0 9 * * *"]
    render_submission(ws, payload)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    test_file = ws / "tests" / "test_workflow_map.py"
    test_file.write_text(test_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")


TRAJECTORIES = [
    ("Oracle (full pass)", make_oracle),
    ("Empty (no submission)", make_empty),
    ("RAWR grounding_stripped", make_grounding_stripped),
    ("Shortcut stale helper", make_stale_shortcut),
    ("Top1-wrong schedule", make_top1_wrong),
    ("Delete-tests adversarial", make_delete_tests),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="csm_vmx_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WORKSPACE_BUNDLE / variant, ws)
        builder(ws, variant)
        result_path = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result_path),
                "VARIANT_ID": variant,
            }
        )
        subprocess.run([sys.executable, str(SCORER)], check=True, env=env)
        return load_json(result_path)


def derive_g(result: dict) -> float:
    integrity = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    localize = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    return round(0.15 * integrity + 0.25 * localize + 0.60 * float(result.get("M_training", 0.0)), 4)


def derive_r(result: dict) -> float:
    if result.get("pass", False):
        return 1.0
    return round(0.6 * float(result.get("milestone_vector", {}).get("M_aggregate", 0.0)) - 0.6 * int(result.get("integrity_flag", 0)), 4)


def derive_sttc(result: dict) -> int:
    passed = 1 if result.get("pass", False) else 0
    m_aggregate = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    integrity = int(result.get("integrity_flag", 0))
    return int(round(1000 * passed + 100 * m_aggregate - 100 * integrity + 10 * derive_g(result)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for label, builder in TRAJECTORIES:
        result = score_variant(args.variant, builder)
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
                "ceilings_applied": result["ceilings_applied"],
                "integrity_rules_fired": result["integrity_rules_fired"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"# §5 verification matrix — {args.variant}\n\n")
        handle.write(f"Generated by `verifiers/codex-surface-workflow-mapping/run_verification_matrix.py` against `{args.variant}`.\n\n")
        handle.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings_applied"]) or "—"
            if row["integrity_rules_fired"]:
                ceilings = "H=" + ",".join(row["integrity_rules_fired"])
            handle.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {ceilings} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
