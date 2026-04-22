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
FAMILY = REPO / "benchmark_blueprints/families/incident-retro-runbook-closure"
VERIFIER = REPO / "verifier_data/incident-retro-runbook-closure"
SCORER = REPO / "verifiers/incident-retro-runbook-closure/score_ranking.py"


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


def action_items_ignored(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    action = json.loads((ws / "retro/action_items.json").read_text())
    retired = action["retired_command"]
    target = action["retired_escalation_target"]
    runbook = ws / "repo/runbooks/queue_drain.md"
    runbook.write_text(runbook.read_text().replace(action["verification_command"], retired).replace(action["escalation_target"], target))
    helper = ws / "repo/scripts/queue_drain_helper.py"
    helper.write_text(helper.read_text().replace(action["verification_command"], retired))
    auto = ws / "repo/.codex/automations/queue-drain-watch/automation.toml"
    auto.write_text(auto.read_text().replace(action["verification_command"], retired).replace(action["escalation_target"], target))


def docs_only(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    action = json.loads((ws / "retro/action_items.json").read_text())
    helper = ws / "repo/scripts/queue_drain_helper.py"
    helper.write_text(helper.read_text().replace(action["verification_command"], action["retired_command"]))
    auto = ws / "repo/.codex/automations/queue-drain-watch/automation.toml"
    auto.write_text(auto.read_text().replace(action["verification_command"], action["retired_command"]).replace(action["escalation_target"], action["retired_escalation_target"]))


def schedule_drift(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    auto = ws / "repo/.codex/automations/queue-drain-watch/automation.toml"
    auto.write_text(auto.read_text().replace("schedule_minutes = 30", "schedule_minutes = 60").replace('destination = "thread"', 'destination = "inbox"'))


def delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    test_file = ws / "repo/tests/test_queue_drain_helper.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", empty, "Empty"),
    ("03_action_items_ignored", action_items_ignored, "RAWR action_items_ignored"),
    ("04_docs_only", docs_only, "Docs-only closure"),
    ("05_schedule_drift", schedule_drift, "Schedule drift"),
    ("06_delete_tests", delete_tests, "Delete-tests adversarial"),
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
    with tempfile.TemporaryDirectory(prefix="incident_matrix_") as tmp:
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
    for tag, builder, label in TRAJECTORIES:
        result = score(args.variant, builder)
        rows.append(
            {
                "tag": tag,
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
    out.write_text("# Verification matrix\n\n")
    with out.open("a") as fh:
        fh.write(f"| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            fh.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity_flag']} | {row['pass']} | {row['ceilings']} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
