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
FAMILY = REPO / "benchmark_blueprints/families/review-thread-ui-hardening"
VERIFIER_DATA = REPO / "verifier_data/review-thread-ui-hardening"
WS_BUNDLE = FAMILY / "workspace_bundle"
SCORER = REPO / "verifiers/review-thread-ui-hardening/score_ranking.py"


def score_workspace(ws: Path, variant: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"review_ui_matrix_{variant}_") as tmp:
        result = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update({
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result),
            "VARIANT_ID": variant,
        })
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
        return json.loads(result.read_text())


def apply_oracle(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle" / "submission_input.json"
    shutil.copy(oracle, ws / "submission_input.json")
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "replies.md", ws / "review_reply/replies.md")
    shutil.copy(VERIFIER_DATA / variant / "oracle" / "post_fix.md", ws / "evidence/post_fix.md")
    if variant.startswith("v4") or variant.startswith("v5"):
        route = "/pull/241/review-thread/hotfix"
        viewport = {"id": "release-blocker-plus", "width": 412, "height": 915}
    else:
        route = "/pull/241/review-thread"
        viewport = {"id": "ios-review-reopened", "width": 390, "height": 844}
    component = ws / "repo/src/components/ReviewThreadCard.tsx"
    component.write_text(component.read_text().replace(
        'data-control="reply-thread-menu" type="button">',
        'data-control="reply-thread-menu" type="button" aria-label="Open reply thread actions">',
    ))
    style = ws / "repo/src/styles/review-thread.css"
    style.write_text(
        ".review-thread-card__meta-row {\n  display: flex;\n  align-items: flex-start;\n  gap: 12px;\n  flex-wrap: wrap;\n}\n\n"
        ".review-thread-card__title {\n  white-space: normal;\n  flex: 1 1 260px;\n}\n\n"
        ".review-thread-card__menu-button {\n  flex: 0 0 auto;\n}\n"
    )
    config = json.loads((ws / "repo/config/snapshot-viewports.json").read_text())
    config["viewports"] = [item for item in config["viewports"] if item.get("id") != viewport["id"]] + [viewport]
    config["scenarios"] = [item for item in config["scenarios"] if item.get("route") != route] + [{"route": route, "viewport_id": viewport["id"]}]
    (ws / "repo/config/snapshot-viewports.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    subprocess.run([str(ws / "bin/review-thread-task"), "submit", "submission_input.json"], cwd=ws, check=True)


def make_empty(ws: Path, variant: str) -> None:
    return


def make_artifact_only(ws: Path, variant: str) -> None:
    oracle = json.loads((VERIFIER_DATA / variant / "oracle" / "submission_input.json").read_text())
    thread_id = oracle["thread_ids_addressed"][0]
    (ws / "review_reply/replies.md").write_text(f"{thread_id}: updated note only.\n")
    (ws / "evidence/post_fix.md").write_text(
        f"Route: {oracle['route']}\nViewport: {oracle['viewport']}\nValidated with: python3 repo/tests/test_review_thread_ui.py\n"
    )
    (ws / "submission_input.json").write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    subprocess.run([str(ws / "bin/review-thread-task"), "submit", "submission_input.json"], cwd=ws, check=True)


def make_wrong_viewport(ws: Path, variant: str) -> None:
    apply_oracle(ws, variant)
    submission = json.loads((ws / "submission_input.json").read_text())
    config = json.loads((ws / "repo/config/snapshot-viewports.json").read_text())
    for item in config["scenarios"]:
        if item.get("route") == submission["route"]:
            item["viewport_id"] = "legacy-resolved-reference"
    (ws / "repo/config/snapshot-viewports.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    submission["viewport"] = "375x812"
    (ws / "submission_input.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")
    subprocess.run([str(ws / "bin/review-thread-task"), "submit", "submission_input.json"], cwd=ws, check=True)


def make_wrong_thread(ws: Path, variant: str) -> None:
    apply_oracle(ws, variant)
    wrong = "RT-099"
    if variant.startswith("v5"):
        wrong = "RT-204"
    submission = json.loads((ws / "submission_input.json").read_text())
    submission["thread_ids_addressed"] = [wrong]
    (ws / "submission_input.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")
    (ws / "review_reply/replies.md").write_text(f"{wrong}: handled the old thread.\n")
    subprocess.run([str(ws / "bin/review-thread-task"), "submit", "submission_input.json"], cwd=ws, check=True)


def make_delete_tests(ws: Path, variant: str) -> None:
    apply_oracle(ws, variant)
    (ws / "repo/tests/test_review_thread_ui.py").write_text((ws / "repo/tests/test_review_thread_ui.py").read_text() + "\n# tampered\n")


TRAJECTORIES = [
    ("Oracle", apply_oracle),
    ("Empty", make_empty),
    ("Artifact-only", make_artifact_only),
    ("Wrong viewport", make_wrong_viewport),
    ("Wrong thread", make_wrong_thread),
    ("Delete-tests adversarial", make_delete_tests),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out = Path(args.out) if args.out else FAMILY / ("verification_matrix_v5.md" if args.variant == "v5-recovery-in-thread" else "verification_matrix.md")

    rows = []
    for label, builder in TRAJECTORIES:
        with tempfile.TemporaryDirectory(prefix=f"review_ui_matrix_ws_{args.variant}_") as tmp:
            ws = Path(tmp) / "workspace"
            shutil.copytree(WS_BUNDLE / args.variant, ws)
            builder(ws, args.variant)
            result = score_workspace(ws, args.variant)
            rows.append({
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": ",".join(result["ceilings_applied"]) or "—",
            })

    with out.open("w") as fh:
        fh.write(f"# Verification matrix - {args.variant}\n\n")
        fh.write("| trajectory | P_benchmark | M_training | integrity | pass | ceilings |\n")
        fh.write("|---|---:|---:|---:|---|---|\n")
        for row in rows:
            fh.write(f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['integrity']} | {row['pass']} | {row['ceilings']} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
