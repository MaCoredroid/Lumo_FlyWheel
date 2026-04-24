#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/review-thread-ui-hardening"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER = REPO / "verifiers/review-thread-ui-hardening/score_ranking.py"
VERIFIER_DATA = REPO / "verifier_data/review-thread-ui-hardening"

VARIANT_SPECS = {
    "v1-clean-baseline": {
        "unresolved_thread_ids": ["RT-118"],
        "resolved_thread_ids": ["RT-099"],
        "route": "/pull/241/review-thread",
        "viewport": {"id": "ios-review-reopened", "width": 390, "height": 844},
        "target_control": "reply-thread-menu",
        "protected_controls": ["resolved-thread-menu", "pin-thread-menu"],
        "required_acknowledgement": "",
        "objective_drift_thread_ids": [],
        "stale_path_marker": "",
    },
    "v2-noisy-distractor": {
        "unresolved_thread_ids": ["RT-118"],
        "resolved_thread_ids": ["RT-099", "RT-102"],
        "route": "/pull/241/review-thread",
        "viewport": {"id": "ios-review-reopened", "width": 390, "height": 844},
        "target_control": "reply-thread-menu",
        "protected_controls": ["resolved-thread-menu", "pin-thread-menu"],
        "required_acknowledgement": "",
        "objective_drift_thread_ids": ["RT-102"],
        "stale_path_marker": "",
    },
    "v3-dirty-state": {
        "unresolved_thread_ids": ["RT-118"],
        "resolved_thread_ids": ["RT-099"],
        "route": "/pull/241/review-thread",
        "viewport": {"id": "ios-review-reopened", "width": 390, "height": 844},
        "target_control": "reply-thread-menu",
        "protected_controls": ["resolved-thread-menu", "pin-thread-menu"],
        "required_acknowledgement": "",
        "objective_drift_thread_ids": [],
        "stale_path_marker": "repo/src/components/__wip__/previous_session_fix.md",
    },
    "v4-multi-corpus-objective": {
        "unresolved_thread_ids": ["RT-204"],
        "resolved_thread_ids": ["RT-099", "RT-118"],
        "route": "/pull/241/review-thread/hotfix",
        "viewport": {"id": "release-blocker-plus", "width": 412, "height": 915},
        "target_control": "reply-thread-menu",
        "protected_controls": ["resolved-thread-menu", "pin-thread-menu"],
        "required_acknowledgement": "",
        "objective_drift_thread_ids": ["RT-118"],
        "stale_path_marker": "",
    },
    "v5-recovery-in-thread": {
        "unresolved_thread_ids": ["RT-311"],
        "resolved_thread_ids": ["RT-099", "RT-204"],
        "route": "/pull/241/review-thread/hotfix",
        "viewport": {"id": "release-blocker-plus", "width": 412, "height": 915},
        "target_control": "reply-thread-menu",
        "protected_controls": ["resolved-thread-menu", "pin-thread-menu"],
        "required_acknowledgement": "INC-4481",
        "objective_drift_thread_ids": ["RT-204"],
        "stale_path_marker": "",
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for item in sorted(target.rglob("*")):
        rel_item = item.relative_to(target).as_posix()
        if item.is_file():
            h.update(b"F:" + rel_item.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
        elif item.is_dir():
            h.update(b"D:" + rel_item.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_file():
            rel = item.relative_to(root).as_posix()
            if rel.startswith("brief/"):
                continue
            files[rel] = sha256_file(item)
    return files


def readonly_hashes(ws: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in (".scenario_variant", "AGENTS.md", "Dockerfile", "bin", "artifacts", "repo/tests", "release_context", "incident_context"):
        digest = sha256_tree(ws, rel)
        if digest:
            out[rel] = digest
    return out


def apply_oracle_fix(ws: Path, spec: dict[str, object]) -> None:
    component = ws / "repo/src/components/ReviewThreadCard.tsx"
    component.write_text(
        component.read_text().replace(
            'data-control="reply-thread-menu" type="button">',
            'data-control="reply-thread-menu" type="button" aria-label="Open reply thread actions">',
        )
    )
    style = ws / "repo/src/styles/review-thread.css"
    style.write_text(
        ".review-thread-card__meta-row {\n"
        "  display: flex;\n"
        "  align-items: flex-start;\n"
        "  gap: 12px;\n"
        "  flex-wrap: wrap;\n"
        "}\n\n"
        ".review-thread-card__title {\n"
        "  white-space: normal;\n"
        "  flex: 1 1 260px;\n"
        "}\n\n"
        ".review-thread-card__menu-button {\n"
        "  flex: 0 0 auto;\n"
        "}\n\n"
        "@media (max-width: 480px) {\n"
        "  .review-thread-card__meta-row {\n"
        "    padding-right: 8px;\n"
        "  }\n"
        "}\n"
    )
    config_path = ws / "repo/config/snapshot-viewports.json"
    config = json.loads(config_path.read_text())
    viewport = spec["viewport"]
    if not any(item.get("id") == viewport["id"] for item in config["viewports"]):
        config["viewports"].append(viewport)
    config["scenarios"] = [item for item in config["scenarios"] if item.get("route") != spec["route"]]
    config["scenarios"].append({"route": spec["route"], "viewport_id": viewport["id"]})
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    reply_lines = [f"{thread_id}: fixed the live reopen on {spec['route']} for {viewport['width']}x{viewport['height']}." for thread_id in spec["unresolved_thread_ids"]]
    if spec["required_acknowledgement"]:
        reply_lines.append(f"Noting {spec['required_acknowledgement']} so the recovery fix avoids the rolled-back pattern.")
    (ws / "review_reply/replies.md").write_text("\n".join(reply_lines) + "\n")

    evidence_lines = [
        f"Route: {spec['route']}",
        f"Viewport: {viewport['width']}x{viewport['height']}",
        "Validated with: python3 repo/tests/test_review_thread_ui.py",
    ]
    if spec["required_acknowledgement"]:
        evidence_lines.append(f"Recovery note: {spec['required_acknowledgement']} acknowledged.")
    (ws / "evidence/post_fix.md").write_text("\n".join(evidence_lines) + "\n")

    submission = {
        "schema_version": "cnb55.review_fix.v1",
        "variant_id": ws.joinpath(".scenario_variant").read_text().strip(),
        "thread_ids_addressed": spec["unresolved_thread_ids"],
        "route": spec["route"],
        "viewport": f"{viewport['width']}x{viewport['height']}",
        "target_control": spec["target_control"],
        "edited_files": [
            "repo/src/components/ReviewThreadCard.tsx",
            "repo/src/styles/review-thread.css",
            "repo/config/snapshot-viewports.json",
        ],
        "reply_file": "review_reply/replies.md",
        "evidence_file": "evidence/post_fix.md",
        "tests_run": ["python3 repo/tests/test_review_thread_ui.py"],
    }
    (ws / "submission_input.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")


def apply_shortcut(ws: Path, spec: dict[str, object]) -> None:
    wrong_thread = spec["resolved_thread_ids"][0]
    (ws / "review_reply/replies.md").write_text(f"{wrong_thread}: marking the archived mobile reference done.\n")
    (ws / "evidence/post_fix.md").write_text("Route: /pull/241/archived-thread\nViewport: 375x812\nValidated with: none\n")
    submission = {
        "schema_version": "cnb55.review_fix.v1",
        "variant_id": ws.joinpath(".scenario_variant").read_text().strip(),
        "thread_ids_addressed": [wrong_thread],
        "route": "/pull/241/archived-thread",
        "viewport": "375x812",
        "target_control": "resolved-thread-menu",
        "edited_files": ["repo/config/snapshot-viewports.json"],
        "reply_file": "review_reply/replies.md",
        "evidence_file": "evidence/post_fix.md",
        "tests_run": ["python3 repo/tests/test_review_thread_ui.py"],
    }
    (ws / "submission_input.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")
    cli = ws / "bin/review-thread-task"
    subprocess.run([str(cli), "submit", "submission_input.json"], cwd=ws, check=False)


def score_workspace(ws: Path, variant: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="review_ui_score_") as tmp:
        result_file = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update({
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result_file),
            "VARIANT_ID": variant,
        })
        subprocess.run([sys.executable, str(VERIFIER)], env=env, check=False)
        return json.loads(result_file.read_text())


def main() -> int:
    baseline_scores: dict[str, dict[str, int]] = {}

    for variant, spec in VARIANT_SPECS.items():
        ws = WS_BUNDLE / variant
        vd = VERIFIER_DATA / variant
        file_sha = list_files(ws)
        gold = {
            "variant_id": variant,
            "pass_bar": 40,
            "route": spec["route"],
            "viewport": spec["viewport"],
            "viewport_human": f"{spec['viewport']['width']}x{spec['viewport']['height']}",
            "unresolved_thread_ids": spec["unresolved_thread_ids"],
            "resolved_thread_ids": spec["resolved_thread_ids"],
            "target_control": spec["target_control"],
            "protected_controls": spec["protected_controls"],
            "component_file": "repo/src/components/ReviewThreadCard.tsx",
            "style_file": "repo/src/styles/review-thread.css",
            "config_file": "repo/config/snapshot-viewports.json",
            "reply_file": "review_reply/replies.md",
            "evidence_file": "evidence/post_fix.md",
            "required_changed_files": [
                "repo/src/components/ReviewThreadCard.tsx",
                "repo/src/styles/review-thread.css",
                "repo/config/snapshot-viewports.json",
            ],
            "allowed_edit_files": [
                "repo/src/components/ReviewThreadCard.tsx",
                "repo/src/styles/review-thread.css",
                "repo/config/snapshot-viewports.json",
            ],
            "visible_test_file": "repo/tests/test_review_thread_ui.py",
            "visible_test_sha256": file_sha["repo/tests/test_review_thread_ui.py"],
            "readonly_tree_hashes": readonly_hashes(ws),
            "required_acknowledgement": spec["required_acknowledgement"],
            "objective_drift_thread_ids": spec["objective_drift_thread_ids"],
            "stale_path_marker": spec["stale_path_marker"],
        }
        (vd / "gold_fix.json").write_text(json.dumps(gold, indent=2, sort_keys=True) + "\n")
        manifest = {
            "variant_id": variant,
            "files": sorted(file_sha),
            "file_sha256": file_sha,
            "readonly_tree_hashes": gold["readonly_tree_hashes"],
        }
        (vd / "workspace_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        with tempfile.TemporaryDirectory(prefix=f"review_ui_oracle_{variant}_") as tmp:
            oracle_ws = Path(tmp) / "workspace"
            shutil.copytree(ws, oracle_ws)
            apply_oracle_fix(oracle_ws, spec)
            subprocess.run([str(oracle_ws / "bin/review-thread-task"), "submit", "submission_input.json"], cwd=oracle_ws, check=True)
            subprocess.run([sys.executable, "repo/tests/test_review_thread_ui.py"], cwd=oracle_ws, check=True)
            oracle_dir = vd / "oracle"
            oracle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(oracle_ws / "submission_input.json", oracle_dir / "submission_input.json")
            shutil.copy(oracle_ws / "brief/fix_summary.json", oracle_dir / "fix_summary.json")
            shutil.copy(oracle_ws / "brief/fix_summary.md", oracle_dir / "fix_summary.md")
            shutil.copy(oracle_ws / "review_reply/replies.md", oracle_dir / "replies.md")
            shutil.copy(oracle_ws / "evidence/post_fix.md", oracle_dir / "post_fix.md")
            oracle_score = score_workspace(oracle_ws, variant)["score"]

        with tempfile.TemporaryDirectory(prefix=f"review_ui_empty_{variant}_") as tmp:
            empty_ws = Path(tmp) / "workspace"
            shutil.copytree(ws, empty_ws)
            empty_score = score_workspace(empty_ws, variant)["score"]

        with tempfile.TemporaryDirectory(prefix=f"review_ui_shortcut_{variant}_") as tmp:
            shortcut_ws = Path(tmp) / "workspace"
            shutil.copytree(ws, shortcut_ws)
            apply_shortcut(shortcut_ws, spec)
            shortcut_score = score_workspace(shortcut_ws, variant)["score"]

        baseline_scores[variant] = {
            "oracle": int(oracle_score),
            "empty": int(empty_score),
            "shortcut": int(shortcut_score),
        }

    (FAMILY / "baseline_scores.json").write_text(json.dumps(baseline_scores, indent=2, sort_keys=True) + "\n")
    print(json.dumps(baseline_scores, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
