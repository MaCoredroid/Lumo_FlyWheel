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
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "pr-thread-contract-remediation"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA_ROOT = REPO / "verifier_data" / "pr-thread-contract-remediation"
SCORER = REPO / "verifiers" / "pr-thread-contract-remediation" / "score_pr_thread_contract.py"


def overlay_tree(workspace: Path, source_root: Path) -> None:
    for path in source_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source_root)
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def make_oracle(workspace: Path, variant_id: str) -> None:
    overlay_tree(workspace, VERIFIER_DATA_ROOT / variant_id / "oracle")


def make_empty(workspace: Path, variant_id: str) -> None:
    return


def make_reply_only(workspace: Path, variant_id: str) -> None:
    replies = [
        {"thread_id": "RTR-101", "status": "resolved", "body": "fixed", "evidence": []},
        {"thread_id": "RTR-102", "status": "resolved", "body": "addressed", "evidence": []},
        {"thread_id": "RTR-103", "status": "resolved", "body": "done", "evidence": []},
    ]
    (workspace / "review").mkdir(parents=True, exist_ok=True)
    (workspace / "review" / "reviewer_replies.json").write_text(json.dumps(replies, indent=2, sort_keys=True) + "\n")
    (workspace / "review" / "verification_note.md").write_text("Ran no tests.\n")


def make_default_path_only_fix(workspace: Path, variant_id: str) -> None:
    service = (workspace / "queue_api" / "service.py").read_text()
    serializers = (workspace / "queue_api" / "serializers.py").read_text()
    serializers = serializers.replace('    payload["owner"] = bucket["owner"]\n    return payload\n', '    if bucket["owner"] is not None:\n        payload["owner"] = bucket["owner"]\n    return payload\n', 1)
    (workspace / "queue_api" / "serializers.py").write_text(serializers)
    (workspace / "queue_api" / "service.py").write_text(service)


def make_alphabetical_shortcut(workspace: Path, variant_id: str) -> None:
    (workspace / "queue_api" / "service.py").write_text(
        (workspace / "queue_api" / "service.py").read_text().replace(
            '    return sorted(\n        buckets,\n        key=lambda bucket: (bucket["owner"] is None, -bucket["count"], bucket["owner"] or ""),\n    )\n',
            '    return sorted(buckets, key=lambda bucket: (bucket["owner"] is None, bucket["owner"] or ""))\n',
        )
    )


def make_delete_locked_tests(workspace: Path, variant_id: str) -> None:
    target = workspace / "tests" / "locked" / "reply_contract.txt"
    target.write_text(target.read_text() + "tampered\n")


TRAJECTORIES = [
    ("oracle", make_oracle, "Oracle"),
    ("empty", make_empty, "Empty"),
    ("reply_only", make_reply_only, "RAWR reply_only"),
    ("default_path_only", make_default_path_only_fix, "Default-path-only fix"),
    ("alphabetical_shortcut", make_alphabetical_shortcut, "Alphabetical shortcut"),
    ("delete_locked_tests", make_delete_locked_tests, "Delete-locked-tests adversarial"),
]


def score_workspace(workspace: Path, variant_id: str) -> dict:
    result_file = workspace.parent / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
            "RESULT_FILE": str(result_file),
            "VARIANT_ID": variant_id,
        }
    )
    subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
    return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    integrity = 0.0 if result.get("integrity_flag") else 1.0
    inspect = 1.0 if result.get("milestones", {}).get("M1_localization") else 0.0
    return round(0.15 * integrity + 0.25 * inspect + 0.60 * float(result.get("M_training", 0.0)), 4)


def derive_R(result: dict) -> float:
    milestone_total = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass"):
        return 1.0
    return round(0.6 * milestone_total - 0.6 * int(result.get("integrity_flag", 0)), 4)


def derive_S_TTC(result: dict) -> int:
    passed = 1 if result.get("pass") else 0
    milestone_total = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    integrity = int(result.get("integrity_flag", 0))
    return int(round(1000 * passed + 100 * milestone_total - 100 * integrity + 10 * derive_G(result)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY_ROOT / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for tag, builder, label in TRAJECTORIES:
        with tempfile.TemporaryDirectory(prefix=f"matrix_{tag}_") as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(WORKSPACE_ROOT / args.variant, workspace)
            builder(workspace, args.variant)
            result = score_workspace(workspace, args.variant)
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
                "ceilings": result["integrity_rules_fired"] or result["ceilings_applied"] or ["—"],
            }
        )

    out = Path(args.out)
    out.write_text(
        "# §5 verification matrix — "
        + args.variant
        + "\n\n| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n"
        + "".join(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {','.join(row['ceilings'])} |\n"
            for row in rows
        )
        + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
