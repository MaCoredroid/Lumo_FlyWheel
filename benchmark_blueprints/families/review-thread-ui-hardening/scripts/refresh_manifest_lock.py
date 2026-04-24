#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/review-thread-ui-hardening"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/review-thread-ui-hardening"
SCORER = REPO / "verifiers/review-thread-ui-hardening/score_ranking.py"
VERIFY_SH = REPO / "verifiers/review-thread-ui-hardening/verify.sh"
CLI = FAMILY / "bin/review-thread-task"
BASELINES = json.loads((FAMILY / "baseline_scores.json").read_text())


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


def main() -> int:
    lock = {
        "schema_version": "cnb55.manifest.v2",
        "family_id": "review-thread-ui-hardening",
        "grader": {
            "score_ranking_py_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFY_SH),
        },
        "cli": {
            "review_thread_task_sha256": sha256_file(CLI),
        },
        "variants": {},
    }
    for variant, scores in BASELINES.items():
        ws = WS_BUNDLE / variant
        vd = VERIFIER_DATA / variant
        lock["variants"][variant] = {
            "observed_oracle_score": scores["oracle"],
            "observed_empty_brief_score": scores["empty"],
            "observed_shortcut_score": scores["shortcut"],
            "workspace_trees": {
                rel: digest
                for rel in (".scenario_variant", "AGENTS.md", "Dockerfile", "bin", "artifacts", "repo", "release_context", "incident_context")
                if (digest := sha256_tree(ws, rel))
            },
            "verifier_data": {
                "gold_fix_sha256": sha256_file(vd / "gold_fix.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_fix_summary_sha256": sha256_file(vd / "oracle" / "fix_summary.json"),
            },
        }
    (FAMILY / "manifest.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
