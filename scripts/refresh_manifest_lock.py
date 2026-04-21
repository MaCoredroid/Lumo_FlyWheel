#!/usr/bin/env python3
"""Refresh benchmark_blueprints/.../manifest.lock.json with updated hashes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAMILY = REPO / "benchmark_blueprints/families/proposal-ranking-manager-judgment"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/proposal-ranking-manager-judgment"
LOCK = FAMILY / "manifest.lock.json"
SCORER = REPO / "verifiers/proposal-ranking-manager-judgment/score_ranking.py"
VERIFY_SH = REPO / "verifiers/proposal-ranking-manager-judgment/verify.sh"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

WS_TREES = (
    ".scenario_variant", "AGENTS.md", "Dockerfile", "bin", "artifacts", "brief",
    "proposals", "release_context", "incident_context", "repo_evidence", "tests",
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    t = root / rel
    if not t.exists():
        return None
    h = hashlib.sha256()
    if t.is_file():
        h.update(b"F")
        h.update(sha256_file(t).encode())
        return h.hexdigest()
    for p in sorted(t.rglob("*")):
        rp = p.relative_to(t).as_posix()
        if p.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
    return h.hexdigest()


def hidden_tests_sha(vd: Path) -> str | None:
    ht = vd / "hidden_tests"
    if not ht.exists():
        return None
    return sha256_tree(ht, ".")


def main() -> int:
    lock = json.loads(LOCK.read_text())

    # Grader hashes
    lock.setdefault("grader", {})
    lock["grader"]["score_ranking_py_sha256"] = sha256_file(SCORER)
    if VERIFY_SH.exists():
        lock["grader"]["verify_sh_sha256"] = sha256_file(VERIFY_SH)

    # CLI hash (new entry — record the shipped cnb55-brief)
    cli = FAMILY / "bin" / "cnb55-brief"
    if cli.exists():
        lock.setdefault("cli", {})["cnb55_brief_sha256"] = sha256_file(cli)

    # Observed scores (from regen_cnb55.py run 2026-04-20)
    observed_oracle = {
        "v1-clean-baseline": 90,
        "v2-noisy-distractor": 90,
        "v3-dirty-state": 90,
        "v4-multi-corpus-objective": 96,
        "v5-recovery-in-thread": 99,
    }

    for v in VARIANTS:
        vd = VER_DATA / v
        ws = WS_BUNDLE / v
        entry = lock["variants"].setdefault(v, {})

        # Observed scores refresh
        entry["observed_oracle_score"] = observed_oracle[v]
        entry["observed_empty_brief_score"] = 0
        entry["observed_shortcut_score"] = 30

        # verifier_data hashes
        vd_hashes = entry.setdefault("verifier_data", {})
        vd_hashes["gold_ranking_sha256"] = sha256_file(vd / "gold_ranking.json")
        vd_hashes["workspace_manifest_sha256"] = sha256_file(vd / "workspace_manifest.json")
        # oracle_brief_sha256 -> now point at canonical JSON (v2 source of truth)
        oracle_json = vd / "oracle" / "manager_brief.json"
        if oracle_json.exists():
            vd_hashes["oracle_brief_json_sha256"] = sha256_file(oracle_json)
        oracle_md = vd / "oracle" / "manager_brief.md"
        if oracle_md.exists():
            vd_hashes["oracle_brief_md_sha256"] = sha256_file(oracle_md)
        # Remove stale v1 field name if still there
        vd_hashes.pop("oracle_brief_sha256", None)

        ht = hidden_tests_sha(vd)
        if ht:
            vd_hashes["hidden_tests_tree_sha256"] = ht

        # workspace_trees: include bin/ and refreshed AGENTS.md
        trees = {}
        for rel in WS_TREES:
            h = sha256_tree(ws, rel)
            if h:
                trees[rel] = h
        entry["workspace_trees"] = trees

    # bump schema version marker so callers notice
    lock["schema_version"] = "cnb55.manifest.v2"
    lock["last_regen_utc"] = "2026-04-20T00:00:00Z"

    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
