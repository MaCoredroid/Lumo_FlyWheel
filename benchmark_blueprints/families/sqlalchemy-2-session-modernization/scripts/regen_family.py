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


REPO_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO_ROOT / "benchmark_blueprints/families/sqlalchemy-2-session-modernization"
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO_ROOT / "verifier_data/sqlalchemy-2-session-modernization"
SCORER = REPO_ROOT / "verifiers/sqlalchemy-2-session-modernization/score_sqlalchemy_session_modernization.py"
VERIFY_SH = REPO_ROOT / "verifiers/sqlalchemy-2-session-modernization/verify.sh"
LOCK_PATH = FAMILY_ROOT / "manifest.lock.json"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

ALLOWED_WRITE_PATHS = [
    "app/api.py",
    "app/repository.py",
    "app/worker.py",
    "app/admin_cli.py",
    "docs/deploy/sqlalchemy2-cutover.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        if target.suffix == ".pyc":
            return None
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
        else:
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
    return h.hexdigest()


def list_workspace_files(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = sha256_file(path)
    return files


def readonly_tree_hashes(root: Path) -> dict[str, str]:
    rels = [
        "AGENTS.md",
        "Dockerfile",
        ".scenario_variant",
        "tests",
        "seed",
        "artifacts",
        "notes",
        "release_context",
        "incident_context",
        "README.md",
    ]
    data = {}
    for rel in rels:
        digest = sha256_tree(root, rel)
        if digest is not None:
            data[rel] = digest
    return data


def write_workspace_manifest(variant: str) -> None:
    root = WORKSPACE_BUNDLE / variant
    out = VERIFIER_DATA / variant / "workspace_manifest.json"
    payload = {
        "variant_id": variant,
        "files": list_workspace_files(root),
        "readonly_tree_hashes": readonly_tree_hashes(root),
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_gold_fix(variant: str) -> None:
    root = WORKSPACE_BUNDLE / variant
    out = VERIFIER_DATA / variant / "gold_fix.json"
    payload = {
        "variant_id": variant,
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
        "readonly_tree_hashes": readonly_tree_hashes(root),
        "oracle_overlay_root": f"verifier_data/sqlalchemy-2-session-modernization/{variant}/oracle/files",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def apply_oracle(workspace: Path, variant: str) -> None:
    oracle_root = VERIFIER_DATA / variant / "oracle" / "files"
    for source in sorted(oracle_root.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(oracle_root)
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)


def apply_query_only_shortcut(workspace: Path, variant: str) -> None:
    apply_oracle(workspace, variant)
    repo = workspace / "app/repository.py"
    text = repo.read_text()
    repo.write_text(text.replace("session.flush()", "session.commit()"))
    worker = workspace / "app/worker.py"
    worker.write_text((WORKSPACE_BUNDLE / variant / "app/worker.py").read_text())
    admin = workspace / "app/admin_cli.py"
    admin.write_text((WORKSPACE_BUNDLE / variant / "app/admin_cli.py").read_text())


def score_workspace(workspace: Path, variant: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"sqla_score_{variant}_") as tmpdir:
        result_file = Path(tmpdir) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(workspace),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], cwd=str(REPO_ROOT), env=env, check=True)
        return json.loads(result_file.read_text())


def scored_overlay(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"sqla_variant_{variant}_") as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        shutil.copytree(WORKSPACE_BUNDLE / variant, workspace)
        builder(workspace, variant)
        return score_workspace(workspace, variant)


def main() -> int:
    lock = {
        "schema_version": "cnb55.manifest.v2",
        "family_id": "sqlalchemy-2-session-modernization",
        "grader": {
            "score_sqlalchemy_session_modernization_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFY_SH),
        },
        "variants": {},
    }

    for variant in VARIANTS:
        write_workspace_manifest(variant)
        write_gold_fix(variant)

        workspace_root = WORKSPACE_BUNDLE / variant
        oracle_score = scored_overlay(variant, apply_oracle)
        empty_score = scored_overlay(variant, lambda ws, _variant: None)
        shortcut_score = scored_overlay(variant, apply_query_only_shortcut)

        workspace_manifest = VERIFIER_DATA / variant / "workspace_manifest.json"
        gold_fix = VERIFIER_DATA / variant / "gold_fix.json"
        hidden_tests_root = VERIFIER_DATA / variant / "hidden_tests"

        lock["variants"][variant] = {
            "observed_oracle_score": oracle_score["P_benchmark"],
            "observed_empty_score": empty_score["P_benchmark"],
            "observed_shortcut_score": shortcut_score["P_benchmark"],
            "oracle_m_training": oracle_score["M_training"],
            "workspace_trees": readonly_tree_hashes(workspace_root),
            "verifier_data": {
                "workspace_manifest_sha256": sha256_file(workspace_manifest),
                "gold_fix_sha256": sha256_file(gold_fix),
                "hidden_tests_tree_sha256": sha256_tree(hidden_tests_root, "."),
            },
        }

    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
