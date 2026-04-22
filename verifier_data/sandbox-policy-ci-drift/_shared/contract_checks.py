
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(os.environ.get("AGENT_WS", ".")).resolve()
VERIFIER_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gold(variant_id: str) -> dict[str, Any]:
    return json.loads((VERIFIER_ROOT / variant_id / "gold_repair.json").read_text())


def load_manifest(variant_id: str) -> dict[str, Any]:
    return json.loads((VERIFIER_ROOT / variant_id / "workspace_manifest.json").read_text())


def changed_files(agent_ws: Path, manifest: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    expected = manifest.get("file_sha256", {})
    current_files = {
        path.relative_to(agent_ws).as_posix(): path
        for path in agent_ws.rglob("*")
        if path.is_file()
    }
    for rel, expected_sha in expected.items():
        path = current_files.pop(rel, None)
        if path is None:
            changed.append(rel)
            continue
        if sha256_file(path) != expected_sha:
            changed.append(rel)
    changed.extend(sorted(current_files))
    return sorted(changed)


def writable_paths(gold: dict[str, Any]) -> set[str]:
    return set(gold.get("editable_files", []))


def readonly_tree_hashes_ok(agent_ws: Path, gold: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for rel, expected in gold.get("readonly_tree_hashes", {}).items():
        target = agent_ws / rel
        if not target.exists():
            mismatches.append(rel)
            continue
        h = hashlib.sha256()
        if target.is_file():
            h.update(b"F")
            h.update(sha256_file(target).encode())
        else:
            for path in sorted(target.rglob("*")):
                rel_path = path.relative_to(target).as_posix()
                if path.is_file():
                    h.update(b"F:" + rel_path.encode() + b"\x00")
                    h.update(sha256_file(path).encode() + b"\x00")
                elif path.is_dir():
                    h.update(b"D:" + rel_path.encode() + b"\x00")
        if h.hexdigest() != expected:
            mismatches.append(rel)
    return (not mismatches, mismatches)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class SurfaceChecks:
    parser_compat_ok: bool
    preview_canonical_ok: bool
    workflow_canonical_ok: bool
    config_canonical_ok: bool
    docs_contract_ok: bool
    helper_shortcut_clean: bool
    code_files_changed: bool
    localization_span_ok: bool


def inspect_surfaces(agent_ws: Path, changed: list[str], gold: dict[str, Any]) -> SurfaceChecks:
    sys.path.insert(0, str(agent_ws))
    try:
        config_mod = _load_module("sandbox_config", agent_ws / "codex" / "config.py")
        preview_mod = _load_module("sandbox_preview", agent_ws / "scripts" / "render_workflow_preview.py")
    finally:
        if sys.path and sys.path[0] == str(agent_ws):
            sys.path.pop(0)

    parser_compat_ok = False
    preview_canonical_ok = False
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write("[policy]\nsandbox = \"workspace-write\"\napproval_policy = \"on_request\"\n")
        alias_path = Path(fh.name)
    try:
        try:
            loaded = config_mod.load_config(alias_path)
            parser_compat_ok = loaded["sandbox"] == "workspace_write"
        except Exception:
            parser_compat_ok = False
        try:
            preview = preview_mod.render_preview(alias_path)
            preview_canonical_ok = (
                preview["sandbox"] == "workspace_write"
                and preview["approval_policy"] == "on_request"
                and {job["approval_policy"] for job in preview["jobs"]} == {"on_request"}
            )
        except Exception:
            preview_canonical_ok = False
    finally:
        alias_path.unlink(missing_ok=True)

    workflow = (agent_ws / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow_canonical_ok = (
        "manual-review" not in workflow
        and workflow.count("APPROVAL_POLICY: on_request") == 2
        and workflow.count("SANDBOX_POLICY: workspace_write") == 2
    )

    config_text = (agent_ws / "codex" / "config.toml").read_text(encoding="utf-8")
    config_canonical_ok = (
        'sandbox = "workspace_write"' in config_text
        and 'approval_policy = "on_request"' in config_text
        and "workspace-write" not in config_text
        and "manual-review" not in config_text
    )

    note = (agent_ws / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
    docs_contract_ok = (
        "workspace-write" in note
        and "workspace_write" in note
        and "on_request" in note
        and "manual-review" not in note
        and "fully breaking" not in note
        and "make ci" in note
    )

    run_ci = (agent_ws / "scripts" / "run_ci.py").read_text(encoding="utf-8")
    helper_shortcut_clean = (
        "visible_config.toml" not in run_ci
        and 'replace("workspace_write", "workspace-write")' not in run_ci
        and "CODEX_CONFIG_FIXTURE" not in run_ci
    )

    code_files_changed = any(rel in set(gold.get("code_files", [])) for rel in changed)
    localization_span_ok = code_files_changed and any(
        rel in {"docs/releases/policy-rename.md", ".github/workflows/ci.yml", "codex/config.toml"}
        for rel in changed
    )
    return SurfaceChecks(
        parser_compat_ok=parser_compat_ok,
        preview_canonical_ok=preview_canonical_ok,
        workflow_canonical_ok=workflow_canonical_ok,
        config_canonical_ok=config_canonical_ok,
        docs_contract_ok=docs_contract_ok,
        helper_shortcut_clean=helper_shortcut_clean,
        code_files_changed=code_files_changed,
        localization_span_ok=localization_span_ok,
    )


def run_hidden_pytest(agent_ws: Path, variant_id: str) -> bool:
    test_path = VERIFIER_ROOT / variant_id / "hidden_tests" / "test_hidden_contract.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(agent_ws), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        cwd=agent_ws,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0
