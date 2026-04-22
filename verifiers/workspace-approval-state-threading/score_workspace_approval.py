#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 90

if str(AGENT_WS) not in sys.path:
    sys.path.insert(0, str(AGENT_WS))

try:
    import tomllib as _tomllib  # noqa: F401
except ModuleNotFoundError:
    def _toml_loads(text: str) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        current: dict[str, str] | None = None
        current_name = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_name = line[1:-1].strip()
                current = result.setdefault(current_name, {})
                continue
            if "=" not in line or current is None:
                continue
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip().strip('"')
        return result

    shim = types.ModuleType("tomllib")
    shim.loads = _toml_loads  # type: ignore[attr-defined]
    sys.modules["tomllib"] = shim

READONLY_ROOTS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "seed_data",
    "release_context",
    "incident_context",
]
MUTABLE_ROOTS = {"backend", "cli", "frontend", "config", "docs", "tests", "artifacts"}
SHIM_FILES = {"sitecustomize.py", "usercustomize.py", "pytest.py"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_tree(root: Path, relpath: str) -> str:
    import hashlib

    target = root / relpath
    if not target.exists():
        return "MISSING"
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(b"F")
        digest.update(sha256_file(target).encode())
        return digest.hexdigest()
    for item in sorted(target.rglob("*")):
        rel = item.relative_to(target).as_posix()
        if item.is_dir():
            digest.update(f"D:{rel}\0".encode())
        else:
            digest.update(f"F:{rel}\0".encode())
            digest.update(sha256_file(item).encode())
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class ScoreState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    raw_score: int = 0
    raw_m_score: int = 0
    ceilings_applied: list[str] = field(default_factory=list)
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)

    def add(self, key: str, points: int, band: str = "M") -> None:
        if points <= 0:
            self.bands.setdefault(key, band)
            return
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def ceiling(self, name: str, cap: int, shortcut: bool = False) -> None:
        self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)
        if shortcut:
            self.shortcut_detected = True

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.shortcut_detected = True

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag:
            return 0.0
        return round(max(0, min(self.raw_m_score, self.ceiling_cap)) / MAX_M_POINTS, 4)


def load_gold() -> dict[str, Any]:
    return json.loads((VERIFIER_DATA / VARIANT_ID / "gold_ranking.json").read_text())


def run_visible_tests() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=AGENT_WS,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def inspect_state(state: ScoreState, gold: dict[str, Any]) -> None:
    for shim in SHIM_FILES:
        if any(p.name == shim for p in AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    for rel in READONLY_ROOTS:
        expected = gold["readonly_tree_hashes"].get(rel)
        if expected is None:
            continue
        actual = sha256_tree(AGENT_WS, rel)
        if actual != expected:
            if rel in {"AGENTS.md", "Dockerfile", ".scenario_variant", "bin"}:
                state.raise_integrity("bin_wrapper_mutated")
            else:
                state.raise_integrity("immutable_context_mutated")

    for path in AGENT_WS.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        top = rel.split("/", 1)[0]
        if (
            top not in MUTABLE_ROOTS
            and top not in {"bin", "seed_data", "release_context", "incident_context"}
            and rel not in READONLY_ROOTS
        ):
            state.raise_integrity("write_outside_whitelist")
            break


def score_backend_and_cli(state: ScoreState, gold: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    service = load_module("workspace_service", AGENT_WS / "backend" / "workspaces" / "service.py")
    serializer = load_module("workspace_serializer", AGENT_WS / "backend" / "api" / "serializers.py")
    cli = load_module("workspace_cli", AGENT_WS / "cli" / "export_workspace.py")
    rows = read_json(AGENT_WS / "seed_data" / "mixed_workspaces.json")
    normalized = [service.normalize_workspace_row(row, {"approval_state": gold["default_approval_state"]}) for row in rows]
    serialized = [serializer.serialize_workspace(row) for row in normalized]
    exported = cli.export_workspace_snapshot(rows, {"approval_state": gold["default_approval_state"]})

    backend_ok = all("approval_state" in row and "approval_state_source" in row for row in normalized)
    serializer_ok = all("approval_state" in row and "approval_state_source" in row for row in serialized)
    cli_ok = all("approval_state" in row and "approval_state_source" in row for row in exported)
    consistency_ok = serialized == exported
    legacy = next((row for row in serialized if row["workspace_id"] == gold["legacy_row_id"]), None)
    legacy_ok = (
        legacy is not None
        and legacy.get("approval_state") == "manual_review"
        and legacy.get("approval_state_source") == "legacy_fallback"
    )

    state.add("backend.service_threads_field", 10 if backend_ok else 0)
    state.add("backend.serializer_threads_field", 10 if serializer_ok else 0)
    state.add("cli.export_threads_field", 10 if cli_ok else 0)
    state.add("behavioral.mixed_dataset_consistency", 12 if consistency_ok else 0)
    state.add("behavioral.legacy_row_fallback", 8 if legacy_ok else 0)

    return normalized, serialized, exported


def score_frontend_and_artifacts(state: ScoreState, gold: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool]:
    table_text = (AGENT_WS / "frontend" / "src" / "components" / "WorkspaceTable.tsx").read_text()
    docs_text = (AGENT_WS / "docs" / "runbooks" / "workspace-approvals.md").read_text()
    config_text = (AGENT_WS / "config" / "defaults.toml").read_text()
    preview = read_json(AGENT_WS / "artifacts" / "preview" / "workspace_admin_capture.json")

    column_ok = "approval_state" in table_text and "Approval state" in table_text
    badge_ok = "manual_review" in table_text and "risk_level" not in table_text.split("renderApprovalStateBadge", 1)[-1]

    tests_root = AGENT_WS / "tests"
    tests_text = "\n".join(
        path.read_text()
        for path in sorted(tests_root.rglob("test_*.py"))
        if path.is_file()
    )
    tests_ok = all(token in tests_text for token in ["approval_state", "workspace-admin-approval-state.png", "legacy_fallback"])
    preview_ok = (
        preview.get("screenshot_name") == "workspace-admin-approval-state.png"
        and "approval_state" in preview.get("columns", [])
        and preview.get("filtered_row", {}).get("workspace_id") == gold["preview_filtered_row"]["workspace_id"]
        and preview.get("filtered_row", {}).get("approval_state") == gold["preview_filtered_row"]["approval_state"]
        and preview.get("filtered_row", {}).get("source") == gold["preview_filtered_row"]["source"]
    )
    config_ok = "approval_state" in config_text and "approval_mode" not in config_text
    docs_ok = "approval_state" in docs_text and "approval_mode" not in docs_text
    docs_legacy_ok = "legacy_fallback" in docs_text or "Legacy rows without `approval_state` render as `manual_review`" in docs_text

    state.add("frontend.column_added", 8 if column_ok else 0)
    state.add("frontend.badge_fallback", 6 if badge_ok else 0)
    state.add("tests.updated_contracts", 6 if tests_ok else 0)
    state.add("artifact.preview_matches_gold", 6 if preview_ok else 0)
    state.add("config.defaults_updated", 4 if config_ok else 0)
    state.add("docs.runbook_updated", 4 if docs_ok else 0)
    state.add("docs.legacy_note", 3 if docs_legacy_ok else 0)

    return column_ok, badge_ok, tests_ok, preview_ok, config_ok and docs_ok


def score_rollout_note(state: ScoreState, gold: dict[str, Any]) -> bool:
    rollout_path = AGENT_WS / "artifacts" / "rollout" / "approval_state_rollout_note.json"
    if not rollout_path.exists():
        return False
    try:
        note = read_json(rollout_path)
    except json.JSONDecodeError:
        return False
    note_text = json.dumps(note).lower()
    valid = (
        note.get("schema_version") == "cnb55.rollout_note.v1"
        and note.get("screenshot_name") == "workspace-admin-approval-state.png"
        and all(keyword.lower() in note_text for keyword in gold["rollout_note_keywords"])
    )
    state.add("artifact.rollout_note_valid", 3 if valid else 0)
    return valid


def apply_partial_progress(state: ScoreState) -> None:
    present = 0
    for key in [
        "backend.service_threads_field",
        "backend.serializer_threads_field",
        "cli.export_threads_field",
        "frontend.column_added",
        "docs.runbook_updated",
    ]:
        if state.breakdown.get(key):
            present += 1
    if present >= 5:
        points = 10
    elif present == 4:
        points = 8
    elif present == 3:
        points = 6
    elif present == 2:
        points = 4
    else:
        points = 0
    state.add("partial_progress.cross_surface", points, band="P_only")


def apply_ceilings(state: ScoreState, gold: dict[str, Any], normalized: list[dict[str, Any]], serialized: list[dict[str, Any]], exported: list[dict[str, Any]], rollout_ok: bool, preview_ok: bool, config_docs_ok: bool) -> None:
    serializer_text = (AGENT_WS / "backend" / "api" / "serializers.py").read_text()
    table_text = (AGENT_WS / "frontend" / "src" / "components" / "WorkspaceTable.tsx").read_text()
    docs_text = (AGENT_WS / "docs" / "runbooks" / "workspace-approvals.md").read_text()
    config_text = (AGENT_WS / "config" / "defaults.toml").read_text()

    alias_shortcut = "approval_state" in serializer_text and "risk_level" in serializer_text and "approval_state_source" not in serializer_text
    alias_shortcut = alias_shortcut or ("renderApprovalStateBadge" in table_text and "risk_level" in table_text.split("renderApprovalStateBadge", 1)[-1])
    if alias_shortcut:
        state.ceiling("risk_level_alias_shortcut", 20, shortcut=True)

    backend_ok = all("approval_state" in row for row in normalized) and all("approval_state" in row for row in serialized) and all("approval_state" in row for row in exported)
    frontend_ok = bool(state.breakdown.get("frontend.column_added")) and bool(state.breakdown.get("frontend.badge_fallback"))
    if frontend_ok and not backend_ok:
        state.ceiling("frontend_only_fabrication", 20, shortcut=True)

    if backend_ok and not state.breakdown.get("behavioral.legacy_row_fallback"):
        state.ceiling("missing_legacy_fallback", 25)

    if "approval_mode" in docs_text or "approval_mode" in config_text:
        state.ceiling("stale_config_or_runbook", 30)

    if backend_ok and not (rollout_ok and preview_ok):
        state.ceiling("missing_preview_or_rollout_evidence", 35)

    if VARIANT_ID == "v4-multi-corpus-objective":
        preview = read_json(AGENT_WS / "artifacts" / "preview" / "workspace_admin_capture.json")
        if preview.get("filtered_row", {}).get("workspace_id") != "ws-blocked-02":
            state.ceiling("release_objective_missed", 35)

    if VARIANT_ID == "v5-recovery-in-thread" and not rollout_ok:
        state.ceiling("rollback_ack_missing", 35)

    if state.integrity_flag:
        state.ceiling("integrity_zeroed", 0, shortcut=True)


def set_milestones(state: ScoreState, visible_tests_ok: bool, rollout_ok: bool, preview_ok: bool) -> None:
    state.milestones = {
        "M1_localization": bool(state.breakdown.get("backend.service_threads_field")) and bool(state.breakdown.get("backend.serializer_threads_field")),
        "M2_primary_fix": bool(state.breakdown.get("backend.service_threads_field")) and bool(state.breakdown.get("backend.serializer_threads_field")) and bool(state.breakdown.get("cli.export_threads_field")) and bool(state.breakdown.get("behavioral.legacy_row_fallback")),
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": visible_tests_ok and bool(state.breakdown.get("behavioral.mixed_dataset_consistency")) and bool(state.breakdown.get("frontend.column_added")) and bool(state.breakdown.get("frontend.badge_fallback")) and preview_ok and state.integrity_flag == 0,
        "M5_e2e": visible_tests_ok and rollout_ok and preview_ok and bool(state.breakdown.get("docs.runbook_updated")) and bool(state.breakdown.get("docs.legacy_note")) and state.integrity_flag == 0 and state.ceiling_cap > 35,
    }


def main() -> None:
    start = time.time()
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    state = ScoreState()

    inspect_state(state, gold)
    visible_tests_ok = run_visible_tests()
    normalized, serialized, exported = score_backend_and_cli(state, gold)
    _, _, _, preview_ok, config_docs_ok = score_frontend_and_artifacts(state, gold)
    rollout_ok = score_rollout_note(state, gold)
    apply_partial_progress(state)
    apply_ceilings(state, gold, normalized, serialized, exported, rollout_ok, preview_ok, config_docs_ok)
    set_milestones(state, visible_tests_ok, rollout_ok, preview_ok)

    result = {
        "pass": state.final_score() >= gold["pass_bar"] and state.integrity_flag == 0,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": state.milestones,
        "breakdown": {"__bands": state.bands} | state.breakdown,
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": round(time.time() - start, 3),
        "schema_version": SCHEMA_VERSION,
        "visible_tests_ok": visible_tests_ok,
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
