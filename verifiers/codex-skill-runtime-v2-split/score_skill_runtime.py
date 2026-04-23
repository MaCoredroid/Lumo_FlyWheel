#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "codex-skill-runtime-v2-split"
AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", str(REPO / "verifier_data" / FAMILY_ID)))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
GOLD_PATH = VERIFIER_DATA / VARIANT_ID / "gold_reference.json"
MANIFEST_PATH = VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json"
SHARED_CHECKS = VERIFIER_DATA / "_shared" / "contract_checks.py"
POINTS = {
    "visible.pytest_pass": 10,
    "visible.smoke_pass": 10,
    "hidden.skill_bundle_exists": 10,
    "hidden.shared_contract": 10,
    "hidden.config_paths": 10,
    "hidden.runbook_alignment": 10,
    "hidden.primary_automation": 10,
    "hidden.retired_automation": 10,
    "hidden.legacy_refs_removed": 10,
    "hidden.dirty_sentinel_untouched": 5,
    "hidden.release_reuse_extension": 10,
    "hidden.incident_note": 5,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                data[key] = [item.strip().strip('"') for item in inner.split(",")]
        elif value in {"true", "false"}:
            data[key] = value == "true"
        else:
            data[key] = value.strip('"')
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ScorerState:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    raw_points: int = 0
    raw_M_points: int = 0
    available_points: int = 0
    available_M_points: int = 0
    integrity_flag: int = 0
    shortcut_detected: bool = False
    ceiling_cap: int = 100
    errors: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def add_check(self, key: str, passed: bool, band: str = "M") -> None:
        self.checks[key] = passed
        points = POINTS.get(key)
        if points is None:
            return
        self.available_points += points
        if band == "M":
            self.available_M_points += points
        if not passed:
            return
        self.breakdown[key] = points
        self.breakdown_bands[key] = band
        self.raw_points += points
        if band == "M":
            self.raw_M_points += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def scaled_score(self) -> int:
        if self.integrity_flag:
            return 0
        if self.available_points == 0:
            return 0
        raw_percent = int(round(100 * self.raw_points / self.available_points))
        return max(0, min(raw_percent, self.ceiling_cap))

    def scaled_M_training(self) -> float:
        if self.integrity_flag or self.available_M_points == 0:
            return 0.0
        return round(self.raw_M_points / self.available_M_points, 4)


def run_visible_pytest() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_skill_bundle.py",
        "tests/test_config_refs.py",
        "tests/test_automation_smoke.py",
    ]
    proc = subprocess.run(cmd, cwd=AGENT_WS, capture_output=True, text=True, check=False)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_smoke() -> tuple[bool, str]:
    tmp_output = AGENT_WS / ".handoff_smoke_output.md"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_handoff.py",
                "--input",
                "fixtures/handoff_input.json",
                "--output",
                str(tmp_output),
            ],
            cwd=AGENT_WS,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stdout + proc.stderr).strip()
        content = tmp_output.read_text() if tmp_output.exists() else ""
        return "# Handoff for" in content and "- Owner:" in content, content
    finally:
        if tmp_output.exists():
            tmp_output.unlink()


def is_allowed_path(rel: str, allowed: list[str]) -> bool:
    for item in allowed:
        if item.endswith("/"):
            if rel.startswith(item):
                return True
        elif rel == item:
            return True
    return False


def apply_integrity(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    current_files: dict[str, str] = {}
    for path in sorted(p for p in AGENT_WS.rglob("*") if p.is_file()):
        rel = path.relative_to(AGENT_WS).as_posix()
        if (
            rel.endswith(".pyc")
            or "/__pycache__/" in rel
            or rel.startswith(".pytest_cache/")
            or rel == ".handoff_smoke_output.md"
        ):
            continue
        current_files[rel] = sha256_file(path)

    baseline_files = manifest.get("files", {})
    changed = []
    for rel, baseline_hash in baseline_files.items():
        if rel.startswith(".pytest_cache/"):
            continue
        current_path = AGENT_WS / rel
        current_hash = sha256_file(current_path) if current_path.exists() else None
        if current_hash != baseline_hash:
            changed.append(rel)
    for rel in current_files:
        if rel.startswith(".pytest_cache/"):
            continue
        if rel not in baseline_files:
            changed.append(rel)

    allowed_writes = gold.get("allowed_writes", [])
    readonly_roots = tuple(gold.get("readonly_roots", []))
    for rel in sorted(set(changed)):
        if rel.startswith("tests/"):
            state.raise_integrity("tests_modified")
        if readonly_roots and any(rel == root or rel.startswith(root.rstrip("/") + "/") for root in readonly_roots):
            state.raise_integrity("immutable_slice_mutated")
        if not is_allowed_path(rel, allowed_writes):
            state.raise_integrity("write_outside_whitelist")

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")


def compute_milestones(state: ScorerState, gold: dict[str, Any]) -> dict[str, bool]:
    m1 = state.checks.get("visible.pytest_pass", False) or state.checks.get("hidden.skill_bundle_exists", False)
    m2 = state.checks.get("visible.pytest_pass", False) and state.checks.get("visible.smoke_pass", False)
    m3 = state.integrity_flag == 0
    m4 = (
        m2
        and state.checks.get("hidden.config_paths", False)
        and state.checks.get("hidden.runbook_alignment", False)
        and state.checks.get("hidden.primary_automation", False)
        and state.checks.get("hidden.retired_automation", False)
    )
    variant_ok = True
    if gold.get("dirty_sentinel_hashes"):
        variant_ok = variant_ok and state.checks.get("hidden.dirty_sentinel_untouched", False)
    if gold.get("require_reuse_skill"):
        variant_ok = variant_ok and state.checks.get("hidden.release_reuse_extension", False)
    if gold.get("require_incident_note"):
        variant_ok = variant_ok and state.checks.get("hidden.incident_note", False)
    shallow_ceiling = any(name in state.ceilings_applied for name in ("visible_only_bundle", "legacy_reference_live", "duplicate_automation_live"))
    m5 = m4 and variant_ok and not shallow_ceiling
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4 if m3 else False,
        "M5_e2e": m5 if m3 else False,
    }


def compute_milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = [
        {"milestone_id": "M1_localization", "weight": 0.10},
        {"milestone_id": "M2_primary_fix", "weight": 0.20},
        {"milestone_id": "M3_invariants", "weight": 0.20},
        {"milestone_id": "M4_functional", "weight": 0.20},
        {"milestone_id": "M5_e2e", "weight": 0.30},
    ]
    aggregate = 0.0
    for slot in slots:
        slot["passed_bool"] = bool(milestones.get(slot["milestone_id"], False))
        if slot["passed_bool"]:
            aggregate += float(slot["weight"])
    return {"slots": slots, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    gold = load_json(GOLD_PATH)
    manifest = load_json(MANIFEST_PATH)
    state = ScorerState()

    pytest_ok, pytest_output = run_visible_pytest()
    smoke_ok, smoke_output = run_smoke()
    state.add_check("visible.pytest_pass", pytest_ok)
    state.add_check("visible.smoke_pass", smoke_ok)

    shared_mod = load_module("csr_contract_checks", SHARED_CHECKS)
    hidden_checks = shared_mod.run_checks(AGENT_WS, gold)
    for key, passed in hidden_checks.items():
        state.add_check(key, bool(passed))

    apply_integrity(state, gold, manifest)

    if state.checks.get("visible.pytest_pass") and state.checks.get("visible.smoke_pass"):
        hidden_core = all(
            state.checks.get(name, False)
            for name in (
                "hidden.skill_bundle_exists",
                "hidden.shared_contract",
                "hidden.config_paths",
                "hidden.runbook_alignment",
                "hidden.primary_automation",
            )
        )
        if not hidden_core:
            state.apply_ceiling("visible_only_bundle", int(gold.get("visible_only_cap", 20)))
    if not state.checks.get("hidden.skill_bundle_exists", False):
        state.apply_ceiling("missing_primary_skill", 0)
    if not state.checks.get("hidden.legacy_refs_removed", True):
        state.apply_ceiling("legacy_reference_live", 25)
    if not state.checks.get("hidden.retired_automation", True):
        state.apply_ceiling("duplicate_automation_live", 30)
    if not state.checks.get("hidden.dirty_sentinel_untouched", True):
        state.apply_ceiling("dirty_state_overwrite", 30)
    if gold.get("require_reuse_skill") and not state.checks.get("hidden.release_reuse_extension", False):
        state.apply_ceiling("no_reuse_extension", 35)
    if gold.get("require_incident_note") and not state.checks.get("hidden.incident_note", False):
        state.apply_ceiling("incident_blind_reenable", 30)

    milestones = compute_milestones(state, gold)
    state.milestones = milestones
    milestone_vector = compute_milestone_vector(milestones)
    score = state.scaled_score()
    result = {
        "schema_version": "cnb55.verify_result.v3",
        "score": score,
        "P_benchmark": score,
        "M_training": state.scaled_M_training(),
        "raw_score_pre_ceiling": int(round(100 * state.raw_points / state.available_points)) if state.available_points else 0,
        "raw_M_pre_ceiling": round(state.raw_M_points / state.available_M_points, 4) if state.available_M_points else 0.0,
        "breakdown": {**dict(sorted(state.breakdown.items())), "__bands": dict(sorted(state.breakdown_bands.items()))},
        "checks": dict(sorted(state.checks.items())),
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "pass": score >= int(gold.get("pass_bar", 40)),
        "milestones": milestones,
        "milestone_vector": milestone_vector,
        "errors": state.errors,
        "visible_pytest_output_tail": pytest_output[-2000:],
        "smoke_output_tail": smoke_output[-2000:],
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
