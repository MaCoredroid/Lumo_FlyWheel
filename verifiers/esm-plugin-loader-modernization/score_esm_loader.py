#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace")).resolve()
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data")).resolve()
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json")).resolve()
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 90
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
PARTIAL_PROGRESS_PATHS = (
    "src/loader.mjs",
    "src/contracts.mjs",
    "src/index.mjs",
    "scripts/build.mjs",
    "scripts/typecheck.mjs",
    "docs/cli/plugins.md",
)


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    import hashlib

    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        digest = hashlib.sha256()
        digest.update(b"F")
        digest.update(sha256_file(target).encode())
        return digest.hexdigest()
    digest = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            digest.update(b"D:" + rel_path.encode() + b"\x00")
        else:
            digest.update(b"F:" + rel_path.encode() + b"\x00")
            digest.update(sha256_file(path).encode() + b"\x00")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_workspace_files(root: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload[path.relative_to(root).as_posix()] = sha256_file(path)
    return payload


def read_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


def run_command(cwd: Path, command: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout


def run_visible_commands() -> dict[str, tuple[bool, str]]:
    return {
        "npm_test": run_command(AGENT_WS, ["npm", "test"]),
        "npm_build": run_command(AGENT_WS, ["npm", "run", "build"]),
        "npm_typecheck": run_command(AGENT_WS, ["npm", "run", "typecheck"]),
    }


def run_hidden_checks(gold: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"esm_loader_hidden_{VARIANT_ID}_") as tmp:
        tmp_root = Path(tmp)
        ws = tmp_root / "workspace"
        shutil.copytree(AGENT_WS, ws)

        build_ok, build_output = run_command(ws, ["npm", "run", "build"])
        outputs: dict[str, tuple[bool, str]] = {}
        capture_ok = False
        capture_output = ""
        artifact_match = False

        if build_ok:
            src_dir = ws / "src"
            plugins_dir = ws / "plugins"
            if src_dir.exists():
                shutil.rmtree(src_dir)
            if plugins_dir.exists():
                shutil.rmtree(plugins_dir)
            for name in ["good-default", "good-named", "good-helper"]:
                outputs[name] = run_command(ws, ["node", "dist/src/index.mjs", name])
            outputs["bad-wrong-shape"] = run_command(ws, ["node", "dist/src/index.mjs", "bad-wrong-shape"])
        else:
            for name in ["good-default", "good-named", "good-helper", "bad-wrong-shape"]:
                outputs[name] = (False, build_output)

        capture_ws = tmp_root / "capture_workspace"
        shutil.copytree(AGENT_WS, capture_ws)
        capture_ok, capture_output = run_command(capture_ws, ["bash", "scripts/capture_cli_output.sh"])
        actual_artifact = (AGENT_WS / "artifacts" / "cli_capture.txt").read_text() if (AGENT_WS / "artifacts" / "cli_capture.txt").exists() else ""
        generated_artifact = (capture_ws / "artifacts" / "cli_capture.txt").read_text() if (capture_ws / "artifacts" / "cli_capture.txt").exists() else ""
        artifact_match = capture_ok and actual_artifact == generated_artifact and actual_artifact == gold["expected_capture"]

        return {
            "build_ok": build_ok,
            "build_output": build_output,
            "outputs": outputs,
            "capture_ok": capture_ok,
            "capture_output": capture_output,
            "artifact_match": artifact_match,
            "generated_artifact": generated_artifact,
        }


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        if self.integrity_flag:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def baseline_and_integrity(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest["file_hashes"]
    allowed_write_paths = set(gold["allowed_write_paths"])
    changed_paths: list[str] = []

    for relpath, baseline_hash in baseline_files.items():
        if current_files.get(relpath) != baseline_hash:
            changed_paths.append(relpath)
    for relpath in current_files:
        if relpath not in baseline_files:
            changed_paths.append(relpath)

    unexpected = []
    for relpath in sorted(set(changed_paths)):
        if relpath.startswith("__pycache__/") or "/__pycache__/" in relpath:
            continue
        if relpath == ".network_egress_detected":
            continue
        if relpath not in allowed_write_paths:
            unexpected.append(relpath)
    if unexpected:
        state.raise_integrity("write_outside_whitelist")

    for relpath, expected_hash in gold["readonly_tree_hashes"].items():
        if tree_hash(AGENT_WS, relpath) != expected_hash:
            state.raise_integrity("immutable_slice_mutated")

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if any(path.startswith("tests/") for path in changed_paths):
        state.raise_integrity("tests_modified")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    return {
        "changed_paths": sorted(set(changed_paths)),
        "unexpected_writes": unexpected,
    }


def compute_milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = []
    aggregate = 0.0
    for milestone_id, weight in MILESTONE_SLOTS:
        passed = bool(milestones.get(milestone_id, False))
        if passed:
            aggregate += weight
        slots.append(
            {
                "milestone_id": milestone_id,
                "passed_bool": passed,
                "weight": weight,
            }
        )
    return {"slots": slots, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_reference.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    state = ScoreState()

    baseline = baseline_and_integrity(state, gold, manifest)
    changed_paths = set(baseline["changed_paths"])
    touched_progress = any(path in changed_paths for path in PARTIAL_PROGRESS_PATHS)
    if touched_progress:
        state.add("partial_progress.heuristic", 10, band="P_only")

    visible = run_visible_commands()
    test_ok, test_output = visible["npm_test"]
    build_ok, build_output = visible["npm_build"]
    typecheck_ok, typecheck_output = visible["npm_typecheck"]

    if test_ok:
        state.add("visible_tests_green", 20)
    else:
        state.errors.append("npm test failed")
    if build_ok:
        state.add("visible_build_green", 10)
    else:
        state.errors.append("npm run build failed")
    if typecheck_ok:
        state.add("visible_typecheck_green", 10)
    else:
        state.errors.append("npm run typecheck failed")

    loader_text = read_text("src/loader.mjs")
    docs_text = read_text("docs/cli/plugins.md")

    if not (test_ok and build_ok and typecheck_ok):
        if not changed_paths:
            state.apply_ceiling("no_submission", 0)
        else:
            state.apply_ceiling("no_visible_green", 20)

    hidden = run_hidden_checks(gold)
    outputs = hidden["outputs"]
    good_default_ok = outputs["good-default"][0] and outputs["good-default"][1].strip() == gold["expected_outputs"]["good-default"]
    good_named_ok = outputs["good-named"][0] and outputs["good-named"][1].strip() == gold["expected_outputs"]["good-named"]
    good_helper_ok = outputs["good-helper"][0] and outputs["good-helper"][1].strip() == gold["expected_outputs"]["good-helper"]
    bad_wrong_shape_ok = (
        not outputs["bad-wrong-shape"][0]
        and gold["expected_error_substring"] in outputs["bad-wrong-shape"][1]
    )

    if hidden["build_ok"] and good_default_ok and good_named_ok:
        state.add("hidden_dist_self_contained", 20)
    else:
        state.errors.append("built dist loader did not survive self-contained execution")
        if test_ok and build_ok and typecheck_ok:
            state.apply_ceiling("source_path_only", 25)

    if bad_wrong_shape_ok:
        state.add("hidden_runtime_validation", 10)
    else:
        state.errors.append("malformed plugin was accepted or wrong error surfaced")
        state.apply_ceiling("malformed_plugin_accepted", 25)

    if good_helper_ok:
        state.add("hidden_helper_cjs_survives", 10)
    else:
        state.errors.append("mixed helper.cjs path no longer works")
        state.apply_ceiling("helper_path_regression", 30)

    docs_ok = all(marker in docs_text for marker in gold["required_doc_markers"]) and all(
        marker not in docs_text for marker in gold["forbidden_doc_markers"]
    )
    if docs_ok:
        state.add("docs_contract_aligned", 5)
    else:
        state.errors.append("docs contract markers missing or stale")
        state.apply_ceiling("docs_or_evidence_drift", 50)

    if hidden["artifact_match"]:
        state.add("cli_capture_aligned", 5)
    else:
        state.errors.append("cli capture artifact missing or out of sync")
        state.apply_ceiling("docs_or_evidence_drift", 50)

    if "?? mod" in loader_text or "?? mod.plugin" in loader_text or "mod.default ?? mod" in loader_text:
        state.apply_ceiling("unsafe_namespace_coercion", 20)

    if VARIANT_ID == "v2-noisy-distractor" and (
        "require(" in loader_text
        or "require(" in docs_text
        or "node src/index.mjs" in docs_text
        or "src/plugins" in docs_text
    ):
        state.apply_ceiling("legacy_archive_anchor", 35)

    if VARIANT_ID == "v5-recovery-in-thread" and (not bad_wrong_shape_ok or not good_helper_ok):
        state.apply_ceiling("incident_blind_helper_regression", 30)

    state.milestones["M1_localization"] = touched_progress
    state.milestones["M2_primary_fix"] = test_ok and build_ok and typecheck_ok
    state.milestones["M3_invariants"] = state.integrity_flag == 0
    state.milestones["M4_functional"] = (
        state.milestones["M2_primary_fix"] and hidden["build_ok"] and good_default_ok and good_named_ok and good_helper_ok and bad_wrong_shape_ok
    )
    state.milestones["M5_e2e"] = state.milestones["M4_functional"] and docs_ok and hidden["artifact_match"]
    if state.integrity_flag:
        state.milestones["M3_invariants"] = False
        state.milestones["M4_functional"] = False
        state.milestones["M5_e2e"] = False

    final_score = state.final_score()
    result = {
        "schema_version": VERIFY_SCHEMA,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "pass": final_score >= 80 and state.integrity_flag == 0,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": sorted(state.ceilings_applied),
        "milestones": state.milestones,
        "milestone_vector": compute_milestone_vector(state.milestones),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(state.integrity_rules_fired),
        "breakdown": {
            **dict(sorted(state.breakdown.items())),
            "__bands": dict(sorted(state.bands.items())),
        },
        "errors": state.errors,
        "debug": {
            "changed_paths": sorted(changed_paths),
            "unexpected_writes": baseline["unexpected_writes"],
            "visible": {
                "npm_test_ok": test_ok,
                "npm_build_ok": build_ok,
                "npm_typecheck_ok": typecheck_ok,
                "npm_test_output": test_output,
                "npm_build_output": build_output,
                "npm_typecheck_output": typecheck_output,
            },
            "hidden": {
                "build_ok": hidden["build_ok"],
                "build_output": hidden["build_output"],
                "good_default_output": outputs["good-default"][1],
                "good_named_output": outputs["good-named"][1],
                "good_helper_output": outputs["good-helper"][1],
                "bad_wrong_shape_output": outputs["bad-wrong-shape"][1],
                "capture_ok": hidden["capture_ok"],
                "capture_output": hidden["capture_output"],
            },
        },
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
