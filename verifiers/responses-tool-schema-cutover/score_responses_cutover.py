#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace")).resolve()
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data")).resolve()
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json")).resolve()
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 100
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    import hashlib

    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    digest = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            digest.update(f"D:{rel_path}\n".encode())
        else:
            digest.update(f"F:{rel_path}\n".encode())
            digest.update(sha256_file(path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_workspace_files(root: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload[path.relative_to(root).as_posix()] = sha256_file(path)
    return payload


def rel_allowed(relpath: str, allowed_paths: list[str]) -> bool:
    return relpath in allowed_paths


def lower_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text().lower()


def text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


def clear_gateway_modules() -> None:
    for name in list(sys.modules):
        if name == "gateway" or name.startswith("gateway."):
            del sys.modules[name]


def load_workspace_modules() -> tuple[Any, Any]:
    clear_gateway_modules()
    sys.path.insert(0, str(AGENT_WS))
    try:
        adapter = importlib.import_module("gateway.adapter")
        reducer = importlib.import_module("gateway.reducer")
        return adapter, reducer
    finally:
        if sys.path and sys.path[0] == str(AGENT_WS):
            sys.path.pop(0)


def run_visible_pytest() -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(AGENT_WS)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_adapter_contract.py",
        "tests/test_replay_render.py",
    ]
    proc = subprocess.run(
        cmd,
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout


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
        return round(max(0, min(self.raw_m_score, self.ceiling_cap)) / MAX_M_POINTS, 4)


def baseline_and_integrity(
    state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest["file_hashes"]
    allowed_write_paths = list(gold["allowed_write_paths"])
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
        if not rel_allowed(relpath, allowed_write_paths):
            unexpected.append(relpath)
    if unexpected:
        state.raise_integrity("write_outside_whitelist")

    for relpath, expected_hash in manifest["readonly_tree_hashes"].items():
        if tree_hash(AGENT_WS, relpath) != expected_hash:
            state.raise_integrity("immutable_slice_mutated")

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    adapter_hash_changed = current_files.get("gateway/adapter.py") != baseline_files.get("gateway/adapter.py")
    reducer_hash_changed = current_files.get("gateway/reducer.py") != baseline_files.get("gateway/reducer.py")
    tests_changed = any(
        current_files.get(rel) != baseline_files.get(rel)
        for rel in ["tests/test_adapter_contract.py", "tests/test_replay_render.py"]
    )

    required_test_markers = [marker.lower() for marker in gold["required_test_markers"]]
    combined_tests = (
        lower_text("tests/test_adapter_contract.py") + "\n" + lower_text("tests/test_replay_render.py")
    )
    if tests_changed and not all(marker in combined_tests for marker in required_test_markers):
        state.raise_integrity("tests_modified")

    return {
        "changed_paths": sorted(set(changed_paths)),
        "adapter_changed": adapter_hash_changed,
        "reducer_changed": reducer_hash_changed,
        "tests_changed": tests_changed,
        "unexpected_writes": unexpected,
    }


def evaluate_cases(adapter: Any, reducer: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    call_id_passes = 0
    render_passes = 0
    preserved_passes = 0
    details = []

    for case in cases:
        fixture = AGENT_WS / case["fixture"]
        normalized = adapter.normalize_events(fixture)
        transcript = reducer.build_replay(fixture)
        rendered = reducer.render_replay(fixture)

        tool_calls = [item for item in normalized if item.get("kind") == "tool_call"]
        transcript_calls = [item for item in transcript if item.get("type") == "tool_call"]
        transcript_results = [item for item in transcript if item.get("type") == "tool_result"]

        call_ids_ok = [item.get("call_id") for item in tool_calls] == case["expected_call_ids"]
        render_ok = rendered == case["expected_render"]
        preserved_ok = (
            len(transcript_calls) == len(case["expected_call_ids"])
            and len(transcript_results) == len(case["expected_call_ids"])
            and [item.get("call_id") for item in transcript_calls] == case["expected_call_ids"]
            and [item.get("call_id") for item in transcript_results] == case["expected_call_ids"]
        )

        if call_ids_ok:
            call_id_passes += 1
        if render_ok:
            render_passes += 1
        if preserved_ok:
            preserved_passes += 1

        details.append(
            {
                "id": case["id"],
                "call_ids_ok": call_ids_ok,
                "render_ok": render_ok,
                "preserved_ok": preserved_ok,
            }
        )

    total = len(cases) if cases else 1
    return {
        "call_id_ratio": call_id_passes / total,
        "render_ratio": render_passes / total,
        "preserved_ratio": preserved_passes / total,
        "details": details,
    }


def score_contracts(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    config_text = text("codex/config.toml")
    docs_text = lower_text("docs/migrations/responses-cutover.md")
    config_ok = all(term in config_text for term in gold["required_config_terms"]) and all(
        term not in config_text for term in gold["forbidden_config_terms"]
    )
    docs_ok = all(term.lower() in docs_text for term in gold["required_doc_phrases"])
    if config_ok:
        state.add("contract.config_updated", 10)
    if docs_ok:
        state.add("contract.docs_updated", 10)
    return {
        "config_ok": config_ok,
        "docs_ok": docs_ok,
    }


def partial_progress_score(changed: dict[str, Any], contracts: dict[str, bool]) -> int:
    points = 0
    if changed["adapter_changed"] or changed["reducer_changed"]:
        points += 4
    if changed["adapter_changed"] and changed["reducer_changed"]:
        points += 2
    if changed["tests_changed"]:
        points += 2
    if contracts["config_ok"] and contracts["docs_ok"]:
        points += 2
    return min(points, 10)


def apply_ceilings(
    state: ScoreState,
    changed: dict[str, Any],
    visible_ok: bool,
    case_scores: dict[str, Any],
    contracts: dict[str, bool],
) -> None:
    if not changed["changed_paths"] and not visible_ok:
        state.apply_ceiling("no_submission", 0)
        return

    if not visible_ok:
        state.apply_ceiling("no_visible_green", 20)

    if not changed["adapter_changed"] and not changed["reducer_changed"]:
        state.apply_ceiling("analysis_only", 20)

    if case_scores["preserved_ratio"] < 1.0:
        state.apply_ceiling("tool_name_only_join", 20)

    ordinal_case = next(
        (detail for detail in case_scores["details"] if detail["id"] == "ordinal-trap"),
        None,
    )
    if ordinal_case and not ordinal_case["call_ids_ok"]:
        state.apply_ceiling("call_id_ordinal_shortcut", 25)

    if changed["adapter_changed"] ^ changed["reducer_changed"]:
        state.apply_ceiling("adapter_or_reducer_gap", 30)

    if case_scores["call_id_ratio"] == 1.0 and case_scores["render_ratio"] == 1.0:
        if not (contracts["config_ok"] and contracts["docs_ok"]):
            state.apply_ceiling("contract_drift", 50)

    if visible_ok and not changed["tests_changed"]:
        state.apply_ceiling("no_test_regression_guard", 35)


def milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = []
    total = 0.0
    for slot_id, weight in MILESTONE_SLOTS:
        passed = bool(milestones.get(slot_id, False))
        if passed:
            total += weight
        slots.append({"slot_id": slot_id, "weight": weight, "passed_bool": passed})
    return {"slots": slots, "M_aggregate": round(total, 4)}


def main() -> int:
    start = time.time()
    state = ScoreState()
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_responses.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    changed = baseline_and_integrity(state, gold, manifest)

    visible_ok, visible_output = run_visible_pytest()
    if visible_ok:
        state.add("phase2.visible_pytest", 20)
    else:
        state.errors.append(visible_output.strip())

    try:
        adapter, reducer = load_workspace_modules()
        visible_case_scores = evaluate_cases(
            adapter,
            reducer,
            [
                {
                    "id": "visible",
                    "fixture": gold["visible_fixture"],
                    "expected_render": gold["visible_render"],
                    "expected_call_ids": gold["visible_call_ids"],
                }
            ],
        )
        hidden_case_scores = evaluate_cases(adapter, reducer, gold["hidden_cases"])
    except Exception as exc:  # pragma: no cover - defensive scoring path
        state.errors.append(f"module_load_or_eval_failed: {exc}")
        visible_case_scores = {"call_id_ratio": 0.0, "render_ratio": 0.0, "preserved_ratio": 0.0, "details": []}
        hidden_case_scores = {"call_id_ratio": 0.0, "render_ratio": 0.0, "preserved_ratio": 0.0, "details": []}

    if visible_case_scores["render_ratio"] == 1.0:
        state.add("core.visible_render_exact", 10)
    if hidden_case_scores["call_id_ratio"] > 0:
        state.add("core.call_id_join", round(15 * hidden_case_scores["call_id_ratio"]))
    if hidden_case_scores["render_ratio"] > 0:
        state.add("core.hidden_replay_oracle", round(15 * hidden_case_scores["render_ratio"]))
    if hidden_case_scores["preserved_ratio"] > 0:
        state.add("core.multi_call_preserved", round(10 * hidden_case_scores["preserved_ratio"]))

    contracts = score_contracts(state, gold)
    if changed["tests_changed"]:
        state.add("repair.regression_tests_updated", 10)

    state.add(
        "partial_progress.heuristic",
        partial_progress_score(changed, contracts),
        band="P_only",
    )

    apply_ceilings(
        state,
        changed,
        visible_ok,
        hidden_case_scores,
        contracts,
    )

    state.milestones["M1_localization"] = changed["adapter_changed"] or changed["reducer_changed"]
    state.milestones["M2_primary_fix"] = (
        visible_ok
        and visible_case_scores["render_ratio"] == 1.0
        and hidden_case_scores["call_id_ratio"] == 1.0
        and hidden_case_scores["preserved_ratio"] == 1.0
    )
    state.milestones["M3_invariants"] = state.integrity_flag == 0
    state.milestones["M4_functional"] = (
        state.milestones["M2_primary_fix"]
        and contracts["config_ok"]
        and contracts["docs_ok"]
    )
    state.milestones["M5_e2e"] = (
        state.milestones["M4_functional"]
        and hidden_case_scores["render_ratio"] == 1.0
        and state.ceiling_cap > 30
    )

    passed = (
        state.integrity_flag == 0
        and visible_ok
        and visible_case_scores["render_ratio"] == 1.0
        and hidden_case_scores["call_id_ratio"] == 1.0
        and hidden_case_scores["render_ratio"] == 1.0
        and hidden_case_scores["preserved_ratio"] == 1.0
        and contracts["config_ok"]
        and contracts["docs_ok"]
    )

    result = {
        "pass": passed,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector(state.milestones),
        "breakdown": {
            "__bands": state.bands,
            **state.breakdown,
        },
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "schema_version": VERIFY_SCHEMA,
        "changed_paths": changed["changed_paths"],
        "visible_case": visible_case_scores,
        "hidden_cases": hidden_case_scores,
        "contracts": contracts,
        "wall_clock_seconds": round(time.time() - start, 2),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
