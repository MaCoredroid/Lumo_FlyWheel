#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 90
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_file() and path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{rel_path}\n".encode())
        else:
            h.update(f"F:{rel_path}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def import_workspace_module(module_name: str, relpath: str) -> Any:
    target = AGENT_WS / relpath
    if str(AGENT_WS) not in sys.path:
        sys.path.insert(0, str(AGENT_WS))
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


def is_transient_runtime_file(relpath: str) -> bool:
    parts = relpath.split("/")
    if "__pycache__" in parts:
        return True
    if relpath.endswith(".pyc"):
        return True
    if relpath.startswith(".pytest_cache/"):
        return True
    if relpath == ".DS_Store" or relpath.endswith("/.DS_Store"):
        return True
    return False


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
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

    def add_error(self, message: str) -> None:
        self.errors.append(message)

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


def trusted_final_state(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, bool]:
    current_hashes: dict[str, str] = {}
    changed_paths: list[str] = []
    for path in sorted(AGENT_WS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        if is_transient_runtime_file(rel):
            continue
        current_hashes[rel] = sha256_file(path)
    baseline_hashes = manifest.get("file_hashes", {})
    for rel, expected in baseline_hashes.items():
        actual = current_hashes.get(rel)
        if actual != expected:
            changed_paths.append(rel)
    for rel in current_hashes:
        if rel not in baseline_hashes:
            changed_paths.append(rel)

    allowed = gold.get("allowed_write_paths", [])
    runtime_changes = any(rel.startswith("replay/") for rel in changed_paths)
    note_changed = "reports/incidents/transcript-merge.md" in changed_paths
    for rel in changed_paths:
        if not any(rel == path or rel.startswith(path.rstrip("/") + "/") for path in allowed):
            state.raise_integrity("write_outside_whitelist")
            if rel.startswith("tests/"):
                state.raise_integrity("tests_modified")

    for rel, expected_hash in manifest.get("readonly_tree_hashes", {}).items():
        actual = sha256_tree(AGENT_WS, rel)
        if actual != expected_hash:
            rule = "tests_modified" if rel.startswith("tests/locked") else "immutable_slice_mutated"
            state.raise_integrity(rule)

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    if note_changed and not runtime_changes:
        state.apply_ceiling("note_only_fix", 10)
    return {"runtime_changes": runtime_changes, "note_changed": note_changed}


def suspicious_shortcuts(state: ScoreState) -> None:
    merge_text = load_text("replay/merge.py")
    render_text = load_text("replay/render.py")
    summary_text = load_text("replay/incident_summary.py")
    if "dict.fromkeys" in render_text or "seen_lines" in render_text or "set()" in render_text and "render_events" in render_text:
        state.apply_ceiling("render_layer_filtering", 20)
        state.shortcut_detected = True
    if "if seen_completion:" in merge_text and "debug_only" not in merge_text:
        state.apply_ceiling("drop_all_post_completion", 20)
        state.shortcut_detected = True
    if "render_events(" in summary_text and '"count_source": "rendered_lines"' in summary_text:
        state.apply_ceiling("summary_still_render_coupled", 40)


def run_visible_tests(state: ScoreState) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AGENT_WS)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_merge",
            "tests.test_render",
            "tests.test_incident_summary",
        ],
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode == 0:
        state.add("visible.tests_pass", 20)
        state.milestones["visible_tests"] = True
        return True
    state.add_error(proc.stdout.strip()[-1000:])
    return False


def run_hidden_checks(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    merge_mod = import_workspace_module("transcript_merge_ws_merge", "replay/merge.py")
    render_mod = import_workspace_module("transcript_merge_ws_render", "replay/render.py")
    summary_mod = import_workspace_module("transcript_merge_ws_summary", "replay/incident_summary.py")

    hidden_root = VERIFIER_DATA_ROOT / "_shared" / "hidden_inputs"
    same_name = merge_mod.merge_paths(
        [
            hidden_root / "same_name_collision_part1.jsonl",
            hidden_root / "same_name_collision_part2.jsonl",
        ]
    )
    same_name_ids = [event["event_id"] for event in same_name if event.get("kind") == "tool_output"]
    same_name_ok = same_name_ids == gold["hidden_expectations"]["same_name_ids"]
    if same_name_ok:
        state.add("hidden.same_name_identity", 15)
        state.milestones["same_name_identity"] = True
    else:
        state.apply_ceiling("same_name_identity_unfixed", 25)

    interleaved = merge_mod.merge_paths(
        [
            hidden_root / "interleaved_fragments_part1.jsonl",
            hidden_root / "interleaved_fragments_part2.jsonl",
        ]
    )
    interleaved_map = {
        event["event_id"]: event.get("content", "")
        for event in interleaved
        if event.get("kind") == "tool_output"
    }
    interleaved_ok = interleaved_map == gold["hidden_expectations"]["interleaved_content"]
    if interleaved_ok:
        state.add("hidden.interleaved_fragments", 10)

    deferred_events = merge_mod.merge_paths([hidden_root / "deferred_output.jsonl"])
    deferred_ids = [event["event_id"] for event in deferred_events if event.get("kind") == "tool_output"]
    deferred_ok = deferred_ids == gold["hidden_expectations"]["deferred_ids"]
    if deferred_ok:
        state.add("hidden.deferred_output_survives", 10)

    rendered = render_mod.render_paths([hidden_root / "deferred_output.jsonl"])
    debug_ok = "debug replay duplicate" not in rendered and "TOOL fetch_logs: live tail output" in rendered
    if debug_ok:
        state.add("hidden.debug_noise_ignored", 10)

    summary = summary_mod.summarize_paths(
        [
            hidden_root / "same_name_collision_part1.jsonl",
            hidden_root / "same_name_collision_part2.jsonl",
        ]
    )
    summary_ok = (
        summary.get("count_source") == "merged_events"
        and summary.get("tool_output_blocks") == 2
    )
    if summary_ok:
        state.add("hidden.summary_stability", 10)
        state.milestones["summary_stability"] = True

    merge_text = load_text("replay/merge.py")
    if "event_id" in merge_text:
        state.add("code.event_identity", 5)
    if "if seen_completion and event.get(\"debug_only\")" in merge_text:
        state.add("code.no_blunt_drop", 5)

    return {
        "same_name_ok": same_name_ok,
        "interleaved_ok": interleaved_ok,
        "deferred_ok": deferred_ok,
        "debug_ok": debug_ok,
        "summary_ok": summary_ok,
    }


def score_note(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    note = load_text("reports/incidents/transcript-merge.md").lower()
    stable_identity = "stable event identity" in note or "`event_id`" in note or "event_id" in note
    deferred_output = (
        "deferred tool output" in note
        or "post-completion fragments" in note
        or "after completion" in note
    )
    merged_event_grounding = (
        "merged events" in note
        or "merged visible events" in note
        or "merged event stream" in note
    )
    required = stable_identity and deferred_output and merged_event_grounding
    variant = all(phrase.lower() in note for phrase in gold["variant_note_phrases"])
    shortcut = "render" in note and "not an acceptable fix" in note and "deferred tool output" in note
    if required:
        state.add("note.core_boundary", 5)
        state.milestones["note_core"] = True
    else:
        state.apply_ceiling("stale_incident_note", 60)
    if variant:
        state.add("note.variant_context", 5, band="P_only")
    if shortcut:
        state.add("note.shortcut_rejection", 5, band="P_only")
    return {"required": required, "variant": variant, "shortcut": shortcut}


def compute_milestones(state: ScoreState, visible_ok: bool, hidden: dict[str, bool], note: dict[str, bool]) -> dict[str, bool]:
    m1 = bool(note["required"] and ("event_identity" in state.breakdown or note["variant"]))
    m2 = bool(hidden["same_name_ok"] and hidden["deferred_ok"] and hidden["debug_ok"])
    m3 = bool(state.integrity_flag == 0 and not any(name in state.ceilings_applied for name in ("render_layer_filtering", "drop_all_post_completion")))
    m4 = bool(visible_ok and hidden["summary_ok"])
    m5 = bool(m2 and m4 and note["required"] and state.final_score() >= 75)
    if state.integrity_flag:
        m3 = False
        m4 = False
        m5 = False
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }


def milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = []
    agg = 0.0
    for name, weight in MILESTONE_SLOTS:
        passed = bool(milestones.get(name, False))
        if passed:
            agg += weight
        slots.append({"slot": name, "weight": weight, "passed_bool": passed})
    return {"slots": slots, "M_aggregate": round(agg, 4)}


def main() -> int:
    started = time.time()
    state = ScoreState()
    variant_root = VERIFIER_DATA_ROOT / VARIANT_ID
    gold = load_json(variant_root / "gold_solution.json")
    manifest = load_json(variant_root / "workspace_manifest.json")

    trusted_final_state(state, gold, manifest)
    suspicious_shortcuts(state)
    visible_ok = run_visible_tests(state)
    hidden = run_hidden_checks(state, gold)
    note = score_note(state, gold)
    milestones = compute_milestones(state, visible_ok, hidden, note)
    result = {
        "schema_version": VERIFY_SCHEMA,
        "variant_id": VARIANT_ID,
        "pass": state.final_score() >= 75 and state.integrity_flag == 0,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": milestones,
        "milestone_vector": milestone_vector(milestones),
        "breakdown": {"__bands": state.bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "wall_clock_seconds": round(time.time() - started, 3),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
