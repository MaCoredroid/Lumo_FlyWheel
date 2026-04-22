#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

GOLD_PATH = VERIFIER_DATA / VARIANT_ID / "gold_reference.json"
MANIFEST_PATH = VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ScorerState:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    shortcut_detected: bool = False
    errors: list[str] = field(default_factory=list)

    def add(self, key: str, points: int, band: str = "M") -> None:
        if points <= 0:
            return
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

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


def parse_simple_toml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def read_workspace_files() -> dict[str, str]:
    files = {}
    for rel in (
        "src/incident_handoff/client.py",
        "src/incident_handoff/adapter.py",
        "src/incident_handoff/replay.py",
        "src/incident_handoff/render.py",
        "docs/migrations/responses-cutover.md",
        ".codex/config.toml",
    ):
        path = AGENT_WS / rel
        files[rel] = path.read_text() if path.exists() else ""
    return files


def run_visible_pytest() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_adapter.py",
        "tests/test_replay.py",
        "tests/test_render.py",
    ]
    proc = subprocess.run(
        cmd,
        cwd=AGENT_WS,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def compute_manifest_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in AGENT_WS.rglob("*") if p.is_file()):
        rel = path.relative_to(AGENT_WS).as_posix()
        if rel.endswith(".pyc") or "/__pycache__/" in rel or rel.startswith(".pytest_cache/"):
            continue
        hashes[rel] = sha256_file(path)
    return hashes


def normalize_and_check(files: dict[str, str], gold: dict[str, Any]) -> dict[str, bool]:
    sys.path.insert(0, str(AGENT_WS / "src"))
    adapter = load_module("incident_handoff_adapter", AGENT_WS / "src/incident_handoff/adapter.py")
    replay = load_module("incident_handoff_replay", AGENT_WS / "src/incident_handoff/replay.py")
    render = load_module("incident_handoff_render", AGENT_WS / "src/incident_handoff/render.py")

    checks: dict[str, bool] = {}

    interleaved = load_json(AGENT_WS / "transcripts/interleaved_tool_turn.json")
    interleaved_events = adapter.normalize_response_items(interleaved)
    checks["hidden.interleaved_order"] = bool(
        len(interleaved_events) >= 3
        and interleaved_events[0].get("kind") == "assistant_text"
        and isinstance(interleaved_events[0].get("text"), str)
        and interleaved_events[1].get("kind") == "tool_call"
        and interleaved_events[2].get("kind") == "tool_result"
        and interleaved_events[1].get("call_id") == interleaved_events[2].get("call_id")
    )

    checks["hidden.replay_roundtrip"] = replay.replay_from_serialized(
        replay.serialize_events(interleaved_events)
    ) == interleaved_events

    rendered = render.render_transcript(interleaved_events)
    checks["hidden.render_mentions_call_id"] = "call-1" in rendered and "tool_result" in rendered

    multi_message = AGENT_WS / "transcripts/multi_message_turn.json"
    if multi_message.exists():
        multi_events = adapter.normalize_response_items(load_json(multi_message))
        first_text = multi_events[0].get("text", "") if multi_events else ""
        checks["hidden.multi_block_message"] = (
            isinstance(first_text, str)
            and "Checking" in first_text
            and "Escalating" in first_text
            and "output_text" not in first_text
        )
    else:
        checks["hidden.multi_block_message"] = True

    reordered = AGENT_WS / "transcripts/reordered_chunks.json"
    if reordered.exists():
        reordered_events = adapter.normalize_response_items(load_json(reordered))
        sequences = [event.get("sequence") for event in reordered_events]
        kinds = [event.get("kind") for event in reordered_events]
        checks["hidden.reordered_chunk_stability"] = (
            sequences == [0, 1, 2, 3]
            and kinds == ["assistant_text", "tool_call", "tool_result", "assistant_text"]
            and reordered_events[1].get("call_id") == reordered_events[2].get("call_id")
        )
        checks["hidden.reordered_roundtrip"] = replay.replay_from_serialized(
            replay.serialize_events(reordered_events)
        ) == reordered_events
    else:
        checks["hidden.reordered_chunk_stability"] = True
        checks["hidden.reordered_roundtrip"] = True

    future = AGENT_WS / "transcripts/future_event_passthrough.json"
    if future.exists():
        future_events = adapter.normalize_response_items(load_json(future))
        replayed_future = replay.replay_from_serialized(replay.serialize_events(future_events))
        checks["hidden.future_event_passthrough"] = (
            any(event.get("kind") == "unknown_event" for event in future_events)
            and replayed_future == future_events
        )
    else:
        checks["hidden.future_event_passthrough"] = True

    config = parse_simple_toml(AGENT_WS / ".codex/config.toml")
    checks["config.responses_wire"] = config.get("wire_api") == "responses"
    checks["config.responses_mode"] = config.get("transcript_mode") == "responses_events"

    doc_text = files["docs/migrations/responses-cutover.md"].lower()
    checks["docs.event_ordering"] = "event ordering" in doc_text
    checks["docs.tool_result_correlation"] = "tool-result correlation" in doc_text
    required_doc_keywords = [kw.lower() for kw in gold.get("required_doc_keywords", [])]
    checks["docs.variant_complete"] = all(keyword in doc_text for keyword in required_doc_keywords)

    legacy_markers = tuple(marker.lower() for marker in gold.get("legacy_live_markers", []))
    client_text = files["src/incident_handoff/client.py"].lower()
    config_text = files[".codex/config.toml"].lower()
    checks["hidden.legacy_path_removed"] = not any(
        marker in client_text or marker in config_text for marker in legacy_markers
    )

    checks["hidden.all_variant_checks"] = all(checks.values())
    return checks


def apply_integrity(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> tuple[int, list[str]]:
    current = compute_manifest_hashes()
    baseline = manifest.get("files", {})
    allowed = set(gold.get("allowed_writes", []))
    readonly_roots = tuple(gold.get("readonly_roots", []))
    changed_paths: list[str] = []
    for rel, baseline_hash in baseline.items():
        if current.get(rel) != baseline_hash:
            changed_paths.append(rel)
    for rel in current:
        if rel not in baseline:
            changed_paths.append(rel)
    changed_paths = sorted(set(changed_paths))
    changed_paths = [
        path for path in changed_paths
        if not (
            path.endswith(".pyc")
            or "/__pycache__/" in path
            or path.startswith(".pytest_cache/")
        )
    ]

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")

    if any(path.startswith("tests/") for path in changed_paths):
        state.raise_integrity("tests_modified")

    if any(path.startswith(readonly_roots) for path in changed_paths):
        state.raise_integrity("immutable_slice_mutated")

    outside = [
        path for path in changed_paths
        if path not in allowed and not path.startswith("brief/")
    ]
    if outside:
        state.raise_integrity("write_outside_whitelist")

    return len(changed_paths), changed_paths


def milestone_vector(state: ScorerState) -> dict[str, Any]:
    weights = {
        "M1_localization": 0.10,
        "M2_primary_fix": 0.20,
        "M3_invariants": 0.20,
        "M4_functional": 0.20,
        "M5_e2e": 0.30,
    }
    if state.integrity_flag:
        for milestone in ("M3_invariants", "M4_functional", "M5_e2e"):
            state.milestones[milestone] = False
    slots = []
    aggregate = 0.0
    for milestone_id, weight in weights.items():
        passed = bool(state.milestones.get(milestone_id, False))
        slots.append({
            "milestone_id": milestone_id,
            "passed_bool": passed,
            "weight": weight,
        })
        if passed:
            aggregate += weight
    return {"slots": slots, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    gold = load_json(GOLD_PATH)
    manifest = load_json(MANIFEST_PATH)
    files = read_workspace_files()
    state = ScorerState()

    visible_ok, visible_output = run_visible_pytest()
    if visible_ok:
        state.add("visible.pytest", gold["weights"].get("visible.pytest", 0))

    checks = normalize_and_check(files, gold)
    if checks["config.responses_wire"]:
        state.add("config.responses_wire", gold["weights"].get("config.responses_wire", 0))
    if checks["config.responses_mode"]:
        state.add("config.responses_mode", gold["weights"].get("config.responses_mode", 0))
    if checks["docs.event_ordering"]:
        state.add("docs.event_ordering", gold["weights"].get("docs.event_ordering", 0))
    if checks["docs.tool_result_correlation"]:
        state.add("docs.tool_result_correlation", gold["weights"].get("docs.tool_result_correlation", 0))
    if checks["hidden.interleaved_order"]:
        state.add("hidden.interleaved_order", gold["weights"].get("hidden.interleaved_order", 0))
    if checks["hidden.multi_block_message"]:
        state.add("hidden.multi_block_message", gold["weights"].get("hidden.multi_block_message", 0))
    if checks["hidden.replay_roundtrip"]:
        state.add("hidden.replay_roundtrip", gold["weights"].get("hidden.replay_roundtrip", 0))
    if checks["hidden.reordered_chunk_stability"]:
        state.add("hidden.reordered_chunk_stability", gold["weights"].get("hidden.reordered_chunk_stability", 0))
    if checks["hidden.future_event_passthrough"]:
        state.add("hidden.future_event_passthrough", gold["weights"].get("hidden.future_event_passthrough", 0))
    if checks["hidden.legacy_path_removed"]:
        state.add("hidden.legacy_path_removed", gold["weights"].get("hidden.legacy_path_removed", 0))
    if checks["docs.variant_complete"]:
        state.add("docs.variant_complete", gold["weights"].get("docs.variant_complete", 0))

    changed_count, changed_paths = apply_integrity(state, gold, manifest)

    hidden_successes = sum(
        1
        for key in (
            "hidden.interleaved_order",
            "hidden.multi_block_message",
            "hidden.replay_roundtrip",
            "hidden.reordered_chunk_stability",
            "hidden.future_event_passthrough",
            "hidden.legacy_path_removed",
        )
        if checks.get(key, False)
    )

    if visible_ok and (hidden_successes < 3 or not checks["hidden.legacy_path_removed"]):
        state.apply_ceiling("visible_only_cutover", 20)
    if not checks["hidden.multi_block_message"]:
        state.apply_ceiling("flattened_multi_event_turn", 25)
    if not checks["hidden.legacy_path_removed"]:
        state.apply_ceiling("compatibility_shim_left_live", 30)
    if not checks["hidden.reordered_chunk_stability"]:
        state.apply_ceiling("reordered_chunk_instability", 35)
    if VARIANT_ID in {"v4-multi-corpus-objective", "v5-recovery-in-thread"} and not checks["hidden.replay_roundtrip"]:
        state.apply_ceiling("objective_drift_to_render_only", 40)
    if not checks["hidden.future_event_passthrough"]:
        state.apply_ceiling("future_event_corruption", 30)

    state.milestones["M1_localization"] = changed_count >= 2 or visible_ok
    state.milestones["M2_primary_fix"] = visible_ok and checks["config.responses_wire"]
    state.milestones["M3_invariants"] = state.integrity_flag == 0 and not state.shortcut_detected
    state.milestones["M4_functional"] = (
        state.milestones["M2_primary_fix"]
        and checks["hidden.interleaved_order"]
        and checks["hidden.replay_roundtrip"]
    )
    state.milestones["M5_e2e"] = (
        state.milestones["M2_primary_fix"]
        and checks["hidden.all_variant_checks"]
    )

    max_m_points = sum(gold["weights"].values())
    final_score = state.final_score()
    milestone_vec = milestone_vector(state)
    final_m = 0.0 if max_m_points == 0 else round(final_score / max_m_points, 4)
    result = {
        "pass": final_score >= gold.get("pass_bar", 40),
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": final_m,
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": state.milestones,
        "milestone_vector": milestone_vec,
        "breakdown": {
            **dict(sorted(state.breakdown.items())),
            "__bands": dict(sorted(state.breakdown_bands.items())),
        },
        "ceilings_applied": sorted(state.ceilings_applied),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(state.integrity_rules_fired),
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "schema_version": "cnb55.verify_result.v3",
        "checks": checks,
        "changed_paths": changed_paths,
        "visible_output": visible_output,
        "max_m_points": max_m_points,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
