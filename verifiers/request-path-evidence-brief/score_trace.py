#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

PATH_MAP_SCHEMA = "cnb55.request_path_map.v1"
VERIFY_RESULT_SCHEMA = "cnb55.verify_result.v3"
READONLY_TREES = (
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "sync_app",
    "config",
    "docs",
    "ops",
    "release_context",
    "incident_context",
    "tests",
)
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
ALLOWED_OUTPUTS = {
    "artifacts/request_path_brief.md",
    "artifacts/path_map.json",
    "artifacts/docs_correction.md",
    "artifacts/README.md",
}
EVIDENCE_TOKEN_RE = re.compile(r"`([^`]+)`")
MAX_M_POINTS = 100


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_of_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_of_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if rel_path == "__pycache__" or rel_path.startswith("__pycache__/"):
            continue
        if path.name.endswith(".pyc"):
            continue
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_of_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_tokens(text: str) -> set[str]:
    return set(EVIDENCE_TOKEN_RE.findall(text))


def lower_contains_any(text: str, needles: list[str] | tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def normalize_symbol_ref(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::")[-1]
    if "/" in text:
        text = text.split("/")[-1]
    if "." in text:
        text = text.split(".")[-1]
    return text.strip()


def collect_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            out.extend(collect_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            out.extend(collect_strings(nested))
    return out


def relation_present(live_steps: list[dict[str, Any]], left_idx: int, right_idx: int) -> bool:
    left = live_steps[left_idx]
    right = live_steps[right_idx]
    left_symbol = normalize_symbol_ref(left.get("symbol"))
    right_symbol = normalize_symbol_ref(right.get("symbol"))
    left_callee = normalize_symbol_ref(left.get("callee_symbol"))
    right_caller = normalize_symbol_ref(right.get("caller_symbol"))
    if left_callee == right_symbol:
        return True
    if right_caller == left_symbol:
        return True
    if right_caller == "sync_item" and left_symbol in {"_resolve_owner", "make_record", "build_routing_key"}:
        return True
    if left_symbol == "sync_item" and right_caller == "sync_item":
        return True
    return False


def field_mentions_target(value: Any, expected_file: str, expected_symbol: str) -> bool:
    strings = collect_strings(value)
    symbol_token = expected_symbol.lower()
    file_token = expected_file.lower()
    for text in strings:
        lowered = text.lower()
        if file_token in lowered and symbol_token in lowered:
            return True
        if normalize_symbol_ref(text).lower() == symbol_token:
            return True
    return False


@dataclass
class ScorerState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    shortcut: bool = False
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    path_map: dict[str, Any] | None = None
    brief_text: str = ""
    correction_text: str = ""
    brief_tokens: set[str] = field(default_factory=set)
    correction_tokens: set[str] = field(default_factory=set)
    live_path_exact_matches: int = 0

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m(self) -> float:
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def load_gold() -> dict[str, Any]:
    return json.loads((VERIFIER_DATA_ROOT / VARIANT_ID / "gold_path.json").read_text())


def load_manifest() -> dict[str, Any]:
    return json.loads((VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json").read_text())


def trusted_final_state(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.apply_ceiling(f"shim:{shim}", 0)
            state.raise_integrity("pytest_shim")
            state.add_error(f"disallowed shim file present: {shim}")

    for rel, expected in gold.get("readonly_tree_hashes", {}).items():
        actual = sha256_of_tree(AGENT_WS, rel)
        if expected and actual != expected:
            state.apply_ceiling(f"readonly_tree:{rel}", 0)
            state.raise_integrity("immutable_slice_mutated")
            state.add_error(f"readonly tree mutated: {rel}")

    known = set(manifest.get("files", []))
    for path in AGENT_WS.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        if rel in known or rel in ALLOWED_OUTPUTS:
            continue
        if rel.startswith(".pytest_cache/"):
            continue
        if rel.endswith(".pyc") or rel.startswith("__pycache__/") or "/__pycache__/" in rel:
            continue
        state.apply_ceiling("wrote_outside_whitelist", 0)
        state.raise_integrity("write_outside_whitelist")
        state.add_error(f"wrote file outside allowed outputs: {rel}")
        break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.apply_ceiling("network_egress", 0)
        state.raise_integrity("network_egress")
        state.add_error("network egress marker present")


def read_outputs(state: ScorerState) -> None:
    brief_path = AGENT_WS / "artifacts" / "request_path_brief.md"
    correction_path = AGENT_WS / "artifacts" / "docs_correction.md"
    path_map_path = AGENT_WS / "artifacts" / "path_map.json"

    if brief_path.exists() and brief_path.stat().st_size > 0:
        state.brief_text = read_text(brief_path)
        state.brief_tokens = extract_tokens(state.brief_text)
        state.add("phase2.brief_exists", 4)
    else:
        state.add_error("artifacts/request_path_brief.md missing or empty")

    if correction_path.exists() and correction_path.stat().st_size > 0:
        state.correction_text = read_text(correction_path)
        state.correction_tokens = extract_tokens(state.correction_text)
        state.add("phase2.correction_exists", 4)
    else:
        state.add_error("artifacts/docs_correction.md missing or empty")

    if not path_map_path.exists() or path_map_path.stat().st_size == 0:
        state.apply_ceiling("no_artifacts", 0)
        state.add_error("artifacts/path_map.json missing or empty")
        return

    try:
        doc = json.loads(read_text(path_map_path))
    except Exception as exc:
        state.apply_ceiling("malformed_path_map", 10)
        state.add_error(f"artifacts/path_map.json is not valid JSON: {exc}")
        return

    if not isinstance(doc, dict):
        state.apply_ceiling("malformed_path_map", 10)
        state.add_error("artifacts/path_map.json must contain a JSON object")
        return

    if doc.get("schema_version") != PATH_MAP_SCHEMA:
        state.apply_ceiling("malformed_path_map", 10)
        state.add_error(
            f"path_map schema_version must be {PATH_MAP_SCHEMA}, got {doc.get('schema_version')!r}"
        )
        return

    state.path_map = doc
    state.add("phase2.path_map_schema", 8)


def score_phase2(state: ScorerState) -> None:
    if state.path_map is None:
        return
    if state.path_map.get("variant_id") == VARIANT_ID:
        state.add("phase2.variant_match", 4)
    else:
        state.add_error("path_map variant_id does not match .scenario_variant")

    required_sections = ("live_path", "field_derivations", "test_observations", "rejected_decoys")
    if all(section in state.path_map for section in required_sections):
        state.add("phase2.sections_present", 6)
    else:
        state.add_error("path_map missing one or more required top-level sections")

    if "wrote_outside_whitelist" not in state.ceilings_applied:
        state.add("phase2.no_stray_files", 4)
    if not any(name.startswith("shim:") for name in state.ceilings_applied):
        state.add("phase2.no_shim", 4)


def score_live_path(state: ScorerState, gold: dict[str, Any]) -> None:
    if state.path_map is None:
        return
    live = state.path_map.get("live_path")
    expected = gold["live_path"]
    if not isinstance(live, list):
        state.add_error("live_path must be a list")
        return

    if not all(isinstance(step, dict) for step in live):
        state.add_error("live_path entries must be objects")
        state.apply_ceiling("missing_symbol_adjacency", 25)
        return

    live_steps = [step for step in live if isinstance(step, dict)]
    expected_nodes = [(step["file"], step["symbol"]) for step in expected]
    observed_positions: list[int | None] = []
    for expected_file, expected_symbol in expected_nodes:
        idx = None
        for pos, step in enumerate(live_steps):
            if step.get("file") == expected_file and step.get("symbol") == expected_symbol:
                idx = pos
                break
        observed_positions.append(idx)

    matched_nodes = sum(1 for pos in observed_positions if pos is not None)

    matched_relations = 0
    for left_pos, right_pos in zip(observed_positions, observed_positions[1:]):
        if left_pos is None or right_pos is None or left_pos >= right_pos:
            continue
        if relation_present(live_steps, left_pos, right_pos):
            matched_relations += 1

    exact_matches = matched_nodes
    state.live_path_exact_matches = exact_matches
    state.add("hidden.live_path_nodes", exact_matches * 2)
    state.add("hidden.live_path_adjacency", matched_relations)

    if matched_nodes == len(expected_nodes) and matched_relations == len(expected_nodes) - 1:
        state.add("hidden.live_path_complete", 3)
        state.milestones["M4_functional"] = True
    else:
        state.add_error(
            f"live_path matched nodes={matched_nodes}/{len(expected_nodes)} edges={matched_relations}/{len(expected_nodes) - 1} for {VARIANT_ID}"
        )
        state.apply_ceiling("missing_symbol_adjacency", 25)


def score_field_derivations(state: ScorerState, gold: dict[str, Any]) -> None:
    if state.path_map is None:
        return
    derivations = state.path_map.get("field_derivations")
    if not isinstance(derivations, dict):
        state.add_error("field_derivations must be an object")
        return

    for field_name, points in (("owner_source", 8), ("routing_key", 8), ("emission", 4)):
        expected = gold["field_derivations"][field_name]
        got = derivations.get(field_name)
        if field_mentions_target(got, expected["file"], expected["symbol"]):
            state.add(f"hidden.field_derivation.{field_name}", points)
        else:
            state.add_error(f"{field_name} derivation mismatch")

    owner_got = derivations.get("owner_source", {})
    owner_blob = json.dumps(owner_got, sort_keys=True).lower()
    if "sync_app/store.py" in owner_blob and "not_derived_from" not in owner_blob and "not_derived_in" not in owner_blob:
        state.apply_ceiling("store_claimed_as_decision_layer", 25)

    routing_got = derivations.get("routing_key", {})
    routing_blob = json.dumps(routing_got, sort_keys=True).lower()
    if (
        ("sync_app/store.py" in routing_blob and "not_derived_in" not in routing_blob)
        or "before owner resolution" in routing_blob
    ):
        state.apply_ceiling("pre_owner_routing_claim", 30)

    if state.breakdown.get("hidden.field_derivation.owner_source", 0) > 0:
        state.milestones["M2_primary_fix"] = True


def score_test_observations(state: ScorerState, gold: dict[str, Any]) -> None:
    if state.path_map is None:
        return
    observations = state.path_map.get("test_observations")
    if not isinstance(observations, list):
        state.add_error("test_observations must be a list")
        return

    matched = 0
    matched_expected: set[tuple[str, str]] = set()
    expected = [
        (entry["file"], entry["symbol"])
        for entry in gold["test_observations"]
    ]
    for item in observations:
        if not isinstance(item, dict):
            continue
        for expected_file, expected_symbol in expected:
            key = (expected_file, expected_symbol)
            if key in matched_expected:
                continue
            if field_mentions_target(item, expected_file, expected_symbol):
                matched_expected.add(key)
                matched += 1
    state.add("hidden.test_observations", min(matched, len(expected)) * 5)
    if matched >= 1:
        state.milestones["M1_localization"] = True
    if matched < len(expected):
        state.add_error(f"matched only {matched}/{len(expected)} expected test observations")


def score_decoy_and_variant_rules(state: ScorerState, gold: dict[str, Any]) -> None:
    if state.path_map is None:
        return
    decoys = state.path_map.get("rejected_decoys")
    if not isinstance(decoys, list) or not decoys:
        state.apply_ceiling("no_decoy_rejection", 35)
        state.add_error("rejected_decoys missing")
        return

    matched = False
    target = gold["required_decoy"]
    for item in decoys:
        if not isinstance(item, dict):
            continue
        if item.get("file") == target["file"] and item.get("symbol") == target["symbol"]:
            reason = str(item.get("reason", ""))
            if lower_contains_any(reason, tuple(target["reason_keywords"])) or lower_contains_any(
                reason, ("legacy", "unused", "not live", "not imported", "not referenced")
            ):
                matched = True
                break
    if matched:
        state.add("hidden.decoy_rejection", 5)
    else:
        state.add_error("required decoy was not rejected with a concrete reason")

    forbidden_positive_files = set(gold.get("forbidden_positive_files", []))
    live_files = {
        step.get("file")
        for step in state.path_map.get("live_path", [])
        if isinstance(step, dict) and isinstance(step.get("file"), str)
    }
    if forbidden_positive_files & live_files:
        variant = VARIANT_ID
        if variant == "v3-dirty-state":
            state.apply_ceiling("dirty_state_trust", 40)
        elif variant == "v4-multi-corpus-objective":
            state.apply_ceiling("future_architecture_anchor", 45)
        elif variant == "v5-recovery-in-thread":
            state.apply_ceiling("incident_blind_reselect", 30)


def score_written_artifacts(state: ScorerState, gold: dict[str, Any]) -> None:
    required_brief_tokens = set(gold.get("required_brief_tokens", []))
    required_correction_tokens = set(gold.get("required_correction_tokens", []))

    brief_hits = len(required_brief_tokens & state.brief_tokens)
    correction_hits = len(required_correction_tokens & state.correction_tokens)
    state.add("hidden.brief_grounding", min(brief_hits, len(required_brief_tokens)))
    state.add("hidden.correction_grounding", min(correction_hits, len(required_correction_tokens)))
    if brief_hits < 3 or correction_hits < 3:
        state.apply_ceiling("weak_markdown_grounding", 45)

    brief_lower = state.brief_text.lower()
    if not (
        "support note" in brief_lower
        and (
            "incorrect" in brief_lower
            or "wrong" in brief_lower
            or "stale" in brief_lower
            or "not correct" in brief_lower
        )
    ):
        state.add_error("brief does not explicitly reject the stale support note")
        state.apply_ceiling("no_support_note_verdict", 35)
    if "scenario_families/" in state.brief_text or "scenario_families/" in state.correction_text:
        state.apply_ceiling("external_evidence", 20)
    if lower_contains_any(state.brief_text + "\n" + state.correction_text, ("https://", "http://", "/Users/")):
        state.apply_ceiling("external_evidence", 20)

    if VARIANT_ID == "v5-recovery-in-thread":
        if lower_contains_any(state.correction_text, ("rollback", "rolled back", "incident")):
            state.add("hidden.v5_incident_ack", 5)
        else:
            state.apply_ceiling("incident_blind_reselect", 30)
            state.add_error("v5 correction did not acknowledge rollback or incident context")


def finalize_milestones(state: ScorerState, gold: dict[str, Any]) -> dict[str, Any]:
    state.milestones.setdefault("M1_localization", False)
    state.milestones.setdefault("M2_primary_fix", False)
    state.milestones["M3_invariants"] = state.integrity_flag == 0
    state.milestones.setdefault("M4_functional", False)
    state.milestones["M5_e2e"] = (
        state.integrity_flag == 0
        and state.final_score() >= int(gold.get("pass_bar", 40))
        and state.path_map is not None
        and bool(state.brief_text)
        and bool(state.correction_text)
    )

    slots = [
        ("M1_localization", 0.10),
        ("M2_primary_fix", 0.20),
        ("M3_invariants", 0.20),
        ("M4_functional", 0.20),
        ("M5_e2e", 0.30),
    ]
    aggregate = 0.0
    slot_rows: list[dict[str, Any]] = []
    for name, weight in slots:
        passed = bool(state.milestones.get(name, False))
        if passed:
            aggregate += weight
        slot_rows.append({"slot": name, "weight": weight, "passed_bool": passed})
    return {"slots": slot_rows, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    start = time.time()
    gold = load_gold()
    manifest = load_manifest()
    state = ScorerState()

    trusted_final_state(state, gold, manifest)
    read_outputs(state)
    score_phase2(state)
    score_live_path(state, gold)
    score_field_derivations(state, gold)
    score_test_observations(state, gold)
    score_decoy_and_variant_rules(state, gold)
    score_written_artifacts(state, gold)

    milestone_vector = finalize_milestones(state, gold)
    result = {
        "schema_version": VERIFY_RESULT_SCHEMA,
        "family_id": "request-path-evidence-brief",
        "variant_id": VARIANT_ID,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "raw_score_pre_ceiling": state.raw_score,
        "M_training": state.final_m(),
        "pass": state.final_score() >= int(gold.get("pass_bar", 40)) and state.integrity_flag == 0,
        "shortcut_detected": state.shortcut,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "ceilings_applied": state.ceilings_applied,
        "errors": state.errors,
        "breakdown": state.breakdown,
        "breakdown_bands": state.breakdown_bands,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "observed_live_path_exact_matches": state.live_path_exact_matches,
        "elapsed_sec": round(time.time() - start, 4),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RESULT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
