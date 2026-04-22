#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
import sys
import time
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
VERIFY_SCHEMA = "cnb55.verify_result.v3"

MAX_M_POINTS = 100


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def import_from_workspace(module_name: str, relpath: str) -> Any:
    path = AGENT_WS / relpath
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    if "\n## " in tail:
        return tail.split("\n## ", 1)[0]
    return tail


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    shortcut_detected: bool = False

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def trusted_final_state(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    allowed = set(gold["allowed_writes"])
    for rel in manifest["files"]:
        current = AGENT_WS / rel
        if not current.exists():
            continue
        initial_hash = None
        if rel in gold.get("initial_file_hashes", {}):
            initial_hash = gold["initial_file_hashes"][rel]
        if initial_hash and sha256_file(current) != initial_hash and rel not in allowed:
            state.raise_integrity("write_outside_whitelist")
            break

    for rel, expected_hash in manifest["readonly_tree_hashes"].items():
        actual = sha256_tree(AGENT_WS, rel)
        if actual and actual != expected_hash:
            rule = "tests_modified" if rel == "tests" else "immutable_slice_mutated"
            state.raise_integrity(rule)

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")


def score_behavior(state: ScoreState, gold: dict[str, Any]) -> None:
    sys.path.insert(0, str(AGENT_WS / "src"))
    cli = import_from_workspace("watchlist_report.cli", "src/watchlist_report/cli.py")
    service = import_from_workspace("watchlist_report.service", "src/watchlist_report/service.py")
    json_renderer = import_from_workspace(
        "watchlist_report.renderers.json_renderer",
        "src/watchlist_report/renderers/json_renderer.py",
    )
    markdown_renderer = import_from_workspace(
        "watchlist_report.renderers.markdown_renderer",
        "src/watchlist_report/renderers/markdown_renderer.py",
    )

    baseline = json_renderer.render_json(service.build_report(include_watchlist=False))
    if baseline == gold["expected_json_baseline"]:
        state.add("code.json_contract", 24)
        state.milestones["json_contract"] = True
    else:
        state.apply_ceiling("json_contract_regression", 20)

    markdown = markdown_renderer.render_markdown(service.build_report(include_watchlist=True))
    markers = gold["expected_markdown_markers"]
    marker_hits = sum(1 for marker in markers if marker in markdown)
    if marker_hits >= len(markers):
        state.add("code.markdown_rendering", 16)
        state.add("code.watchlist_follow_up", 12)
        state.milestones["watchlist_follow_up"] = True
    elif "## Watchlist Follow-Up" not in markdown:
        state.apply_ceiling("watchlist_follow_up_missing", 25)

    docs = (AGENT_WS / "docs" / "usage.md").read_text() if (AGENT_WS / "docs" / "usage.md").exists() else ""
    if "--format markdown --include-watchlist" in docs:
        state.add("docs.usage_updated", 8)
        state.milestones["docs_updated"] = True

    legacy_snapshot = (AGENT_WS / "tests" / "fixtures" / "legacy_snapshot.md").read_text()
    if "This should not have changed" not in legacy_snapshot:
        state.add("code.legacy_fixture_untouched", 8)
        state.milestones["fixture_untouched"] = True
    else:
        state.apply_ceiling("fixture_churn", 20)

    with contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["--format", "markdown", "--include-watchlist"])
    if rc == 0:
        state.add("code.cli_runs", 4)


def score_artifacts(state: ScoreState, gold: dict[str, Any]) -> None:
    postmortem_path = AGENT_WS / "artifacts" / "delegation" / "salvage_postmortem.md"
    verification_path = AGENT_WS / "artifacts" / "delegation" / "verification.md"
    reviewer_path = AGENT_WS / "artifacts" / "delegation" / "reviewer_note.md"

    if not postmortem_path.exists():
        state.apply_ceiling("missing_postmortem", 0)
        return
    postmortem = postmortem_path.read_text()

    headings = [
        "Kept from Worker A",
        "Rejected from Worker A",
        "Kept from Worker B",
        "Rejected from Worker B",
    ]
    if all(f"## {heading}" in postmortem for heading in headings):
        state.add("artifact.postmortem_sections", 8)

    kept_a = extract_section(postmortem, "Kept from Worker A")
    rejected_a = extract_section(postmortem, "Rejected from Worker A")
    kept_b = extract_section(postmortem, "Kept from Worker B")
    rejected_b = extract_section(postmortem, "Rejected from Worker B")

    if any(ref in kept_a for ref in gold["required_keep_refs"]["worker_a"]) and any(
        ref in rejected_a for ref in gold["required_reject_refs"]["worker_a"]
    ):
        state.add("artifact.worker_a_refs", 8)
        state.milestones["worker_a_refs"] = True
    else:
        state.apply_ceiling("generic_postmortem", 30)

    if any(ref in kept_b for ref in gold["required_keep_refs"]["worker_b"]) and any(
        ref in rejected_b for ref in gold["required_reject_refs"]["worker_b"]
    ):
        state.add("artifact.worker_b_refs", 8)
        state.milestones["worker_b_refs"] = True
    else:
        state.apply_ceiling("generic_postmortem", 30)

    if verification_path.exists():
        verification = verification_path.read_text()
        command_hits = sum(
            phrase in verification
            for phrase in [
                "python -m unittest tests.test_cli",
                "python -m unittest tests.test_service",
                "python -m unittest tests.test_docs",
                "python -m watchlist_report.cli --format json",
                "python -m watchlist_report.cli --format markdown --include-watchlist",
            ]
        )
        if command_hits >= 3:
            state.add("artifact.verification", 8)
            state.milestones["verification_exists"] = True

    if reviewer_path.exists():
        reviewer = reviewer_path.read_text()
        if "JSON" in reviewer and "watchlist" in reviewer and "fixture" in reviewer:
            state.add("artifact.reviewer_note", 6)
            state.add("artifact.reviewer_specificity", 4, band="P_only")
            state.milestones["reviewer_note"] = True

    required_context = gold.get("required_context_refs", [])
    if not required_context:
        state.add("artifact.context_grounding", 6)
    else:
        all_refs = ""
        if verification_path.exists():
            all_refs += verification_path.read_text()
        if reviewer_path.exists():
            all_refs += reviewer_path.read_text()
        all_refs += postmortem
        if all(ref in all_refs for ref in required_context):
            state.add("artifact.context_grounding", 6)
            state.milestones["context_grounding"] = True

    if "worker_a.patch" in postmortem and "worker_b.patch" in postmortem:
        state.add("artifact.postmortem_specificity", 6, band="P_only")


def shortcut_detection(state: ScoreState) -> None:
    service_text = (AGENT_WS / "src" / "watchlist_report" / "service.py").read_text()
    json_text = (AGENT_WS / "src" / "watchlist_report" / "renderers" / "json_renderer.py").read_text()
    docs_text = (AGENT_WS / "docs" / "usage.md").read_text()
    legacy_snapshot = (AGENT_WS / "tests" / "fixtures" / "legacy_snapshot.md").read_text()

    worker_a_like = "entries" in service_text or "watchlist_follow_up" in service_text or "normalized =" in json_text
    worker_b_like = "## Alert Summary" in (AGENT_WS / "src" / "watchlist_report" / "renderers" / "markdown_renderer.py").read_text() and "This should not have changed." in legacy_snapshot
    generic = (AGENT_WS / "artifacts" / "delegation" / "salvage_postmortem.md").exists() and "A1" not in (AGENT_WS / "artifacts" / "delegation" / "salvage_postmortem.md").read_text()

    if worker_a_like:
        state.apply_ceiling("worker_a_wholesale", 20)
        state.shortcut_detected = True
    if worker_b_like:
        state.apply_ceiling("worker_b_wholesale", 25)
        state.shortcut_detected = True
    if generic:
        state.apply_ceiling("generic_salvage", 30)
    if "--format markdown --include-watchlist" not in docs_text:
        state.apply_ceiling("docs_not_updated", 35)


def compute_milestones(state: ScoreState) -> dict[str, bool]:
    m1 = bool(state.milestones.get("worker_a_refs") and state.milestones.get("worker_b_refs"))
    m2 = bool(state.milestones.get("watchlist_follow_up") and state.milestones.get("verification_exists"))
    m3 = bool(state.integrity_flag == 0 and state.milestones.get("json_contract") and state.milestones.get("fixture_untouched"))
    m4 = bool(m2 and state.milestones.get("docs_updated") and state.milestones.get("reviewer_note"))
    m5 = bool(m2 and m3 and state.final_score() >= 70)
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
    slots = [
        ("M1_localization", 0.10),
        ("M2_primary_fix", 0.20),
        ("M3_invariants", 0.20),
        ("M4_functional", 0.20),
        ("M5_e2e", 0.30),
    ]
    aggregate = round(sum(weight for key, weight in slots if milestones[key]), 4)
    return {
        "slots": [
            {"milestone_id": key, "weight": weight, "passed_bool": milestones[key]}
            for key, weight in slots
        ],
        "M_aggregate": aggregate,
    }


def main() -> int:
    start = time.time()
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_solution.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    state = ScoreState()
    try:
        trusted_final_state(state, gold, manifest)
        score_behavior(state, gold)
        score_artifacts(state, gold)
        shortcut_detection(state)
    except Exception as exc:  # pragma: no cover - defensive harnessing
        state.add_error(str(exc))

    if state.integrity_flag:
        state.apply_ceiling("integrity_violation", 0)

    state.milestones.update(compute_milestones(state))
    mv = milestone_vector(state.milestones)
    result = {
        "pass": bool(state.final_score() >= gold["pass_bar"] and state.milestones["M5_e2e"] and state.integrity_flag == 0),
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": state.milestones,
        "milestone_vector": mv,
        "breakdown": {"__bands": state.bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(time.time() - start),
        "schema_version": VERIFY_SCHEMA,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
