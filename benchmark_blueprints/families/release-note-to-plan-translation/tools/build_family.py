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

REPO = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "release-note-to-plan-translation"
VERIFIER_ROOT = REPO / "verifiers" / "release-note-to-plan-translation"
VERIFIER_DATA_ROOT = REPO / "verifier_data" / "release-note-to-plan-translation"
CLI_ROOT = FAMILY_ROOT / "bin" / "cnb55-brief"
SCORER = VERIFIER_ROOT / "score_ranking.py"

BRIEF_SCHEMA_VERSION = "cnb55.release_plan_brief.v1"

TRACK = 10
TRACK_NAME = "Strategic Management & Long-Horizon Evolution"
DOCKER_BASE = "python:3.12-bookworm@sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3"


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
    for p in sorted(target.rglob("*")):
        rel_p = p.relative_to(target).as_posix()
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def write_text(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


CLI_TEXT = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cnb55.release_plan_brief.v1"
ASSUMPTION_STATUSES = ("observed", "to_verify", "missing")
EVIDENCE_ROOTS = ("release_notes", "repo_inventory", "release_context", "incident_context")


def schema_object() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CNB-55 Release Plan Brief Input",
        "type": "object",
        "required": [
            "schema_version",
            "variant_id",
            "first_milestone_id",
            "ordered_steps",
            "dependency_notes",
            "primary_risk",
            "assumption_ledger",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "variant_id": {"type": "string"},
            "first_milestone_id": {"type": "string", "minLength": 2},
            "ordered_steps": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "required": [
                        "step_id",
                        "rank",
                        "title",
                        "summary",
                        "bounded_deliverable",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "dependency_notes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["before", "after", "reason"],
                    "additionalProperties": False,
                },
            },
            "primary_risk": {
                "type": "object",
                "required": ["statement", "evidence", "mitigations"],
                "additionalProperties": False,
            },
            "assumption_ledger": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["topic", "status", "note"],
                    "additionalProperties": False,
                },
            },
        },
    }


class ValidationError(Exception):
    def __init__(self, code: int, messages: list[str]):
        super().__init__("\\n".join(messages))
        self.code = code
        self.messages = messages


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(2, [f"input file missing: {path}"]) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(2, [f"input file is not valid JSON: {exc}"]) from exc


def validate_document(doc: Any, workspace: Path) -> None:
    errors: list[str] = []
    if not isinstance(doc, dict):
        raise ValidationError(3, ["top-level value must be an object"])
    required = {
        "schema_version",
        "variant_id",
        "first_milestone_id",
        "ordered_steps",
        "dependency_notes",
        "primary_risk",
        "assumption_ledger",
    }
    for key in sorted(required - set(doc)):
        errors.append(f"missing required field: {key}")
    for key in sorted(set(doc) - required):
        errors.append(f"unexpected field: {key}")
    if errors:
        raise ValidationError(3, errors)
    if doc["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    variant_marker = (workspace / ".scenario_variant").read_text().strip()
    if doc["variant_id"] != variant_marker:
        errors.append(f"variant_id must match .scenario_variant ({variant_marker})")
    steps = doc["ordered_steps"]
    if not isinstance(steps, list) or len(steps) < 3:
        errors.append("ordered_steps must be an array of at least 3 entries")
        steps = []
    ranks = []
    step_ids = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"ordered_steps[{idx}] must be an object")
            continue
        needed = {"step_id", "rank", "title", "summary", "bounded_deliverable", "evidence"}
        for key in sorted(needed - set(step)):
            errors.append(f"ordered_steps[{idx}].{key} is required")
        for key in sorted(set(step) - needed):
            errors.append(f"ordered_steps[{idx}].{key} is not allowed")
        step_id = step.get("step_id")
        rank = step.get("rank")
        evidence = step.get("evidence", [])
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"ordered_steps[{idx}].step_id must be a non-empty string")
        else:
            step_ids.append(step_id)
        if not isinstance(rank, int):
            errors.append(f"ordered_steps[{idx}].rank must be an int")
        else:
            ranks.append(rank)
        for field in ("title", "summary", "bounded_deliverable"):
            if not isinstance(step.get(field), str) or len(step[field].strip()) < 8:
                errors.append(f"ordered_steps[{idx}].{field} must be a string >= 8 chars")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"ordered_steps[{idx}].evidence must be a non-empty array")
        else:
            for rel in evidence:
                if not isinstance(rel, str):
                    errors.append(f"ordered_steps[{idx}].evidence entries must be strings")
                    continue
                if not any(rel == root or rel.startswith(root + "/") for root in EVIDENCE_ROOTS):
                    errors.append(f"evidence path outside allowed roots: {rel}")
                if not (workspace / rel).exists():
                    errors.append(f"evidence path does not exist: {rel}")
    if sorted(ranks) != list(range(1, len(steps) + 1)):
        errors.append("ordered_steps ranks must be contiguous 1..N")
    if len(set(step_ids)) != len(step_ids):
        errors.append("ordered_steps step_id values must be unique")
    if steps and doc["first_milestone_id"] != steps[0].get("step_id"):
        errors.append("first_milestone_id must match ordered_steps[0].step_id")
    dependency_notes = doc["dependency_notes"]
    if not isinstance(dependency_notes, list) or not dependency_notes:
        errors.append("dependency_notes must be a non-empty array")
    else:
        valid_ids = set(step_ids)
        for idx, note in enumerate(dependency_notes):
            if not isinstance(note, dict):
                errors.append(f"dependency_notes[{idx}] must be an object")
                continue
            if note.get("before") not in valid_ids:
                errors.append(f"dependency_notes[{idx}].before must reference a known step_id")
            if note.get("after") not in valid_ids:
                errors.append(f"dependency_notes[{idx}].after must reference a known step_id")
            if not isinstance(note.get("reason"), str) or len(note["reason"].strip()) < 8:
                errors.append(f"dependency_notes[{idx}].reason must be a string >= 8 chars")
    primary_risk = doc["primary_risk"]
    if not isinstance(primary_risk, dict):
        errors.append("primary_risk must be an object")
    else:
        if not isinstance(primary_risk.get("statement"), str) or len(primary_risk["statement"].strip()) < 20:
            errors.append("primary_risk.statement must be a string >= 20 chars")
        evidence = primary_risk.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("primary_risk.evidence must be a non-empty array")
        else:
            for rel in evidence:
                if not isinstance(rel, str) or not (workspace / rel).exists():
                    errors.append(f"primary_risk evidence path missing: {rel}")
        mitigations = primary_risk.get("mitigations")
        if not isinstance(mitigations, list) or len(mitigations) < 2:
            errors.append("primary_risk.mitigations must have at least 2 items")
    assumption_ledger = doc["assumption_ledger"]
    if not isinstance(assumption_ledger, list) or not assumption_ledger:
        errors.append("assumption_ledger must be a non-empty array")
    else:
        if not any(isinstance(row, dict) and row.get("status") == "missing" for row in assumption_ledger):
            errors.append("assumption_ledger must include at least one status='missing' row")
        for idx, row in enumerate(assumption_ledger):
            if not isinstance(row, dict):
                errors.append(f"assumption_ledger[{idx}] must be an object")
                continue
            if row.get("status") not in ASSUMPTION_STATUSES:
                errors.append(f"assumption_ledger[{idx}].status invalid: {row.get('status')!r}")
            for field in ("topic", "note"):
                if not isinstance(row.get(field), str) or len(row[field].strip()) < 3:
                    errors.append(f"assumption_ledger[{idx}].{field} must be a string >= 3 chars")
    if errors:
        raise ValidationError(3, errors)


def render_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Release Plan Brief",
        "",
        f"- Variant: `{doc['variant_id']}`",
        f"- First milestone: `{doc['first_milestone_id']}`",
        "",
        "## Ordered Plan",
        "",
    ]
    for step in sorted(doc["ordered_steps"], key=lambda item: item["rank"]):
        lines.extend(
            [
                f"### {step['rank']}. {step['step_id']} — {step['title']}",
                "",
                step["summary"],
                "",
                f"Bounded deliverable: {step['bounded_deliverable']}",
                "",
                "Evidence:",
            ]
        )
        for rel in step["evidence"]:
            lines.append(f"- `{rel}`")
        lines.append("")
    lines.extend(["## Dependency Notes", ""])
    for note in doc["dependency_notes"]:
        lines.append(f"- `{note['before']}` before `{note['after']}`: {note['reason']}")
    lines.extend(["", "## Primary Risk", "", doc["primary_risk"]["statement"], "", "Evidence:"])
    for rel in doc["primary_risk"]["evidence"]:
        lines.append(f"- `{rel}`")
    lines.extend(["", "Mitigations:"])
    for item in doc["primary_risk"]["mitigations"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Assumption Ledger", ""])
    for row in doc["assumption_ledger"]:
        lines.append(f"- {row['topic']} [{row['status']}]: {row['note']}")
    lines.append("")
    return "\\n".join(lines)


def cmd_schema() -> int:
    json.dump(schema_object(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\\n")
    return 0


def cmd_validate(file_path: Path) -> int:
    workspace = Path.cwd()
    doc = _load_json(file_path)
    validate_document(doc, workspace)
    return 0


def cmd_submit(file_path: Path) -> int:
    workspace = Path.cwd()
    doc = _load_json(file_path)
    validate_document(doc, workspace)
    out_dir = workspace / "brief"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manager_brief.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
    (out_dir / "manager_brief.md").write_text(render_markdown(doc) + "\\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    for name in ("validate", "submit"):
        p = sub.add_parser(name)
        p.add_argument("file")
    args = parser.parse_args()
    try:
        if args.cmd == "schema":
            return cmd_schema()
        if args.cmd == "validate":
            return cmd_validate(Path(args.file))
        return cmd_submit(Path(args.file))
    except ValidationError as exc:
        for msg in exc.messages:
            print(msg, file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
"""

AGENTS_TEXT = """# Agent Instructions — `release-note-to-plan-translation`

## Task

Translate frozen release notes plus repo-state evidence into an ordered
implementation plan. The plan must identify the smallest meaningful first
milestone, the prerequisite dependency chain, and the main user-visible risk
if the work is sequenced incorrectly.

## Inputs

- `release_notes/` — the release-note bullets that must be translated into a
  concrete execution order. Read dates and status notes carefully; some notes
  are stale or superseded.
- `repo_inventory/` — current repo-state evidence: dependency notes, test
  status, partial patches, and rollout constraints. This is the source of
  hidden coupling and user-visible risk.
- `release_context/` — optional. If present, read it before deciding; the
  operating objective has shifted since the original notes were written.
- `incident_context/` — optional. If present, read it before deciding; a prior
  plan order was rolled back and should change what you pick first.
- `.scenario_variant` — the variant id string. Use it in your brief.
- `tests/test_plan_brief.py` — visible checks. Do not modify it.

## Output — use the CLI

Use `./bin/cnb55-brief` to validate and submit a structured JSON brief.
Do not hand-write `brief/manager_brief.md`.

1. Write `brief_input.json` at the workspace root.
2. Inspect the schema if needed:
   `./bin/cnb55-brief schema`
3. Validate:
   `./bin/cnb55-brief validate brief_input.json`
4. Submit:
   `./bin/cnb55-brief submit brief_input.json`

The canonical schema version is `cnb55.release_plan_brief.v1`.

Your brief must contain:
- `first_milestone_id`
- `ordered_steps` with contiguous ranks
- `dependency_notes` tying prerequisite work to downstream steps
- `primary_risk` naming the user-visible failure mode if the order is wrong
- `assumption_ledger` with at least one honest `missing` row

## What strong planning looks like

- The first milestone is bounded: it should reduce ambiguity or de-risk the
  path, not attempt the full launch.
- Dependencies are explicit. If dashboard or rollout work depends on schema,
  fixtures, or guardrails, that relation should appear both in ordering and in
  `dependency_notes`.
- Risk is user-visible. Tie it to the evidence in `repo_inventory/`, not to an
  abstract engineering concern.
- If a note is stale, a draft is abandoned, the objective changed, or an
  incident invalidated a prior order, say so in the plan rather than anchoring
  on the old path.

## Rules

- Do not modify `release_notes/`, `repo_inventory/`, `release_context/`,
  `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  or `bin/`.
- Do not write files outside `brief/`. The only extra root file allowed is
  `brief_input.json`, which the CLI reads.
- Do not fetch network resources.
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.

Any of those trips the integrity detector and fails the run.
"""

DOCKERFILE_TEXT = f"""# syntax=docker/dockerfile:1.6
FROM {DOCKER_BASE}

RUN pip install --no-cache-dir "pytest==8.3.3"

WORKDIR /workspace
COPY . /workspace
RUN test -f /workspace/.scenario_variant && test -d /workspace/release_notes

CMD ["bash"]
"""

TEST_TEXT = """from __future__ import annotations

import json
import pathlib

import pytest

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]
BRIEF_JSON = WS / "brief" / "manager_brief.json"
BRIEF_MD = WS / "brief" / "manager_brief.md"
VARIANT_MARKER = WS / ".scenario_variant"


def _load() -> dict:
    if not BRIEF_JSON.exists():
        pytest.fail(f"brief json missing at {BRIEF_JSON}")
    try:
        return json.loads(BRIEF_JSON.read_text())
    except json.JSONDecodeError as exc:
        pytest.fail(f"brief json invalid: {exc}")


def test_brief_files_exist():
    assert BRIEF_JSON.exists()
    assert BRIEF_MD.exists()
    assert BRIEF_MD.read_text().strip()


def test_variant_matches_marker():
    data = _load()
    assert data["variant_id"] == VARIANT_MARKER.read_text().strip()


def test_first_milestone_matches_rank_1():
    data = _load()
    ordered = data["ordered_steps"]
    assert ordered[0]["step_id"] == data["first_milestone_id"]


def test_ordered_steps_are_contiguous():
    data = _load()
    ordered = data["ordered_steps"]
    assert len(ordered) >= 3
    ranks = sorted(step["rank"] for step in ordered)
    assert ranks == list(range(1, len(ordered) + 1))
    assert len({step["step_id"] for step in ordered}) == len(ordered)


def test_dependency_notes_reference_known_steps():
    data = _load()
    step_ids = {step["step_id"] for step in data["ordered_steps"]}
    assert data["dependency_notes"]
    for note in data["dependency_notes"]:
        assert note["before"] in step_ids
        assert note["after"] in step_ids
        assert note["reason"].strip()


def test_primary_risk_and_assumptions_present():
    data = _load()
    primary_risk = data["primary_risk"]
    assert primary_risk["statement"].strip()
    assert len(primary_risk["evidence"]) >= 1
    assert len(primary_risk["mitigations"]) >= 2
    assert any(row["status"] == "missing" for row in data["assumption_ledger"])
"""

HIDDEN_TEST_TEXT = """from __future__ import annotations

import json
from pathlib import Path


def test_oracle_and_gold_align():
    root = Path(__file__).resolve().parents[1]
    gold = json.loads((root / "gold_ranking.json").read_text())
    oracle = json.loads((root / "oracle" / "manager_brief.json").read_text())
    assert oracle["first_milestone_id"] == gold["first_milestone_id"]
    assert [step["step_id"] for step in oracle["ordered_steps"]] == gold["gold_order"]
"""

MILESTONE_SCRIPT_TEXTS = {
    "m1_localize.sh": """#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {}).get("M1_localization", False) else 1)
PY
""",
    "m2_primary_fix.sh": """#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {}).get("M2_primary_fix", False) else 1)
PY
""",
    "m3_invariants.sh": """#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M3_invariants", False))
sys.exit(0 if ok else 1)
PY
""",
    "m4_functional.sh": """#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {}).get("M4_functional", False) else 1)
PY
""",
    "m5_e2e.sh": """#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {}).get("M5_e2e", False) else 1)
PY
""",
}


def variant_specs() -> list[dict]:
    base_steps = [
        ("RN-101", "Audit translator schema drift", "Freeze the step-id schema and map the hidden dependency edges before any downstream rollout work."),
        ("RN-102", "Backfill dependency-graph fixtures", "Extend plan-contract fixtures so the new dependency edges fail loudly before release."),
        ("RN-103", "Enable gated dashboard summary rollout", "Turn on the translated release-plan summary behind a kill switch once prerequisites are stable."),
        ("RN-104", "Update operator runbook and launch checklist", "Document the new summary workflow and the rollback path once the rollout path is known."),
    ]
    return [
        {
            "variant_id": "v1-clean-baseline",
            "steps": base_steps,
            "release_notes": {
                "release_notes/release_notes_2026_04.md": """# Frozen release notes

- RN-101 Audit translator schema drift before removing the legacy parser shim.
- RN-102 Backfill dependency-graph fixtures so the plan contract catches ordering regressions.
- RN-103 Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.
- RN-104 Update the operator runbook and launch checklist after the summary path is proven in canary.
""",
            },
            "repo_inventory": {
                "repo_inventory/dependency_map.md": """# Dependency map

- The dashboard summary consumes the step-id schema produced by the translator.
- The dashboard summary must not ship until dependency-graph fixtures reflect the new schema.
- The runbook only stabilizes after the rollout and rollback shape are known.
""",
                "repo_inventory/test_inventory.md": """# Test inventory

- `tests/plan_contract.py` currently fails when new step ids are introduced without fixture updates.
- Canary-only coverage exists for the dashboard summary, but not for bad dependency ordering.
""",
                "repo_inventory/repo_state.md": """# Repo state

- Legacy parser shim still masks schema drift in local smoke runs.
- The release dashboard is user-visible and will render whatever order the translator emits.
""",
            },
            "release_context": {},
            "incident_context": {},
            "gold": {
                "first_milestone_id": "RN-101",
                "gold_order": ["RN-101", "RN-102", "RN-103", "RN-104"],
                "required_dependency_pairs": [
                    {"before": "RN-101", "after": "RN-103"},
                    {"before": "RN-102", "after": "RN-103"},
                    {"before": "RN-103", "after": "RN-104"},
                ],
                "large_first_milestone_ids": ["RN-103", "RN-104"],
                "required_risk_term_groups": [["dashboard", "summary"], ["dependency", "ordering", "order"], ["user", "customer", "operator"]],
                "user_visible_risk_term_groups": [["dashboard", "summary"], ["incorrect", "wrong", "misordered"]],
                "objective_term_groups": [],
                "incident_term_groups": [],
                "ceiling_rules": [],
                "tau_threshold": 0.8,
                "min_unique_evidence": 4,
                "pass_bar": 40,
                "matrix_pick_ceiling_id": "RN-103",
                "shortcut_first_milestone_id": "RN-103",
            },
        },
        {
            "variant_id": "v2-noisy-distractor",
            "steps": base_steps + [
                ("RN-105", "Revive deprecated cache experiment", "An older perf note suggests a cache path, but the note predates the current translator flow and should not lead the plan."),
            ],
            "release_notes": {
                "release_notes/release_notes_2026_04.md": """# Frozen release notes

- RN-101 Audit translator schema drift before removing the legacy parser shim.
- RN-102 Backfill dependency-graph fixtures so the plan contract catches ordering regressions.
- RN-103 Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.
- RN-104 Update the operator runbook and launch checklist after the summary path is proven in canary.
- RN-105 Revisit the January cache experiment for possible quick wins.
""",
                "release_notes/stale_note_jan_2026.md": """# January 2026 perf note

The cache experiment predates the translator schema rewrite and was recorded before the dashboard summary existed.
Treat it as stale context, not as the current primary path.
""",
            },
            "repo_inventory": {
                "repo_inventory/dependency_map.md": """# Dependency map

- Schema audit and fixture backfill still gate the dashboard rollout.
- The cache experiment does not address the current release dashboard ordering bug.
""",
                "repo_inventory/test_inventory.md": """# Test inventory

- `tests/plan_contract.py` still fails on step-id drift.
- No test currently asserts that stale release-note experiments are ranked last.
""",
                "repo_inventory/repo_state.md": """# Repo state

- The user-visible dashboard summary is still the current release objective.
- The January cache experiment was superseded by the translator rewrite.
""",
            },
            "release_context": {},
            "incident_context": {},
            "gold": {
                "first_milestone_id": "RN-101",
                "gold_order": ["RN-101", "RN-102", "RN-103", "RN-104", "RN-105"],
                "required_dependency_pairs": [
                    {"before": "RN-101", "after": "RN-103"},
                    {"before": "RN-102", "after": "RN-103"},
                    {"before": "RN-103", "after": "RN-104"},
                ],
                "large_first_milestone_ids": ["RN-103", "RN-104"],
                "required_risk_term_groups": [["dashboard", "summary"], ["dependency", "ordering", "order"], ["stale", "superseded"]],
                "user_visible_risk_term_groups": [["dashboard", "summary"], ["incorrect", "wrong", "misordered"]],
                "objective_term_groups": [],
                "incident_term_groups": [],
                "ceiling_rules": [
                    {"id": "ignored_stale_release_note", "cap": 30, "bad_first_milestone_ids": ["RN-105"]},
                ],
                "tau_threshold": 0.8,
                "min_unique_evidence": 4,
                "pass_bar": 40,
                "matrix_pick_ceiling_id": "RN-105",
                "shortcut_first_milestone_id": "RN-105",
            },
        },
        {
            "variant_id": "v3-dirty-state",
            "steps": base_steps + [
                ("RN-105", "Salvage abandoned rollout draft", "There is an in-progress patch from a departed engineer, but it is a sunk-cost trap unless the prerequisites are already settled."),
            ],
            "release_notes": {
                "release_notes/release_notes_2026_04.md": """# Frozen release notes

- RN-101 Audit translator schema drift before removing the legacy parser shim.
- RN-102 Backfill dependency-graph fixtures so the plan contract catches ordering regressions.
- RN-103 Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.
- RN-104 Update the operator runbook and launch checklist after the summary path is proven in canary.
- RN-105 Review whether the abandoned rollout draft is even worth finishing.
""",
            },
            "repo_inventory": {
                "repo_inventory/dependency_map.md": """# Dependency map

- The abandoned rollout draft depends on the same unresolved schema and fixture work as RN-103.
- Treat it as optional cleanup after the safe path is clear.
""",
                "repo_inventory/test_inventory.md": """# Test inventory

- The regression suite still does not protect against hidden dependency reordering.
- The abandoned draft never added coverage for rollback behavior.
""",
                "repo_inventory/repo_state.md": """# Repo state

- The release dashboard is still user-visible.
- A half-finished rollout draft exists under `repo_inventory/in_progress_patch/`.
""",
                "repo_inventory/in_progress_patch/README.md": """# In-progress patch

Started by an engineer who has since transferred teams.
The patch assumes the translator schema is already frozen, which is false today.
""",
            },
            "release_context": {},
            "incident_context": {},
            "gold": {
                "first_milestone_id": "RN-101",
                "gold_order": ["RN-101", "RN-102", "RN-103", "RN-104", "RN-105"],
                "required_dependency_pairs": [
                    {"before": "RN-101", "after": "RN-103"},
                    {"before": "RN-102", "after": "RN-103"},
                    {"before": "RN-103", "after": "RN-104"},
                ],
                "large_first_milestone_ids": ["RN-103", "RN-104"],
                "required_risk_term_groups": [["dashboard", "summary"], ["dependency", "ordering", "order"], ["abandoned", "sunk", "draft"]],
                "user_visible_risk_term_groups": [["dashboard", "summary"], ["incorrect", "wrong", "misordered"]],
                "objective_term_groups": [],
                "incident_term_groups": [],
                "ceiling_rules": [
                    {"id": "sunk_cost_finish", "cap": 30, "bad_first_milestone_ids": ["RN-105"]},
                ],
                "tau_threshold": 0.8,
                "min_unique_evidence": 4,
                "pass_bar": 40,
                "matrix_pick_ceiling_id": "RN-105",
                "shortcut_first_milestone_id": "RN-105",
            },
        },
        {
            "variant_id": "v4-multi-corpus-objective",
            "steps": base_steps + [
                ("RN-105", "Re-sequence change window communication", "The current release objective is safe enterprise rollout windows, not raw dashboard velocity."),
            ],
            "release_notes": {
                "release_notes/release_notes_2026_04.md": """# Frozen release notes

- RN-101 Audit translator schema drift before removing the legacy parser shim.
- RN-102 Backfill dependency-graph fixtures so the plan contract catches ordering regressions.
- RN-103 Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.
- RN-104 Update the operator runbook and launch checklist after the summary path is proven in canary.
- RN-105 Send the enterprise change-window update once the safe plan order is locked.
""",
            },
            "repo_inventory": {
                "repo_inventory/dependency_map.md": """# Dependency map

- Enterprise change-window communication is blocked on correct ordering guarantees.
- Dashboard rollout is no longer the only objective; safety and rollback clarity now dominate.
""",
                "repo_inventory/test_inventory.md": """# Test inventory

- Existing tests catch schema drift but still need dependency-fixture coverage.
- There is no customer-facing safety check until the fixture backfill lands.
""",
                "repo_inventory/repo_state.md": """# Repo state

- The release dashboard remains visible to users.
- The current quarter objective changed after enterprise escalations.
""",
            },
            "release_context": {
                "release_context/current_objective.md": """# Current objective shift

Enterprise customers escalated incorrect release-plan ordering during guarded change windows.
The success metric is now safety and rollback confidence, not dashboard speed.
That makes fixture backfill and operator clarity the first-order work.
""",
            },
            "incident_context": {},
            "gold": {
                "first_milestone_id": "RN-102",
                "gold_order": ["RN-102", "RN-101", "RN-105", "RN-103", "RN-104"],
                "required_dependency_pairs": [
                    {"before": "RN-102", "after": "RN-103"},
                    {"before": "RN-101", "after": "RN-103"},
                    {"before": "RN-105", "after": "RN-103"},
                ],
                "large_first_milestone_ids": ["RN-103", "RN-104"],
                "required_risk_term_groups": [["enterprise", "change window"], ["rollback", "confidence", "guardrail"], ["dashboard", "summary"]],
                "user_visible_risk_term_groups": [["customer", "enterprise"], ["incorrect", "wrong", "misordered"]],
                "objective_term_groups": [["enterprise", "change window"], ["safety", "rollback"]],
                "incident_term_groups": [],
                "ceiling_rules": [
                    {"id": "objective_drift", "cap": 30, "bad_first_milestone_ids": ["RN-101", "RN-103"]},
                ],
                "tau_threshold": 0.8,
                "min_unique_evidence": 5,
                "pass_bar": 40,
                "matrix_pick_ceiling_id": "RN-103",
                "shortcut_first_milestone_id": "RN-103",
            },
        },
        {
            "variant_id": "v5-recovery-in-thread",
            "steps": base_steps + [
                ("RN-105", "Re-approve rollout after rollback review", "A prior attempt shipped the dashboard summary too early and was rolled back; the plan must show the safer order before re-enabling it."),
            ],
            "release_notes": {
                "release_notes/release_notes_2026_04.md": """# Frozen release notes

- RN-101 Audit translator schema drift before removing the legacy parser shim.
- RN-102 Backfill dependency-graph fixtures so the plan contract catches ordering regressions.
- RN-103 Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.
- RN-104 Update the operator runbook and launch checklist after the summary path is proven in canary.
- RN-105 Re-approve the summary rollout after the rollback review finishes.
""",
            },
            "repo_inventory": {
                "repo_inventory/dependency_map.md": """# Dependency map

- The re-approval review depends on the fixture backfill, the schema audit, and an updated rollback/runbook path.
- Re-enabling the summary before those land repeats the earlier incident.
""",
                "repo_inventory/test_inventory.md": """# Test inventory

- Dependency-fixture coverage is still missing.
- The rollback path is only described in docs and has no structured launch checklist yet.
""",
                "repo_inventory/repo_state.md": """# Repo state

- The release dashboard is still user-visible.
- A rollback incident invalidated the previous rollout-first plan.
""",
            },
            "release_context": {
                "release_context/current_objective.md": """# Current objective shift

Safe recovery from the rollback takes precedence over rollout velocity.
Do not restore the dashboard summary until dependency ordering is verified and the rollback notes are folded into the runbook.
""",
            },
            "incident_context": {
                "incident_context/rollback_incident.md": """# Rollback incident

The team enabled the dashboard summary before dependency-graph fixtures were updated.
Users saw a misordered execution plan, support escalated within the hour, and the summary was rolled back.
The follow-up requires fixture backfill first, then schema audit, then runbook/rollback instructions, then any re-enable decision.
""",
            },
            "gold": {
                "first_milestone_id": "RN-102",
                "gold_order": ["RN-102", "RN-101", "RN-104", "RN-105", "RN-103"],
                "required_dependency_pairs": [
                    {"before": "RN-102", "after": "RN-103"},
                    {"before": "RN-101", "after": "RN-103"},
                    {"before": "RN-104", "after": "RN-105"},
                ],
                "large_first_milestone_ids": ["RN-103", "RN-105"],
                "required_risk_term_groups": [["rollback", "incident"], ["misordered", "incorrect", "wrong"], ["user", "support", "customer"]],
                "user_visible_risk_term_groups": [["support", "customer", "user"], ["misordered", "wrong"]],
                "objective_term_groups": [["safe", "recovery"], ["rollback", "runbook"]],
                "incident_term_groups": [["rollback", "incident"], ["misordered", "support"]],
                "ceiling_rules": [
                    {"id": "incident_blind_reselect", "cap": 30, "bad_first_milestone_ids": ["RN-103", "RN-105"]},
                    {"id": "incident_blind_reselect", "cap": 30, "require_all_term_groups": [["rollback", "incident"], ["misordered", "support"]]},
                ],
                "tau_threshold": 0.8,
                "min_unique_evidence": 5,
                "pass_bar": 40,
                "matrix_pick_ceiling_id": "RN-103",
                "shortcut_first_milestone_id": "RN-103",
            },
        },
    ]


def oracle_for(spec: dict) -> dict:
    notes_md = next(iter(spec["release_notes"]))
    repo_keys = list(spec["repo_inventory"].keys())
    order = spec["gold"]["gold_order"]
    step_defs = {step_id: (title, summary) for step_id, title, summary in spec["steps"]}
    dependency_evidence = [notes_md, "repo_inventory/dependency_map.md", "repo_inventory/test_inventory.md"]
    ordered_steps = []
    for rank, step_id in enumerate(order, start=1):
        title, summary = step_defs[step_id]
        evidence = [notes_md, "repo_inventory/dependency_map.md", "repo_inventory/repo_state.md"]
        if rank == 1:
            evidence.append("repo_inventory/test_inventory.md")
        if step_id == "RN-105":
            if spec["release_context"]:
                evidence.append(next(iter(spec["release_context"])))
            if spec["incident_context"]:
                evidence.append(next(iter(spec["incident_context"])))
            if "stale" in summary.lower():
                evidence.append("release_notes/stale_note_jan_2026.md")
            if "abandoned" in summary.lower():
                evidence.append("repo_inventory/in_progress_patch/README.md")
        if spec["release_context"] and step_id in {"RN-102", "RN-105"}:
            evidence.append(next(iter(spec["release_context"])))
        if spec["incident_context"] and step_id in {"RN-102", "RN-104", "RN-105"}:
            evidence.append(next(iter(spec["incident_context"])))
        ordered_steps.append(
            {
                "step_id": step_id,
                "rank": rank,
                "title": title,
                "summary": summary,
                "bounded_deliverable": (
                    "Freeze the exact prerequisite and produce one reviewable artifact for this step only."
                    if rank == 1
                    else "Land the minimum scoped change plus the directly blocked follow-on evidence."
                ),
                "evidence": sorted(dict.fromkeys(evidence)),
            }
        )
    dependency_notes = []
    for pair in spec["gold"]["required_dependency_pairs"]:
        dependency_notes.append(
            {
                "before": pair["before"],
                "after": pair["after"],
                "reason": "Repo-state evidence says this prerequisite must land first to avoid shipping the wrong user-visible order.",
            }
        )
    risk_statement = "If the team ships the dashboard summary before the prerequisite ordering work is locked, users will see a wrong or misleading execution order during release rollout."
    if spec["release_context"]:
        risk_statement = "If the team optimizes for rollout speed instead of the current enterprise safety objective, customers will see the wrong plan order during guarded change windows."
    if spec["incident_context"]:
        risk_statement = "If the team re-enables the dashboard summary before the rollback follow-up lands, users and support will see the same misordered plan that triggered the incident."
    primary_risk_evidence = [notes_md, "repo_inventory/dependency_map.md", "repo_inventory/repo_state.md"]
    if spec["release_context"]:
        primary_risk_evidence.append(next(iter(spec["release_context"])))
    if spec["incident_context"]:
        primary_risk_evidence.append(next(iter(spec["incident_context"])))
    return {
        "schema_version": BRIEF_SCHEMA_VERSION,
        "variant_id": spec["variant_id"],
        "first_milestone_id": spec["gold"]["first_milestone_id"],
        "ordered_steps": ordered_steps,
        "dependency_notes": dependency_notes,
        "primary_risk": {
            "statement": risk_statement,
            "evidence": primary_risk_evidence,
            "mitigations": [
                "Keep the summary behind a kill switch until prerequisites are green.",
                "Use the plan-contract fixtures as the rollout gate for dependency ordering.",
                "Update the operator runbook before broadening exposure.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "Exact dashboard traffic split for the kill switch ramp",
                "status": "to_verify",
                "note": "Repo inventory names the gate but not the final percentage ladder.",
            },
            {
                "topic": "Whether any hidden customer workflow still depends on the legacy parser shim",
                "status": "missing",
                "note": "Current notes imply hidden coupling but do not enumerate every downstream consumer.",
            },
        ],
    }


def shortcut_for(spec: dict) -> dict:
    oracle = oracle_for(spec)
    bad_id = spec["gold"]["shortcut_first_milestone_id"]
    by_id = {step["step_id"]: step for step in oracle["ordered_steps"]}
    bad_first = by_id[bad_id]
    ordered = [bad_first] + [step for step in oracle["ordered_steps"] if step["step_id"] != bad_id]
    for rank, step in enumerate(ordered, start=1):
        step["rank"] = rank
    oracle["ordered_steps"] = ordered
    oracle["first_milestone_id"] = bad_id
    return oracle


def render_task_spec() -> str:
    return """# Release Note To Plan Translation

## Task Identity
- `task_id`: `release-note-to-plan-translation/implementation-plan`
- `family_id`: `release-note-to-plan-translation`
- `scenario_type`: `strategic_management`

## Task Prompt
Convert frozen release notes, repo state, and optional release / incident context
into a concrete implementation plan. The plan must identify prerequisite work,
dependency order, a bounded first milestone, and the main user-visible risk of
sequencing the release incorrectly.

## Workspace Bundle
- `release_notes/`
- `repo_inventory/`
- `release_context/` when present
- `incident_context/` when present
- `tests/test_plan_brief.py`
- `bin/cnb55-brief`

## Structured Output Contract
- Write `brief_input.json` at the workspace root.
- Validate with `./bin/cnb55-brief validate brief_input.json`.
- Submit with `./bin/cnb55-brief submit brief_input.json`.
- The canonical schema version is `cnb55.release_plan_brief.v1`.
- The CLI writes `brief/manager_brief.json` and `brief/manager_brief.md`.

## Required Fields
- `first_milestone_id`
- `ordered_steps[]` with `step_id`, `rank`, `title`, `summary`,
  `bounded_deliverable`, and evidence paths
- `dependency_notes[]`
- `primary_risk`
- `assumption_ledger[]` with at least one `missing` row

## Variant Progression
- `v1-clean-baseline`: clean dependency chain, bounded-first-milestone test
- `v2-noisy-distractor`: stale release-note experiment should not anchor the plan
- `v3-dirty-state`: abandoned draft / sunk-cost trap should not become the first milestone
- `v4-multi-corpus-objective`: current objective shifts from speed to enterprise-safe rollout
- `v5-recovery-in-thread`: prior rollback incident invalidates rollout-first ordering

## Saturation And Renewal
If this family's mean `P_benchmark` exceeds 80 for two consecutive training
rounds, the family is due for renewal. Renewal queue:
1. Add a mid-trajectory state-change variant where a staffing or rollout gate changes after the first plan draft.
2. Add a contradictory-evidence variant where two repo-inventory files disagree and the agent must explicitly reconcile them.
"""


def render_evaluator_contract() -> str:
    return """# `release-note-to-plan-translation` Evaluator Contract

## Evaluation Goal
Score whether a solver can translate release notes into a dependency-aware
implementation plan with a bounded first milestone and an evidence-backed
user-visible risk statement.

## Visible Checks
- `pytest -q tests/test_plan_brief.py`

## Hidden / Deterministic Checks
- Gold first milestone matches the safe bounded milestone.
- Ordered steps preserve the key prerequisite pairs.
- Risk statement captures the variant's hidden user-visible failure mode.
- Objective drift and incident recovery are acknowledged when present.

## 100-Point Breakdown
- `31` structural plan validity
- `10` first milestone correctness
- `12` ordering fidelity (Kendall tau threshold)
- `10` dependency-pair correctness
- `8` bounded first milestone
- `8` risk surface
- `6` grounding depth
- `5` objective acknowledgement
- `4` incident acknowledgement
- `4` user-visible risk specificity
- `2` markdown partial-progress signal (`P_only` band)

## Ceilings
- `oversized_first_milestone`: cap 35 when the solver starts with a launch / rollout step instead of a bounded prerequisite.
- `ignored_stale_release_note`: cap 30 when a stale experiment is chosen first in V2.
- `sunk_cost_finish`: cap 30 when the abandoned draft becomes the first milestone in V3.
- `objective_drift`: cap 30 when V4 ignores the current enterprise-safe objective.
- `incident_blind_reselect`: cap 30 when V5 reselects the rolled-back path or ignores the rollback context.
- `plan_without_grounding`: cap 25 when the plan is thinly grounded in evidence files.

## Baselines
- Oracle: `>= 90`
- Empty brief: `0`
- Shortcut brief: `<= 35`
"""


def render_family_yaml() -> str:
    return """family_id: release-note-to-plan-translation
track: 10
schema_version: cnb55.family.v1
layer_a_status: in_progress
layer_b_status: implemented
grader_ref: verifiers/release-note-to-plan-translation/score_ranking.py
milestone_config_ref: verifier_data/release-note-to-plan-translation/{variant_id}/milestones/
milestones:
  M1_localization:
    weight: 0.10
  M2_primary_fix:
    weight: 0.20
  M3_invariants:
    weight: 0.20
  M4_functional:
    weight: 0.20
  M5_e2e:
    weight: 0.30
capability_tags:
  shared_core:
    required: [localize, inspect, modify, verify, respect_invariants]
  per_variant:
    v1-clean-baseline:
      required: [inspect:evidence_triage, modify:policy_tradeoff]
    v2-noisy-distractor:
      required: [inspect:evidence_triage]
    v3-dirty-state:
      required: [inspect:prioritize, inspect:evidence_triage]
    v4-multi-corpus-objective:
      required: [modify:policy_tradeoff, verify:assumption_honesty]
    v5-recovery-in-thread:
      required: [modify:policy_tradeoff, verify:assumption_honesty]
tool_call_overrides:
  - command: "cnb55-brief schema"
    tags: [inspect]
  - command: "cnb55-brief validate"
    tags: [verify]
  - command: "cnb55-brief submit"
    tags: [modify]
    terminal: true
state_delta:
  kind: json_deliverable
  deliverable_path: brief/manager_brief.json
  states: [absent, present_and_invalid, present_and_valid]
  transitions:
    - {from: absent, to: absent, trigger: "read-only action", delta: 0.0}
    - {from: absent, to: present_and_valid, trigger: "cnb55-brief submit", delta: 0.7}
    - {from: absent, to: present_and_invalid, trigger: "raw brief write with invalid json", delta: 0.0}
    - {from: present_and_invalid, to: present_and_valid, trigger: "cnb55-brief submit", delta: 0.4}
    - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
  aggregate_clamp: [0.0, 1.0]
integrity_rules:
  - id: write_outside_whitelist
  - id: immutable_slice_mutated
  - id: pytest_shim
  - id: tests_modified
  - id: network_egress
llm_judge_quarantine:
  partial_progress_markdown:
    max_points: 2
    band: P_benchmark_only
  total_quarantined_points: 2
seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false
initial_state:
  workspace_bundle_root: benchmark_blueprints/families/release-note-to-plan-translation/workspace_bundle/
  manifest_lock: benchmark_blueprints/families/release-note-to-plan-translation/manifest.lock.json
  pinning: manifest_locked
saturation:
  trigger: mean P_benchmark > 80 for 2 consecutive rounds
  renewal_queue:
    - add mid-run state change after the first plan draft
    - add contradictory repo-inventory evidence requiring explicit reconciliation
"""


def build_workspace(spec: dict) -> None:
    variant_root = FAMILY_ROOT / "workspace_bundle" / spec["variant_id"]
    if variant_root.exists():
        shutil.rmtree(variant_root)
    write_text(variant_root / ".scenario_variant", spec["variant_id"] + "\n")
    write_text(variant_root / "AGENTS.md", AGENTS_TEXT)
    write_text(variant_root / "Dockerfile", DOCKERFILE_TEXT)
    write_text(
        variant_root / "artifacts" / "README.md",
        "This directory is intentionally empty. The scored deliverable is written under brief/.\n",
    )
    write_text(variant_root / "bin" / "cnb55-brief", CLI_TEXT, mode=0o755)
    write_text(variant_root / "tests" / "test_plan_brief.py", TEST_TEXT)
    for rel, text in spec["release_notes"].items():
        write_text(variant_root / rel, text)
    for rel, text in spec["repo_inventory"].items():
        write_text(variant_root / rel, text)
    for rel, text in spec["release_context"].items():
        write_text(variant_root / rel, text)
    for rel, text in spec["incident_context"].items():
        write_text(variant_root / rel, text)


def generate_oracle_artifacts(spec: dict) -> None:
    variant_root = FAMILY_ROOT / "workspace_bundle" / spec["variant_id"]
    oracle_dir = VERIFIER_DATA_ROOT / spec["variant_id"] / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_input = oracle_for(spec)
    write_json(oracle_dir / "brief_input.json", oracle_input)
    with tempfile.TemporaryDirectory(prefix=f"{spec['variant_id']}_oracle_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(variant_root, ws)
        write_json(ws / "brief_input.json", oracle_input)
        subprocess.run([sys.executable, str(ws / "bin" / "cnb55-brief"), "submit", "brief_input.json"], cwd=ws, check=True)
        shutil.copy(ws / "brief" / "manager_brief.json", oracle_dir / "manager_brief.json")
        shutil.copy(ws / "brief" / "manager_brief.md", oracle_dir / "manager_brief.md")


def build_verifier_data(spec: dict) -> None:
    vd = VERIFIER_DATA_ROOT / spec["variant_id"]
    if vd.exists():
        shutil.rmtree(vd)
    (vd / "oracle").mkdir(parents=True, exist_ok=True)
    (vd / "hidden_tests").mkdir(parents=True, exist_ok=True)
    (vd / "milestones").mkdir(parents=True, exist_ok=True)
    generate_oracle_artifacts(spec)
    write_text(vd / "hidden_tests" / "test_plan_properties.py", HIDDEN_TEST_TEXT)
    for name in MILESTONE_SCRIPT_TEXTS:
        target = Path("../_milestones_shared") / name
        os.symlink(target, vd / "milestones" / name)


def build_shared_files() -> None:
    write_text(CLI_ROOT, CLI_TEXT, mode=0o755)
    write_text(FAMILY_ROOT / "task_spec.md", render_task_spec())
    write_text(FAMILY_ROOT / "evaluator_contract.md", render_evaluator_contract())
    write_text(FAMILY_ROOT / "family.yaml", render_family_yaml())
    write_text(
        FAMILY_ROOT / "codex" / "config.toml",
        """[family]
family_id = "release-note-to-plan-translation"
task_spec = "task_spec.md"
evaluator_contract = "evaluator_contract.md"

[solver]
model = "gpt-5.4"
reasoning_effort = "high"
workspace_root = "."
required_surfaces = ["shell", "structured_reasoning"]
preferred_skill = "skills/release-plan-conversion/SKILL.md"
evidence_required = true
assumption_budget = "low"

[solver.rules]
require_ordered_schedule = true
require_assumption_ledger = true
forbid_live_web_or_external_evidence = true
forbid_invented_fixture_claims = true

[grading]
target_attack_band = "15-25"
cap_without_runtime_fixtures = 20
max_naive_score = 30
""",
    )
    shared = VERIFIER_DATA_ROOT / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for name, text in MILESTONE_SCRIPT_TEXTS.items():
        write_text(shared / name, text, mode=0o755)
    write_text(
        VERIFIER_DATA_ROOT / "_rubrics" / "partial_progress.md",
        "# Partial progress rubric\n\nPresent for Layer B dual-band bookkeeping only; the scorer keeps this family deterministic.\n",
    )


def list_files(root: Path) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            if rel.startswith("brief/"):
                continue
            out.append(rel)
    return out


def readonly_hashes(ws: Path) -> dict[str, str]:
    rels = [
        ".scenario_variant",
        "AGENTS.md",
        "Dockerfile",
        "bin",
        "release_notes",
        "repo_inventory",
        "release_context",
        "incident_context",
        "tests",
    ]
    out: dict[str, str] = {}
    for rel in rels:
        digest = sha256_tree(ws, rel)
        if digest:
            out[rel] = digest
    return out


def update_gold_and_manifest(spec: dict) -> None:
    ws = FAMILY_ROOT / "workspace_bundle" / spec["variant_id"]
    vd = VERIFIER_DATA_ROOT / spec["variant_id"]
    gold = dict(spec["gold"])
    gold["variant_id"] = spec["variant_id"]
    gold["readonly_tree_hashes"] = readonly_hashes(ws)
    gold["test_plan_brief_sha256"] = sha256_file(ws / "tests" / "test_plan_brief.py")
    write_json(vd / "gold_ranking.json", gold)
    write_json(
        vd / "workspace_manifest.json",
        {
            "variant_id": spec["variant_id"],
            "files": list_files(ws),
            "readonly_tree_hashes": gold["readonly_tree_hashes"],
            "test_plan_brief_sha256": gold["test_plan_brief_sha256"],
        },
    )


def score_variant(spec: dict, mode: str) -> int:
    with tempfile.TemporaryDirectory(prefix=f"{spec['variant_id']}_{mode}_") as tmp:
        ws = Path(tmp) / "workspace"
        results = Path(tmp) / "results"
        shutil.copytree(FAMILY_ROOT / "workspace_bundle" / spec["variant_id"], ws)
        results.mkdir(parents=True, exist_ok=True)
        (ws / "brief").mkdir(parents=True, exist_ok=True)
        if mode == "oracle":
            shutil.copy(VERIFIER_DATA_ROOT / spec["variant_id"] / "oracle" / "manager_brief.json", ws / "brief" / "manager_brief.json")
            shutil.copy(VERIFIER_DATA_ROOT / spec["variant_id"] / "oracle" / "manager_brief.md", ws / "brief" / "manager_brief.md")
        elif mode == "shortcut":
            write_json(ws / "brief_input.json", shortcut_for(spec))
            subprocess.run([sys.executable, str(ws / "bin" / "cnb55-brief"), "submit", "brief_input.json"], cwd=ws, check=True)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
                "RESULT_FILE": str(results / "verify_result.json"),
                "VARIANT_ID": spec["variant_id"],
                "CNB55_SEED": "42",
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        result = json.loads((results / "verify_result.json").read_text())
        return int(result["score"])


def build_manifest_lock(specs: list[dict]) -> None:
    variants = {}
    for spec in specs:
        ws = FAMILY_ROOT / "workspace_bundle" / spec["variant_id"]
        vd = VERIFIER_DATA_ROOT / spec["variant_id"]
        variants[spec["variant_id"]] = {
            "observed_oracle_score": score_variant(spec, "oracle"),
            "observed_empty_brief_score": score_variant(spec, "empty"),
            "observed_shortcut_score": score_variant(spec, "shortcut"),
            "verifier_data": {
                "gold_ranking_sha256": sha256_file(vd / "gold_ranking.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_brief_json_sha256": sha256_file(vd / "oracle" / "manager_brief.json"),
                "oracle_brief_md_sha256": sha256_file(vd / "oracle" / "manager_brief.md"),
                "hidden_tests_tree_sha256": sha256_tree(vd, "hidden_tests"),
            },
            "workspace_trees": {
                rel: digest
                for rel, digest in readonly_hashes(ws).items()
            },
        }
    write_json(
        FAMILY_ROOT / "manifest.lock.json",
        {
            "schema_version": "cnb55.manifest.v2",
            "created_at_utc": "2026-04-21T00:00:00Z",
            "last_regen_utc": "2026-04-21T00:00:00Z",
            "family_id": "release-note-to-plan-translation",
            "track": TRACK,
            "track_name": TRACK_NAME,
            "docker_base": DOCKER_BASE,
            "cli": {"cnb55_brief_sha256": sha256_file(CLI_ROOT)},
            "grader": {"score_ranking_py_sha256": sha256_file(SCORER)},
            "determinism": {"cnb55_seed": 42, "result_key_sort": "sorted"},
            "model_under_test": {"model": "gpt-5.4", "provider": "codex", "reasoning_effort": "high"},
            "calibration_targets": {
                "family_mean_center": 20,
                "family_mean_window": [15, 25],
                "max_variant_score": 40,
                "min_hard_variant_score": 10,
                "monotonicity_tolerance": 3,
            },
            "variants": variants,
        },
    )


def main() -> int:
    specs = variant_specs()
    build_shared_files()
    for spec in specs:
        build_workspace(spec)
        build_verifier_data(spec)
        update_gold_and_manifest(spec)
    build_manifest_lock(specs)
    print("built release-note-to-plan-translation family artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
