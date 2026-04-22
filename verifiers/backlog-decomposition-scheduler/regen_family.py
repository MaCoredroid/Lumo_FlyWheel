#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/backlog-decomposition-scheduler"
VERIFIERS = REPO / "verifiers/backlog-decomposition-scheduler"
VERIFIER_DATA = REPO / "verifier_data/backlog-decomposition-scheduler"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"

CLI_NAME = "cnb55-schedule"
BRIEF_SCHEMA_VERSION = "cnb55.schedule_brief.v1"
LANES = ("core", "platform", "ops", "release")
READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "backlog",
    "repo_evidence",
    "release_context",
    "incident_context",
    "tests",
]
IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}


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
        if any(part in IGNORED_NAMES for part in p.parts):
            continue
        if p.suffix == ".pyc":
            continue
        rel_p = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            if any(part in IGNORED_NAMES for part in p.parts):
                continue
            if p.suffix == ".pyc":
                continue
            rel = p.relative_to(root).as_posix()
            if rel.startswith("brief/"):
                continue
            out.append(rel)
    return out


CLI_TEMPLATE = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "{schema_version}"
LANES = {lanes}
CONSTRAINT_TAGS = ("dependency", "capacity", "rollout", "objective", "incident")
STATUS_VALUES = ("observed", "to_verify", "missing")
EVIDENCE_ROOTS = ("backlog", "repo_evidence", "release_context", "incident_context")


def item_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "backlog").glob("B*.md"))


def schema() -> dict[str, Any]:
    return {{
        "type": "object",
        "required": [
            "schema_version",
            "variant_id",
            "objective_focus",
            "schedule",
            "scarce_role_plan",
            "risk_gate",
            "assumption_ledger",
        ],
        "additionalProperties": False,
    }}


class ValidationError(Exception):
    def __init__(self, code: int, messages: list[str]):
        super().__init__("\\n".join(messages))
        self.code = code
        self.messages = messages


def validate(doc: Any, root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["top-level value must be an object"]
    required = set(schema()["required"])
    missing = sorted(required - set(doc))
    extra = sorted(set(doc) - required)
    for name in missing:
        errors.append(f"missing field: {{name}}")
    for name in extra:
        errors.append(f"unexpected field: {{name}}")
    if errors:
        return errors
    if doc["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {{SCHEMA_VERSION}}")
    variant = (root / ".scenario_variant").read_text().strip()
    if doc["variant_id"] != variant:
        errors.append("variant_id mismatch")
    if not isinstance(doc["objective_focus"], str) or len(doc["objective_focus"].strip()) < 8:
        errors.append("objective_focus must be a non-empty sentence")
    items = item_ids(root)
    schedule = doc["schedule"]
    if not isinstance(schedule, list) or len(schedule) != len(items):
        errors.append("schedule must cover every backlog item exactly once")
    else:
        seen = set()
        slots = []
        for entry in schedule:
            if not isinstance(entry, dict):
                errors.append("schedule entries must be objects")
                continue
            for field in ("item_id", "slot", "lane", "summary", "citations", "constraint_tags"):
                if field not in entry:
                    errors.append(f"schedule entry missing {{field}}")
            item_id = entry.get("item_id")
            if item_id not in items:
                errors.append(f"unknown item_id: {{item_id}}")
            if item_id in seen:
                errors.append(f"duplicate item_id: {{item_id}}")
            seen.add(item_id)
            slot = entry.get("slot")
            if not isinstance(slot, int) or slot < 1:
                errors.append(f"bad slot for {{item_id}}")
            else:
                slots.append(slot)
            if entry.get("lane") not in LANES:
                errors.append(f"bad lane for {{item_id}}")
            if not isinstance(entry.get("summary"), str) or len(entry["summary"].strip()) < 12:
                errors.append(f"summary too short for {{item_id}}")
            citations = entry.get("citations")
            if not isinstance(citations, list) or not citations:
                errors.append(f"citations missing for {{item_id}}")
            else:
                for rel in citations:
                    if not isinstance(rel, str):
                        errors.append(f"citation must be string for {{item_id}}")
                        continue
                    if not any(rel == prefix or rel.startswith(prefix + "/") for prefix in EVIDENCE_ROOTS):
                        errors.append(f"citation outside evidence roots: {{rel}}")
                    if not (root / rel).exists():
                        errors.append(f"missing citation path: {{rel}}")
            tags = entry.get("constraint_tags")
            if not isinstance(tags, list) or not tags or not set(tags).issubset(CONSTRAINT_TAGS):
                errors.append(f"bad constraint_tags for {{item_id}}")
        if slots and set(slots) != set(range(1, max(slots) + 1)):
            errors.append("slots must be contiguous starting at 1")
    scarce = doc["scarce_role_plan"]
    if not isinstance(scarce, dict):
        errors.append("scarce_role_plan must be object")
    else:
        for field in ("role", "protected_items", "note"):
            if field not in scarce:
                errors.append(f"scarce_role_plan missing {{field}}")
    risk = doc["risk_gate"]
    if not isinstance(risk, dict):
        errors.append("risk_gate must be object")
    else:
        for field in ("risky_item_id", "must_follow", "note"):
            if field not in risk:
                errors.append(f"risk_gate missing {{field}}")
    ledger = doc["assumption_ledger"]
    if not isinstance(ledger, list) or not ledger:
        errors.append("assumption_ledger must be non-empty")
    else:
        if not any(isinstance(row, dict) and row.get("status") == "missing" for row in ledger):
            errors.append("assumption_ledger needs a missing row")
        for row in ledger:
            if not isinstance(row, dict):
                errors.append("assumption row must be object")
                continue
            if row.get("status") not in STATUS_VALUES:
                errors.append("assumption status invalid")
    return errors


def render_md(doc: dict[str, Any]) -> str:
    lines = [
        "# Schedule Brief",
        "",
        f"- variant: `{{doc['variant_id']}}`",
        f"- objective: {{doc['objective_focus']}}",
        "",
        "## Scheduled Slots",
        "",
        "| slot | lane | item | summary |",
        "|---:|---|---|---|",
    ]
    for entry in sorted(doc["schedule"], key=lambda x: (x["slot"], x["item_id"])):
        lines.append(f"| {{entry['slot']}} | {{entry['lane']}} | `{{entry['item_id']}}` | {{entry['summary']}} |")
    lines.extend(
        [
            "",
            "## Scarce Role Plan",
            "",
            f"- role: `{{doc['scarce_role_plan']['role']}}`",
            f"- protected_items: {{', '.join(doc['scarce_role_plan']['protected_items'])}}",
            f"- note: {{doc['scarce_role_plan']['note']}}",
            "",
            "## Risk Gate",
            "",
            f"- risky_item_id: `{{doc['risk_gate']['risky_item_id']}}`",
            f"- must_follow: {{', '.join(doc['risk_gate']['must_follow'])}}",
            f"- note: {{doc['risk_gate']['note']}}",
            "",
            "## Assumption Ledger",
            "",
            "| status | topic | note |",
            "|---|---|---|",
        ]
    )
    for row in doc["assumption_ledger"]:
        lines.append(f"| {{row['status']}} | {{row['topic']}} | {{row['note']}} |")
    return "\\n".join(lines) + "\\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    for name in ("validate", "submit"):
        sp = sub.add_parser(name)
        sp.add_argument("input_file")
    args = ap.parse_args()
    root = Path.cwd()
    if args.cmd == "schema":
        print(json.dumps(schema(), indent=2, sort_keys=True))
        return 0
    path = Path(args.input_file)
    if not path.exists():
        raise ValidationError(2, [f"missing input file: {{path}}"])
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(2, [f"invalid json: {{exc}}"])
    errors = validate(doc, root)
    if errors:
        raise ValidationError(3, errors)
    if args.cmd == "validate":
        return 0
    brief_dir = root / "brief"
    brief_dir.mkdir(exist_ok=True)
    (brief_dir / "schedule_brief.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
    (brief_dir / "schedule_brief.md").write_text(render_md(doc))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        for line in exc.messages:
            print(line, file=sys.stderr)
        raise SystemExit(exc.code)
"""

VISIBLE_TEST = """from __future__ import annotations

import json
from pathlib import Path

WS = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
BRIEF = WS / "brief" / "schedule_brief.json"


def load_brief() -> dict:
    assert BRIEF.exists(), f"missing brief: {BRIEF}"
    return json.loads(BRIEF.read_text())


def backlog_ids() -> list[str]:
    return sorted(p.stem for p in (WS / "backlog").glob("B*.md"))


def test_brief_exists_and_parses():
    brief = load_brief()
    assert brief["schema_version"] == "__SCHEMA_VERSION__"


def test_schedule_covers_all_items():
    brief = load_brief()
    scheduled = sorted(entry["item_id"] for entry in brief["schedule"])
    assert scheduled == backlog_ids()


def test_slots_are_contiguous():
    brief = load_brief()
    slots = [entry["slot"] for entry in brief["schedule"]]
    assert set(slots) == set(range(1, max(slots) + 1))


def test_risk_gate_present():
    brief = load_brief()
    assert brief["risk_gate"]["risky_item_id"]
    assert brief["risk_gate"]["must_follow"]


def test_assumption_ledger_has_missing_row():
    brief = load_brief()
    assert any(row["status"] == "missing" for row in brief["assumption_ledger"])
""".replace("__SCHEMA_VERSION__", BRIEF_SCHEMA_VERSION)

HIDDEN_TEST = """from __future__ import annotations

import json
from pathlib import Path

WS = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[2]


def test_citations_reference_workspace_files():
    brief = json.loads((WS / "brief" / "schedule_brief.json").read_text())
    for entry in brief["schedule"]:
        for rel in entry["citations"]:
            assert (WS / rel).exists(), rel
"""

AGENTS_TEMPLATE = """# Agent Instructions — `backlog-decomposition-scheduler`

## Task

Build a dependency-aware execution schedule for this frozen backlog. The right
answer respects prerequisites, keeps the migration specialist out of conflicting
parallel work, and keeps the risky cutover after its observability and dry-run
gates.

## Inputs

- `backlog/` — one markdown file per backlog item.
- `repo_evidence/` — current objective, dependency notes, staffing, and risk context.
- `release_context/` — optional. If present, read it before locking the schedule.
- `incident_context/` — optional. If present, read it before locking the schedule.
- `.scenario_variant` — the variant id.
- `tests/test_schedule_brief.py` — visible checks. Do not modify it.

## Output — use the CLI

Use `./bin/{cli}`. Do not hand-write `brief/schedule_brief.json`.

1. Write `brief_input.json` at the workspace root.
2. Validate:

   ```bash
   ./bin/{cli} validate brief_input.json
   ```

3. Submit:

   ```bash
   ./bin/{cli} submit brief_input.json
   ```

The JSON must include:

- `schema_version = "{schema_version}"`
- `variant_id`
- `objective_focus`
- `schedule[]` with `item_id`, `slot`, `lane`, `summary`, `citations[]`, `constraint_tags[]`
- `scarce_role_plan`
- `risk_gate`
- `assumption_ledger[]`

## What strong work looks like

- Dependencies land before blocked work.
- Items needing the migration SRE do not share a slot.
- The risky rollout lands after the dry-run and observability work.
- If release or incident context changes the objective, the schedule reflects the current objective rather than stale planning.
- Any inferred staffing, timing, or rollout detail is marked in the assumption ledger.

## Rules

- Do not modify `backlog/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`, or `bin/`.
- Do not write outside `brief/` other than the required root `brief_input.json`.
- Do not fetch network resources.
- Do not add test shims.
""".format(cli=CLI_NAME, schema_version=BRIEF_SCHEMA_VERSION)

DOCKERFILE = """# syntax=docker/dockerfile:1.6
FROM python:3.12-bookworm@sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3

RUN pip install --no-cache-dir "pytest==8.3.3"

WORKDIR /workspace
COPY . /workspace
RUN test -f /workspace/.scenario_variant && test -d /workspace/backlog

CMD ["bash"]
"""

ARTIFACTS_README = """# artifacts/

This directory is intentionally empty in the shipped workspace. Gold schedules
live only under `verifier_data/backlog-decomposition-scheduler/<variant>/`.
"""


def item_md(item: dict) -> str:
    deps = ", ".join(item.get("depends_on", [])) or "none"
    role = item.get("scarce_role") or "none"
    return f"""# {item['item_id']} — {item['title']}

- lane: `{item['lane']}`
- depends_on: {deps}
- scarce_role: {role}

{item['body']}
"""


BASE_ITEMS = {
    "B1": {"title": "dependency-map", "lane": "core", "depends_on": [], "body": "Map the migration prerequisites and freeze the dependency graph before downstream work."},
    "B2": {"title": "customer-id-backfill", "lane": "core", "depends_on": ["B1"], "scarce_role": "migration-sre", "body": "Backfill customer IDs needed for the shadow dry-run; this consumes the migration SRE."},
    "B3": {"title": "shadow-dry-run", "lane": "core", "depends_on": ["B2"], "body": "Run the dry-run that proves the cutover path is safe enough to enter wave-1 rollout."},
    "B4": {"title": "cutover-observability", "lane": "platform", "depends_on": ["B1"], "scarce_role": "migration-sre", "body": "Install the guardrail dashboards and rollback alarms needed before wave-1 cutover."},
    "B5": {"title": "wave1-cutover", "lane": "release", "depends_on": ["B3", "B4"], "body": "Ship the risky customer-visible cutover once the dry-run and observability gates are green."},
    "B6": {"title": "support-playbook", "lane": "ops", "depends_on": ["B1"], "body": "Prepare the operator playbook so support can absorb wave-1 fallout without blocking the critical path."},
    "B7": {"title": "legacy-fastlane-toggle", "lane": "platform", "depends_on": ["B1"], "scarce_role": "migration-sre", "body": "A tempting fast-lane path backed by stale evidence; in the current cycle it should stay behind the real cutover gates."},
    "B8": {"title": "contract-compatibility-probe", "lane": "core", "depends_on": ["B1"], "body": "Compatibility probe required after the rollback incident before re-entering the fast-lane path."},
}

VARIANTS = {
    "v1-clean-baseline": {
        "items": ["B1", "B2", "B3", "B4", "B5", "B6"],
        "slot_targets": {"B1": 1, "B2": 2, "B6": 2, "B4": 3, "B3": 4, "B5": 5},
        "dependency_edges": [["B1", "B2"], ["B1", "B4"], ["B2", "B3"], ["B3", "B5"], ["B4", "B5"], ["B1", "B6"]],
        "objective_keywords": ["wave-1", "cutover", "dry-run"],
        "objective_order_pairs": [["B2", "B3"], ["B4", "B5"]],
        "critical_path": ["B1", "B2", "B3", "B5"],
        "preferred_prefix": ["B1", "B2", "B4"],
        "repo_docs": {
            "current_objective.md": "# Current objective\\n\\nLand the wave-1 import cutover safely this sprint. That means unblocking the dry-run and getting observability in before the cutover itself.\\n",
            "dependency_notes.md": "# Dependency notes\\n\\n- B2 and B4 both wait on B1.\\n- B3 waits on B2.\\n- B5 waits on both B3 and B4.\\n- B6 can start once B1 is done.\\n",
            "staffing.md": "# Staffing\\n\\nPriya is the only migration SRE this sprint. Work that needs Priya cannot share a slot.\\n",
            "risk_register.md": "# Risk register\\n\\nWave-1 cutover is the risky rollout item. Do not schedule it before dry-run proof and observability.\\n",
        },
    },
    "v2-noisy-distractor": {
        "items": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "slot_targets": {"B1": 1, "B2": 2, "B6": 2, "B4": 3, "B3": 4, "B5": 5, "B7": 6},
        "dependency_edges": [["B1", "B2"], ["B1", "B4"], ["B2", "B3"], ["B3", "B5"], ["B4", "B5"], ["B1", "B6"], ["B1", "B7"]],
        "objective_keywords": ["wave-1", "cutover", "dry-run"],
        "objective_order_pairs": [["B2", "B3"], ["B4", "B5"], ["B5", "B7"]],
        "critical_path": ["B1", "B2", "B3", "B5"],
        "preferred_prefix": ["B1", "B2", "B4"],
        "stale_priority_item": "B7",
        "stale_anchor_after": ["B3", "B4"],
        "repo_docs": {
            "current_objective.md": "# Current objective\\n\\nThe current objective is still the safe wave-1 cutover. Ignore older planning that optimized for the legacy fast-lane.\\n",
            "dependency_notes.md": "# Dependency notes\\n\\nThe real gating path is unchanged from V1.\\n",
            "staffing.md": "# Staffing\\n\\nPriya remains the only migration SRE this sprint.\\n",
            "risk_register.md": "# Risk register\\n\\nFast-lane rollout is not approved until after the real cutover gates land.\\n",
            "q1_fastlane_memo.md": "# Q1 fast-lane memo (stale)\\n\\nThis note pre-dates the current staffing freeze and argues for the fast-lane toggle first. Treat it as stale.\\n",
        },
    },
    "v3-dirty-state": {
        "items": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "slot_targets": {"B1": 1, "B2": 2, "B6": 2, "B4": 3, "B3": 4, "B5": 5, "B7": 6},
        "dependency_edges": [["B1", "B2"], ["B1", "B4"], ["B2", "B3"], ["B3", "B5"], ["B4", "B5"], ["B1", "B6"], ["B1", "B7"]],
        "objective_keywords": ["wave-1", "cutover", "dry-run"],
        "objective_order_pairs": [["B2", "B3"], ["B4", "B5"], ["B5", "B7"]],
        "critical_path": ["B1", "B2", "B3", "B5"],
        "preferred_prefix": ["B1", "B2", "B4"],
        "stale_priority_item": "B7",
        "stale_anchor_after": ["B3", "B4"],
        "sunk_cost_item": "B7",
        "sunk_cost_after": ["B5"],
        "repo_docs": {
            "current_objective.md": "# Current objective\\n\\nLand the safe wave-1 cutover. The abandoned fast-lane patch is not a substitute for the current objective.\\n",
            "dependency_notes.md": "# Dependency notes\\n\\nThe cutover path is still B1 -> B2 -> B3 -> B5 with B4 guarding B5.\\n",
            "staffing.md": "# Staffing\\n\\nPriya remains the only migration SRE this sprint.\\n",
            "risk_register.md": "# Risk register\\n\\nDo not front-load B7 because there is a partial patch on disk.\\n",
            "abandoned_patch.md": "# Abandoned fast-lane patch\\n\\nA prior session left a half-finished B7 patch. It is not production-ready and should be treated as sunk cost.\\n",
        },
    },
    "v4-multi-corpus-objective": {
        "items": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "slot_targets": {"B1": 1, "B4": 2, "B6": 2, "B2": 3, "B3": 4, "B5": 5, "B7": 6},
        "dependency_edges": [["B1", "B2"], ["B1", "B4"], ["B2", "B3"], ["B3", "B5"], ["B4", "B5"], ["B1", "B6"], ["B1", "B7"]],
        "objective_keywords": ["reliability", "observability", "rollback"],
        "objective_order_pairs": [["B4", "B2"], ["B4", "B5"], ["B5", "B7"]],
        "critical_path": ["B1", "B4", "B2", "B3", "B5"],
        "preferred_prefix": ["B1", "B4", "B2"],
        "stale_priority_item": "B7",
        "stale_anchor_after": ["B5"],
        "objective_drift_active": True,
        "repo_docs": {
            "current_objective.md": "# Current objective\\n\\nThe backlog still mentions throughput, but the release team reopened reliability as the current objective.\\n",
            "dependency_notes.md": "# Dependency notes\\n\\nB4 guards the risky rollout. B2 still feeds B3, but observability must move first this cycle.\\n",
            "staffing.md": "# Staffing\\n\\nPriya is still the only migration SRE.\\n",
            "risk_register.md": "# Risk register\\n\\nReliability-first sequencing is required this cycle.\\n",
        },
        "release_docs": {
            "release_notes.md": "# Release context\\n\\nLast cycle's throughput push increased on-call noise. This cycle is reliability-first until observability closes the gap.\\n",
            "test_inventory.md": "# Test inventory\\n\\nRollback alarms are incomplete until B4 lands.\\n",
        },
    },
    "v5-recovery-in-thread": {
        "items": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        "slot_targets": {"B1": 1, "B4": 2, "B8": 2, "B2": 3, "B6": 3, "B3": 4, "B5": 5, "B7": 6},
        "dependency_edges": [["B1", "B2"], ["B1", "B4"], ["B1", "B8"], ["B2", "B3"], ["B3", "B5"], ["B4", "B5"], ["B1", "B6"], ["B1", "B7"]],
        "objective_keywords": ["reliability", "recovery", "rollback"],
        "objective_order_pairs": [["B4", "B2"], ["B8", "B7"], ["B4", "B5"], ["B5", "B7"]],
        "critical_path": ["B1", "B4", "B2", "B3", "B5"],
        "preferred_prefix": ["B1", "B4", "B8"],
        "stale_priority_item": "B7",
        "stale_anchor_after": ["B5"],
        "objective_drift_active": True,
        "incident_reselect_item": "B7",
        "incident_guard_item": "B8",
        "repo_docs": {
            "current_objective.md": "# Current objective\\n\\nStay in recovery mode: restore a safe rollout path before reopening the fast-lane.\\n",
            "dependency_notes.md": "# Dependency notes\\n\\nB8 is now required before B7 because the rollback showed a compatibility hole.\\n",
            "staffing.md": "# Staffing\\n\\nPriya is still the only migration SRE.\\n",
            "risk_register.md": "# Risk register\\n\\nThe prior fast-lane attempt was rolled back. Re-entering that lane before B8 repeats the failure.\\n",
        },
        "release_docs": {
            "release_notes.md": "# Release context\\n\\nRecovery mode remains active until the compatibility probe lands.\\n",
        },
        "incident_docs": {
            "incident_rollback.md": "# Incident rollback\\n\\nThe last B7 fast-lane attempt was rolled back because contract compatibility checks were missing.\\n",
            "prior_schedule.md": "# Prior schedule\\n\\nThe previous plan front-loaded B7 and failed in production.\\n",
        },
    },
}


def oracle_brief(variant: str, spec: dict) -> dict:
    focus = "Land the safe wave-1 cutover." if variant.startswith("v1") or variant.startswith("v2") or variant.startswith("v3") else "Recovery-first reliability plan before reopening fast-lane work."
    if variant == "v4-multi-corpus-objective":
        focus = "Reliability-first sequencing: observability before throughput work."
    schedule = []
    for item_id, slot in sorted(spec["slot_targets"].items(), key=lambda x: (x[1], x[0])):
        item = BASE_ITEMS[item_id]
        schedule.append(
            {
                "item_id": item_id,
                "slot": slot,
                "lane": item["lane"],
                "summary": f"Place {item['title']} in slot {slot} because it respects the live dependency, capacity, and rollout constraints.",
                "citations": [f"backlog/{item_id}.md", "repo_evidence/current_objective.md", "repo_evidence/dependency_notes.md"],
                "constraint_tags": ["dependency", "objective"] if item_id not in ("B4", "B5", "B7") else ["rollout", "capacity", "objective"],
            }
        )
    return {
        "schema_version": BRIEF_SCHEMA_VERSION,
        "variant_id": variant,
        "objective_focus": focus,
        "schedule": schedule,
        "scarce_role_plan": {
            "role": "migration-sre",
            "protected_items": ["B2", "B4"] if variant != "v5-recovery-in-thread" else ["B2", "B4"],
            "note": "Keep migration-SRE work in different slots so Priya is not double-booked.",
        },
        "risk_gate": {
            "risky_item_id": "B5",
            "must_follow": ["B3", "B4"],
            "note": "Wave-1 cutover waits for the dry-run and observability gates.",
        },
        "assumption_ledger": [
            {"topic": "Customer slice size", "status": "to_verify", "note": "Need PM sign-off on the exact wave-1 batch size."},
            {"topic": "Rollback staffing coverage", "status": "missing", "note": "No document names the backup approver if Priya is unexpectedly unavailable."},
        ],
    }


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def render_variant(variant: str, spec: dict) -> None:
    ws = WORKSPACE_BUNDLE / variant
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "brief").mkdir(parents=True)
    write(ws / "AGENTS.md", AGENTS_TEMPLATE)
    write(ws / "Dockerfile", DOCKERFILE)
    write(ws / ".scenario_variant", variant + "\n")
    write(ws / "artifacts/README.md", ARTIFACTS_README)
    write(ws / "bin" / CLI_NAME, CLI_TEMPLATE.format(schema_version=BRIEF_SCHEMA_VERSION, lanes=LANES))
    (ws / "bin" / CLI_NAME).chmod(0o755)
    write(ws / "tests/test_schedule_brief.py", VISIBLE_TEST)

    for item_id in spec["items"]:
        item = {"item_id": item_id, **BASE_ITEMS[item_id]}
        write(ws / "backlog" / f"{item_id}.md", item_md(item))

    for name, content in spec["repo_docs"].items():
        write(ws / "repo_evidence" / name, content)
    for name, content in spec.get("release_docs", {}).items():
        write(ws / "release_context" / name, content)
    for name, content in spec.get("incident_docs", {}).items():
        write(ws / "incident_context" / name, content)


def build_golds_and_manifests() -> dict[str, dict]:
    observed_scores: dict[str, dict] = {}
    for variant, spec in VARIANTS.items():
        ws = WORKSPACE_BUNDLE / variant
        vd = VERIFIER_DATA / variant
        if vd.exists():
            shutil.rmtree(vd)
        (vd / "oracle").mkdir(parents=True)
        (vd / "hidden_tests").mkdir(parents=True)
        (vd / "milestones").mkdir(parents=True)

        readonly = {}
        for rel in READONLY_TREES:
            digest = sha256_tree(ws, rel)
            if digest:
                readonly[rel] = digest

        gold = {
            "variant_id": variant,
            "pass_bar": 40,
            "items": [{"item_id": item_id} for item_id in spec["items"]],
            "slot_targets": spec["slot_targets"],
            "dependency_edges": spec["dependency_edges"],
            "hard_dependency_edges": spec["dependency_edges"],
            "scarce_role_name": "migration-sre",
            "scarce_role_items": ["B2", "B4"],
            "risky_item_id": "B5",
            "risk_must_follow": ["B3", "B4"],
            "objective_keywords": spec["objective_keywords"],
            "objective_order_pairs": spec["objective_order_pairs"],
            "critical_path": spec["critical_path"],
            "preferred_prefix": spec["preferred_prefix"],
            "stale_priority_item": spec.get("stale_priority_item"),
            "stale_anchor_after": spec.get("stale_anchor_after", []),
            "sunk_cost_item": spec.get("sunk_cost_item"),
            "sunk_cost_after": spec.get("sunk_cost_after", []),
            "objective_drift_active": spec.get("objective_drift_active", False),
            "incident_reselect_item": spec.get("incident_reselect_item"),
            "incident_guard_item": spec.get("incident_guard_item"),
            "readonly_tree_hashes": readonly,
            "test_schedule_brief_sha256": sha256_file(ws / "tests/test_schedule_brief.py"),
        }
        write(vd / "gold_schedule.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
        write(vd / "workspace_manifest.json", json.dumps({"variant_id": variant, "files": list_files(ws), "readonly_tree_hashes": readonly, "test_schedule_brief_sha256": gold["test_schedule_brief_sha256"]}, indent=2, sort_keys=True) + "\n")
        write(vd / "hidden_tests/test_schedule_hidden.py", HIDDEN_TEST)

        for idx, name in enumerate(("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"), start=1):
            body = "#!/usr/bin/env bash\nset -euo pipefail\npython3 - <<'PY'\nimport json, os, sys\nr=json.load(open(os.environ['RESULT_FILE']))\n"
            key = {
                1: "M1_localization",
                2: "M2_primary_fix",
                3: "M3_invariants",
                4: "M4_functional",
                5: "M5_e2e",
            }[idx]
            body += f"sys.exit(0 if r['milestones'].get('{key}') else 1)\nPY\n"
            write(vd / "milestones" / name, body)
            (vd / "milestones" / name).chmod(0o755)

        oracle = oracle_brief(variant, spec)
        write(vd / "oracle/brief_input.json", json.dumps(oracle, indent=2, sort_keys=True) + "\n")
        with tempfile.TemporaryDirectory(prefix=f"bds_regen_{variant}_") as tmp:
            scratch = Path(tmp) / "workspace"
            shutil.copytree(ws, scratch)
            input_file = scratch / "brief_input.json"
            input_file.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
            subprocess.run([sys.executable, str(scratch / "bin" / CLI_NAME), "submit", str(input_file)], cwd=scratch, check=True)
            shutil.copy(scratch / "brief" / "schedule_brief.json", vd / "oracle" / "schedule_brief.json")
            shutil.copy(scratch / "brief" / "schedule_brief.md", vd / "oracle" / "schedule_brief.md")

            score_script = VERIFIERS / "score_schedule.py"
            result_path = Path(tmp) / "result.json"
            env = {
                **dict(os.environ),
                "AGENT_WS": str(scratch),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result_path),
                "VARIANT_ID": variant,
            }
            subprocess.run([sys.executable, str(score_script)], check=True, env=env)
            oracle_score = json.loads(result_path.read_text())["P_benchmark"]

            # Empty baseline.
            shutil.rmtree(scratch / "brief")
            (scratch / "brief").mkdir()
            subprocess.run([sys.executable, str(score_script)], check=True, env=env)
            empty_score = json.loads(result_path.read_text())["P_benchmark"]

            # Shortcut baseline: front-load risky rollout.
            shortcut = json.loads((vd / "oracle" / "schedule_brief.json").read_text())
            for entry in shortcut["schedule"]:
                if entry["item_id"] == "B5":
                    entry["slot"] = 2
            (scratch / "brief" / "schedule_brief.json").write_text(json.dumps(shortcut, indent=2, sort_keys=True) + "\n")
            subprocess.run([sys.executable, str(score_script)], check=True, env=env)
            shortcut_score = json.loads(result_path.read_text())["P_benchmark"]

        observed_scores[variant] = {
            "oracle": oracle_score,
            "empty": empty_score,
            "shortcut": shortcut_score,
        }
    return observed_scores


def refresh_manifest_lock(observed_scores: dict[str, dict]) -> None:
    lock = {
        "family_id": "backlog-decomposition-scheduler",
        "schema_version": "cnb55.manifest.v2",
        "last_regen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "grader": {
            "score_schedule_py_sha256": sha256_file(VERIFIERS / "score_schedule.py"),
            "verify_sh_sha256": sha256_file(VERIFIERS / "verify.sh"),
            "regen_family_py_sha256": sha256_file(VERIFIERS / "regen_family.py"),
        },
        "variants": {},
    }
    for variant in VARIANTS:
        vd = VERIFIER_DATA / variant
        ws = WORKSPACE_BUNDLE / variant
        lock["variants"][variant] = {
            "observed_oracle_score": observed_scores[variant]["oracle"],
            "observed_empty_brief_score": observed_scores[variant]["empty"],
            "observed_shortcut_score": observed_scores[variant]["shortcut"],
            "verifier_data": {
                "gold_schedule_sha256": sha256_file(vd / "gold_schedule.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_schedule_json_sha256": sha256_file(vd / "oracle" / "schedule_brief.json"),
                "oracle_schedule_md_sha256": sha256_file(vd / "oracle" / "schedule_brief.md"),
            },
            "workspace_trees": {rel: sha256_tree(ws, rel) for rel in READONLY_TREES if sha256_tree(ws, rel)},
        }
    write(FAMILY / "manifest.lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main() -> int:
    WORKSPACE_BUNDLE.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)
    for variant, spec in VARIANTS.items():
        render_variant(variant, spec)
    observed_scores = build_golds_and_manifests()
    refresh_manifest_lock(observed_scores)
    print(json.dumps(observed_scores, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import os

    raise SystemExit(main())
