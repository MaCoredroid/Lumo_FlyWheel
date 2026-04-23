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

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/codex-surface-workflow-mapping"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/codex-surface-workflow-mapping"
VERIFIER_ROOT = REPO / "verifiers/codex-surface-workflow-mapping"
SCORER = VERIFIER_ROOT / "score_workflow_mapping.py"

FAMILY_ID = "codex-surface-workflow-mapping"
TRACK = 2
TASK_ID = "t2_codex_surface_daily_triage_mapping"
SCHEMA_VERSION = "cnb55.workflow_mapping.v1"
VERIFY_RESULT_VERSION = "cnb55.verify_result.v3"
ENTRYPOINT_COMMAND = "make codex-daily-triage"
DIRECT_COMMAND = "python3 scripts/triage.py --window active --emit-md reports/daily_triage.md"
STALE_COMMAND = "python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md"
SCHEDULE_LITERAL = "0 9 * * 1-5"
TIMEZONE = "America/Los_Angeles"
IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "label": "clean-baseline",
        "difficulty_summary": "One live workflow, one stale skill draft, one conflicting old schedule note.",
        "scope": "Triage the open incident queue for the on-call rotation and summarize the live blockers.",
        "required_scope_keywords": ["triage", "on-call"],
        "required_paths": [
            "Makefile",
            "scripts/triage.py",
            "docs/ops/daily_triage.md",
            "ops/automation/triage.cron",
            "fixtures/open_incidents.json",
        ],
        "rejected_candidates": [
            {
                "path": "skills/legacy/daily_triage/SKILL.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "stale",
            }
        ],
        "release_paths": [],
        "incident_paths": [],
    },
    {
        "id": "v2-noisy-distractor",
        "label": "noisy-distractor",
        "difficulty_summary": "Adds a polished legacy helper path and a migration memo that tempts grep-first mapping.",
        "scope": "Triage the open incident queue for the on-call rotation and summarize the live blockers.",
        "required_scope_keywords": ["triage", "on-call"],
        "required_paths": [
            "Makefile",
            "scripts/triage.py",
            "docs/ops/daily_triage.md",
            "ops/automation/triage.cron",
            "fixtures/open_incidents.json",
            "docs/ops/migration_notes.md",
        ],
        "rejected_candidates": [
            {
                "path": "skills/legacy/daily_triage/SKILL.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "stale",
            },
            {
                "path": "scripts/triage_legacy.py",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "deprecated",
            },
        ],
        "release_paths": [],
        "incident_paths": [],
    },
    {
        "id": "v3-dirty-state",
        "label": "dirty-state",
        "difficulty_summary": "Adds an abandoned Codex artifact patch that points at the wrong helper and looks nearly reusable.",
        "scope": "Triage the open incident queue for the on-call rotation and summarize the live blockers.",
        "required_scope_keywords": ["triage", "on-call"],
        "required_paths": [
            "Makefile",
            "scripts/triage.py",
            "docs/ops/daily_triage.md",
            "ops/automation/triage.cron",
            "fixtures/open_incidents.json",
            "drafts/codex_skill_patch.md",
        ],
        "rejected_candidates": [
            {
                "path": "skills/legacy/daily_triage/SKILL.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "stale",
            },
            {
                "path": "drafts/codex_skill_patch.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "abandoned",
            },
        ],
        "release_paths": [],
        "incident_paths": [],
    },
    {
        "id": "v4-multi-corpus-objective",
        "label": "multi-corpus-objective",
        "difficulty_summary": "Release context changes the right workflow scope from generic queue review to blocker-first triage.",
        "scope": "Triage blocker incidents for the on-call rotation before any general queue cleanup.",
        "required_scope_keywords": ["triage", "blocker", "on-call"],
        "required_paths": [
            "Makefile",
            "scripts/triage.py",
            "docs/ops/daily_triage.md",
            "ops/automation/triage.cron",
            "fixtures/open_incidents.json",
            "release_context/release_notes_2026_04.md",
            "release_context/objective_shift.md",
        ],
        "rejected_candidates": [
            {
                "path": "skills/legacy/daily_triage/SKILL.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "stale",
            },
            {
                "path": "drafts/codex_skill_patch.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "abandoned",
            },
        ],
        "release_paths": [
            "release_context/release_notes_2026_04.md",
            "release_context/objective_shift.md",
        ],
        "incident_paths": [],
    },
    {
        "id": "v5-recovery-in-thread",
        "label": "recovery-in-thread",
        "difficulty_summary": "Incident context shows the weekend helper-based automation was rolled back and must not be reintroduced.",
        "scope": "Triage blocker incidents for the on-call rotation before any general queue cleanup.",
        "required_scope_keywords": ["triage", "blocker", "on-call"],
        "required_paths": [
            "Makefile",
            "scripts/triage.py",
            "docs/ops/daily_triage.md",
            "ops/automation/triage.cron",
            "fixtures/open_incidents.json",
            "release_context/release_notes_2026_04.md",
            "release_context/objective_shift.md",
            "incident_context/weekend_rollback.md",
            "incident_context/recovered_plan.md",
        ],
        "rejected_candidates": [
            {
                "path": "skills/legacy/daily_triage/SKILL.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "stale",
            },
            {
                "path": "drafts/codex_skill_patch.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "abandoned",
            },
            {
                "path": "incident_context/weekend_rollback.md",
                "command_literal": STALE_COMMAND,
                "reason_keyword": "rollback",
            },
        ],
        "release_paths": [
            "release_context/release_notes_2026_04.md",
            "release_context/objective_shift.md",
        ],
        "incident_paths": [
            "incident_context/weekend_rollback.md",
            "incident_context/recovered_plan.md",
        ],
    },
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write(path, json.dumps(payload, indent=2, sort_keys=True))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for entry in sorted(target.rglob("*")):
        if any(part in IGNORED_NAMES for part in entry.parts):
            continue
        if entry.suffix == ".pyc":
            continue
        rel_entry = entry.relative_to(target).as_posix()
        if entry.is_dir():
            h.update(f"D:{rel_entry}\n".encode())
        elif entry.is_file():
            h.update(f"F:{rel_entry}\n".encode())
            h.update(sha256_file(entry).encode())
            h.update(b"\n")
    return h.hexdigest()


def list_files(root: Path) -> list[str]:
    files: list[str] = []
    for entry in sorted(root.rglob("*")):
        if entry.is_file():
            if any(part in IGNORED_NAMES for part in entry.parts):
                continue
            if entry.suffix == ".pyc":
                continue
            files.append(entry.relative_to(root).as_posix())
    return files


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def task_spec_text() -> str:
    return f"""# `{FAMILY_ID}` Task Spec

**Track:** 02 — Codebase Understanding
**Family id:** `{FAMILY_ID}`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (v1 through v5)

## Task Prompt (canonical)

You are dropped into a small repo that already runs a script-driven triage workflow. The benchmark is not asking you to invent a workflow. It is asking you to map the live repo workflow into Codex-native artifacts without anchoring on stale drafts, deprecated helpers, or outdated schedule notes.

Produce the family's four required artifacts through the structured-output CLI:

- `artifacts/SKILL.md`
- `artifacts/codex_triage.toml`
- `artifacts/automation_proposal.md`
- `artifacts/mapping_note.md`

The canonical submission path is:

```
./bin/cnb55-workflow-map schema
./bin/cnb55-workflow-map validate workflow_input.json
./bin/cnb55-workflow-map submit workflow_input.json
```

The agent writes `workflow_input.json` at workspace root. The CLI validates it, writes `artifacts/workflow_map.json` as the canonical scored payload, and renders the four human-facing artifacts listed above.

### Required structured-output schema

- `schema_version`: `{SCHEMA_VERSION}`
- `variant_id`: exact contents of `.scenario_variant`
- `skill.entrypoint_command_literal`
- `toml.entrypoint_command_literal`
- `automation.kind`
- `automation.schedule_literal`
- `automation.command_literal`
- `automation.task_prompt`
- `mapping_note.decisions[]`
- `rejected_candidates[]`

Every cited `source_paths[]` entry must be a real file inside the provided workspace bundle. Every `command_literal` must appear verbatim in at least one cited source file. Schedule literals are validated the same way. Out-of-bundle evidence is invalid.

## Scenario Type

`codebase_understanding` — the agent must read the repo, resolve which workflow path is live, distinguish stale vs. current surfaces, and express that mapping consistently across multiple deliverables.

## Required Surfaces

- Shell for repo inspection and optional test execution.
- File reads across `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, and any variant-specific `release_context/` or `incident_context/`.
- Structured-output CLI usage via `./bin/cnb55-workflow-map`.
- Codex artifact authoring via the CLI-rendered outputs under `artifacts/`.

No network, no browser, no sibling-family evidence, no benchmark-authoring note scavenging.

## Workspace Bundle (per variant)

Every variant ships:

```
.scenario_variant
AGENTS.md
Dockerfile
Makefile
bin/cnb55-workflow-map
scripts/
docs/
ops/automation/
fixtures/
skills/legacy/
tests/test_workflow_map.py
artifacts/README.md
```

Variant-specific files add noise or state pressure:

- V2 adds a migration memo and a more tempting deprecated helper path.
- V3 adds an abandoned Codex patch draft under `drafts/`.
- V4 adds `release_context/` proving the workflow scope is blocker-first, not generic queue review.
- V5 adds `incident_context/` showing a weekend helper-based automation was rolled back.

## Difficulty Ladder

### v1 — clean-baseline

One live path (`{ENTRYPOINT_COMMAND}`), one stale skill draft, one conflicting legacy schedule note.

### v2 — noisy-distractor

Adds a migration memo and deprecated helper that look current on superficial grep.

### v3 — dirty-state

Adds an abandoned Codex patch that points at the wrong helper. The agent must not treat partial work as proof of the live path.

### v4 — multi-corpus-objective

Adds `release_context/` showing the workflow has shifted to blocker-first triage for the on-call rotation. A generic “daily queue sweep” mapping is now wrong.

### v5 — recovery-in-thread

Adds `incident_context/` showing the weekend helper-based automation caused noise and was rolled back. Re-using that path is a judgment failure, not an acceptable alternate interpretation.

## Expected Deliverables

- `artifacts/workflow_map.json` — canonical grader input written by the CLI.
- `artifacts/SKILL.md` — repo-local skill pointing at the live workflow.
- `artifacts/codex_triage.toml` — Codex config artifact consistent with the same entrypoint and scope.
- `artifacts/automation_proposal.md` — schedule and task semantics separated cleanly.
- `artifacts/mapping_note.md` — artifact-by-artifact explanation citing exact source paths and exact command literals.

## Grader Contract

### Visible checks

```
pytest -q tests/test_workflow_map.py
```

Visible checks are capped at 30 points. They verify the canonical JSON exists, parses, matches `{SCHEMA_VERSION}`, writes the four deliverables, keeps schedule semantics separate from the task section, and includes one rejected stale candidate.

### Hidden checks

Hidden scoring verifies:

- the live entrypoint is `{ENTRYPOINT_COMMAND}`, not `{STALE_COMMAND}`
- skill, TOML, and automation all point at the same live entrypoint
- the mapping is grounded in the shipped repo files
- schedule evidence resolves to `{SCHEDULE_LITERAL}` and is kept out of `automation.task_prompt`
- stale and dirty-state candidates are explicitly rejected
- V4+ reads `release_context/` and preserves blocker-first scope
- V5 reads `incident_context/` and does not re-introduce the rolled-back helper path

## Partial-Credit Ceilings

- `no_submission` ≤ 0
- `malformed_submission` ≤ 10
- `inconsistent_entrypoints` ≤ 0
- `stale_entrypoint` ≤ 20
- `wrong_schedule` ≤ 25
- `schedule_in_prompt` ≤ 30
- `ungrounded_mapping` ≤ 25
- `generic_mapping` ≤ 25
- `dirty_state_reuse` ≤ 35
- `objective_drift` ≤ 30
- `incident_blind_reuse` ≤ 25

## Saturation and Renewal Plan

Trigger when mean `P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:

1. Mid-run staffing/cadence change injected after the first inspection turn.
2. Contradictory repo-vs-incident automation evidence that must be flagged explicitly.
3. Retire V1 once the floor check saturates and promote a harder V2-derived baseline.
"""


def evaluator_contract_text() -> str:
    return f"""# `{FAMILY_ID}` Evaluator Contract

**Track:** 02 — Codebase Understanding
**Spec conformance:** CNB-55 v1.0 §7
**Scorer:** deterministic stdlib-only Python

## Evaluation Goal

Score whether the agent can identify the live Codex-facing workflow in a noisy repo and map it consistently across multiple artifacts without using stale drafts, deprecated helpers, or outdated schedule claims.

## Visible Checks (30 points max)

```
pytest -q tests/test_workflow_map.py
```

| Check | Points |
| --- | ---: |
| `artifacts/workflow_map.json` exists | 4 |
| canonical JSON parses and matches `{SCHEMA_VERSION}` | 6 |
| rendered artifacts exist and are non-empty | 4 |
| skill / toml / automation sections are present | 4 |
| mapping note covers all four artifacts | 4 |
| at least one rejected candidate is recorded | 4 |
| automation proposal separates Task and Schedule sections | 4 |

## Hidden Checks (70 points)

| Check | Points |
| --- | ---: |
| `skill.entrypoint_command_literal == "{ENTRYPOINT_COMMAND}"` | 10 |
| `toml.entrypoint_command_literal == "{ENTRYPOINT_COMMAND}"` | 10 |
| `automation.command_literal == "{ENTRYPOINT_COMMAND}"` | 5 |
| `automation.schedule_literal == "{SCHEDULE_LITERAL}"` | 10 |
| `automation.task_prompt` keeps schedule semantics out of the task body | 5 |
| mapping decisions ground commands and schedule in cited repo files | 15 |
| stale / dirty / incident candidates are rejected correctly for the variant | 10 |
| workflow scope is variant-correct and repo-specific | 5 |

## Trusted Final-State Checks

These are pass/fail integrity gates:

- no writes outside `artifacts/*` and `workflow_input.json`
- no modification of `Makefile`, `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, `skills/`, `drafts/`, `release_context/`, `incident_context/`, `tests/`, `.scenario_variant`, `AGENTS.md`, `Dockerfile`, or `bin/`
- no `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim
- no `.network_egress_detected` marker

Any integrity violation forces `shortcut_detected=true` and caps the score at 0.

## Partial-Credit Ceilings

| Ceiling | Max points | Trigger |
| --- | ---: | --- |
| `no_submission` | 0 | canonical JSON missing |
| `malformed_submission` | 10 | invalid JSON or wrong schema / variant |
| `inconsistent_entrypoints` | 0 | skill / toml / automation point at different entrypoints |
| `stale_entrypoint` | 20 | any primary artifact uses the deprecated helper path |
| `wrong_schedule` | 25 | schedule literal does not match `{SCHEDULE_LITERAL}` |
| `schedule_in_prompt` | 30 | automation task prompt embeds schedule semantics |
| `ungrounded_mapping` | 25 | commands or schedule are not grounded in cited files |
| `generic_mapping` | 25 | source coverage too thin or scope wording is repo-agnostic |
| `dirty_state_reuse` | 35 | V3+ abandoned draft not rejected |
| `objective_drift` | 30 | V4+ blocker-first release context ignored |
| `incident_blind_reuse` | 25 | V5 rollback context ignored or helper path reintroduced |

## Baselines

- Oracle submission: expected `100`
- Empty workspace submission: expected `0`
- Stale-helper shortcut submission: expected `20`

## Dual-Band Emission

- `P_benchmark`: 0-100 benchmark score
- `M_training`: normalized deterministic score in `[0, 1]`
- No LLM-judge checks are active in this family; all awarded points remain in the deterministic training band.
"""


def agents_text(variant: dict[str, object]) -> str:
    extra = []
    if variant["release_paths"]:
        extra.append("- `release_context/` exists in this variant. Read it before you decide the workflow scope.")
    if variant["incident_paths"]:
        extra.append("- `incident_context/` exists in this variant. Read it before you reuse any prior automation shape.")
    if any(item["path"] == "drafts/codex_skill_patch.md" for item in variant["rejected_candidates"]):
        extra.append("- `drafts/` contains abandoned Codex work. Treat it as evidence to evaluate, not a shortcut to reuse.")
    extra_text = "\n".join(extra)
    return f"""# Agent Instructions — `{FAMILY_ID}`

## Task

Map the live triage workflow in this repo into Codex-native artifacts. Do not invent a new workflow. Do not anchor on stale drafts or deprecated helpers.

## Inputs

- `Makefile`
- `scripts/`
- `docs/ops/`
- `ops/automation/`
- `fixtures/`
- `skills/legacy/`
- `.scenario_variant`
- `tests/test_workflow_map.py`
{extra_text}

## Output — use the CLI

The only supported submission path is `./bin/cnb55-workflow-map`.

1. Write `workflow_input.json` at workspace root.
2. Validate it:

   ```
   ./bin/cnb55-workflow-map validate workflow_input.json
   ```

3. Submit it:

   ```
   ./bin/cnb55-workflow-map submit workflow_input.json
   ```

The CLI writes:

- `artifacts/workflow_map.json` (canonical, scored by the grader)
- `artifacts/SKILL.md`
- `artifacts/codex_triage.toml`
- `artifacts/automation_proposal.md`
- `artifacts/mapping_note.md`

## Rules

- Use only evidence from this workspace bundle.
- Do not write outside `artifacts/` except for `workflow_input.json`.
- Do not modify `Makefile`, `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, `skills/`, `drafts/`, `release_context/`, `incident_context/`, `tests/`, `.scenario_variant`, `AGENTS.md`, `Dockerfile`, or `bin/`.
- Do not fetch network resources.

## What strong work looks like

- Skill, TOML, and automation all point at the same live entrypoint: `{ENTRYPOINT_COMMAND}`.
- Exact command literals are quoted from real files, not guessed.
- The automation proposal keeps task semantics separate from schedule semantics.
- Stale candidates are explicitly rejected with evidence-backed reasons.
- Variant-specific context (dirty state, release context, incident rollback) changes the mapping when it should.
"""


CLI_SCRIPT = f"""#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "{SCHEMA_VERSION}"
ARTIFACTS = ("skill", "toml", "automation", "mapping_note")
ALLOWED_KINDS = {{"heartbeat", "cron"}}


class ValidationError(Exception):
    def __init__(self, code: int, messages: list[str]):
        super().__init__("\\n".join(messages))
        self.code = code
        self.messages = messages


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(2, [f"input file missing: {{path}}"]) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(2, [f"input file is not valid JSON: {{exc}}"]) from exc
    if not isinstance(payload, dict):
        raise ValidationError(2, ["top-level JSON value must be an object"])
    return payload


def ensure_rel_file(workspace: Path, rel: str) -> Path:
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValidationError(4, [f"invalid source path: {{rel!r}}"])
    path = workspace / rel
    if not path.is_file():
        raise ValidationError(4, [f"source path does not exist: {{rel}}"])
    return path


def command_grounded(workspace: Path, command: str, source_paths: list[str]) -> bool:
    if not command.strip():
        return False
    for rel in source_paths:
        text = ensure_rel_file(workspace, rel).read_text(encoding="utf-8")
        if command in text:
            return True
    return False


def validate_payload(workspace: Path, payload: dict) -> None:
    errors: list[str] = []
    required = {{"schema_version", "variant_id", "skill", "toml", "automation", "mapping_note", "rejected_candidates"}}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    for key in missing:
        errors.append(f"missing required field: {{key}}")
    for key in extra:
        errors.append(f"unexpected top-level field: {{key}}")
    if errors:
        raise ValidationError(3, errors)

    if payload["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {{SCHEMA_VERSION}}")

    variant_file = (workspace / ".scenario_variant").read_text(encoding="utf-8").strip()
    if payload["variant_id"] != variant_file:
        errors.append("variant_id does not match .scenario_variant")

    for section_name in ("skill", "toml"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"{{section_name}} must be an object")
            continue
        for field in ("entrypoint_command_literal", "workflow_scope", "source_paths"):
            if field not in section:
                errors.append(f"{{section_name}} missing {{field}}")
        if not isinstance(section.get("source_paths"), list) or not section["source_paths"]:
            errors.append(f"{{section_name}}.source_paths must be a non-empty array")
        else:
            for rel in section["source_paths"]:
                ensure_rel_file(workspace, rel)
            if not command_grounded(workspace, section.get("entrypoint_command_literal", ""), section["source_paths"]):
                errors.append(f"{{section_name}} entrypoint command is not grounded in cited files")

    automation = payload.get("automation")
    if not isinstance(automation, dict):
        errors.append("automation must be an object")
    else:
        for field in ("kind", "schedule_literal", "command_literal", "task_prompt", "source_paths"):
            if field not in automation:
                errors.append(f"automation missing {{field}}")
        if automation.get("kind") not in ALLOWED_KINDS:
            errors.append("automation.kind must be heartbeat or cron")
        if not isinstance(automation.get("source_paths"), list) or not automation["source_paths"]:
            errors.append("automation.source_paths must be a non-empty array")
        else:
            for rel in automation["source_paths"]:
                ensure_rel_file(workspace, rel)
            if not command_grounded(workspace, automation.get("command_literal", ""), automation["source_paths"]):
                errors.append("automation command literal is not grounded in cited files")
            if not command_grounded(workspace, automation.get("schedule_literal", ""), automation["source_paths"]):
                errors.append("automation schedule literal is not grounded in cited files")

    mapping_note = payload.get("mapping_note")
    if not isinstance(mapping_note, dict) or not isinstance(mapping_note.get("decisions"), list):
        errors.append("mapping_note.decisions must be an array")
    else:
        seen_artifacts: set[str] = set()
        for decision in mapping_note["decisions"]:
            if not isinstance(decision, dict):
                errors.append("mapping_note decision entries must be objects")
                continue
            artifact = decision.get("artifact")
            if artifact not in ARTIFACTS:
                errors.append(f"invalid mapping artifact: {{artifact!r}}")
            else:
                seen_artifacts.add(artifact)
            source_paths = decision.get("source_paths")
            command_literals = decision.get("command_literals")
            if not isinstance(source_paths, list) or not source_paths:
                errors.append("mapping_note decision source_paths must be a non-empty array")
                continue
            for rel in source_paths:
                ensure_rel_file(workspace, rel)
            if not isinstance(command_literals, list) or not command_literals:
                errors.append("mapping_note decision command_literals must be a non-empty array")
                continue
            for command in command_literals:
                if not command_grounded(workspace, command, source_paths):
                    errors.append(f"mapping_note command literal is not grounded: {{command}}")
        if seen_artifacts != set(ARTIFACTS):
            errors.append("mapping_note.decisions must cover skill, toml, automation, and mapping_note exactly once")

    rejected = payload.get("rejected_candidates")
    if not isinstance(rejected, list) or not rejected:
        errors.append("rejected_candidates must be a non-empty array")
    else:
        for entry in rejected:
            if not isinstance(entry, dict):
                errors.append("rejected candidate entries must be objects")
                continue
            for field in ("path", "command_literal", "reason", "source_paths"):
                if field not in entry:
                    errors.append(f"rejected candidate missing {{field}}")
            if isinstance(entry.get("source_paths"), list):
                for rel in entry["source_paths"]:
                    ensure_rel_file(workspace, rel)
                if not command_grounded(workspace, entry.get("command_literal", ""), entry["source_paths"]):
                    errors.append(f"rejected candidate command is not grounded: {{entry.get('command_literal', '')}}")

    if errors:
        raise ValidationError(3, errors)


def render_skill(payload: dict) -> str:
    skill = payload["skill"]
    return "\\n".join(
        [
            "# Workflow Mapping",
            "",
            "## Objective",
            "",
            skill["workflow_scope"],
            "",
            "## Entrypoint",
            "",
            f"`{{skill['entrypoint_command_literal']}}`",
            "",
            "## Evidence",
            "",
            *[f"- `{{path}}`" for path in skill["source_paths"]],
            "",
            "## Procedure",
            "",
            "1. Inspect the cited workflow files before reusing any draft artifact.",
            "2. Use the exact entrypoint above across the config and automation artifacts.",
            "3. Keep schedule semantics out of the task body.",
        ]
    )


def render_toml(payload: dict) -> str:
    toml = payload["toml"]
    must_read = toml.get("must_read", [])
    rendered = [
        '[workflow]',
        f'schema_version = "{SCHEMA_VERSION}"',
        f'entrypoint = "{{toml["entrypoint_command_literal"]}}"',
        f'workflow_scope = "{{toml["workflow_scope"]}}"',
        'preferred_output_dir = "artifacts"',
        '',
        '[workflow.must_read]',
        "paths = [" + ", ".join(json.dumps(item) for item in must_read) + "]",
        '',
        '[automation]',
        f'kind = "{{payload["automation"]["kind"]}}"',
        f'schedule_literal = "{{payload["automation"]["schedule_literal"]}}"',
        f'command = "{{payload["automation"]["command_literal"]}}"',
    ]
    return "\\n".join(rendered)


def render_automation(payload: dict) -> str:
    automation = payload["automation"]
    return "\\n".join(
        [
            "# Automation Proposal",
            "",
            "## Task",
            "",
            automation["task_prompt"],
            "",
            "## Schedule",
            "",
            f"- kind: `{{automation['kind']}}`",
            f"- schedule_literal: `{{automation['schedule_literal']}}`",
            f"- timezone: `{TIMEZONE}`",
            "",
            "## Command",
            "",
            f"`{{automation['command_literal']}}`",
            "",
            "## Evidence",
            "",
            *[f"- `{{path}}`" for path in automation["source_paths"]],
        ]
    )


def render_mapping_note(payload: dict) -> str:
    lines = ["# Mapping Note", ""]
    for decision in payload["mapping_note"]["decisions"]:
        lines.append(f"## {{decision['artifact']}}")
        lines.append("")
        lines.append(decision["rationale"])
        lines.append("")
        lines.append("Source paths:")
        lines.extend(f"- `{{path}}`" for path in decision["source_paths"])
        lines.append("")
        lines.append("Command literals:")
        lines.extend(f"- `{{command}}`" for command in decision["command_literals"])
        lines.append("")
    lines.append("## Rejected Candidates")
    lines.append("")
    for entry in payload["rejected_candidates"]:
        lines.append(f"- `{{entry['path']}}`: {{entry['reason']}}")
    return "\\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {{"schema", "validate", "submit"}}:
        print("usage: cnb55-workflow-map [schema|validate|submit] workflow_input.json", file=sys.stderr)
        return 2
    command = argv[1]
    workspace = Path.cwd()
    if command == "schema":
        schema = {{
            "schema_version": SCHEMA_VERSION,
            "top_level_fields": ["skill", "toml", "automation", "mapping_note", "rejected_candidates"],
            "automation_kind": ["heartbeat", "cron"],
        }}
        print(json.dumps(schema, indent=2, sort_keys=True))
        return 0

    input_path = Path(argv[2])
    try:
        payload = load_json(input_path)
        validate_payload(workspace, payload)
    except ValidationError as exc:
        for line in exc.messages:
            print(line, file=sys.stderr)
        return exc.code

    if command == "validate":
        print("valid")
        return 0

    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "workflow_map.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    (artifacts / "SKILL.md").write_text(render_skill(payload) + "\\n", encoding="utf-8")
    (artifacts / "codex_triage.toml").write_text(render_toml(payload) + "\\n", encoding="utf-8")
    (artifacts / "automation_proposal.md").write_text(render_automation(payload) + "\\n", encoding="utf-8")
    (artifacts / "mapping_note.md").write_text(render_mapping_note(payload) + "\\n", encoding="utf-8")
    print("submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
"""


VISIBLE_TEST = """from __future__ import annotations

import json
import pathlib

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]


def test_canonical_json_exists():
    target = WS / "artifacts" / "workflow_map.json"
    assert target.exists(), "artifacts/workflow_map.json missing"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cnb55.workflow_mapping.v1"
    assert payload["variant_id"] == (WS / ".scenario_variant").read_text(encoding="utf-8").strip()


def test_rendered_artifacts_exist():
    for rel in [
        "artifacts/SKILL.md",
        "artifacts/codex_triage.toml",
        "artifacts/automation_proposal.md",
        "artifacts/mapping_note.md",
    ]:
        path = WS / rel
        assert path.exists(), f"missing {rel}"
        assert path.read_text(encoding="utf-8").strip(), f"{rel} is empty"


def test_toml_parses():
    text = (WS / "artifacts" / "codex_triage.toml").read_text(encoding="utf-8")
    assert 'entrypoint = "' in text
    assert 'schedule_literal = "' in text


def test_mapping_note_has_four_decisions():
    payload = json.loads((WS / "artifacts" / "workflow_map.json").read_text(encoding="utf-8"))
    artifacts = {item["artifact"] for item in payload["mapping_note"]["decisions"]}
    assert artifacts == {"skill", "toml", "automation", "mapping_note"}


def test_rejected_candidates_present():
    payload = json.loads((WS / "artifacts" / "workflow_map.json").read_text(encoding="utf-8"))
    assert payload["rejected_candidates"], "need at least one rejected candidate"


def test_automation_proposal_splits_task_and_schedule():
    text = (WS / "artifacts" / "automation_proposal.md").read_text(encoding="utf-8")
    assert "## Task" in text
    assert "## Schedule" in text
"""


def dockerfile_text() -> str:
    return """FROM python:3.11-slim
WORKDIR /workspace
"""


def makefile_text() -> str:
    return f"""codex-daily-triage:
\t{DIRECT_COMMAND}

daily-triage-legacy:
\t{STALE_COMMAND}
"""


def triage_py() -> str:
    return """from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="active")
    parser.add_argument("--emit-md", required=True)
    args = parser.parse_args()

    incidents = json.loads(Path("fixtures/open_incidents.json").read_text(encoding="utf-8"))
    blockers = [item for item in incidents if item.get("severity") == "blocker"]
    out = Path(args.emit_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Daily triage", "", f"window={args.window}", ""]
    for item in blockers:
        lines.append(f"- {item['id']}: {item['owner']} -> {item['summary']}")
    out.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def legacy_triage_py() -> str:
    return f"""from __future__ import annotations

import argparse
from pathlib import Path

# Deprecated invocation preserved for rollback diffing only:
# {STALE_COMMAND}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="today")
    parser.add_argument("--emit-md", required=True)
    args = parser.parse_args()
    out = Path(args.emit_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Legacy daily triage\\n\\nThis helper was kept for rollback diffs only.\\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def docs_daily_text(variant: dict[str, object]) -> str:
    scope = variant["scope"]
    return f"""# Daily Triage Workflow

The live Codex-facing entrypoint is `{ENTRYPOINT_COMMAND}`.

The Make target shells out to:

`{DIRECT_COMMAND}`

The automation cadence is pinned in `ops/automation/triage.cron` as:

`{SCHEDULE_LITERAL}`

Workflow scope:

- {scope}
- use `fixtures/open_incidents.json` as the sample evidence surface for local dry runs
- keep task semantics separate from schedule semantics when drafting Codex automation prompts
"""


def docs_legacy_text() -> str:
    return f"""# Legacy Schedule Note

This note pre-dates the current automation. It still mentions a seven-day sweep and the deprecated helper:

`{STALE_COMMAND}`

Legacy cron literal:

`0 9 * * *`

Do not use this note as the source of truth when mapping the live workflow.
"""


def docs_migration_text() -> str:
    return """# Migration Notes

During the codex-native migration we temporarily kept the legacy helper around for diffing outputs.

That did not make it the live entrypoint. The live workflow stayed on the Make target while the helper remained rollback-only.
"""


def cron_text() -> str:
    return f"{SCHEDULE_LITERAL} {DIRECT_COMMAND}\n"


def fixtures_text() -> str:
    return json.dumps(
        [
            {"id": "INC-410", "severity": "blocker", "owner": "oncall-a", "summary": "response export misses owner_source"},
            {"id": "INC-411", "severity": "warning", "owner": "oncall-b", "summary": "non-blocking routing drift"},
            {"id": "INC-412", "severity": "blocker", "owner": "oncall-a", "summary": "weekend automation page noise"},
        ],
        indent=2,
        sort_keys=True,
    )


def stale_skill_text() -> str:
    return f"""# Daily Triage (legacy draft)

This draft predates the current repo migration and still references:

`{STALE_COMMAND}`

It is kept only so reviewers can compare old vs. new workflow wording.
"""


def draft_patch_text() -> str:
    return f"""# Abandoned Codex Patch

This half-finished patch tried to wrap the deprecated helper:

`{STALE_COMMAND}`

It was never shipped and should not be treated as proof of the live workflow.
"""


def release_notes_text() -> str:
    return """# Release Notes 2026-04

The triage workflow now prioritizes blocker incidents for the on-call rotation before any general backlog review.

Weekend queue sweeps were deprioritized after noisy pages in March.
"""


def objective_shift_text() -> str:
    return """# Objective Shift

The old workflow narrative talked about a broad daily queue sweep. The current objective is narrower:

- blockers first
- on-call ownership clarity
- summary output that can feed the response review stand-up
"""


def incident_rollback_text() -> str:
    return f"""# Weekend Rollback

The weekend helper-based automation was rolled back after it reintroduced noisy non-blocker pages.

Rollback target:

`{STALE_COMMAND}`

The live recovery plan returned to the weekday blocker-first Make target.
"""


def recovered_plan_text() -> str:
    return f"""# Recovered Plan

Post-rollback, the only approved automation route is:

`{ENTRYPOINT_COMMAND}`

Weekday cadence stays pinned to `{SCHEDULE_LITERAL}`.
"""


def artifacts_readme() -> str:
    return """# Artifacts

The CLI writes scored outputs here.
"""


def family_yaml_text() -> str:
    return f"""family_id: {FAMILY_ID}
track: {TRACK}
schema_version: cnb55.family.v1
layer_a_status: probe_pending
layer_b_status: green

grader_ref: verifiers/{FAMILY_ID}/score_workflow_mapping.py
milestone_config_ref: verifier_data/{FAMILY_ID}/{{variant_id}}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: enough cited source paths to prove the mapping came from repo evidence
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: canonical workflow_map.json exists and parses
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: no integrity violation and no stale primary entrypoint
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: skill, TOML, and automation align on the live workflow
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: schedule, scope, and stale-candidate rejection all match the gold mapping

capability_tags:
  shared_core:
    required:
      - localize
      - inspect
      - modify
      - verify
      - respect_invariants
    recommended:
      - inspect:evidence_triage
      - modify:artifact_authoring
      - verify:cross_surface_consistency
  per_variant:
    v1-clean-baseline: {{}}
    v2-noisy-distractor:
      variant_notes: stale helper and migration memo must be triaged correctly
    v3-dirty-state:
      variant_notes: abandoned Codex draft must be rejected explicitly
    v4-multi-corpus-objective:
      variant_notes: release context changes the workflow scope to blocker-first
    v5-recovery-in-thread:
      variant_notes: incident rollback forbids reusing the helper-based weekend flow

tool_call_overrides:
  - command: "cnb55-workflow-map schema"
    tags: [inspect]
  - command: "cnb55-workflow-map validate"
    tags: [verify]
  - command: "cnb55-workflow-map submit"
    tags: [modify]
    terminal: true

state_delta:
  kind: json_deliverable
  deliverable_path: artifacts/workflow_map.json
  states:
    - absent
    - present_and_invalid
    - present_and_valid
  transitions:
    - {{from: absent, to: absent, trigger: "read-only turn", delta: 0.0}}
    - {{from: absent, to: absent, trigger: "validate success", delta: 0.3}}
    - {{from: absent, to: present_and_valid, trigger: "submit success", delta: 0.7}}
    - {{from: absent, to: present_and_invalid, trigger: "raw invalid write", delta: 0.0}}
    - {{from: present_and_invalid, to: present_and_valid, trigger: "subsequent valid submit", delta: 0.4}}
    - {{from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}}
  aggregate_clamp: [0.0, 1.0]

integrity_rules:
  - id: write_outside_whitelist
    rule: any write outside artifacts/* and workflow_input.json
  - id: immutable_slice_mutated
    rule: readonly tree hash changed
  - id: pytest_shim
    rule: shim file present
  - id: tests_modified
    rule: visible test checksum changed
  - id: network_egress
    rule: .network_egress_detected marker present

llm_judge_quarantine:
  total_quarantined_points: 0

seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false

initial_state:
  type: manifest_locked
  ref: benchmark_blueprints/families/{FAMILY_ID}/manifest.lock.json

rawr_modes:
  grounding_stripped: implemented
  citation_fabricated: declared_not_yet_implemented
  constraint_named_not_respected: declared_not_yet_implemented

saturation:
  threshold_mean_P: 80
  sustained_rounds: 2
  renewal_queue:
    - midrun-cadence-change
    - contradictory-docs-vs-incident
    - retire-v1
"""


def build_oracle_input(variant: dict[str, object]) -> dict:
    source_paths = sorted(set(variant["required_paths"]))
    decisions = [
        {
            "artifact": "skill",
            "source_paths": ["Makefile", "docs/ops/daily_triage.md", "scripts/triage.py"],
            "command_literals": [ENTRYPOINT_COMMAND, DIRECT_COMMAND],
            "rationale": "The skill uses the live Make target and cites the direct script only as backing evidence.",
        },
        {
            "artifact": "toml",
            "source_paths": ["Makefile", "docs/ops/daily_triage.md", "fixtures/open_incidents.json"],
            "command_literals": [ENTRYPOINT_COMMAND],
            "rationale": "The TOML config keeps the same live entrypoint and points the agent at the repo surfaces it must read.",
        },
        {
            "artifact": "automation",
            "source_paths": ["ops/automation/triage.cron", "docs/ops/daily_triage.md"],
            "command_literals": [ENTRYPOINT_COMMAND, SCHEDULE_LITERAL],
            "rationale": "The automation proposal keeps task text separate from the weekday schedule literal grounded in the cron file.",
        },
        {
            "artifact": "mapping_note",
            "source_paths": source_paths,
            "command_literals": [ENTRYPOINT_COMMAND, DIRECT_COMMAND, SCHEDULE_LITERAL],
            "rationale": "The mapping note ties each artifact back to the exact repo files and exact literals that justify the decision.",
        },
    ]
    rejected: list[dict[str, object]] = []
    for item in variant["rejected_candidates"]:
        source_paths = [item["path"]]
        if item["path"].startswith("incident_context/"):
            source_paths.append("incident_context/recovered_plan.md")
        elif item["path"].startswith("scripts/"):
            source_paths.append("docs/ops/migration_notes.md")
        else:
            source_paths.append("docs/ops/legacy_schedule.md")
        rejected.append(
            {
                "path": item["path"],
                "command_literal": item["command_literal"],
                "reason": f"Rejected because this path is {item['reason_keyword']} relative to the live workflow mapping.",
                "source_paths": sorted(set(source_paths)),
            }
        )
    must_read = ["docs/ops/daily_triage.md", "ops/automation/triage.cron", "fixtures/open_incidents.json"]
    if variant["release_paths"]:
        must_read.append("release_context/objective_shift.md")
    if variant["incident_paths"]:
        must_read.append("incident_context/weekend_rollback.md")
    return {
        "schema_version": SCHEMA_VERSION,
        "variant_id": variant["id"],
        "skill": {
            "entrypoint_command_literal": ENTRYPOINT_COMMAND,
            "workflow_scope": variant["scope"],
            "source_paths": ["Makefile", "docs/ops/daily_triage.md", "scripts/triage.py"],
        },
        "toml": {
            "entrypoint_command_literal": ENTRYPOINT_COMMAND,
            "workflow_scope": variant["scope"],
            "must_read": must_read,
            "source_paths": ["Makefile", "docs/ops/daily_triage.md", "fixtures/open_incidents.json"],
        },
        "automation": {
            "kind": "heartbeat",
            "schedule_literal": SCHEDULE_LITERAL,
            "command_literal": ENTRYPOINT_COMMAND,
            "task_prompt": variant["scope"],
            "source_paths": ["ops/automation/triage.cron", "docs/ops/daily_triage.md"],
        },
        "mapping_note": {"decisions": decisions},
        "rejected_candidates": rejected,
    }


def build_shortcut_input(variant: dict[str, object]) -> dict:
    payload = build_oracle_input(variant)
    payload["skill"]["entrypoint_command_literal"] = STALE_COMMAND
    payload["skill"]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "scripts/triage_legacy.py"]
    payload["toml"]["entrypoint_command_literal"] = STALE_COMMAND
    payload["toml"]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "docs/ops/legacy_schedule.md"]
    payload["automation"]["command_literal"] = STALE_COMMAND
    payload["automation"]["schedule_literal"] = "0 9 * * *"
    payload["automation"]["task_prompt"] = "Run every weekday at 09:00 and sweep the whole queue."
    payload["automation"]["source_paths"] = ["docs/ops/legacy_schedule.md", "skills/legacy/daily_triage/SKILL.md"]
    payload["mapping_note"]["decisions"][0]["command_literals"] = [STALE_COMMAND]
    payload["mapping_note"]["decisions"][0]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "scripts/triage_legacy.py"]
    payload["mapping_note"]["decisions"][1]["command_literals"] = [STALE_COMMAND]
    payload["mapping_note"]["decisions"][1]["source_paths"] = ["skills/legacy/daily_triage/SKILL.md", "docs/ops/legacy_schedule.md"]
    payload["mapping_note"]["decisions"][2]["command_literals"] = [STALE_COMMAND, "0 9 * * *"]
    payload["mapping_note"]["decisions"][2]["source_paths"] = ["docs/ops/legacy_schedule.md", "skills/legacy/daily_triage/SKILL.md"]
    payload["mapping_note"]["decisions"][3]["command_literals"] = [STALE_COMMAND, "0 9 * * *"]
    payload["mapping_note"]["decisions"][3]["source_paths"] = [
        "skills/legacy/daily_triage/SKILL.md",
        "docs/ops/legacy_schedule.md",
        "scripts/triage_legacy.py",
    ]
    return payload


def variant_files(variant: dict[str, object]) -> dict[str, str]:
    files: dict[str, str] = {
        ".scenario_variant": f"{variant['id']}\n",
        "AGENTS.md": agents_text(variant),
        "Dockerfile": dockerfile_text(),
        "Makefile": makefile_text(),
        "bin/cnb55-workflow-map": CLI_SCRIPT,
        "scripts/triage.py": triage_py(),
        "scripts/triage_legacy.py": legacy_triage_py(),
        "docs/ops/daily_triage.md": docs_daily_text(variant),
        "docs/ops/legacy_schedule.md": docs_legacy_text(),
        "ops/automation/triage.cron": cron_text(),
        "fixtures/open_incidents.json": fixtures_text(),
        "skills/legacy/daily_triage/SKILL.md": stale_skill_text(),
        "tests/test_workflow_map.py": VISIBLE_TEST,
        "artifacts/README.md": artifacts_readme(),
    }
    if variant["id"] in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        files["docs/ops/migration_notes.md"] = docs_migration_text()
    if variant["id"] in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        files["drafts/codex_skill_patch.md"] = draft_patch_text()
    if variant["release_paths"]:
        files["release_context/release_notes_2026_04.md"] = release_notes_text()
        files["release_context/objective_shift.md"] = objective_shift_text()
    if variant["incident_paths"]:
        files["incident_context/weekend_rollback.md"] = incident_rollback_text()
        files["incident_context/recovered_plan.md"] = recovered_plan_text()
    return files


def build_workspace_manifest(variant: dict[str, object], ws: Path) -> dict:
    files = list_files(ws)
    readonly_keys = [
        ".scenario_variant",
        "AGENTS.md",
        "Dockerfile",
        "Makefile",
        "bin",
        "scripts",
        "docs",
        "ops",
        "fixtures",
        "skills",
        "drafts",
        "release_context",
        "incident_context",
        "tests",
    ]
    readonly_tree_hashes = {key: sha256_tree(ws, key) for key in readonly_keys if (ws / key).exists()}
    return {
        "variant_id": variant["id"],
        "files": files,
        "readonly_tree_hashes": readonly_tree_hashes,
        "test_workflow_map_sha256": sha256_file(ws / "tests" / "test_workflow_map.py"),
    }


def render_oracle_outputs(variant: dict[str, object], ws: Path, oracle_dir: Path) -> None:
    oracle_input = build_oracle_input(variant)
    with tempfile.TemporaryDirectory(prefix="csm_oracle_") as tmp:
        temp_ws = Path(tmp) / "workspace"
        shutil.copytree(ws, temp_ws)
        input_path = temp_ws / "workflow_input.json"
        write_json(input_path, oracle_input)
        subprocess.run([sys.executable, str(temp_ws / "bin" / "cnb55-workflow-map"), "submit", str(input_path)], check=True, cwd=temp_ws)
        oracle_dir.mkdir(parents=True, exist_ok=True)
        write_json(oracle_dir / "workflow_input.json", oracle_input)
        for name in ["workflow_map.json", "SKILL.md", "codex_triage.toml", "automation_proposal.md", "mapping_note.md"]:
            shutil.copy2(temp_ws / "artifacts" / name, oracle_dir / name)


def make_shortcut_workspace(variant: dict[str, object], ws_src: Path, dst: Path) -> Path:
    temp_ws = dst / "workspace"
    shutil.copytree(ws_src, temp_ws)
    write_json(temp_ws / "workflow_input.json", build_shortcut_input(variant))
    subprocess.run([sys.executable, str(temp_ws / "bin" / "cnb55-workflow-map"), "submit", str(temp_ws / "workflow_input.json")], check=True, cwd=temp_ws)
    return temp_ws


def score_workspace(variant_id: str, ws: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="csm_score_") as tmp:
        result_path = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result_path),
                "VARIANT_ID": variant_id,
            }
        )
        subprocess.run([sys.executable, str(SCORER)], check=True, env=env)
        return json.loads(result_path.read_text(encoding="utf-8"))


def write_milestones(variant_id: str) -> None:
    milestone_dir = VERIFIER_DATA / variant_id / "milestones"
    milestone_dir.mkdir(parents=True, exist_ok=True)
    for key in ["M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e"]:
        script = f"""#!/bin/sh
set -eu
python3 - "$RESULT_FILE" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path, "r", encoding="utf-8"))
sys.exit(0 if payload.get("milestones", {{}}).get("{key}", False) else 1)
PY
"""
        target = milestone_dir / f"{key.lower()}.sh"
        write(target, script)
        target.chmod(0o755)


def build_gold(variant: dict[str, object], workspace_manifest: dict) -> dict:
    return {
        "variant_id": variant["id"],
        "expected_entrypoint_command": ENTRYPOINT_COMMAND,
        "expected_direct_command": DIRECT_COMMAND,
        "expected_schedule_literal": SCHEDULE_LITERAL,
        "allowed_automation_kinds": ["heartbeat", "cron"],
        "required_scope_keywords": variant["required_scope_keywords"],
        "required_paths": variant["required_paths"],
        "rejected_candidates": variant["rejected_candidates"],
        "release_paths": variant["release_paths"],
        "incident_paths": variant["incident_paths"],
        "readonly_tree_hashes": workspace_manifest["readonly_tree_hashes"],
        "test_workflow_map_sha256": workspace_manifest["test_workflow_map_sha256"],
    }


def build_manifest_lock(observed_scores: dict[str, dict]) -> dict:
    return {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest.v2",
        "last_regen_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "grader": {
            "regen_family_py_sha256": sha256_file(Path(__file__)),
            "score_workflow_mapping_py_sha256": sha256_file(SCORER),
            "run_verification_matrix_py_sha256": sha256_file(VERIFIER_ROOT / "run_verification_matrix.py"),
        },
        "variants": observed_scores,
    }


def main() -> int:
    if WORKSPACE_BUNDLE.exists():
        shutil.rmtree(WORKSPACE_BUNDLE)
    if VERIFIER_DATA.exists():
        shutil.rmtree(VERIFIER_DATA)

    write(FAMILY / "task_spec.md", task_spec_text())
    write(FAMILY / "evaluator_contract.md", evaluator_contract_text())
    write(FAMILY / "family.yaml", family_yaml_text())

    observed_scores: dict[str, dict] = {}
    for variant in VARIANTS:
        ws = WORKSPACE_BUNDLE / variant["id"]
        for rel, text in variant_files(variant).items():
            write(ws / rel, text)
        (ws / "bin" / "cnb55-workflow-map").chmod(0o755)

        workspace_manifest = build_workspace_manifest(variant, ws)
        variant_verifier = VERIFIER_DATA / variant["id"]
        variant_verifier.mkdir(parents=True, exist_ok=True)
        write_json(variant_verifier / "workspace_manifest.json", workspace_manifest)
        write_json(variant_verifier / "gold_workflow.json", build_gold(variant, workspace_manifest))
        render_oracle_outputs(variant, ws, variant_verifier / "oracle")
        write_milestones(variant["id"])

        empty_result = score_workspace(variant["id"], ws)
        with tempfile.TemporaryDirectory(prefix="csm_oracle_score_") as tmp:
            oracle_ws = Path(tmp) / "workspace"
            shutil.copytree(ws, oracle_ws)
            shutil.copy2(variant_verifier / "oracle" / "workflow_map.json", oracle_ws / "artifacts" / "workflow_map.json")
            shutil.copy2(variant_verifier / "oracle" / "SKILL.md", oracle_ws / "artifacts" / "SKILL.md")
            shutil.copy2(variant_verifier / "oracle" / "codex_triage.toml", oracle_ws / "artifacts" / "codex_triage.toml")
            shutil.copy2(variant_verifier / "oracle" / "automation_proposal.md", oracle_ws / "artifacts" / "automation_proposal.md")
            shutil.copy2(variant_verifier / "oracle" / "mapping_note.md", oracle_ws / "artifacts" / "mapping_note.md")
            oracle_result = score_workspace(variant["id"], oracle_ws)
        with tempfile.TemporaryDirectory(prefix="csm_shortcut_") as tmp:
            shortcut_ws = make_shortcut_workspace(variant, ws, Path(tmp))
            shortcut_result = score_workspace(variant["id"], shortcut_ws)
        observed_scores[variant["id"]] = {
            "observed_empty_score": empty_result["P_benchmark"],
            "observed_oracle_score": oracle_result["P_benchmark"],
            "observed_shortcut_score": shortcut_result["P_benchmark"],
            "verifier_data": {
                "gold_workflow_sha256": sha256_file(variant_verifier / "gold_workflow.json"),
                "oracle_input_sha256": sha256_file(variant_verifier / "oracle" / "workflow_input.json"),
                "oracle_workflow_map_sha256": sha256_file(variant_verifier / "oracle" / "workflow_map.json"),
                "workspace_manifest_sha256": sha256_file(variant_verifier / "workspace_manifest.json"),
            },
            "workspace_trees": workspace_manifest["readonly_tree_hashes"],
        }

    write_json(FAMILY / "manifest.lock.json", build_manifest_lock(observed_scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
