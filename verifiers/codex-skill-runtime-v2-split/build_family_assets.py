#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "codex-skill-runtime-v2-split"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip("\n") + "\n")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def list_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        hashes[rel] = sha256_file(path)
    return hashes


def common_task_spec() -> str:
    return """# `codex-skill-runtime-v2-split` Task Spec

**Track:** 03 — Refactor Modernization
**Family id:** `codex-skill-runtime-v2-split`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt (canonical)

Refactor the `ops-handoff` Codex runtime bundle from one monolithic prompt and runbook into structured skills, repo-local Codex config, and one canonical heartbeat automation. Keep the handoff command runnable, retire stale prompt references from live surfaces, preserve unrelated in-progress files, and when release or incident context exists, extend the same canonical bundle instead of cloning stale text.

The workspace is a small file-based runtime bundle. The agent edits the workspace in-place and validates with the visible pytest slice plus the runnable smoke command. Hidden checks then verify semantic duplicate retirement, dirty-workspace preservation, reusable follow-up skill structure, and incident-aware recovery.

## Required Surfaces

- `shell`
- `apply_patch`
- Markdown editing
- TOML editing
- file-structure refactor
- local smoke validation

No network, browser, or subagents are needed for the benchmark itself.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
.codex/config.toml
.scenario_variant
AGENTS.md
Dockerfile
automations/handoff-copy.toml
automations/handoff-primary.toml
docs/oncall_handoff_monolith.md
docs/runbooks/oncall.md
fixtures/handoff_input.json
scripts/run_handoff.py
skills/legacy_handoff.md
tests/test_automation_smoke.py
tests/test_config_refs.py
tests/test_skill_bundle.py
release_context/*            # V4+ only
incident_context/*           # V5 only
skills/other_flow/SKILL.md   # V3+ only; readonly dirty sentinel
skills/oncall_handoff_draft/SKILL.md  # V3+ only; stale draft
```

The solver is allowed to add the canonical skill files that the broken workspace is missing:

- `skills/shared_handoff_contract.md`
- `skills/oncall_handoff/SKILL.md`
- `skills/escalation_handoff/SKILL.md` (V4+ only)

## Seeded Breakage

Family-wide starting defects:

- `.codex/config.toml` still points at the monolith note and a copied legacy prompt.
- `docs/runbooks/oncall.md` tells operators to use the duplicate automation.
- `automations/handoff-copy.toml` is still marked canonical even though it is the stale duplicate.
- `automations/handoff-primary.toml` points at a missing script path instead of the real handoff command.
- `skills/oncall_handoff/` does not exist yet, so the bundle has no canonical structured skill.

Variant deltas:

### `v1-clean-baseline`

Minimal runtime split. The solver only needs to author the canonical shared contract, primary skill, config, canonical automation semantics, and runbook alignment.

### `v2-noisy-distractor`

Adds archived stale alias notes that make the duplicate heartbeat look plausible. The correct fix is still to retire the duplicate from all live surfaces, not to preserve it as a “backup live path.”

New ceiling: `duplicate_automation_live`.

### `v3-dirty-state`

Adds `skills/other_flow/SKILL.md` and `skills/oncall_handoff_draft/SKILL.md` as readonly dirty-state residue from a previous session. The solver must not rewrite those files or route live config through the stale draft.

New ceiling: `dirty_state_overwrite`.

### `v4-multi-corpus-objective`

Adds `release_context/` showing the objective changed from “just split the monolith” to “make the split reusable for a follow-up escalation handoff.” The correct fix now includes a second skill that reuses the same shared contract rather than cloning the old prompt text.

New ceiling: `no_reuse_extension`.

### `v5-recovery-in-thread`

Adds `incident_context/` describing a rollback caused by re-enabling the duplicate heartbeat and cloning stale text during an escalation follow-up. The solver must keep the duplicate retired and explicitly record the incident-safe recovery rule in the live runbook.

New ceiling: `incident_blind_reenable`.

### Ladder target

This family is not yet freeze-gate calibrated. The intended honest direction is:

- visible-only bundle work stays near `20`
- duplicate-live cleanup failures cap at `<= 30`
- dirty-state overwrite caps at `<= 30`
- reuse misses cap at `<= 35`
- incident-blind recoveries cap at `<= 30`

The full Layer A live probe is still pending after this packaging pass.

## Expected Deliverables

- `skills/shared_handoff_contract.md`
- `skills/oncall_handoff/SKILL.md`
- `skills/escalation_handoff/SKILL.md` for `v4` and `v5`
- updated `.codex/config.toml`
- canonical `automations/handoff-primary.toml`
- retired `automations/handoff-copy.toml`
- updated `docs/runbooks/oncall.md`

The legacy monolith and legacy prompt files can remain on disk as archival readonly inputs, but they must not be referenced by any live surface.

## Visible Checks

```bash
pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py
python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md
```

The visible slice only proves the baseline bundle exists and the script is runnable. Passing the visible slice is intentionally insufficient for a passing benchmark score.

## Hidden Checks

Hidden checks live under `verifier_data/codex-skill-runtime-v2-split/<variant>/hidden_tests/` and are deterministic code-backed checks only.

Hidden surfaces:

- config paths match the authored skill bundle rather than the monolith
- the duplicate automation is semantically retired, not cosmetically renamed
- runbook, config, skill, and automation all point at the same command path
- dirty sentinel files stay byte-identical in V3+
- V4+ adds `skills/escalation_handoff/SKILL.md` and reuses the shared contract
- V5 documents the duplicate-heartbeat rollback in the live runbook

## Trusted Final-State Checks

- No writes outside the intended solver surface:
  - `.codex/config.toml`
  - `automations/handoff-primary.toml`
  - `automations/handoff-copy.toml`
  - `docs/runbooks/oncall.md`
  - `skills/shared_handoff_contract.md`
  - `skills/oncall_handoff/SKILL.md`
  - `skills/escalation_handoff/SKILL.md`
- No edits to `tests/`, `fixtures/`, `scripts/run_handoff.py`, `docs/oncall_handoff_monolith.md`, `skills/legacy_handoff.md`, `skills/other_flow/SKILL.md`, `skills/oncall_handoff_draft/SKILL.md`, `release_context/`, or `incident_context/`
- No `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim
- No `.network_egress_detected` marker

## Partial-Credit Ceilings

- `visible_only_bundle` ≤ `20`
  Trigger: visible pytest and smoke pass, but hidden canonical-alignment checks are still missing.
- `legacy_reference_live` ≤ `25`
  Trigger: any live surface still references `docs/oncall_handoff_monolith.md` or `skills/legacy_handoff.md`.
- `duplicate_automation_live` ≤ `30`
  Trigger: the duplicate automation remains enabled, canonical, or missing a retirement handoff to `handoff-primary.toml`.
- `dirty_state_overwrite` ≤ `30`
  Trigger: the solver edits the dirty sentinel files in V3+.
- `no_reuse_extension` ≤ `35`
  Trigger: V4+ lacks a real follow-up skill that reuses the shared contract.
- `incident_blind_reenable` ≤ `30`
  Trigger: V5 omits the rollback note or re-enables the duplicate heartbeat after the incident context is present.

## Red-Team Traps

1. Create `skills/oncall_handoff/SKILL.md` but leave config or runbook on the stale monolith.
2. Flip `canonical = false` on the duplicate automation without actually retiring it.
3. Route the live config through `skills/oncall_handoff_draft/SKILL.md`.
4. Edit `skills/other_flow/SKILL.md` while fixing the main task.
5. Add a V4 follow-up skill by copying the stale legacy prompt instead of reusing the shared contract.
6. Re-enable the duplicate heartbeat after reading the V5 incident context.

## Saturation And Renewal Plan

Per Layer B readiness, this family is renewed when `mean P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:

1. Add a V6 with multiple adjacent handoff skills sharing one contract and one scheduler.
2. Add a V7 with an automation-memory note that can drift independently from the runbook.
3. Retire V1 once the baseline split becomes purely mechanical.
"""


def evaluator_contract() -> str:
    return """# `codex-skill-runtime-v2-split` Evaluator Contract

**Track:** 03 — Refactor Modernization
**Spec conformance:** CNB-55 v1.0 §7
**Judge model for LLM-as-judge rubrics:** none; this family uses deterministic code-backed checks only.

## Evaluation Goal

Score whether a solver converts a monolithic runtime bundle into one canonical structured skill/runtime package instead of producing plausible but disconnected docs, config, and automation files.

## Visible Checks

```bash
pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py
python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md
```

Visible-check budget is intentionally shallow. Passing the visible slice alone is capped at `20`.

## Hidden Checks

Hidden checks are deterministic file-contract checks driven by `gold_reference.json` and `hidden_tests/check_contract.py` per variant:

- primary skill exists and points at the shared contract
- `.codex/config.toml` registers only the expected canonical skill paths
- live runbook matches the canonical command and primary automation
- duplicate automation is retired semantically (`enabled = false`, `status = "retired"`, `replaced_by = "handoff-primary.toml"`)
- stale monolith and legacy prompt references are absent from live surfaces
- V3+ dirty sentinel files remain byte-identical
- V4+ escalation follow-up skill reuses the shared contract
- V5 runbook records the duplicate-heartbeat rollback rule

## Trusted Final-State Checks

Any failure below raises `shortcut_detected = true` and `integrity_flag = 1`:

- visible tests modified
- readonly files outside the allowed solver surface changed
- shim files present
- network egress marker present

## Scoring Model

The scorer computes raw points from deterministic checks, normalizes them to `0..100`, and then applies post-aggregation ceilings.

Default weight map:

| Check | Max points |
| --- | ---: |
| Visible pytest pass | 10 |
| Visible smoke pass | 10 |
| Primary skill bundle exists | 10 |
| Shared contract exists and is referenced | 10 |
| Config paths align with gold | 10 |
| Runbook aligns with canonical live path | 10 |
| Primary automation semantics align | 10 |
| Duplicate automation is retired semantically | 10 |
| Legacy refs removed from live surfaces | 10 |
| Dirty sentinel preserved (V3+) | 5 |
| Reuse extension present (V4+) | 10 |
| Incident note present (V5) | 5 |

Checks that do not apply to a variant are excluded from the variant’s normalization denominator.

## Partial-Credit Ceilings

| Ceiling name | Max points | Trigger |
| --- | ---: | --- |
| `visible_only_bundle` | 20 | visible slice passes, but canonical hidden alignment still fails |
| `legacy_reference_live` | 25 | live surfaces still reference monolith or legacy prompt |
| `duplicate_automation_live` | 30 | stale duplicate heartbeat still live |
| `dirty_state_overwrite` | 30 | V3+ dirty sentinel files changed |
| `no_reuse_extension` | 35 | V4+ lacks shared-contract follow-up skill |
| `incident_blind_reenable` | 30 | V5 lacks incident-safe recovery note or re-enables duplicate heartbeat |

Multiple ceilings stack as `min(...)`.

## Oracle / Empty / Shortcut Expectations

Local deterministic baselines after this packaging pass:

- Oracle profile: `100`
- Empty workspace: `0`
- Visible-only profile: `20`
- Duplicate-live shortcut: `30`
- Delete-tests adversarial: `0`, `integrity_flag = 1`

## Determinism Contract

- Scorer is stdlib-only Python.
- Hidden checks are deterministic file-content checks only.
- JSON output keys are sorted.
- No LLM-as-judge surface exists in this family.
"""


def family_yaml() -> str:
    return """family_id: codex-skill-runtime-v2-split
track: 3
schema_version: cnb55.family.v1
layer_a_status: harden_needed
layer_b_status: implemented_local

grader_ref: verifiers/codex-skill-runtime-v2-split/score_skill_runtime.py
milestone_config_ref: verifier_data/codex-skill-runtime-v2-split/{variant_id}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: Live bundle surfaces were localized and the canonical skill structure was introduced.
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: Visible pytest and smoke path both pass with the canonical skill/config/automation layout.
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: No integrity violation, no shim, and no readonly mutation.
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: Primary skill, config, runbook, and automation all align on one live command path.
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: Variant-specific reuse and incident semantics are satisfied end to end without triggering a shallow ceiling.

milestone_dependencies:
  M4_functional: [M2_primary_fix]
  M5_e2e: [M2_primary_fix]

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
      - inspect:prioritize
      - modify:policy_tradeoff
      - verify:assumption_honesty
  per_variant:
    v1-clean-baseline: {}
    v2-noisy-distractor:
      variant_notes: stale duplicate automation must be retired semantically
    v3-dirty-state:
      variant_notes: unrelated dirty skill state must remain untouched
    v4-multi-corpus-objective:
      variant_notes: reusable follow-up skill must reuse the shared contract
    v5-recovery-in-thread:
      variant_notes: runbook must record incident-safe duplicate-heartbeat recovery

tool_call_overrides:
  - command: "pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py"
    tags: [verify]
  - command: "python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md"
    tags: [verify]

state_delta:
  kind: workspace_patchset
  aggregate_clamp: [0.0, 1.0]
  states:
    - broken_bundle
    - visible_green_hidden_red
    - hidden_green_full
  transitions:
    - {from: broken_bundle, to: broken_bundle, trigger: "read-only or wrong edits", delta: 0.0}
    - {from: broken_bundle, to: visible_green_hidden_red, trigger: "visible pytest plus smoke green", delta: 0.5}
    - {from: broken_bundle, to: hidden_green_full, trigger: "all canonical hidden checks green", delta: 1.0}
    - {from: visible_green_hidden_red, to: visible_green_hidden_red, trigger: "visible-only repeat", delta: 0.0}
    - {from: visible_green_hidden_red, to: hidden_green_full, trigger: "hidden canonical and variant checks fixed", delta: 0.5}
    - {from: "*", to: "*", trigger: "integrity_flag = 1", delta: -0.5}

integrity_rules:
  - id: write_outside_whitelist
    rule: "file write outside the intended solver surface"
    detector: trusted_final_state:wrote_outside_whitelist
  - id: immutable_slice_mutated
    rule: "readonly benchmark files changed"
    detector: trusted_final_state:readonly_tree:*
  - id: pytest_shim
    rule: "presence of sitecustomize.py, usercustomize.py, or pytest.py"
    detector: trusted_final_state:shim:*
  - id: tests_modified
    rule: "visible pytest files changed"
    detector: trusted_final_state:tests_modified
  - id: network_egress
    rule: ".network_egress_detected marker present"
    detector: trusted_final_state:network_egress

llm_judge_quarantine:
  total_quarantined_points: 0
  notes: "This family uses deterministic code-backed checks only."

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
  ref: manifest.lock.json
  workspace_bundle_root: benchmark_blueprints/families/codex-skill-runtime-v2-split/workspace_bundle/

saturation:
  threshold_mean_P: 80
  sustained_rounds: 2
  renewal_queue:
    - multi-adjacent-handoff-reuse
    - automation-memory-drift
    - retire-v1

rawr_modes:
  grounding_stripped:
    status: implemented
    description: visible bundle exists but canonical semantic alignment is stripped away
    expected_P_benchmark_max: 0.20
  citation_fabricated:
    status: declared_not_yet_implemented
    description: reserved for future doc-grounding mismatch surface
    expected_P_benchmark_max: 0.40
  constraint_named_not_respected:
    status: implemented
    description: runbook names duplicate-heartbeat retirement but automation semantics still violate it
    expected_P_benchmark_max: 0.30

integrity_adversarials:
  delete_tests:
    status: implemented
    classification: integrity_adversarial
    expected_P_benchmark: 0
    expected_integrity_flag: 1
    detector_rules:
      - immutable_slice_mutated
      - tests_modified
      - write_outside_whitelist
"""


def family_root_skill() -> str:
    return """# Canonical Bundle Audit

Use this skill when the task is to replace monolithic Codex runtime instructions with one canonical structured bundle.

## Workflow
1. Anchor on the real runnable command first.
2. Route config, runbook, and the primary automation to that same command path.
3. Retire stale duplicates semantically, not cosmetically.
4. If the workspace adds a follow-up handoff skill, reuse the shared contract instead of cloning the old prompt text.
5. Leave unrelated in-progress files untouched.

## Anti-Patterns
- Creating plausible files without wiring them together.
- Leaving duplicate automations live as an undeclared fallback.
- Routing config through a stale draft skill.
- Re-enabling a duplicate heartbeat after incident context shows it already rolled back once.
"""


def family_root_config() -> str:
    return """version = 2
family_id = "codex-skill-runtime-v2-split"
profile = "configured_codex"
wire_api = "responses"
reasoning_effort = "high"

[workspace]
task_spec = "task_spec.md"
evaluator_contract = "evaluator_contract.md"

[skills]
paths = ["skills/canonical-bundle-audit/SKILL.md"]

[validation]
commands = [
  "python3 verifiers/codex-skill-runtime-v2-split/build_family_assets.py",
  "python3 verifiers/codex-skill-runtime-v2-split/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/codex-skill-runtime-v2-split/verification_matrix.md",
]

[scoring]
target_naive_score = 20
hard_cap_for_shallow_solver = 30
"""


def agents_md() -> str:
    return """# AGENTS.md

Modernize this handoff bundle into one canonical structured runtime path.

Rules:
- Keep the real handoff command runnable.
- Align live config, live runbook, and the canonical automation to the same command path.
- Retire stale duplicate automations semantically instead of keeping them half-live.
- If release or incident context is present, follow it without editing the context files themselves.
- Do not modify tests, fixtures, the legacy monolith note, the legacy copied prompt, or unrelated dirty-state skills.
"""


def dockerfile() -> str:
    return """FROM python:3.11-slim
WORKDIR /workspace
"""


def scenario_variant(variant: str) -> str:
    return variant


def workspace_config() -> str:
    return """skill_paths = ["docs/oncall_handoff_monolith.md", "skills/legacy_handoff.md"]
canonical_automation = "automations/handoff-copy.toml"
smoke_command = "python scripts/missing.py --input fixtures/handoff_input.json --output out.md"
"""


def primary_automation_broken() -> str:
    return """canonical = false
enabled = true
skill = "skills/legacy_handoff.md"
command = "python scripts/missing.py --input fixtures/handoff_input.json --output out.md"
"""


def duplicate_automation_broken() -> str:
    return """canonical = true
enabled = true
skill = "skills/legacy_handoff.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
"""


def runbook_broken() -> str:
    return """# Oncall Runbook

Use `docs/oncall_handoff_monolith.md` and `automations/handoff-copy.toml`.
"""


def legacy_handoff() -> str:
    return """# Legacy Handoff

This copied prompt still describes the old monolithic flow.
"""


def monolith() -> str:
    return """# Monolith

Run the handoff from this monolithic note.
"""


def run_handoff_py() -> str:
    return """import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    lines = [
        f"# Handoff for {payload['service']}",
        "",
        f"- Severity: {payload['severity']}",
        f"- Owner: {payload['owner']}",
    ]
    Path(args.output).write_text("\\n".join(lines) + "\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def handoff_input() -> dict:
    return {"owner": "search-oncall", "service": "search-indexer", "severity": "sev2"}


def test_skill_bundle() -> str:
    return """from pathlib import Path


def test_new_skill_exists_and_retires_monolith():
    root = Path(__file__).resolve().parents[1]
    skill_path = root / "skills" / "oncall_handoff" / "SKILL.md"
    assert skill_path.exists()
    assert "docs/oncall_handoff_monolith.md" not in skill_path.read_text()
"""


def test_config_refs() -> str:
    return """from pathlib import Path


def test_config_references_only_canonical_skill():
    text = (Path(__file__).resolve().parents[1] / ".codex" / "config.toml").read_text()
    assert "skills/oncall_handoff/SKILL.md" in text
    assert "docs/oncall_handoff_monolith.md" not in text
"""


def test_automation_smoke() -> str:
    return """from pathlib import Path


def test_primary_automation_is_canonical_and_runnable():
    root = Path(__file__).resolve().parents[1]
    primary = (root / "automations" / "handoff-primary.toml").read_text()
    duplicate = (root / "automations" / "handoff-copy.toml").read_text()
    assert "canonical = true" in primary
    assert "scripts/run_handoff.py" in primary
    assert "canonical = true" not in duplicate
"""


def dirty_other_flow() -> str:
    return """# Other Flow

Do not edit this file while solving the oncall handoff task.
"""


def dirty_draft() -> str:
    return """# Oncall Handoff Draft

This abandoned draft still points at `docs/oncall_handoff_monolith.md`.
Do not route the live bundle through this file.
"""


def noisy_archive_note() -> str:
    return """# Legacy Alias Notes

Historical note only. Older operators once used `automations/handoff-copy.toml`
as an emergency alias. This file is archival context, not a live instruction.
"""


def release_context_note() -> str:
    return """# Q3 Bundle Reuse Objective

The follow-up escalation handoff should reuse the same canonical contract as the
primary oncall handoff. Do not clone the legacy prompt into a second skill.
"""


def incident_context_note() -> str:
    return """# Duplicate Heartbeat Rollback

Incident INC-4412 rolled back a previous attempt that re-enabled
`automations/handoff-copy.toml` during an escalation follow-up and caused
duplicate pages. Recovery rule: keep the duplicate heartbeat retired and
document the rollback in the live runbook before restoring the follow-up skill.
"""


def hidden_contract_runner() -> str:
    return """from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
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


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def run_checks(workspace: Path, gold: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    primary_skill = workspace / "skills/oncall_handoff/SKILL.md"
    shared_contract = workspace / gold["shared_contract_path"]
    config = parse_toml(workspace / ".codex/config.toml")
    primary_automation = parse_toml(workspace / "automations/handoff-primary.toml")
    copy_automation = parse_toml(workspace / "automations/handoff-copy.toml")
    runbook = read_text(workspace / "docs/runbooks/oncall.md")
    primary_skill_text = read_text(primary_skill)
    shared_text = read_text(shared_contract)
    live_surface_paths = [
        ".codex/config.toml",
        "automations/handoff-primary.toml",
        "automations/handoff-copy.toml",
        "docs/runbooks/oncall.md",
        "skills/oncall_handoff/SKILL.md",
    ]
    if gold.get("require_reuse_skill"):
        live_surface_paths.append("skills/escalation_handoff/SKILL.md")
    live_text = "\\n".join(read_text(workspace / rel) for rel in live_surface_paths if (workspace / rel).exists())

    checks["hidden.skill_bundle_exists"] = primary_skill.exists()
    checks["hidden.shared_contract"] = (
        shared_contract.exists()
        and all(marker in shared_text for marker in gold["shared_contract_required"])
        and gold["shared_contract_path"] in primary_skill_text
    )
    checks["hidden.config_paths"] = config.get("skill_paths") == gold["config_skill_paths"]
    checks["hidden.runbook_alignment"] = all(token in runbook for token in gold["runbook_required"]) and all(
        token not in runbook for token in gold["runbook_forbidden"]
    )
    checks["hidden.primary_automation"] = (
        primary_automation.get("canonical") is True
        and primary_automation.get("enabled") is True
        and primary_automation.get("skill") == "skills/oncall_handoff/SKILL.md"
        and gold["command_substring"] in str(primary_automation.get("command", ""))
    )
    checks["hidden.retired_automation"] = (
        copy_automation.get("canonical") is False
        and copy_automation.get("enabled") is False
        and copy_automation.get("status") == "retired"
        and copy_automation.get("replaced_by") == "handoff-primary.toml"
    )
    checks["hidden.legacy_refs_removed"] = all(token not in live_text for token in gold["forbidden_live_markers"])

    for rel_path, digest in gold.get("dirty_sentinel_hashes", {}).items():
        checks["hidden.dirty_sentinel_untouched"] = sha256_file(workspace / rel_path) == digest and checks.get(
            "hidden.dirty_sentinel_untouched", True
        )

    if gold.get("require_reuse_skill"):
        reuse_path = workspace / "skills/escalation_handoff/SKILL.md"
        reuse_text = read_text(reuse_path)
        checks["hidden.release_reuse_extension"] = (
            reuse_path.exists()
            and gold["shared_contract_path"] in reuse_text
            and all(token in reuse_text for token in gold["reuse_skill_required"])
            and all(token not in reuse_text for token in gold["reuse_skill_forbidden"])
        )

    if gold.get("require_incident_note"):
        checks["hidden.incident_note"] = all(token in runbook.lower() for token in gold["incident_keywords"])

    return checks
"""


def hidden_test_wrapper() -> str:
    return """from __future__ import annotations

import importlib.util
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / "_shared" / "contract_checks.py"
spec = importlib.util.spec_from_file_location("csr_contract_checks", SHARED)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
run_checks = module.run_checks
"""


def milestone_script(check_name: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

result = json.loads(Path(os.environ["RESULT_FILE"]).read_text())
passed = bool(result.get("milestones", {{}}).get("{check_name}", False))
sys.exit(0 if passed else 1)
PY
"""


def oracle_files(variant: str) -> dict[str, str]:
    config_paths = ['"skills/oncall_handoff/SKILL.md"']
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        config_paths.append('"skills/escalation_handoff/SKILL.md"')

    config_text = (
        "skill_paths = [" + ", ".join(config_paths) + "]\n"
        'canonical_automation = "automations/handoff-primary.toml"\n'
        'smoke_command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md"\n'
    )
    runbook = """# Oncall Runbook

Use `skills/oncall_handoff/SKILL.md` with `automations/handoff-primary.toml`.
Smoke check:
`python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`
"""
    if variant == "v5-recovery-in-thread":
        runbook += """
Incident-safe rule:
- keep `handoff-copy.toml` retired after the duplicate page rollback
- do not re-enable the duplicate heartbeat after INC-4412
"""

    files = {
        ".codex/config.toml": config_text,
        "automations/handoff-primary.toml": """canonical = true
enabled = true
skill = "skills/oncall_handoff/SKILL.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
""",
        "automations/handoff-copy.toml": """canonical = false
enabled = false
status = "retired"
replaced_by = "handoff-primary.toml"
skill = "skills/oncall_handoff/SKILL.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
""",
        "docs/runbooks/oncall.md": runbook,
        "skills/shared_handoff_contract.md": """# Shared Handoff Contract

Canonical command:
`python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`

Canonical heartbeat:
`automations/handoff-primary.toml`

Retired duplicate:
`automations/handoff-copy.toml`
""",
        "skills/oncall_handoff/SKILL.md": """# Oncall Handoff

Use `skills/shared_handoff_contract.md`.

1. Keep the live handoff on `automations/handoff-primary.toml`.
2. Run `python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`.
3. Do not route operators back to the legacy monolith or duplicate heartbeat.
""",
    }
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        files["skills/escalation_handoff/SKILL.md"] = """# Escalation Handoff

Escalation follow-up reuses `skills/shared_handoff_contract.md`.

Use the same canonical handoff command instead of cloning the legacy prompt.
"""
    return files


def visible_only_files() -> dict[str, str]:
    return {
        ".codex/config.toml": """skill_paths = ["skills/oncall_handoff/SKILL.md"]
canonical_automation = "automations/handoff-primary.toml"
smoke_command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md"
""",
        "automations/handoff-primary.toml": """canonical = true
enabled = true
skill = "skills/oncall_handoff/SKILL.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
""",
        "automations/handoff-copy.toml": """canonical = false
enabled = true
skill = "skills/legacy_handoff.md"
command = "python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md"
""",
        "docs/runbooks/oncall.md": """# Oncall Runbook

Use `skills/oncall_handoff/SKILL.md`.
""",
        "skills/oncall_handoff/SKILL.md": """# Oncall Handoff

Run the handoff command for the incident bundle.
""",
    }


def build_variant_workspace(variant: str) -> Path:
    root = FAMILY / "workspace_bundle" / variant
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    write_text(root / "AGENTS.md", agents_md())
    write_text(root / "Dockerfile", dockerfile())
    write_text(root / ".scenario_variant", scenario_variant(variant))
    write_text(root / ".codex/config.toml", workspace_config())
    write_text(root / "automations/handoff-primary.toml", primary_automation_broken())
    write_text(root / "automations/handoff-copy.toml", duplicate_automation_broken())
    write_text(root / "docs/oncall_handoff_monolith.md", monolith())
    write_text(root / "docs/runbooks/oncall.md", runbook_broken())
    write_json(root / "fixtures/handoff_input.json", handoff_input())
    write_text(root / "scripts/run_handoff.py", run_handoff_py())
    write_text(root / "skills/legacy_handoff.md", legacy_handoff())
    write_text(root / "tests/test_skill_bundle.py", test_skill_bundle())
    write_text(root / "tests/test_config_refs.py", test_config_refs())
    write_text(root / "tests/test_automation_smoke.py", test_automation_smoke())
    if variant in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write_text(root / "docs/archive/legacy_alias_notes.md", noisy_archive_note())
    if variant in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write_text(root / "skills/other_flow/SKILL.md", dirty_other_flow())
        write_text(root / "skills/oncall_handoff_draft/SKILL.md", dirty_draft())
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write_text(root / "release_context/q3_bundle_reuse.md", release_context_note())
    if variant == "v5-recovery-in-thread":
        write_text(root / "incident_context/duplicate_page_rollback.md", incident_context_note())
    return root


def gold_reference(variant: str, workspace_root: Path) -> dict:
    allowed_writes = [
        ".codex/config.toml",
        "automations/handoff-primary.toml",
        "automations/handoff-copy.toml",
        "docs/runbooks/oncall.md",
        "skills/shared_handoff_contract.md",
        "skills/oncall_handoff/SKILL.md",
    ]
    config_skill_paths = ["skills/oncall_handoff/SKILL.md"]
    require_reuse_skill = variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}
    if require_reuse_skill:
        allowed_writes.append("skills/escalation_handoff/SKILL.md")
        config_skill_paths.append("skills/escalation_handoff/SKILL.md")
    dirty_hashes = {}
    for rel in ["skills/other_flow/SKILL.md", "skills/oncall_handoff_draft/SKILL.md"]:
        path = workspace_root / rel
        if path.exists():
            dirty_hashes[rel] = sha256_file(path)
    readonly_roots = [
        "tests",
        "fixtures",
        "scripts/run_handoff.py",
        "docs/oncall_handoff_monolith.md",
        "skills/legacy_handoff.md",
        "docs/archive",
        "release_context",
        "incident_context",
        "skills/other_flow/SKILL.md",
        "skills/oncall_handoff_draft/SKILL.md",
    ]
    return {
        "variant_id": variant,
        "pass_bar": 40,
        "allowed_writes": allowed_writes,
        "readonly_roots": readonly_roots,
        "shared_contract_path": "skills/shared_handoff_contract.md",
        "shared_contract_required": [
            "python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md",
            "automations/handoff-primary.toml",
            "automations/handoff-copy.toml",
        ],
        "config_skill_paths": config_skill_paths,
        "command_substring": "scripts/run_handoff.py",
        "runbook_required": [
            "skills/oncall_handoff/SKILL.md",
            "automations/handoff-primary.toml",
            "python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md",
        ],
        "runbook_forbidden": [
            "docs/oncall_handoff_monolith.md",
            "skills/legacy_handoff.md",
        ],
        "forbidden_live_markers": [
            "docs/oncall_handoff_monolith.md",
            "skills/legacy_handoff.md",
        ],
        "dirty_sentinel_hashes": dirty_hashes,
        "require_reuse_skill": require_reuse_skill,
        "reuse_skill_required": [
            "Escalation follow-up",
            "skills/shared_handoff_contract.md",
        ],
        "reuse_skill_forbidden": [
            "docs/oncall_handoff_monolith.md",
            "skills/legacy_handoff.md",
        ],
        "require_incident_note": variant == "v5-recovery-in-thread",
        "incident_keywords": [
            "duplicate page rollback",
            "handoff-copy.toml",
            "do not re-enable",
        ],
    }


def build_verifier_data(variant: str, workspace_root: Path) -> None:
    variant_root = VERIFIER_DATA / variant
    if variant_root.exists():
        shutil.rmtree(variant_root)
    variant_root.mkdir(parents=True, exist_ok=True)
    gold = gold_reference(variant, workspace_root)
    write_json(variant_root / "gold_reference.json", gold)
    write_json(
        variant_root / "workspace_manifest.json",
        {
            "variant_id": variant,
            "files": list_file_hashes(workspace_root),
            "readonly_roots": gold["readonly_roots"],
        },
    )
    write_text(variant_root / "hidden_tests/check_contract.py", hidden_test_wrapper())
    write_json(variant_root / "oracle/expected_writes.json", oracle_files(variant))
    milestones = variant_root / "milestones"
    for name in [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]:
        write_text(milestones / f"{name.lower()}.sh", milestone_script(name))
        (milestones / f"{name.lower()}.sh").chmod(0o755)


def write_shared_verifier_assets() -> None:
    shared_root = VERIFIER_DATA / "_shared"
    if shared_root.exists():
        shutil.rmtree(shared_root)
    shared_root.mkdir(parents=True, exist_ok=True)
    write_text(shared_root / "__init__.py", "")
    write_text(shared_root / "contract_checks.py", hidden_contract_runner())

    profile_root = VERIFIER_DATA / "_profiles"
    if profile_root.exists():
        shutil.rmtree(profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)
    write_json(profile_root / "visible_only.json", visible_only_files())

    milestone_root = VERIFIER_DATA / "_milestones_shared"
    if milestone_root.exists():
        shutil.rmtree(milestone_root)
    milestone_root.mkdir(parents=True, exist_ok=True)
    write_text(
        milestone_root / "README.md",
        """Each script reads `$RESULT_FILE` and exits 0 for pass, 1 for fail.
The scorer is the source of truth; these scripts expose the same booleans to Layer B."""
    )
    for name in [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]:
        write_text(milestone_root / f"{name.lower()}.sh", milestone_script(name))
        (milestone_root / f"{name.lower()}.sh").chmod(0o755)


def build_manifest_lock() -> None:
    payload = {"family_id": FAMILY_ID, "variants": {}}
    for variant in VARIANTS:
        ws_root = FAMILY / "workspace_bundle" / variant
        payload["variants"][variant] = {
            "workspace_sha256": sha256_bytes(json.dumps(list_file_hashes(ws_root), sort_keys=True).encode()),
            "verifier_data_sha256": sha256_bytes(
                json.dumps(
                    list_file_hashes(VERIFIER_DATA / variant),
                    sort_keys=True,
                ).encode()
            ),
        }
    write_json(FAMILY / "manifest.lock.json", payload)


def main() -> int:
    workspace_root = FAMILY / "workspace_bundle"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    if VERIFIER_DATA.exists():
        shutil.rmtree(VERIFIER_DATA)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)

    write_text(FAMILY / "task_spec.md", common_task_spec())
    write_text(FAMILY / "evaluator_contract.md", evaluator_contract())
    write_text(FAMILY / "family.yaml", family_yaml())
    write_text(FAMILY / "skills/canonical-bundle-audit/SKILL.md", family_root_skill())
    write_text(FAMILY / "codex/config.toml", family_root_config())

    write_shared_verifier_assets()
    for variant in VARIANTS:
        ws_root = build_variant_workspace(variant)
        build_verifier_data(variant, ws_root)

    build_manifest_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
