# `codex-skill-runtime-v2-split` Task Spec

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
