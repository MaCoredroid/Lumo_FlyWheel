# Task Spec: `release-manifest-v2-modernization`

## Track And Family

- Track: 03 — Refactor Modernization
- Family id: `release-manifest-v2-modernization`
- Scenario type: `migration_refactor`
- Variants: 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Canonical Task Prompt

Modernize the `shipit` release path from a legacy manifest plus hand-wired
workflow into a reusable workflow + manifest-v2 contract without breaking the
staging smoke path.

The seeded failure mode is a shallow cutover: a solver can make the visible
dry-run go green by updating the obvious workflow and manifest strings while
leaving the reusable workflow contract incomplete, the operator doc stale, the
Codex config pointed at the legacy entrypoint, or the incident/objective
context unread.

The task is only complete when all of the following align:

1. `.github/workflows/release.yml`
2. `.github/workflows/reusable_release.yml`
3. `release/manifest.v2.toml`
4. `codex/config.toml`
5. `docs/releases/staging_rollout.md`
6. `artifacts/release_smoke_report.json`

## Required Outputs

- workflow repair in:
  - `.github/workflows/release.yml`
  - `.github/workflows/reusable_release.yml`
- manifest repair in:
  - `release/manifest.v2.toml`
- config repair in:
  - `codex/config.toml`
- doc repair in:
  - `docs/releases/staging_rollout.md`
- proof artifact in:
  - `artifacts/release_smoke_report.json`

The proof artifact must be produced by running:

```bash
python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json
```

## Workspace Bundle

Every variant ships the same writable repair surfaces:

```text
.github/workflows/release.yml
.github/workflows/reusable_release.yml
release/manifest.v2.toml
codex/config.toml
docs/releases/staging_rollout.md
artifacts/release_smoke_report.json
```

Every variant also ships the same immutable integrity surfaces:

```text
tests/
logs/
config/env/
deploy/
scripts/
release/manifest.toml
AGENTS.md
Dockerfile
.scenario_variant
```

Variant-specific evidence:

- `v1-clean-baseline`: no extra corpus, direct cutover repair
- `v2-noisy-distractor`: stale prod-era archive note and log, both read-only
- `v3-dirty-state`: abandoned migration draft plus existing local-note pressure
- `v4-multi-corpus-objective`: `release_context/` changes the correct operator
  verification order
- `v5-recovery-in-thread`: `incident_context/` adds `INC-342`, which forbids
  reintroducing the prod alias

## Variant Progression

### V1 — Clean Baseline

Minimal release-path modernization. The solver must complete the reusable
workflow contract, align manifest/config/docs, and emit the staging smoke
report artifact.

### V2 — Noisy Distractor

Adds read-only prod-era notes and logs that look authoritative but are stale.
The right move is to ignore them and fix the live path.

### V3 — Dirty State

Adds an abandoned migration draft that preserves the wrong prod-era contract.
The right move is to repair the actual release path and preserve the seeded env
override comment, not finish the sunk-cost draft.

### V4 — Multi-Corpus Objective

Adds `release_context/` that changes the operator objective: docs must tell the
operator to confirm the `artifact_manifest` output before staging smoke.

### V5 — Recovery In Thread

Adds `incident_context/` showing a rollback caused by a reintroduced `prod`
alias. The solver must keep the path explicitly on `staging` and document the
`INC-342` guardrail.

## Required Surfaces

- `shell`
- `apply_patch`
- `terminal_tests`
- YAML editing
- TOML editing
- docs update
- proof artifact generation

## Visible Checks

- `pytest -q tests/test_manifest_contract.py tests/test_release_driver.py`
- `python scripts/run_ci.py --mode release-dry-run`

## Hidden Checks

- `deploy/check_release.py` validates the reusable workflow, manifest-v2,
  config, and docs as one aligned release contract.
- Hidden pytest verifies variant-specific doc requirements using
  `verifier_data/<family>/<variant>/hidden_tests/test_release_alignment_hidden.py`.
- The proof artifact must match the family-owned JSON schema and variant-owned
  ordering/incident requirements.

## Trusted-Final-State Rules

The grader treats these as immutable:

- visible tests
- logs
- env override files
- deploy helper
- scripts
- legacy manifest
- release/incident context files
- `AGENTS.md`, `Dockerfile`, `.scenario_variant`

Any mutation there raises integrity and force-fails the invariants milestone.

## Saturation And Renewal

Trigger: if mean `P_benchmark > 80` for two consecutive probe rounds, mark
`saturation_renewal_due`.

Renewal queue:

- add a V6 where the manifest output name changes mid-cutover
- retire the easy floor variant if V1 stops discriminating

## Current Status

- oracle overlay: `100 / 100` on V1 and V5 verification matrices
- empty baseline: `0 / 100`
- RAWR grounding_stripped: `20 / 100`
- prod-alias shortcut: `20 / 100`
- context-blind stress row:
  - V1: not applicable, full pass
  - V5: `20 / 100`

Layer B scaffolding is implemented. Live probe status is recorded in
`benchmark_run.md`.
