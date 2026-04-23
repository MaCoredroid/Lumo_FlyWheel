# `runbook-code-reconciliation` Task Spec

**Track:** 02 — Codebase Understanding
**Family id:** `runbook-code-reconciliation`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt (canonical)

The on-call runbook for generating the daily release preview drifted after a CLI rename and one config-path change. Operators now report that the documented commands no longer work reliably. Reconcile the runbook against the real repo behavior.

Use the bundle-local code and bundle-local CLI help as the source of truth. README fragments are evidence, not authority. One helper script still supports a legacy alias for backwards compatibility, but the runbook must not present that alias as the primary operator path.

Produce exactly these deliverables:

- a patch to `docs/runbooks/release_preview.md`
- `artifacts/verification_notes.md`
- `artifacts/deploy_note.md`
- `artifacts/reconciliation_facts.json`

`reconciliation_facts.json` must contain these exact keys:

- `primary_entrypoint`
- `legacy_alias`
- `current_flag`
- `deprecated_flag`
- `current_env_var`
- `deprecated_env_var`
- `evidence_commands`

Field-shape requirements:

- `primary_entrypoint` and `legacy_alias` are entrypoint commands only, without appended default flags or config paths.
- `evidence_commands` must include these exact minimum commands verbatim:
  - `python src/release_preview/cli.py --help`
  - `python src/release_preview/cli.py generate --help`
  - `python scripts/release_preview_helper.py build-preview --help`
  - `pytest -q tests/test_release_preview_cli.py`

`artifacts/verification_notes.md` must contain exactly these section headings:

- `## Checked directly`
- `## Inferred from code`
- `## Remaining caveats`

The notes must list the exact commands actually run and must explicitly say when conflicting README prose was overruled by code or live help.

Edit only the four files above. Do not modify code, tests, helper scripts, configs, or README fragments.

## Scenario Type

`docs_from_codebase` — a documentation-and-evidence repair where the correct operator path must be recovered from code and live help instead of copied from stale prose.

## Required Surfaces

- `shell`
- `apply_patch`
- local pytest execution
- CLI help inspection
- deterministic file inspection

No network, browser, MCP, or subagents are needed for the benchmark itself. The family-local probe harness may use `codex exec` to calibrate the family.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
.env.example
Makefile
configs/release_preview.toml
docs/runbooks/release_preview.md
README_fragments/current_path.md
README_fragments/legacy_path.md
src/release_preview/cli.py
src/release_preview/__init__.py
scripts/release_preview_helper.py
tests/test_release_preview_cli.py
artifacts/reconciliation_facts.schema.json
release_context/                 # v4+ only
incident_context/               # v5 only
```

The only writable surfaces are:

- `docs/runbooks/release_preview.md`
- `artifacts/verification_notes.md`
- `artifacts/deploy_note.md`
- `artifacts/reconciliation_facts.json`

Everything else is immutable. Any edit outside those four files is an integrity violation.

## Variant Ladder

### `v1-clean-baseline`

The code and CLI help are internally consistent. The runbook still recommends the legacy helper alias plus the deprecated config flag and env var. One README fragment is current and one is stale.

- Primary stress: recover the live path from code instead of the familiar alias.
- Honest failure mode: patch only the main command and leave the stale flag or env var behind.

### `v2-noisy-distractor`

V1 plus a second prose fragment that mixes the current entrypoint with the deprecated env var. A shallow “split the difference” harmonization pass is wrong.

- Primary stress: reject stale prose even when it looks almost current.
- Honest failure mode: combine the new entrypoint with the old config surface.

### `v3-dirty-state`

V2 plus a half-updated runbook draft that keeps both the legacy alias and the current CLI path as acceptable “for now”. The helper alias still works, which makes that compromise tempting.

- Primary stress: close the loop cleanly instead of preserving dual-path ambiguity.
- Honest failure mode: leave the legacy alias in the primary steps because it is still compatible.

### `v4-multi-corpus-objective`

V3 plus `release_context/` that emphasizes operator speed and “minimal doc churn”. The authoritative code path is unchanged: the runbook still needs the current CLI and current config surface.

- Primary stress: resist objective drift toward the compatibility helper.
- Honest failure mode: recommend the legacy helper alias because it is “fast enough”.

### `v5-recovery-in-thread`

V4 plus `incident_context/` documenting a prior bad fix that removed helper compatibility text and incorrectly claimed the alias no longer existed. The current task is still docs-only: describe the current primary path accurately without inventing a code behavior change.

- Primary stress: distinguish “deprecated” from “removed”.
- Honest failure mode: write a deploy note that claims the helper alias or deprecated env var stopped working.

### Ladder Monotonicity Target

Calibration target for GPT-5.4/high after hardening:

- `v1-clean-baseline`: `25`
- `v2-noisy-distractor`: `23`
- `v3-dirty-state`: `19`
- `v4-multi-corpus-objective`: `15`
- `v5-recovery-in-thread`: `10`

Family mean target: `18.4`, inside the Layer A freeze window `[15, 25]`.

## Expected Deliverables

- A runbook patch that names the live primary entrypoint.
- A verification note that clearly separates direct checks from code inferences and records exact commands.
- A deploy note that explains what operators should do now without claiming a code behavior change.
- A structured facts artifact that exactly separates current vs deprecated names.

## Visible Checks

```bash
pytest -q tests/test_release_preview_cli.py
```

The visible slice validates only the CLI contract:

- the current entrypoint is `python src/release_preview/cli.py generate`
- the visible primary flag is `--config`
- the helper alias still exists for compatibility

Visible success is intentionally insufficient for a passing score.

## Hidden Checks

Hidden checks are deterministic and code-backed. No LLM judge is used anywhere in this family.

They validate:

- the runbook uses the current entrypoint, current flag, and current env var
- the runbook does not present the legacy alias as the default flow
- `reconciliation_facts.json` matches the authoritative code surfaces exactly
- `verification_notes.md` contains direct command checks, code-backed inferences, and a bounded caveat
- `deploy_note.md` describes documentation/instruction drift only
- no immutable slice was modified

## Trusted Final-State Checks

- only the four writable surfaces may change
- `src/`, `scripts/`, `tests/`, `README_fragments/`, `configs/`, `.env.example`, and `Makefile` remain untouched
- no `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim
- no network marker

## 100-Point Breakdown

| Sub-metric | Points | Source |
| --- | ---: | --- |
| Runbook path / flag / env correctness | 35 | deterministic hidden checks |
| Facts artifact exactness | 20 | deterministic hidden checks |
| Verification notes quality | 20 | deterministic hidden checks |
| Deploy note accuracy | 15 | deterministic hidden checks |
| Evidence discipline | 10 | deterministic hidden checks |

## Partial-Credit Ceilings

- `no_direct_verification` ≤ `20`
  Trigger: no direct CLI help or pytest command is recorded.
- `facts_unvalidated` ≤ `25`
  Trigger: facts file is internally consistent but not backed by direct checks.
- `legacy_primary_path` ≤ `30`
  Trigger: the runbook keeps the helper alias or deprecated flag/env in the primary flow.
- `docs_harmonization_only` ≤ `15`
  Trigger: the runbook reads like a README merge instead of a code-grounded reconciliation.
- `invented_behavior_change` ≤ `10`
  Trigger: the deploy note claims the alias or deprecated env var was removed rather than deprecated.

## Red-Team Traps

1. Copy the closest README fragment instead of checking the CLI help.
2. Treat helper compatibility as proof that the legacy alias is still the right default.
3. Update the runbook but hand-wave the facts artifact.
4. Write verification notes that only describe searches, not checked commands.
5. Claim a code or compatibility change that the task did not make.

## Quality Gate

- Oracle repair must score at least `90`.
- Empty / untouched workspace must score `0`.
- Legacy-primary shortcut must score at most `30`.
- Delete-tests adversarial run must raise integrity and zero higher milestones.

## Saturation And Renewal Plan

Per Layer B readiness, this family is renewed when `mean P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:

1. add a V6 where helper compatibility and `.env.example` drift in different directions
2. add a V7 where a generated help excerpt is stale but code and runtime behavior are current
3. retire V1 once it becomes purely mechanical and promote V2 as the new floor
