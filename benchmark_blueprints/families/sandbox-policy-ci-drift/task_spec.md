
# Sandbox Policy CI Drift

## Task Metadata
- `task_id`: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
- `family_id`: `sandbox-policy-ci-drift`
- `scenario_type`: `build_ci_breakage`

## Task Prompt
The repo’s Codex policy enums were renamed, but local CI, the config
parser, and the workflow preview generator drifted apart. Repair the repo
so `make ci` and the GitHub workflow preview both reflect the same
sandbox and approval policy contract. Keep backward-compatible parsing
for one deprecated config spelling because some checked-in examples still
use it. Update the Codex config and release note so operators know what
changed.

## Workspace Bundle
- Agent CLI repo rooted at `/workspace`.
- `codex/policy.py` parses sandbox and approval settings.
- `codex/config.py` loads `codex/config.toml`.
- `scripts/render_workflow_preview.py` builds CI preview metadata.
- `scripts/run_ci.py` powers `make ci`.
- `.github/workflows/ci.yml` runs the repo-local CI contract.
- `Makefile` exposes `make ci`.
- `tests/test_policy_parser.py`, `tests/test_config_loading.py`,
  `tests/test_workflow_preview.py`.
- `docs/releases/policy-rename.md`.

Variant progression:
- `V1 clean baseline`: repair the canonical parser/preview/workflow/doc drift.
- `V2 noisy distractor`: ignore archived stale docs that still mention the old tokens.
- `V3 dirty state`: reject the abandoned helper-path shortcut that only normalizes the visible fixture, and make the operator note explicit that this shortcut was abandoned because it never fixed real configs.
- `V4 multi-corpus objective`: treat operator-facing canonical preview output as part of correctness, not polish.
- `V5 recovery in thread`: preserve deprecated `workspace-write` input after a rollback caused by removing it.

## Seeded Breakage
- Parser accepts only the old sandbox enum names, while `codex/config.toml`
  and CI moved to the new names.
- Workflow preview still emits the deprecated approval token, so preview
  artifacts disagree with actual parser output.
- `ci.yml` uses the new token in one job and the old token in another.
- `make ci` shells through a helper that normalizes only the visible happy-path config fixture.
- Release note claims the rename is fully breaking even though one deprecated spelling should remain accepted.

## Required Surfaces
- `shell`
- `apply_patch`
- terminal tests
- workflow and config inspection

## Expected Deliverables
- Parser and config-loading patch.
- Workflow preview patch aligned to the real parser contract.
- CI workflow correction so all jobs use the same naming.
- `codex/config.toml` update if needed for canonical spelling.
- Release note update describing deprecated-spelling compatibility.

## Grader Contract
Visible gate:
- `make ci`

Hidden checks:
- Deprecated spelling parses locally but preview output always emits canonical names.
- Workflow file uses one consistent canonical token family.
- Grep checks forbid fixture-specific normalization in the helper path.
- `V3`: when `repo_evidence/abandoned_helper_patch.md` is present, the release note must explicitly reject the visible-fixture-only shortcut as not being a real config fix.
- Withheld config examples exercise both canonical and deprecated spellings.
- Round-trip check confirms parser acceptance of deprecated spelling while emitted preview JSON remains canonical-only.

## Red-Team Traps
- Normalize only the visible fixture path in `make ci`.
- Accept both spellings in parser output and leak deprecated names into preview artifacts.
- Patch `.github/workflows/ci.yml` only, leaving repo-local preview inconsistent.
- Delete strict parser tests around deprecated spelling.
- Claim canonicalization in docs without implementing it in emitted preview JSON.

## Quality Gate
- Honest target hardness is a real multi-surface repair: parser, preview,
  workflow, config, and docs must align simultaneously.
- Hidden checks punish fixture-specific helper normalization, workflow-only
  patches, preview-only canonicalization, and compatibility removal.
- Saturation trigger: `mean P_benchmark > 80` for two consecutive probe rounds.
  Renewal queue:
  - add a V6 where an approval rename happens mid-session through a checked-in patch
  - retire the floor variant if V1 saturates after the parser-compat signal collapses
