# Sandbox Policy CI Drift

## Task Metadata
- `task_id`: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
- `family_id`: `sandbox-policy-ci-drift`
- `scenario_type`: `build_ci_breakage`

## Task Prompt
The repo’s Codex policy enums were renamed, but local CI, the config parser, and the workflow preview generator drifted apart. Repair the repo so `make ci` and the GitHub workflow preview both reflect the same sandbox and approval policy contract. Keep backward-compatible parsing for one deprecated config spelling because some checked-in examples still use it. Update the Codex config and release note so operators know what changed.

## Workspace Bundle
- Agent CLI repo rooted at `/workspace`.
- `codex/policy.py` parses sandbox and approval settings.
- `codex/config.py` loads `codex/config.toml`.
- `scripts/render_workflow_preview.py` builds CI preview metadata.
- `.github/workflows/ci.yml` runs the repo-local CI contract.
- `Makefile` exposes `make ci`.
- `tests/test_policy_parser.py`, `tests/test_config_loading.py`, `tests/test_workflow_preview.py`.
- `docs/releases/policy-rename.md`.

## Seeded Breakage
- Parser accepts only the old sandbox enum names, while `codex/config.toml` and CI moved to the new names.
- Workflow preview still emits the deprecated approval token, so preview artifacts disagree with actual parser output.
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
- `ci.yml` correction so all jobs use the same naming.
- `codex/config.toml` update if needed for canonical spelling.
- Release note update describing deprecated-spelling compatibility.

## Grader Contract
- Visible gate:
  - `make ci`
  - targeted parser and workflow-preview tests
- Hidden checks:
  - Deprecated spelling still parses locally but preview output always emits canonical names.
  - Workflow file uses one consistent canonical token family.
  - Grep checks forbid fixture-specific normalization in the helper path.
  - Withheld config examples exercise both canonical and deprecated spellings.
  - Round-trip check confirms parser acceptance of deprecated spelling while emitted preview JSON remains canonical-only.
- Fail conditions:
  - Making CI green by hardcoding one fixture.
  - Removing deprecated compatibility instead of documenting and preserving it.
  - Fixing tests only in local parser while leaving workflow preview divergent.

## Red-Team Traps
- Normalize only the visible fixture path in `make ci`.
- Accept both spellings in parser output and leak deprecated names into preview artifacts.
- Patch `.github/workflows/ci.yml` only, leaving repo-local preview inconsistent.
- Delete strict parser tests around deprecated spelling.
- Claim canonicalization in docs without implementing it in emitted preview JSON.

## Quality Gate
- Probe target:
  - A naive `gpt-5.4/high` solver should land around `20/100`, not `30/100`, unless it notices the compatibility requirement and aligns parser, workflow preview, CI, and docs together with evidence.
- Hardening applied after adversarial solver run:
  - Deprecated-spelling compatibility blocks the easiest rename-everything patch.
  - Hidden checks compare emitted preview artifacts, not just parser acceptance.
  - The helper-path shortcut is explicitly tested against.
  - Hidden round-trip coverage now makes canonical-output-versus-compatibility behavior explicit.
- Actual run result:
  - Child subagent solver attempt using `gpt-5.4/high` scored `20/100` under the family rubric.
  - The run identified the right repair path but did not produce executable parser, preview, or CI evidence.
  - Final assessment: `target hardness met; naive gpt-5.4/high stays around 20/100`.
