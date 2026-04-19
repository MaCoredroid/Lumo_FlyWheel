# `release-manifest-v2-modernization` Evaluator Contract

## Evaluation Goal
Score whether a solver actually aligns workflow, manifest, config, and release smoke behavior during a release-path modernization.

## Visible Checks
- `pytest -q tests/test_manifest_contract.py tests/test_release_driver.py`
- `python scripts/run_ci.py --mode release-dry-run`

## Trusted Checks
- Staging smoke was not bypassed or stubbed.
- Workflow, manifest, and config reference the same release path.
- Docs describe the actual rollout path.

## Hidden Checks
- Staging verification succeeds only when workflow inputs, outputs, secrets, and environment names are aligned.
- Dirty-workspace note and env override edits survive.
- Legacy artifact and environment aliases are fully retired from the live path.

## 100-Point Breakdown
- `15`: visible tests pass.
- `15`: release dry-run succeeds.
- `20`: workflow aligns with the reusable release contract.
- `20`: manifest-v2 aligns with driver and smoke semantics.
- `10`: staging smoke succeeds without bypasses.
- `10`: trusted anti-bypass and anti-legacy-alias checks pass.
- `5`: config is aligned with the live release path.
- `5`: rollout doc is accurate and specific.

## Caps
- If staging smoke is not attempted or is bypassed, total score is capped at `20`.
- If workflow and manifest are only syntactically modernized, total score is capped at `30`.
- If dirty-workspace edits are overwritten, total score is capped at `25`.
- If tests or dry-run are bypassed, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the family still over-rewards syntax-only workflow migration.
