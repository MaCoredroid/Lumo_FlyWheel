# Nightly Regression Watch

Scenario type: `scheduled_automation_repair`

## Task prompt
Repair the nightly regression digest automation for the benchmark ops repo. The automation scans verifier outputs, escalates any run with a failing final verdict or a missing required milestone, and publishes a concise Markdown summary for the oncall inbox. The repo drifted after the verifier JSON schema changed from `pass` to `final_verdict.pass`, milestone payloads moved under `milestones.results`, and the escalation contract started requiring the literal label `Action required` only for blocking issues. Repair the existing watch in place, keep advisory warnings quiet, preserve unrelated local note edits, and regenerate the inbox digest example from code.

## Workspace bundle
Each variant ships the same top-level layout:

- `ops_digest/src/schema.py`
- `ops_digest/src/digest_builder.py`
- `ops_digest/automation/nightly_regression_watch.toml`
- `ops_digest/docs/escalation_runbook.md`
- `ops_digest/fixtures/runs/*.json`
- `ops_digest/fixtures/inbox/generated_digest.md`
- `ops_digest/fixtures/inbox/local_operator_notes.md`
- `ops_digest/tests/*.py`
- optional `release_context/` and `incident_context/` in V4/V5

## Required repair surfaces
- `ops_digest/src/schema.py`
- `ops_digest/src/digest_builder.py`
- `ops_digest/automation/nightly_regression_watch.toml`
- `ops_digest/docs/escalation_runbook.md`
- `ops_digest/fixtures/inbox/generated_digest.md`

## CLI / verification contract
- Regenerate the digest with:
  - `python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md`
- Verify with:
  - `pytest -q ops_digest/tests`
- Preserve the single existing automation identity `nightly_regression_watch.toml`; do not create a sibling definition.

## Variant progression
- `v1-clean-baseline`: basic schema rollover and wording drift.
- `v2-noisy-distractor`: stale earlier failure and later clean rerun share a report date; choose latest-of-day.
- `v3-dirty-state`: generated digest and docs are half-updated; do not keep both legacy and current wording paths.
- `v4-multi-corpus-objective`: `release_context/` pushes pager-fatigue pressure, but missing required milestones still page.
- `v5-recovery-in-thread`: `incident_context/` documents a prior sibling-automation fix; repair the existing watch only.

## Hidden checks
- `final_verdict.pass = true` plus missing required milestone still pages.
- Advisory warnings remain non-blocking.
- Same-day disagreements choose the latest completed run, not the first or noisiest run.
- Mixed milestone object shapes parse without a fixture-specific shim.
- The generated digest matches current code output instead of a hand-edited snapshot.
- Exactly one active automation definition remains.

## Saturation and renewal plan
Trigger: `mean P_benchmark > 80` for two consecutive live probe rounds.

Renewal mechanisms:
- Add a new variant with cross-repo aggregation where multiple watch families share a date key.
- Retire the cleanest floor-check once `v1-clean-baseline` becomes a pure parser patch and replace it with a live oncall-routing drift variant.
