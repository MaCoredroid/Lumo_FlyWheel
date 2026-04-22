# Evaluator Contract: Nightly Regression Watch

## Purpose
Evaluate whether the solver repairs the nightly regression watch end to end, not just the visible markdown example.

## Visible checks (`20` points)
- `20` points: `pytest -q ops_digest/tests` passes.

## Hidden checks (`80` points)
- `20` points: parser/classification logic handles `final_verdict.pass`, `milestones.results`, mixed milestone shapes, and missing-required-milestone blockers.
- `15` points: latest-of-day dedupe selects the newest completed run per `report_date` and suppresses stale duplicate outputs.
- `10` points: automation prompt contract repaired in place with exactly one active watch definition.
- `10` points: regenerated digest artifact matches current code output.
- `10` points: runbook command and wording contract match the repaired behavior.
- `15` points: hidden pytest robustness passes across advisory-only, mixed-shape, and same-day reversal scenarios.

## Named ceilings
- `docs_only_repair` cap `25`: runbook or prompt repaired, but code files still fail the schema/dedupe contract.
- `missing_required_milestone_ignored` cap `30`: final verdict pass + required milestone miss is still treated as healthy.
- `latest_of_day_wrong` cap `35`: same-day reversals keep the wrong run or duplicate stale output.
- `stale_generated_digest` cap `20`: generated digest example is hand-edited or stale relative to current code.
- `second_automation_created` cap `20`: solver adds a sibling automation instead of repairing the existing watch.

## Shortcut defenses
- The scorer recomputes the generated digest from current workspace code and compares it with the checked-in example.
- Hidden scenarios mutate milestone object shapes and same-day ordering.
- Read-only hashes cover visible tests, run fixtures, local notes, and release/incident context.
