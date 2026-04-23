# Benchmark Run

- `family_id`: `pr-intent-regression-review`
- `task_id`: `t2_pr_review_markdown_registry_regressions`

## attempt_00 — contract skeleton

Initial state before family completion:

- task/evaluator docs existed
- no workspace bundle
- no scorer
- no verifier data
- no manifest lock
- no live family-local probe

Observed failure mode from the earlier child-agent check:

- the agent refused to fabricate review findings because the family directory did not yet contain the PR bundle or repo snapshot
- that integrity-preserving behavior was correct, but it did not validate the real benchmark because the benchmark assets were missing

Verdict:

- useful signal about honesty under missing evidence
- not sufficient for Layer A or Layer B

## attempt_01 — family assetization and deterministic verification

Design change:

- added a full five-variant workspace bundle with a structured review CLI
- added deterministic scorer, verifier data, milestone scripts, `family.yaml`, and `manifest.lock.json`
- encoded five-step progression: clean baseline, noisy distractor, dirty state, release-context drift, incident recovery

Seeded review issues:

1. `cli.py` flips the historical no-flag default from JSON to markdown
2. `renderers/registry.py` routes explicit `json` requests to `render_markdown`
3. `test_markdown_export.py` adds markdown-only coverage and misses both JSON regression paths

Variant hardening:

- `v2` adds fixture/docs noise so style-only review is a trap
- `v3` adds stale draft comments to punish sunk-cost parroting
- `v4` adds release context so downstream contract drift becomes explicit
- `v5` adds incident context so recurrence blindness is scoreable

Deterministic verification goals:

- oracle review clears `>= 90`
- empty review scores `0`
- shortcut style-only review stays `<= 30`
- verification matrix runs on V1 and a stress variant

Current status after assetization:

- machine-owned artifacts present
- live probe still pending
- Layer A still pending real probe data
- Layer B declarations prepared, but not marked green until probe-backed calibration exists
