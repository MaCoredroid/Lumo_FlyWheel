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

## attempt_02a — first live probe and scorer diagnosis

Command:

```bash
python3 verifiers/pr-intent-regression-review/run_live_probe.py \
  --attempt attempt_02a_live_probe_initial \
  --n 1 \
  --timeout 1200
```

Observed scores:

| Variant | scores | mean |
|---|---|---:|
| `v1-clean-baseline` | `[20]` | `20.00` |
| `v2-noisy-distractor` | `[20]` | `20.00` |
| `v3-dirty-state` | `[85]` | `85.00` |
| `v4-multi-corpus-objective` | `[20]` | `20.00` |
| `v5-recovery-in-thread` | `[20]` | `20.00` |

Layer A gate values:

- `family_mean = 33.00`
- `max_variant_mean = 85.00`
- `min_variant_mean = 20.00`
- `monotonic_ok = false`
- `monotonic_breaks = ['v2-noisy-distractor (20.00) < v3-dirty-state (85.00) beyond +/-3.0']`

Design read:

- `v3` was too easy because the scorer still let a two-code-issue review land high without the missing-regression-test blocker.
- `v5` was not hard enough because recurrence blindness still capped too softly.

Family-local hardening that followed:

- added `missed_second_seeded_issue` cap at `20`
- added `missing_regression_guard` cap at `35`
- added `sunk_cost_finish` cap at `20` on `v3`
- tightened `incident_blind_repeat` cap from `30` to `10` on `v5`
- documented the changes in `task_spec.md` and `evaluator_contract.md`
- regenerated assets with:

```bash
python3 verifiers/pr-intent-regression-review/build_family_assets.py
```

## attempt_02b — invalid counted run discarded

Command:

```bash
python3 verifiers/pr-intent-regression-review/run_live_probe.py \
  --attempt attempt_02b_live_probe_counted \
  --n 3 \
  --timeout 1200
```

This run was intentionally discarded before completion. Early rows showed false highs caused by the scorer still rewarding one-major-only reviews too much after the first hardening pass. The run was stopped with:

```bash
pkill -f 'pr-intent-regression-review_20260423T232207Z' || true
```

Discard reason:

- partial counted rows were not trustworthy for calibration
- only the post-change full-family rerun should count

## attempt_02d — counted whole-family live probe after scorer hardening

Commands:

```bash
python3 verifiers/pr-intent-regression-review/build_family_assets.py
python3 verifiers/pr-intent-regression-review/run_live_probe.py \
  --attempt attempt_02d_live_probe_counted_final \
  --n 3 \
  --timeout 1200
python3 verifiers/pr-intent-regression-review/run_verification_matrix.py \
  --variant v1-clean-baseline \
  --out benchmark_blueprints/families/pr-intent-regression-review/verification_matrix.md
python3 verifiers/pr-intent-regression-review/run_verification_matrix.py \
  --variant v5-recovery-in-thread \
  --out benchmark_blueprints/families/pr-intent-regression-review/verification_matrix_v5.md
```

Probe artifacts:

- `benchmark_blueprints/families/pr-intent-regression-review/report/attempt_02d_live_probe_counted_final/probe_runs.jsonl`
- `benchmark_blueprints/families/pr-intent-regression-review/report/attempt_02d_live_probe_counted_final/probe_summary.json`
- `benchmark_blueprints/families/pr-intent-regression-review/report/attempt_02d_live_probe_counted_final/probe_report.txt`

Per-variant numeric results:

| Variant | scores | mean | stdev | mean `M_training` |
|---|---|---:|---:|---:|
| `v1-clean-baseline` | `[20, 20, 10]` | `16.67` | `5.77` | `0.1852` |
| `v2-noisy-distractor` | `[20, 20, 20]` | `20.00` | `0.00` | `0.2222` |
| `v3-dirty-state` | `[20, 20, 20]` | `20.00` | `0.00` | `0.2222` |
| `v4-multi-corpus-objective` | `[20, 20, 20]` | `20.00` | `0.00` | `0.2222` |
| `v5-recovery-in-thread` | `[10, 10, 10]` | `10.00` | `0.00` | `0.1111` |

Layer A gate values from the counted run:

- `family_mean = 17.33`
- `family_mean_ok = true`
- `max_variant_mean = 20.00`
- `max_variant_ok = true`
- `min_variant_mean = 10.00`
- `hard_variant_ok = true`
- `monotonic_ok = false`
- `monotonic_breaks = ['v1-clean-baseline (16.67) < v2-noisy-distractor (20.00) beyond +/-3.0']`
- `all_pass = false`

What the counted run says honestly:

- the family is now in the intended low band on mean/max/floor
- `v5` is stably hard at `10`, so the incident-recurrence ceiling is doing real work
- `v2`/`v3`/`v4` flatten at `20`, which means the current scorer family has a mechanical low-band plateau for this model after the second-core-issue and missing-test guards fire
- the remaining Layer A miss is narrow and specific: `v1` averages `16.67`, but `v2` averages `20.00`, exceeding the allowed `+/-3` monotonic tolerance by `0.33`

Why this follow-up stops here:

- the user requested the real whole-family live `codex exec` probe/calibration chain and exact recorded results
- that chain is now complete and family-local artifacts are on disk
- any additional hardening would invalidate the counted run and require another full family-local reprobe
- the remaining failure mode is now concrete and localized enough for the next hardening loop to target precisely instead of guessing

Post-probe verification matrix outputs:

- `verification_matrix.md` (`v1-clean-baseline`)
  - Oracle: `P_benchmark=100`, `M_training=1.0000`, `integrity=0`, `pass=true`
  - Empty: `0`, `0.0000`, `0`, `false`
  - RAWR grounding_stripped: `20`, `0.2222`, `0`, ceilings `unanchored_or_wrong_findings,no_cross_surface_linkage,missed_second_seeded_issue`
  - Style-only shortcut: `10`, `0.1111`, `0`, ceilings `style_only_review,generated_fixture_fixation,unanchored_or_wrong_findings,no_cross_surface_linkage,ci_complacency,missed_second_seeded_issue`
  - One-major-only: `20`, `0.2222`, `0`, ceiling `missed_second_seeded_issue`
  - Delete-tests adversarial: `100`, `0.0000`, `1`, `false`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`)
  - Oracle: `P_benchmark=100`, `M_training=1.0000`, `integrity=0`, `pass=true`
  - Empty: `0`, `0.0000`, `0`, `false`
  - RAWR grounding_stripped: `10`, `0.1111`, `0`, ceilings `unanchored_or_wrong_findings,no_cross_surface_linkage,release_context_blindness,incident_blind_repeat,missed_second_seeded_issue`
  - Style-only shortcut: `10`, `0.1111`, `0`, ceilings `style_only_review,generated_fixture_fixation,unanchored_or_wrong_findings,no_cross_surface_linkage,ci_complacency,release_context_blindness,incident_blind_repeat,missed_second_seeded_issue`
  - One-major-only: `10`, `0.1111`, `0`, ceilings `incident_blind_repeat,missed_second_seeded_issue`
  - Delete-tests adversarial: `100`, `0.0000`, `1`, `false`
