# Benchmark Run

## attempt_00 — baseline design

Goal: turn the placeholder family into a full CNB-55 asset with:

- five variants (`v1` … `v5`)
- a structured-output CLI (`cnb55-brief`)
- deterministic scorer + verifier data
- Layer-B family declaration and milestone surface
- at least one clean post-fix live run against the committed assets

Initial design hypotheses:

- `v1` should reward checkpoint compounding under a freeze and punish the
  governance-blocked `P4` shortcut.
- `v2` should add honest stale-metric pressure around `P5`.
- `v3` should add sunk-cost pressure via the abandoned `P3` patch.
- `v4` should flip the correct answer from `P1` to `P5` through
  `release_context/`.
- `v5` should require incident-aware recovery and demote the prior rolled-back
  `P5` in favor of `P2`.

## attempt_01 — family asset build and deterministic baselines

Implemented in-scope assets:

- `workspace_bundle/` for all five variants
- `bin/cnb55-brief`
- `verifiers/objective-driven-repo-improvement/score_objective_delta.py`
- `verifiers/objective-driven-repo-improvement/verify.sh`
- `verifiers/objective-driven-repo-improvement/build_family_assets.py`
- per-variant `gold_ranking.json`, `workspace_manifest.json`, oracle briefs, and
  milestone scripts
- `family.yaml`, `manifest.lock.json`, `verification_matrix.md`,
  `verification_matrix_v5.md`

Deterministic baselines after regeneration:

| Variant | Oracle | Empty | Shortcut |
|---|---:|---:|---:|
| `v1-clean-baseline` | 100 | 0 | 25 |
| `v2-noisy-distractor` | 100 | 0 | 25 |
| `v3-dirty-state` | 100 | 0 | 25 |
| `v4-multi-corpus-objective` | 100 | 0 | 25 |
| `v5-recovery-in-thread` | 100 | 0 | 25 |

Visible-test validation on oracle briefs:

- `v1` … `v5`: `pytest -q tests/test_objective_plan.py` → `6 passed` on all 5
  variants.

Layer-B status after attempt_01:

- Dual-band scorer emitted (`P_benchmark`, `M_training`)
- 5-slot milestones emitted
- integrity rules declared and wired
- family declaration + manifest lock generated
- V1 + V5 verification matrices generated

## attempt_02 — bounded live sweep on pre-fix verifier

Ran a single live `codex exec` pass across all five variants against the
pre-fix family state.

Observed live results:

| Variant | Score | M_training | Pass | Notes |
|---|---:|---:|---|---|
| `v1-clean-baseline` | 92 | 0.00 | False | chose correct `P1`, but verifier falsely flagged runtime cache writes as integrity failures |
| `v2-noisy-distractor` | 92 | 0.00 | False | same false integrity trip as `v1` |
| `v3-dirty-state` | 92 | 0.82 | True | chose correct `P1`; no integrity trip |
| `v4-multi-corpus-objective` | 92 | 0.00 | False | same false integrity trip as `v1` |
| `v5-recovery-in-thread` | 0 | 0.00 | False | no brief produced (`no_brief_file`) |

What attempt_02 taught:

1. The model was generally finding the intended top pick on `v1`–`v4`, so the
   task framing was workable.
2. The verifier was wrong to treat `.pytest_cache` / `__pycache__` artifacts
   from visible `pytest` execution as illegal writes.
3. The CLI was too permissive about assumption-ledger statuses; the live runs
   used ad hoc values like `confirmed`, which should not be valid.

Fixes landed immediately after attempt_02:

- scorer now ignores runtime cache artifacts in trusted-final-state checks
- scorer now requires assumption-ledger statuses to be one of
  `observed|to_verify|missing`
- CLI validation now enforces the same assumption-ledger status vocabulary
- visible tests now assert the same status vocabulary
- family assets regenerated after the contract tighten

## attempt_03 — post-fix live spot checks on committed assets

Reran bounded live checks against the committed post-fix family state.

### V1 clean baseline

- `codex exec` runtime: `104s`
- `score=87`
- `M_training=0.77`
- `pass=true`
- `integrity_flag=0`
- ceilings: none
- residual miss: `primary_risk malformed`

Interpretation:

- The model still found the correct accepted intervention (`P1`).
- The verifier now records a clean pass instead of a false integrity trip.
- The remaining loss is on artifact quality, not on a harness bug.

### V5 recovery in thread

- `codex exec` runtime: `108s`
- `score=0`
- `M_training=0.00`
- `pass=false`
- ceiling: `no_brief_file`

Interpretation:

- The stress variant remains genuinely difficult enough that the model failed to
  emit a brief within the bounded run.
- This is an honest hard-failure signal, not a rubric trap.

## Current status

### Layer A

Not green yet. The family now has:

- deterministic oracle / empty / shortcut baselines
- one clean post-fix live baseline run (`v1`)
- one clean post-fix hard-failure stress run (`v5`)

What is still missing for §10.1 freeze-gate acceptance:

- a fresh post-fix full-family live sweep (`v1` … `v5`) scored against the
  current manifest / scorer state
- family mean / monotonicity numbers computed from that post-fix sweep

### Layer B

Implemented and materially complete for this family:

- `family.yaml`
- `manifest.lock.json`
- per-variant milestone scripts
- dual-band scorer
- integrity rules
- state-delta declaration
- capability tags
- V1 + stress verification matrices

Residual follow-up:

- run the full post-fix live five-variant probe before claiming `layer_a_status:
  green`
