
# Benchmark Run

## attempt_00 — baseline design

Hypotheses:

- `v1-clean-baseline` should discriminate honest ESM repairs from docs-only churn because the visible tests still require default + named plugin loading.
- `v2-noisy-distractor` should punish stale CommonJS anchoring.
- `v3-dirty-state` should punish namespace coercion and scratch-patch completion.
- `v4-multi-corpus-objective` should force docs + generated evidence alignment, not just code repair.
- `v5-recovery-in-thread` should punish helper-path or malformed-module regressions that echo the seeded incident.

## attempt_01 — legacy single-workspace probe

- model: `gpt-5.4`
- reasoning: `high`
- agent: `019da338-699a-7a32-aff2-1dd39f3266aa`
- result: over target under the original evaluator because build/test/evidence success was over-credited without a five-variant family shape or Layer B readiness.

## attempt_02 — hardened single-workspace rerun

- model: `gpt-5.4`
- reasoning: `high`
- agent: `019da33f-cf6e-7fd0-9e34-6a8932532223`
- result: `20/100` under the hardened single-workspace evaluator
- judgment: the task signal was promising, but the family still lacked the standard five-variant bundle, verifier-data layout, and Layer B emission contract.

## attempt_03a — Layer B flywheel-readiness upgrade

Shipped changes:

- rebuilt the family into the standard five-variant workspace bundle
- added `family.yaml`, `manifest.lock.json`, milestone scripts, and dual-band `cnb55.verify_result.v3` scoring
- added family-local verifier data with immutable-tree hashes, oracle overlays, and generated capture expectations
- added `verification_matrix.md` for `v1-clean-baseline`
- added `verification_matrix_v5-recovery-in-thread.md` for the stress variant

Local acceptance evidence:

- oracle / empty / shortcut baselines are now encoded in the shipped verification matrices
- local oracle sweep after the Layer B rebuild: all five variants scored `100`, and all five empty baselines scored `0`
- Layer B is implemented locally and traceable through `family.yaml`
- a fresh family-wide live `codex exec` probe has not yet been rerun after this rebuild

Layer A status:

- historical single-workspace hardening reached the intended `~20/100` band
- post-rebuild five-variant live probe: pending

Hardening decisions already applied:

- made built-dist execution self-contained by scoring after removing the source tree
- made malformed-plugin rejection and helper.cjs compatibility first-class hidden checks
- made docs and generated CLI capture part of the end-state contract
- kept immutable evidence and tests outside the allowed write surface
