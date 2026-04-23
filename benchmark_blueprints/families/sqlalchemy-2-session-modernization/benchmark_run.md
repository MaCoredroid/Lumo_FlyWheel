# Benchmark Run

## attempt_00 — baseline design

- Original family existed only as a single loose workspace plus shallow docs.
- No five-variant ladder, no family-local scorer, no verifier data, no
  manifests, no Layer-B declaration.
- Hypothesis: visible worker/admin failures alone would not give an honest
  `~20/100` signal because the bundle did not yet encode batch, context, or
  integrity pressure.

## attempt_01 — pre-rebuild hardening snapshot

- Earlier family-local notes already documented a hardened visible/hidden split
  that targeted `20/100` for a shallow GPT-5.4/high solver.
- That work remained partial because the shipped family layout still lacked the
  CNB-55 five-variant bundle and Layer-B artifacts.

## attempt_02 — family rebuild for CNB-55 shape

- Replaced the single runnable bundle with five explicit variants:
  `v1-clean-baseline` through `v5-recovery-in-thread`.
- Added immutable context files for stale-shim noise, dirty local edits,
  release-objective drift, and rollback incident recovery.
- Added family-local scorer, hidden tests, oracle file set, milestone scripts,
  and variant-local verifier-data directories.
- Rewrote `task_spec.md` and `evaluator_contract.md` to match the new family
  shape and narrow write allowlist.

## attempt_03 — regeneration / acceptance status

- Ran `scripts/regen_family.py` end to end.
- Generated per-variant `workspace_manifest.json`, `gold_fix.json`, and
  `manifest.lock.json`.
- Observed baseline scores on all five variants:
  - oracle: `100`
  - untouched / empty: `0`
  - query-only shortcut: `20`
- Generated verification matrices:
  - `verification_matrix.md` for `v1-clean-baseline`
  - `verification_matrix_v5.md` for `v5-recovery-in-thread`
- Stress-matrix spot check:
  - V5 `Visible-only repair` lands at `35` with `batch_atomicity_missing`
    still firing, which is the intended shape for the batch / incident stress
    row.
- Live probe status: pending.
- Next gate: run a real `codex exec` family probe and decide whether Layer A is
  already honest enough to keep or still needs another hardening pass.
