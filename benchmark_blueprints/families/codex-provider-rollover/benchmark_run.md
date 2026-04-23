# Benchmark Run

## Attempt 00 — design-only stub

- Context: this family existed only as a design shell (`task_spec.md`,
  `evaluator_contract.md`, `benchmark_run.md`, family-local skill, benchmark
  config).
- Recorded pre-implementation solver result: `21/100`.
- What that number meant: a solver could describe the right repair shape, but
  there was no runnable workspace bundle, scorer, verifier data, or manifest.
- Outcome: design intent existed, but neither Layer A nor Layer B was actually
  implemented.

## Attempt 01 — family-owned bundle + verifier implementation

### Scope landed

- five runnable workspace variants under `workspace_bundle/`
- family scorer: `verifiers/codex-provider-rollover/score_provider_rollover.py`
- family generator: `verifiers/codex-provider-rollover/regen_family.py`
- family verification-matrix runner:
  `verifiers/codex-provider-rollover/run_verification_matrix.py`
- per-variant oracle + hidden-fixture + manifest data under
  `verifier_data/codex-provider-rollover/`
- family manifest lock and Layer-B declaration

### Commands run for this attempt

```bash
python3 verifiers/codex-provider-rollover/regen_family.py
python3 verifiers/codex-provider-rollover/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/codex-provider-rollover/verification_matrix.md
python3 verifiers/codex-provider-rollover/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/codex-provider-rollover/verification_matrix_v5-recovery-in-thread.md
python3 - <<'PY'
# copied each variant to a temp workspace, overlaid verifier_data/<variant>/oracle/,
# ran verifiers/codex-provider-rollover/score_provider_rollover.py, and checked
# that each oracle scored 100 with no ceilings applied
PY
```

### Local verification results

`v1-clean-baseline` matrix:

- Oracle: `100`, `M_training=1.0000`, integrity `0`, pass `True`
- Empty: `0`, integrity `0`, ceilings `no_visible_green`, `legacy_default_selected`, `proxy_route_incorrect`, `continuity_not_exact`, `config_rewritten`, `docs_unaligned`
- RAWR grounding_stripped: `25`, ceilings `no_visible_green`, `docs_unaligned`
- Pick-ceiling: `15`, ceilings `no_visible_green`, `legacy_default_selected`, `proxy_route_incorrect`, `continuity_not_exact`, `config_rewritten`, `docs_unaligned`
- Top1-wrong: `30`, ceiling `config_rewritten`
- Delete-tests adversarial: `0`, integrity `1`

`v5-recovery-in-thread` matrix:

- Oracle: `100`, `M_training=1.0000`, integrity `0`, pass `True`
- Empty: `0`, integrity `0`, ceilings `no_visible_green`, `legacy_default_selected`, `proxy_route_incorrect`, `continuity_not_exact`, `config_rewritten`, `docs_unaligned`, `rollback_blind`
- RAWR grounding_stripped: `25`, ceilings `no_visible_green`, `docs_unaligned`, `rollback_blind`
- Pick-ceiling: `15`, ceilings `no_visible_green`, `legacy_default_selected`, `proxy_route_incorrect`, `continuity_not_exact`, `config_rewritten`, `docs_unaligned`
- Top1-wrong: `30`, ceiling `config_rewritten`
- Delete-tests adversarial: `0`, integrity `1`

All five oracle workspaces were then spot-checked through the family scorer and
each variant returned `100` with no ceilings applied:

- `v1-clean-baseline`
- `v2-noisy-distractor`
- `v3-dirty-state`
- `v4-multi-corpus-objective`
- `v5-recovery-in-thread`

### Layer status after attempt 01

- Layer B: implemented locally. The family now has five runnable workspace
  variants, deterministic scorer + wrapper, family manifest lock, milestone
  scripts, hidden negative smoke fixtures, and both required verification
  matrices (`verification_matrix.md` and
  `verification_matrix_v5-recovery-in-thread.md`).
- Layer A: still pending a counted whole-family live `codex exec` probe across
  all five variants. No live probe was launched in this turn.

### Honest next step

Run the family-local whole-family live probe against the committed bundle to
measure real per-variant means and decide whether the calibration lands in the
target `[15, 25]` band or needs hardening. This step is intentionally still
pending.
