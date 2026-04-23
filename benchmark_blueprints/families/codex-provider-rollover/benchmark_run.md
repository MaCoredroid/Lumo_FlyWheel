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

## Attempt 02 — counted whole-family live `codex exec` probe

### Why this attempt exists

- Reviewer follow-up required a real whole-family live probe after the scaffold
  commit.
- A family-local runner was added at
  `verifiers/codex-provider-rollover/run_live_probe.py` so the family can run
  the 5-variant probe without touching shared probe infrastructure.
- A one-run V1 smoke (`attempt_02a_smoke_v1`) confirmed the runner and scorer
  path before the counted probe. It scored `40` with `docs_unaligned`.

### Commands run

```bash
python3 verifiers/codex-provider-rollover/run_live_probe.py --attempt attempt_02a_smoke_v1 --n 1 --model gpt-5.4 --reasoning-effort high --timeout 900 --variants v1-clean-baseline
python3 verifiers/codex-provider-rollover/run_live_probe.py --attempt attempt_02 --n 3 --model gpt-5.4 --reasoning-effort high --timeout 900
python3 scripts/probe_report.py benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/probe_runs.jsonl --probe-run-id 20260423T193329Z --emit-json | tee benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/probe_report.txt
```

### Probe artifacts

- run ledger: `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/probe_runs.jsonl`
- text report: `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/probe_report.txt`
- metadata: `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/probe_meta.json`
- per-run verifier outputs / diffs / logs / workspaces:
  `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/<variant>/run_0{1,2,3}/`

### Whole-family live results

Probe run id: `20260423T193329Z`

| Variant | Scores | Mean | Stdev | Ceiling hits |
| --- | --- | ---: | ---: | --- |
| `v1-clean-baseline` | `[40, 40, 40]` | `40.00` | `0.00` | `docs_unaligned x3` |
| `v2-noisy-distractor` | `[35, 35, 35]` | `35.00` | `0.00` | `canary_anchoring x3` |
| `v3-dirty-state` | `[100, 30, 30]` | `53.33` | `40.41` | `template_overwrite x2` |
| `v4-multi-corpus-objective` | `[45, 45, 45]` | `45.00` | `0.00` | `maintenance_profile_drift x3` |
| `v5-recovery-in-thread` | `[100, 100, 100]` | `100.00` | `0.00` | `-` |

### Layer A gate values

- `family_mean = 54.67` vs required `[15, 25]` -> `FAIL`
- `max_variant_mean = 100.00` vs required `<= 40` -> `FAIL`
- `min_variant_mean = 35.00` vs required `<= 10` -> `FAIL`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/- 3` -> `FAIL`
  - `v2-noisy-distractor (35.0) < v3-dirty-state (53.3)` beyond `+/-3.0`
  - `v4-multi-corpus-objective (45.0) < v5-recovery-in-thread (100.0)` beyond `+/-3.0`

### Verification matrix status at the time of live probe

The scorer and manifests used for the live probe matched the family-local
Layer B matrices generated in attempt 01:

- `benchmark_blueprints/families/codex-provider-rollover/verification_matrix.md`
  - Oracle `100`, Empty `0`, RAWR `25`, Pick-ceiling `15`, Top1-wrong `30`, Delete-tests `0` with integrity `1`
- `benchmark_blueprints/families/codex-provider-rollover/verification_matrix_v5-recovery-in-thread.md`
  - Oracle `100`, Empty `0`, RAWR `25`, Pick-ceiling `15`, Top1-wrong `30`, Delete-tests `0` with integrity `1`

### Spot-check diagnosis

- `v5-recovery-in-thread/run_01` scored `100` because the child solver really
  completed the task: it selected `responses_proxy`, preserved the tuning
  block, strengthened the smoke to validate request and response
  `previous_response_id`, rejected store/provider drift, and documented the
  rollback context. See:
  - `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/v5-recovery-in-thread/run_01/verify_result.json`
  - `benchmark_blueprints/families/codex-provider-rollover/report/attempt_02/v5-recovery-in-thread/run_01/workspace.diff`
- `v3-dirty-state/run_01` also scored `100` by avoiding the template rewrite
  trap and preserving the tuning block while fixing the smoke generically. Runs
  02 and 03 hit `template_overwrite` and were capped at `30`.
- `v1` and `v2` runs generally repaired the runtime and hidden smoke behavior,
  but missed the exact variant doc keywords, so they landed at the documented
  ceiling values rather than passing.

### Outcome

- Layer B remains implemented and now has counted live-probe evidence.
- Layer A is failed, not pending. The first real whole-family probe is too easy
  for the target band: capable solvers can complete the provider rollover
  task, and the current residual failures are mostly doc-keyword misses.
- No post-probe hardening was applied in this attempt. A legitimate next
  attempt should redesign the family around stronger hidden runtime dimensions
  or retire/widen the calibration target; simply lowering caps further would
  turn real successful repairs into artificial failures.
