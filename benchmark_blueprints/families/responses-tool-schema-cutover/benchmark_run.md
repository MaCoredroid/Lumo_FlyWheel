# Benchmark Run

## Attempt 00 — design-only calibration stub

- Context: this family existed only as a design shell (`task_spec.md`, `evaluator_contract.md`, `benchmark_run.md`, family-local skill, benchmark config).
- Recorded pre-implementation solver result: `20/100`
- What that number meant: a child solver could describe the right repair shape (`call_id` joins, replay chronology, config/doc updates) but had no runnable family bundle, no scorer, no oracle, and no verifier data.
- Outcome: the design intent was reasonable, but neither Layer A nor Layer B was actually implemented.

## Attempt 01 — family-owned bundle + verifier implementation

### Scope landed

- five runnable workspace variants under `workspace_bundle/`
- family scorer: `verifiers/responses-tool-schema-cutover/score_responses_cutover.py`
- family generator: `verifiers/responses-tool-schema-cutover/regen_family.py`
- family verification-matrix runner: `verifiers/responses-tool-schema-cutover/run_verification_matrix.py`
- per-variant oracle + hidden-test + manifest data under `verifier_data/responses-tool-schema-cutover/`
- family manifest lock and Layer-B declaration

### Commands run

```bash
python3 verifiers/responses-tool-schema-cutover/regen_family.py
python3 verifiers/responses-tool-schema-cutover/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/responses-tool-schema-cutover/verification_matrix.md
python3 verifiers/responses-tool-schema-cutover/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/responses-tool-schema-cutover/verification_matrix_v5-recovery-in-thread.md
```

### Local verification results

`v1-clean-baseline` matrix:

- Oracle: `100`, `M_training=1.0000`, integrity `0`, pass `True`
- Empty: `0`, integrity `0`, ceiling `no_submission`
- RAWR grounding_stripped: `50`, ceiling `contract_drift`
- Adapter-only shortcut: `19`, ceilings `no_visible_green`, `tool_name_only_join`, `adapter_or_reducer_gap`
- Chronology-blind fix: `20`, ceilings `no_visible_green`, `tool_name_only_join`
- Delete-tests adversarial: `0`, integrity `1` via `write_outside_whitelist`, `pytest_shim`

`v5-recovery-in-thread` matrix:

- Oracle: `100`, `M_training=1.0000`, integrity `0`, pass `True`
- Empty: `0`, integrity `0`, ceiling `no_submission`
- RAWR grounding_stripped: `50`, ceiling `contract_drift`
- Adapter-only shortcut: `19`, ceilings `no_visible_green`, `tool_name_only_join`, `adapter_or_reducer_gap`
- Chronology-blind fix: `20`, ceilings `no_visible_green`, `tool_name_only_join`
- Delete-tests adversarial: `0`, integrity `1` via `write_outside_whitelist`, `pytest_shim`

### Layer status after attempt 01

- Layer B: implemented locally. The family now emits `cnb55.verify_result.v3`, has 5-slot milestones, integrity rules, state-delta declarations, verifier manifests, milestone scripts, and both required verification matrices (`verification_matrix.md` and `verification_matrix_v5-recovery-in-thread.md`).
- Layer A: materially advanced but not closed in this rollout. Oracle / empty / shortcut baselines are now in place, but the live family-wide `codex exec` probe loop across all five variants was intentionally not launched in this turn.

### Honest next step

Run the family probe loop against the implemented bundle to measure actual per-variant means and verify the §10.1 freeze-gate math with live solver attempts. That step is deliberately left unlaunched here.

## Attempt 02 — whole-family live `codex exec` probe

### Commands run

```bash
python3 verifiers/responses-tool-schema-cutover/run_live_probe.py --attempt attempt_02 --n 3 --model gpt-5.4 --reasoning-effort high
python3 scripts/probe_report.py benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/probe_runs.jsonl --probe-run-id 20260422T214841Z --emit-json | tee benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/probe_report.txt
```

### Probe artifacts

- run ledger: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/probe_runs.jsonl`
- text report: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/probe_report.txt`
- metadata: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/probe_meta.json`
- per-run verifier outputs / diffs / logs: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/<variant>/run_0{1,2,3}/`

### Whole-family live results

Probe run id: `20260422T214841Z`

| Variant | Scores | Mean | Stdev | Ceiling hits |
| --- | --- | ---: | ---: | --- |
| `v1-clean-baseline` | `[50, 0, 0]` | `16.67` | `28.87` | `contract_drift x3`, `no_test_regression_guard x2` |
| `v2-noisy-distractor` | `[50, 60, 60]` | `56.67` | `5.77` | `contract_drift x1`, `no_test_regression_guard x3` |
| `v3-dirty-state` | `[60, 50, 50]` | `53.33` | `5.77` | `contract_drift x2`, `no_test_regression_guard x3` |
| `v4-multi-corpus-objective` | `[50, 60, 0]` | `36.67` | `32.15` | `contract_drift x2`, `no_test_regression_guard x2` |
| `v5-recovery-in-thread` | `[60, 50, 60]` | `56.67` | `5.77` | `contract_drift x1`, `no_test_regression_guard x3` |

### Layer A gate values

- `family_mean = 44.00` vs required `[15, 25]` -> `FAIL`
- `max_variant_mean = 56.67` vs required `<= 40` -> `FAIL`
- `min_variant_mean = 16.67` vs required `<= 10` -> `FAIL`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/- 3` -> `FAIL`
  - `v1-clean-baseline (16.7) < v2-noisy-distractor (56.7)` beyond `+/- 3.0`
  - `v4-multi-corpus-objective (36.7) < v5-recovery-in-thread (56.7)` beyond `+/- 3.0`

### Spot-check diagnosis from live artifacts

- `v2-noisy-distractor/run_02` scored `60` because the solver repaired adapter/reducer/config/docs and passed visible + hidden replay checks, but it did not strengthen either visible regression test file. The run therefore stayed capped by `no_test_regression_guard`. See:
  - `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/v2-noisy-distractor/run_02/verify_result.json`
  - `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/v2-noisy-distractor/run_02/workspace.diff`
- `v1-clean-baseline/run_02` and `v4-multi-corpus-objective/run_03` collapsed to `0` because the solver wrote helper pytest files outside the whitelist (`tests/conftest.py` / `conftest.py`). Those runs were correctly forced to `integrity_flag = 1` by `write_outside_whitelist`. See:
  - `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/v1-clean-baseline/run_02/verify_result.json`
  - `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_02/v4-multi-corpus-objective/run_03/verify_result.json`

### Verification matrix status at the time of live probe

The scorer and manifests used for the live probe matched the family-local Layer B matrices already generated in attempt 01:

- `benchmark_blueprints/families/responses-tool-schema-cutover/verification_matrix.md`
  - Oracle `100`, Empty `0`, RAWR `50`, Adapter-only `19`, Chronology-blind `20`, Delete-tests `0` with integrity `1`
- `benchmark_blueprints/families/responses-tool-schema-cutover/verification_matrix_v5-recovery-in-thread.md`
  - Oracle `100`, Empty `0`, RAWR `50`, Adapter-only `19`, Chronology-blind `20`, Delete-tests `0` with integrity `1`

### Outcome

- Layer B remains implemented and evidenced locally.
- Layer A is still open after the first real whole-family live probe. The measured signal is not a missing-artifact problem anymore; it is a calibration problem driven mainly by `no_test_regression_guard` / `contract_drift` ceilings plus integrity variance from out-of-whitelist pytest helper writes.

## Attempt 03 — post-attestation-fix whole-family live rerun

### Why this attempt exists

- Family-local metadata changed in `family.yaml` to make Layer A / Layer B attestation and RAWR taxonomy canonical and honest for review.
- Per the current review-round rule, any family-local change requires a fresh whole-family live `codex exec` probe after the fix, even if the scorer and workspace bundle are otherwise unchanged.

### Commands run

```bash
python3 verifiers/responses-tool-schema-cutover/run_live_probe.py --attempt attempt_03 --n 3 --model gpt-5.4 --reasoning-effort high
python3 scripts/probe_report.py benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/probe_runs.jsonl --probe-run-id 20260423T000148Z --emit-json | tee benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/probe_report.txt
python3 - <<'PY'
import json
from pathlib import Path
import yaml
p = Path('benchmark_blueprints/families/responses-tool-schema-cutover/family.yaml')
data = yaml.safe_load(p.read_text())
assert data['layer_a_status'] == 'failed_freeze_gate'
assert data['layer_b_status'] == 'implemented_pending_review'
assert data['seeds']['current_observed_stdev_M_training'] == 0.189
assert data['seeds']['escalation_currently_active'] is True
expected = {
    'grounding_stripped': 'implemented',
    'citation_fabricated': 'declared_not_yet_implemented',
    'constraint_named_not_respected': 'implemented',
}
modes = {row['id']: row for row in data['rawr_modes']}
assert set(modes) == set(expected)
for mode_id, status in expected.items():
    assert modes[mode_id]['status'] == status
print(json.dumps({
    'parsed': True,
    'layer_a_status': data['layer_a_status'],
    'layer_b_status': data['layer_b_status'],
    'current_observed_stdev_M_training': data['seeds']['current_observed_stdev_M_training'],
    'escalation_currently_active': data['seeds']['escalation_currently_active'],
    'rawr_modes': data['rawr_modes'],
}, indent=2))
PY
```

### Probe artifacts

- run ledger: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/probe_runs.jsonl`
- text report: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/probe_report.txt`
- metadata: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/probe_meta.json`
- per-run verifier outputs / diffs / logs: `benchmark_blueprints/families/responses-tool-schema-cutover/report/attempt_03/<variant>/run_0{1,2,3}/`

### Whole-family live results

Probe run id: `20260423T000148Z`

| Variant | Scores | Mean | Stdev | Ceiling hits |
| --- | --- | ---: | ---: | --- |
| `v1-clean-baseline` | `[100, 50, 60]` | `70.00` | `26.46` | `contract_drift x1`, `no_test_regression_guard x2` |
| `v2-noisy-distractor` | `[60, 0, 50]` | `36.67` | `32.15` | `contract_drift x2`, `no_test_regression_guard x3` |
| `v3-dirty-state` | `[50, 60, 50]` | `53.33` | `5.77` | `contract_drift x2`, `no_test_regression_guard x3` |
| `v4-multi-corpus-objective` | `[50, 50, 60]` | `53.33` | `5.77` | `contract_drift x2`, `no_test_regression_guard x3` |
| `v5-recovery-in-thread` | `[60, 60, 50]` | `56.67` | `5.77` | `contract_drift x1`, `no_test_regression_guard x3` |

### Layer A gate values

- `family_mean = 54.00` vs required `[15, 25]` -> `FAIL`
- `max_variant_mean = 70.00` vs required `<= 40` -> `FAIL`
- `min_variant_mean = 36.67` vs required `<= 10` -> `FAIL`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/- 3` -> `FAIL`
  - `v2-noisy-distractor (36.7) < v3-dirty-state (53.3)` beyond `+/- 3.0`
  - `v4-multi-corpus-objective (53.3) < v5-recovery-in-thread (56.7)` beyond `+/- 3.0`

### Metadata honesty updates tied to this attempt

- `family.yaml` attestation now reflects recorded reality:
  - `layer_a_status: failed_freeze_gate`
  - `layer_b_status: implemented_pending_review`
- canonical RAWR taxonomy in `family.yaml` is now:
  - `grounding_stripped` -> `implemented`
  - `citation_fabricated` -> `declared_not_yet_implemented`
  - `constraint_named_not_respected` -> `implemented`
- the latest full probe observed `current_observed_stdev_M_training = 0.189`, so `escalation_currently_active` is now `true`

### Legacy matrix naming note

- The older attempt-01 verification matrices still include a legacy `Chronology-blind fix` row name. That row remains preserved as historical family-local evidence, but it is **not** the canonical HLD §4 `citation_fabricated` RAWR mode and is no longer used as the family’s taxonomy declaration.

### Outcome

- Layer B declaration is now more honest than the prior `green` claim because the canonical taxonomy and latest variance state are explicitly recorded in `family.yaml`, but reviewer acceptance is still pending.
- Layer A remains open after the required post-fix rerun. The latest fresh live probe is even easier than `attempt_02`, with `family_mean = 54.00` and no variant mean below `36.67`.

## Attempt 04 — scorer hardening only, no live rerun in this turn

### Why this change

- Reviewer follow-up asked for a family-local hardening pass that reduces
  `no_test_regression_guard` saturation and improves `v1` discrimination.
- `attempt_03` showed that runs which repaired runtime/config/docs and passed
  hidden replay checks, but still never strengthened visible regression tests,
  were routinely landing in the `50-60` band.
- That was too generous for the declared task contract because visible test
  strengthening is already an explicit deliverable rather than an optional
  polish step.

### Hardening applied

- lowered `no_test_regression_guard` cap from `60` to `35` in the family scorer
- updated `evaluator_contract.md` to document the new cap and the reason it was
  changed

### Scope

- family-local only
- no live probe rerun in this turn by explicit user instruction

### Honest status after attempt 04

- This is a design-only hardening step recorded ahead of the next rerun.
- `attempt_03` remains the latest live evidence on file until a later whole-family
  probe is run against the hardened scorer.
