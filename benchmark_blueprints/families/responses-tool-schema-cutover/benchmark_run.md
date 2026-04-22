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
