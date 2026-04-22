
# Benchmark Run

## attempt_00 — baseline design
- Family: `sandbox-policy-ci-drift`
- Goal: turn the stub into a full family-local bundle with five variants, deterministic scoring, verifier data, and Layer-B metadata.
- Hypothesis:
  - V1 should discriminate parser/preview/workflow alignment.
  - V2 should punish archive noise leakage.
  - V3 should cap helper-only shortcuts near the target band.
  - V4 should force config/doc/operator-contract alignment.
  - V5 should punish rename-only fixes that drop `workspace-write` compatibility.
- Status: asset build complete; live Layer-A probe not launched here by instruction.

## attempt_01 — family-local asset validation
- Generator:
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/regen_family.py`
- Scope completed:
  - Added `workspace_bundle/v1..v5`, `family.yaml`, `manifest.lock.json`,
    family-local verifier data, hidden tests, milestone scripts, scorer, and
    verification-matrix tooling.
- Oracle sweep:
  - `v1-clean-baseline`: `P=100`, `M=1.0000`, `pass=True`
  - `v2-noisy-distractor`: `P=100`, `M=1.0000`, `pass=True`
  - `v3-dirty-state`: `P=100`, `M=1.0000`, `pass=True`
  - `v4-multi-corpus-objective`: `P=100`, `M=1.0000`, `pass=True`
  - `v5-recovery-in-thread`: `P=100`, `M=1.0000`, `pass=True`
- Verification matrix spot checks:
  - `verification_matrix.md` (`v1-clean-baseline`)
    - Oracle: `P=100`, `M=1.0000`, `pass=True`
    - Empty: `P=18`, `M=0.1800`, `pass=False`
    - RAWR grounding_stripped: `P=10`, `M=0.1000`, `pass=False`
    - Pick-ceiling drop compatibility: `P=20`, `M=0.2000`, `pass=False`
    - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`, `pass=False`
  - `verification_matrix_v5.md` (`v5-recovery-in-thread`)
    - Oracle: `P=100`, `M=1.0000`, `pass=True`
    - Empty: `P=18`, `M=0.1800`, `pass=False`
    - Pick-ceiling drop compatibility: `P=20`, `M=0.2000`, `pass=False`
    - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`, `pass=False`
- Layer status after this attempt:
  - Layer B: implemented and locally validated through scorer, milestones,
    state-delta metadata, integrity rules, and V1/V5 verification matrices.
  - Layer A: still `implemented_pending_probe`; no live `codex exec` probe was
    launched here because the task explicitly said not to launch the next loop step.

## attempt_02a — live-probe smoke and diagnosis
- Live probe command:
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/probe_family.py --attempt attempt_02a_smoke --n 1`
- Family-local verifier fix before trusting the smoke:
  - ignored runtime cache files (`__pycache__`, `.pyc`, `.pytest_cache`) in the trusted-final-state checks
  - stopped treating any mention of `workspace-write` in `codex/config.toml` comments as a scorer failure; the scorer now checks parsed config semantics instead
- Smoke results:
  - `v1-clean-baseline`: `P=20`, `M=0.2000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v2-noisy-distractor`: `P=20`, `M=0.2000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v3-dirty-state`: `P=20`, `M=0.2000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v4-multi-corpus-objective`: `P=50`, `M=0.5000`, ceilings=`docs_contract_stale`
  - `v5-recovery-in-thread`: `P=20`, `M=0.2000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
- Layer A checks from the smoke:
  - family mean: `26.00` -> `FAIL`
  - max variant mean: `50.00` -> `FAIL`
  - at least one variant mean <= `10`: `FAIL`
  - monotonic `V1>=V2>=V3>=V4>=V5` within `+/-3`: `FAIL`
- Spot-check diagnosis:
  - `v4` was the leak. The solver could repair parser, preview, workflow, and helper behavior while leaving the release note generic. That was enough to land at `50`, which was too high for the `multi-corpus/objective-drift` rung.
  - `v5` was already directionally correct as the fail rung, but it was only reaching `20` because the scorer did not yet turn the rollback-aware release-note requirement into an explicit ceiling.
- Next-step hypothesis:
  - add one family-local hardening change in the scorer only: when `release_context/preview-consumer-contract.md` exists, missing operator/preview-consumer language in the release note should cap the run at `20`; when `incident_context/rollback_2026_04.md` exists, missing rollback acknowledgement should cap the run at `10`

## attempt_02b — family-local hardening plus full live probe
- Hardening change applied:
  - scorer now emits `release_consumer_contract_missed` (cap `20`) when the release note misses the `release_context` operator-preview contract
  - scorer now emits `rollback_context_unacknowledged` (cap `10`) when the release note misses the `v5` rollback rationale
  - contract documentation updated in `evaluator_contract.md`
- Exact live probe command:
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/probe_family.py --attempt attempt_02b_live_probe --n 3`
- Exact probe-report command:
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/probe_report.py --attempt-dir benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe`
- Full `N=3` live results:
  - `v1-clean-baseline`: `scores=[100,20,20]`, `mean P=46.67`, `stdev P=37.71`, `mean M=0.4667`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v2-noisy-distractor`: `scores=[50,20,20]`, `mean P=30.00`, `stdev P=14.14`, `mean M=0.3000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v3-dirty-state`: `scores=[20,20,50]`, `mean P=30.00`, `stdev P=14.14`, `mean M=0.3000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`
  - `v4-multi-corpus-objective`: `scores=[20,20,20]`, `mean P=20.00`, `stdev P=0.00`, `mean M=0.2000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`, `release_consumer_contract_missed`
  - `v5-recovery-in-thread`: `scores=[10,10,10]`, `mean P=10.00`, `stdev P=0.00`, `mean M=0.1000`, ceilings=`docs_contract_stale`, `fixture_specific_normalization`, `release_consumer_contract_missed`, `rollback_context_unacknowledged`
- Layer A checks:
  - family mean: `27.33` -> `FAIL`
  - max variant mean: `46.67` -> `FAIL`
  - at least one variant mean <= `10`: `PASS`
  - monotonic `V1>=V2>=V3>=V4>=V5` within `+/-3`: `PASS`
- Layer B probe metadata:
  - family mean `M_training`: `0.2733`
  - current observed stdev `M_training`: `0.2265`
- Verification-matrix refresh after the scorer change:
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/run_verification_matrix.py --variant v1-clean-baseline`
  - `python3 benchmark_blueprints/families/sandbox-policy-ci-drift/tools/run_verification_matrix.py --variant v5-recovery-in-thread`
  - `verification_matrix.md` (`v1-clean-baseline`)
    - Oracle: `P=100`, `M=1.0000`, `pass=True`
    - Empty: `P=10`, `M=0.1000`, `pass=False`
    - RAWR grounding_stripped: `P=10`, `M=0.1000`, `pass=False`
    - Pick-ceiling drop compatibility: `P=20`, `M=0.2000`, `pass=False`
    - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`, `pass=False`
  - `verification_matrix_v5.md` (`v5-recovery-in-thread`)
    - Oracle: `P=100`, `M=1.0000`, `pass=True`
    - Empty: `P=18`, `M=0.1800`, `pass=False`
    - RAWR grounding_stripped: `P=10`, `M=0.1000`, `pass=False`
    - Pick-ceiling drop compatibility: `P=20`, `M=0.2000`, `pass=False`
    - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`, `pass=False`
- Honest read after the full live probe:
  - the new hardening did the intended work on the stress variants: `v4` is now stable at `20` and `v5` is now stable at `10`
  - the remaining Layer-A misses come from the clean floor, not from rubric leakage: a strong `gpt-5.4 high` run can still fully solve `v1` and intermittently clear the middle rungs without violating any hidden invariant
  - pushing `v1` below the strict `max<=40` gate would now require adding a new hidden requirement to an already fully solved clean-rung repair, which would risk fake ambiguity rather than legitimate difficulty
- Family-local artifact paths for this attempt:
  - `benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe/probe_runs.jsonl`
  - `benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe/summary.json`
  - `benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe/attempt_02b_live_probe_probe_report.txt`
  - `benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe/logs/`
  - `benchmark_blueprints/families/sandbox-policy-ci-drift/report/attempt_02b_live_probe/artifacts/`
