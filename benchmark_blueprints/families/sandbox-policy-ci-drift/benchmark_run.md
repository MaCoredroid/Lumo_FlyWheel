
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
