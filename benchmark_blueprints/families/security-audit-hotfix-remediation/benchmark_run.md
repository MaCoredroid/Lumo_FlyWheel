# Benchmark Run

## attempt_00 — blueprint-only baseline
- `date`: `2026-04-18`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `bundle_state`: only `task_spec.md`, `evaluator_contract.md`, `skills/.../SKILL.md`, and `benchmark_run.md` existed; the workspace bundle and verifier fixtures were missing
- `observed_raw_score`: `77/100`
- `applied_cap`: `artifact_grounding_missing -> 20/100`
- `verdict`: the doc-only blueprint could describe the intended exploit but could not support artifact-grounded triage or workspace-backed calibration

## attempt_01 — family-local scaffold + Layer B implementation
- Implemented the missing family-local assets:
  - `workspace_bundle/v1..v5`
  - `verifier_data/security-audit-hotfix-remediation/...`
  - `verifiers/security-audit-hotfix-remediation/score_hotfix.py`
  - `verifiers/security-audit-hotfix-remediation/run_verification_matrix.py`
  - `verifiers/security-audit-hotfix-remediation/run_live_probe.py`
  - `family.yaml` and `manifest.lock.json`
- Hardening decisions already applied:
  - structured triage and deploy-note artifacts replace vague markdown-only scoring
  - hidden containment cases exercise encoded separators, drive-qualified paths, absolute paths, double-encoded separators, and symlink escapes
  - release and incident corpora gate `v4` and `v5`
  - dirty-state corpora gate `v3`+
- Layer A status: pending live probe
- Layer B status: verification matrix pending generation at regen time

## attempt_02a — smoke live probe + scorer hardening
- `probe_artifact_root`: `benchmark_blueprints/families/security-audit-hotfix-remediation/report/attempt_02_smoke_live_probe/`
- `observed_live_scores_before_hardening`:
  - `v1-clean-baseline`: `84`
  - `v2-noisy-distractor`: `85`
  - `v3-dirty-state`: `85`
  - `v4-multi-corpus-objective`: `20`
- `diagnosis`:
  - the live solver consistently fixed the code, added regression tests, and validated the structured artifacts
  - the scorer was still letting wrong triage dispositions ride as non-binding raw-point misses
  - `M4` and `M5` were not actually gated on `M2_primary_fix`, despite the family contract saying they must be
- `post-smoke hardening`:
  - added binding ceiling `triage_misclassification -> 20`
  - tightened `incident_blind_reselect -> 10`
  - enforced `M4`/`M5` dependency on `M2_primary_fix`
  - tightened release and incident acknowledgement checks to require real corpus acknowledgement rather than generic filler

## attempt_02b — counted whole-family live probe
- `date`: `2026-04-23`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `command`:
  - `python3 verifiers/security-audit-hotfix-remediation/run_live_probe.py --repeats 3 --timeout 900 --jsonl-out benchmark_blueprints/families/security-audit-hotfix-remediation/report/attempt_02b_live_probe/probe_runs.jsonl --summary-out benchmark_blueprints/families/security-audit-hotfix-remediation/report/attempt_02b_live_probe/summary.json`
- `verification_matrix_outputs`:
  - `benchmark_blueprints/families/security-audit-hotfix-remediation/verification_matrix.md`
  - `benchmark_blueprints/families/security-audit-hotfix-remediation/verification_matrix_v5.md`
- `probe_report_artifact`:
  - `benchmark_blueprints/families/security-audit-hotfix-remediation/report/attempt_02b_live_probe/attempt_02b_probe_report.txt`

### Counted Results
| variant | scores | raw_scores | mean | ceiling_hits |
|---|---|---|---:|---|
| `v1-clean-baseline` | `[20, 20, 20]` | `[79, 77, 77]` | `20.00` | `artifact_grounding_missing x2`, `blanket_triage x1`, `triage_misclassification x3` |
| `v2-noisy-distractor` | `[20, 20, 20]` | `[79, 74, 80]` | `20.00` | `artifact_grounding_missing x2`, `triage_misclassification x3` |
| `v3-dirty-state` | `[20, 20, 20]` | `[85, 85, 85]` | `20.00` | `triage_misclassification x3` |
| `v4-multi-corpus-objective` | `[20, 20, 20]` | `[85, 85, 75]` | `20.00` | `artifact_grounding_missing x1`, `release_freeze_unacknowledged x1`, `triage_misclassification x3` |
| `v5-recovery-in-thread` | `[20, 20, 20]` | `[85, 80, 80]` | `20.00` | `artifact_grounding_missing x1`, `triage_misclassification x3` |

### Layer A Gate Verdict
- `family_mean`: `20.00` -> pass (`15 <= mean <= 25`)
- `max_variant_mean`: `20.00` -> pass (`<= 40`)
- `min_variant_mean`: `20.00` -> fail (`must be <= 10`)
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`: pass
- overall §10.1 verdict: `failed_freeze_gate`

### Spot-Check Diagnosis
- `v1-clean-baseline` run 1:
  - the live solver implemented the correct resolved-root containment fix, added regression coverage, validated the JSON artifacts, and passed `make test`
  - the score still collapsed to `20` because the triage note deferred the JWT work but did not correctly classify every seeded SARIF or AppSec item, so `triage_misclassification` fired
- `v4-multi-corpus-objective` run 3:
  - the solver again fixed the code and kept nested exports live, but the deploy note only partially absorbed `release_context/freeze_window.md`
  - `release_freeze_unacknowledged` fired once, but the dominant floor was still the family-wide `triage_misclassification` ceiling

### Honest-Signal Conclusion
- The family is now fully scaffolded and live-probe runnable, and the live signal is honest: `gpt-5.4/high` reliably solves the code-repair core but still misses full artifact-grounded triage.
- Layer A is **not** accepted yet because the family lacks a hard variant below `10`.
- The current counted evidence says the variant progression is too uniform: one legitimate ceiling (`triage_misclassification`) binds across all 15 runs, flattening every variant at `20`.
- Next legitimate hardening step, if continued later: add a variant-specific binding obligation that only `v5` can fail honestly (for example, stricter incident-thread recovery accounting in the triage artifact itself), rather than lowering the global triage cap and collapsing the whole family below the target mean.
