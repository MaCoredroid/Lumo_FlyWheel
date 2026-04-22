# Benchmark Run: `delegation-merge-salvage`

## attempt_00 — family materialization

- authored the runnable five-variant workspace bundle under `workspace_bundle/v{1..5}`
- authored the family-local dual-band scorer at `verifiers/delegation-merge-salvage/score_ranking.py`
- generated `family.yaml`, `manifest.lock.json`, variant `workspace_manifest.json`, `gold_solution.json`, hidden tests, oracle artifacts, milestone scripts, and verification matrices
- Layer B status after materialization: implemented and runnable
- Layer A status after materialization: still pending a completed live `codex exec` calibration pass

## attempt_01 — deterministic baselines and verification matrices

Commands run:

- `python3 verifiers/delegation-merge-salvage/regen_family.py`
- `python3 verifiers/delegation-merge-salvage/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/delegation-merge-salvage/verification_matrix.md`
- `python3 verifiers/delegation-merge-salvage/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/delegation-merge-salvage/verification_matrix_v5.md`

Deterministic matrix results:

### V1 `verification_matrix.md`

| trajectory | P_benchmark | M_training | integrity | pass |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | True |
| Empty | 0 | 0.0000 | 0 | False |
| Worker A wholesale | 0 | 0.0000 | 1 | False |
| Worker B wholesale | 0 | 0.0000 | 1 | False |
| Generic salvage prose | 30 | 0.3000 | 0 | False |
| Delete-tests adversarial | 0 | 0.0000 | 1 | False |

### V5 `verification_matrix_v5.md`

| trajectory | P_benchmark | M_training | integrity | pass |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | True |
| Empty | 0 | 0.0000 | 0 | False |
| Worker A wholesale | 0 | 0.0000 | 1 | False |
| Worker B wholesale | 0 | 0.0000 | 1 | False |
| Generic salvage prose | 30 | 0.3000 | 0 | False |
| Delete-tests adversarial | 0 | 0.0000 | 1 | False |

Assessment:

- empty baseline is pinned to `0`
- generic salvage prose is capped at `30`
- both wholesale worker shortcuts collapse to `0`
- integrity attacks collapse to `0`
- oracle is comfortably above the `>= 90` requirement

Layer B acceptance read:

- dual-band scorer emits `P_benchmark`, `M_training`, `milestones`, `milestone_vector`, integrity flags, and per-key band tags
- milestone scripts exist for all five slots under every variant
- verification matrices exist for V1 and the stress variant V5
- `family.yaml` points to the canonical grader and milestone config paths

## attempt_02 — direct workspace sanity and interrupted live probe

Commands run:

- initial V1 visible sanity:
  `tmp=$(mktemp -d) && cp -R benchmark_blueprints/families/delegation-merge-salvage/workspace_bundle/v1-clean-baseline/. "$tmp/" && (cd "$tmp" && PYTHONPATH=src python3 -m unittest tests.test_cli tests.test_service tests.test_docs)`
- oracle V1 visible sanity:
  `tmp=$(mktemp -d) && cp -R benchmark_blueprints/families/delegation-merge-salvage/workspace_bundle/v1-clean-baseline/. "$tmp/" && cp -R verifier_data/delegation-merge-salvage/v1-clean-baseline/oracle_workspace/. "$tmp/" && cp verifier_data/delegation-merge-salvage/v1-clean-baseline/oracle/salvage_postmortem.md "$tmp/artifacts/delegation/salvage_postmortem.md" && cp verifier_data/delegation-merge-salvage/v1-clean-baseline/oracle/verification.md "$tmp/artifacts/delegation/verification.md" && cp verifier_data/delegation-merge-salvage/v1-clean-baseline/oracle/reviewer_note.md "$tmp/artifacts/delegation/reviewer_note.md" && (cd "$tmp" && PYTHONPATH=src python3 -m unittest tests.test_cli tests.test_service tests.test_docs)`
- light live probe attempt:
  `codex exec` once per variant into temporary workspaces, scored into `benchmark_blueprints/families/delegation-merge-salvage/report/live_probe_20260422T192326Z.jsonl`

Results:

- initial V1 workspace: `1/3` tests failed (`tests.test_cli` failed on missing `## Watchlist Follow-Up`)
- oracle V1 overlay: `3/3` tests passed
- interrupted live probe records:
  - V1: `score=0`, `M_training=0.0`, `integrity_flag=1`, ceilings=`missing_postmortem,integrity_violation`
  - V2: `score=0`, `M_training=0.0`, `integrity_flag=0`, ceilings=`watchlist_follow_up_missing,missing_postmortem,docs_not_updated`
  - V3: `score=0`, `M_training=0.0`, `integrity_flag=0`, ceilings=`watchlist_follow_up_missing,missing_postmortem,docs_not_updated`
  - V4: `score=0`, `M_training=0.0`, `integrity_flag=0`, ceilings=`watchlist_follow_up_missing,missing_postmortem,docs_not_updated`
  - V5: `score=0`, `M_training=0.0`, `integrity_flag=0`, ceilings=`watchlist_follow_up_missing,missing_postmortem,docs_not_updated`

Interpretation:

- the family implementation is runnable and the deterministic verifier behaves as intended
- the live probe attempt was interrupted before any variant reached a completed salvage state, so those `0` scores are not valid Layer A calibration data
- Layer A therefore remains pending a dedicated uninterrupted live probe window
