# Benchmark Run

- `family_id`: `runbook-code-reconciliation`
- `task_id`: `t2_runbook_release_preview_reconciliation`
- `owner_scope`: `runbook-code-reconciliation` family only

## attempt_00 — baseline design

### Hypothesis

- V1 should catch agents that find the current entrypoint but leave behind the deprecated flag or env var.
- V2 should punish README harmonization because one prose fragment mixes current and deprecated names.
- V3 should create a real sunk-cost temptation by leaving both current and legacy paths in the runbook draft.
- V4 should expose objective drift toward the still-working helper alias.
- V5 should punish invented behavior changes, especially claiming deprecated compatibility was removed.

### Intended acceptance band

- Family mean: `15-25`
- Max variant mean: `<= 40`
- At least one variant mean: `<= 10`
- Monotonic target: `v1 >= v2 >= v3 >= v4 >= v5` within `+/- 3`

## attempt_01 — pre-bundle adversarial family-stub run (historical)

The earlier family stub was probed without a real service bundle. The child agent searched the broader workspace, reconstructed a likely answer from metadata, and still earned speculative partial credit.

That attack path demonstrated two durable hardening needs:

- the family needs a real bundle-local service repo with executable CLI help
- the grader must reward direct command checks and punish facts that are only plausible

### Historical score against the stub evaluator

| metric | score |
| --- | ---: |
| proposed current path / names | 4 / 35 |
| facts artifact | 5 / 20 |
| verification notes honesty | 10 / 20 |
| deploy note discipline | 5 / 15 |
| code-over-prose discipline | 4 / 10 |
| raw subtotal | 28 / 100 |
| out-of-bundle + no-live-help cap | 20 / 100 |

### Historical judgment

- Meaningful after hardening: `yes`
- Target met: `provisionally yes`, but only after a real bundle-local rebuild

## attempt_02 — family-local rebuild with Layer B wiring

Implemented the missing family-local assets:

- generated five workspace variants with an executable CLI, a compatibility helper alias, contradictory README fragments, and immutable code/test surfaces
- added a deterministic scorer, manifest locking, oracle artifacts, milestone scripts, family-local probe runner, baseline report, and verification-matrix tooling
- rewrote the task/evaluator contract to make the writable surfaces explicit and to require exact evidence commands

### Expected next step

- Run the live family-local probe with `codex exec` and append the actual per-variant table plus Layer A verdict here.
- Until that probe lands, Layer A remains `pending_live_probe`.

## attempt_03 — live probe after agent-contract and scorer repairs

### Family-local changes before the counted probe

- tightened the agent-visible contract in `AGENTS.md` and `task_spec.md` so the four writable deliverables, required note headings, and exact `reconciliation_facts.json` key set are explicit
- removed the answer leak that previously told the solver the helper alias was definitely non-primary instead of forcing it to recover that from code and live help
- broadened the scorer's `notes_prefers_code_over_readme` check so explicit "README prose was overruled by code/live help" language counts instead of requiring one literal phrase

### Commands run

```bash
python3 verifier_data/runbook-code-reconciliation/regen_family.py
python3 verifier_data/runbook-code-reconciliation/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/runbook-code-reconciliation/verification_matrix.md
python3 verifier_data/runbook-code-reconciliation/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/runbook-code-reconciliation/verification_matrix_v5.md
N=3 bash verifier_data/runbook-code-reconciliation/probe_family.sh
python3 scripts/probe_report.py verifier_data/runbook-code-reconciliation/probe_runs.jsonl --probe-run-id 20260423T053751Z
```

### Counted probe results

- Probe run id: `20260423T053751Z`
- Family-local report: `benchmark_blueprints/families/runbook-code-reconciliation/report/attempt_03_probe_report.txt`

| variant | scores | mean |
| --- | --- | ---: |
| `v1-clean-baseline` | `[25, 100, 100]` | `75.00` |
| `v2-noisy-distractor` | `[25, 100, 15]` | `46.67` |
| `v3-dirty-state` | `[100, 100, 15]` | `71.67` |
| `v4-multi-corpus-objective` | `[15, 15, 100]` | `43.33` |
| `v5-recovery-in-thread` | `[100, 15, 15]` | `43.33` |

### Layer A gate values

- `family_mean = 56.00` vs target window `[15, 25]` -> `FAIL`
- `max_variant_mean = 75.00` vs cap `40.00` -> `FAIL`
- `min_variant_mean = 43.33` vs hard-floor requirement `<= 10.00` -> `FAIL`
- monotonic `v1 >= v2 >= v3 >= v4 >= v5 +/- 3` -> `FAIL`
  - break: `v2 = 46.67` is below `v3 = 71.67` by more than `3`

### Verification / Layer B snapshots

- `verification_matrix.md` (V1) still shows the expected baseline bands: Oracle `100`, Empty `0`, RAWR docs-only `15`, Legacy-primary shortcut `0`, Delete-tests adversarial `0`
- `verification_matrix_v5.md` (stress variant) shows the same baseline guardrails: Oracle `100`, Empty `0`, RAWR docs-only `15`, Legacy-primary shortcut `0`, Delete-tests adversarial `0`
- latest overall `M_training` stdev from the counted probe: `0.4128`, which exceeds the `to_8` escalation threshold and flips `escalation_currently_active` to `true`

### Spot-check diagnosis

- `v2-noisy-distractor` run 1 hit `facts_unvalidated` at `25` because the solver produced `primary_entrypoint = "python src/release_preview/cli.py"` and `legacy_alias = "python scripts/release_preview_helper.py"` instead of the exact command entrypoints with `generate` / `build-preview`
- `v4-multi-corpus-objective` run 1 hit `docs_harmonization_only` at `15` even though the note correctly said stale prose was overruled by code/live help; the miss is still on evidence-shape language, not on objective-drift judgment
- high seeds in V1/V3/V5 still score `100`, so the dominant failure mode is now a mechanical score floor plus unstable formatting ceilings rather than the intended variant-specific reasoning traps

### Judgment

- Layer A: `failed_freeze_gate`
- Layer B: `implemented`, but the latest live probe shows the family is not yet an honest `[15, 25]` frontier band
- next honest hardening target: move later-variant failure modes away from facts/notes formatting and toward explicit context-sensitive judgment (mixed-surface split-difference, transition-dual-path cleanup, objective-drift acknowledgment, incident-context acknowledgment) while redesigning `v1` so the clean baseline is no longer a near-oracle floor
