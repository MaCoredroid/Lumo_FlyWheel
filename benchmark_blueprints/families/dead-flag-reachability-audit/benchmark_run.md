# Benchmark Run — `dead-flag-reachability-audit`

## Model under test

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

## Layer A target

- family mean in `[15, 25]`
- max variant mean `<= 40`
- at least one variant mean `<= 10`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`
- oracle `>= 90`
- empty `= 0`
- shortcut `<= 30`

## attempt_00 — runtime family build and static verification

Scope completed inside this family only:

- authored `workspace_bundle/v1..v5`
- added family-local CLI `bin/cnb55-flag-audit`
- added deterministic scorer `verifiers/dead-flag-reachability-audit/score_reachability.py`
- added family-local regen, live-probe, and verification-matrix runners
- generated `verifier_data/dead-flag-reachability-audit/v1..v5`
- added `family.yaml`, `manifest.lock.json`, and verification-matrix artifacts

Commands run:

```bash
python3 verifiers/dead-flag-reachability-audit/regen_family.py
python3 verifiers/dead-flag-reachability-audit/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/dead-flag-reachability-audit/verification_matrix.md
python3 verifiers/dead-flag-reachability-audit/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/dead-flag-reachability-audit/verification_matrix_v5.md
```

Observed static sanity:

| variant | oracle | empty | shortcut |
| --- | ---: | ---: | ---: |
| v1-clean-baseline | 100 | 0 | 20 |
| v2-noisy-distractor | 100 | 0 | 20 |
| v3-dirty-state | 100 | 0 | 20 |
| v4-multi-corpus-objective | 100 | 0 | 20 |
| v5-recovery-in-thread | 100 | 0 | 20 |

Verification matrix headline results:

V1 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_audit_file` |
| RAWR grounding_stripped | 30 | 0.3000 | 0 | `grep_only_reasoning` |
| Alias collapse | 25 | 0.2500 | 0 | `classification_miss`, `alias_collapse` |
| Force-legacy as live | 20 | 0.2000 | 0 | `classification_miss`, `parser_equals_runtime` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

V5 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_audit_file` |
| RAWR grounding_stripped | 30 | 0.3000 | 0 | `grep_only_reasoning` |
| Alias collapse | 25 | 0.2500 | 0 | `classification_miss`, `alias_collapse` |
| Force-legacy as live | 20 | 0.2000 | 0 | `classification_miss`, `parser_equals_runtime`, `sunk_cost_finish` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

Interpretation:

- Layer B scaffolding exists and is executable.
- Oracle, empty, shortcut, and adversarial integrity behaviors are all in the
  expected bands.
- Static verification alone does not say whether Layer A is calibrated; the
  family still needs a real live probe against the authored bundle.

## attempt_01 — live probe

Status at commit time: pending until the family-local live runner finishes or
is intentionally deferred.

One `--repeats 1` launch was started during authoring and then stopped without
recording results because the first Codex workspace run had not completed yet.
No scored live-probe outcome is claimed in this file.

Planned command:

```bash
python3 verifiers/dead-flag-reachability-audit/run_live_probe.py --repeats 1
```
