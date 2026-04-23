# `request-path-evidence-brief` Evaluator Contract

**Track:** 02 — Codebase Understanding  
**Family id:** `request-path-evidence-brief`  
**Spec conformance:** CNB-55 v1.0 §7  
**Scorer:** `verifiers/request-path-evidence-brief/score_trace.py`

## Evaluation Goal

Score whether the solver can reconstruct a live request path from code and tests, reject a plausible decoy, and write a narrow correction note without inventing or patching behavior.

This family is deterministic. There is no LLM-as-judge component in the active score path.

## Visible Checks (30 points max)

Run:

```bash
pytest -q tests/test_sync.py tests/test_docs.py tests/test_trace_outputs.py
```

Allocated as:

| Check | Points |
| --- | ---: |
| `artifacts/request_path_brief.md` exists | 4 |
| `artifacts/docs_correction.md` exists | 4 |
| `artifacts/path_map.json` parses and uses `cnb55.request_path_map.v1` | 8 |
| `variant_id` matches `.scenario_variant` | 4 |
| required `path_map.json` sections exist | 6 |
| no stray writes outside the allowed artifact files | 4 |

## Hidden Checks (70 points)

### Live-path reconstruction

| Check | Points |
| --- | ---: |
| exact ordered hop sequence in `live_path[]` | 18 |
| live-path completeness bonus | 2 |

### Field derivations

| Check | Points |
| --- | ---: |
| `owner_source` derivation matches live service step | 8 |
| `routing_key` derivation matches live serializer step | 8 |
| emission step matches final serializer hop | 4 |

### Test evidence

| Check | Points |
| --- | ---: |
| explicit-owner proving test cited | 5 |
| default-owner proving test cited | 5 |

### Decoy rejection

| Check | Points |
| --- | ---: |
| required variant decoy rejected with a concrete non-liveness reason | 5 |

### Markdown grounding

| Check | Points |
| --- | ---: |
| brief contains the required `path::symbol` evidence tokens | 5 |
| docs correction contains the required `path::symbol` evidence tokens | 5 |

### Variant-specific recovery

| Check | Points |
| --- | ---: |
| V5 correction acknowledges rollback or incident context | 5 |

V1–V4 do not use the final row; their deterministic oracle ceiling remains 99 rather than 100.

## Partial-Credit Ceilings

Ceilings are hard caps applied after raw aggregation.

| Ceiling | Max score | Trigger |
| --- | ---: | --- |
| `no_artifacts` | 0 | `artifacts/path_map.json` missing |
| `malformed_path_map` | 10 | `path_map.json` is not valid JSON or has the wrong schema |
| `missing_symbol_adjacency` | 25 | ordered hop sequence or `caller_symbol` / `callee_symbol` adjacency is broken |
| `store_claimed_as_decision_layer` | 25 | submission says storage decides `owner_source` |
| `pre_owner_routing_claim` | 30 | submission says `routing_key` is derived before owner resolution or outside serializer |
| `no_support_note_verdict` | 35 | brief never explicitly rejects the stale support note |
| `no_decoy_rejection` | 35 | no plausible decoy is rejected |
| `weak_markdown_grounding` | 45 | markdown outputs lack enough concrete `path::symbol` grounding |
| `dirty_state_trust` | 40 | V3 submission treats the abandoned previous-session draft as live evidence |
| `future_architecture_anchor` | 45 | V4 submission treats future release planning notes as implemented behavior |
| `incident_blind_reselect` | 30 | V5 submission repeats the rolled-back correction without incident acknowledgment |
| `external_evidence` | 20 | cites sibling scenario bundles, URLs, or machine-local paths outside the bundle |

## Trusted Final-State Checks

Failure triggers integrity flag `H=1` and force-fails later milestones:

- writes outside the allowed artifact files,
- mutation of readonly repo inputs,
- test edits,
- pytest shims,
- network-egress marker presence.

Integrity rules live in `family.yaml` and must match scorer call sites 1:1.

## Milestones / Dual-Band Output

The scorer emits `cnb55.verify_result.v3` with:

- `score` / `P_benchmark`
- `M_training`
- `milestones`
- `milestone_vector`
- `integrity_flag`
- `ceilings_applied`

Milestones:

- `M1_localization`: path map and test observations exist
- `M2_primary_fix`: field-derivation claims localize the live owner-selection step
- `M3_invariants`: readonly inputs stay unchanged
- `M4_functional`: live path matches the gold hop sequence
- `M5_e2e`: all artifacts exist and the final score clears the pass bar

This family currently keeps all active points in the deterministic band. `llm_judge_quarantine.total_quarantined_points = 0`.

## Oracle / Empty / Shortcut Targets

Observed from the regenerated manifest:

| Variant set | Oracle | Empty | Shortcut | Grounding stripped |
| --- | ---: | ---: | ---: | ---: |
| V1–V4 | 99 | 0 | 25 | 35 |
| V5 | 100 | 0 | 25 | 30 |

These satisfy the authored baseline goals:

- oracle near ceiling,
- empty at 0,
- shortcut at or below 30,
- markdown-grounding failure clearly below pass.

## Meaningfulness Check

Reject the family if any future edit makes the live path under-specified. A careful solver must still be able to answer the task from the shipped code, tests, and notes alone; otherwise this stops being a trace task and becomes an ambiguity trap.
