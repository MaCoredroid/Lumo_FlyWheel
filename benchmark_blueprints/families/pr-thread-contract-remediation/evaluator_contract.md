# `pr-thread-contract-remediation` Evaluator Contract

**Track:** 04 — Review Remediation
**Spec conformance:** CNB-55 v1.0 §7
**Judge type:** deterministic only, no LLM-as-judge

## Evaluation Goal

Measure whether an agent can read a mixed-state PR review export, implement the
minimum real remediation, preserve request-side semantics, document the
response-contract nuance, and close only the actionable threads with
evidence-backed replies.

## Visible Checks

```bash
python3 -m unittest tests.test_handlers tests.test_service
```

Visible budget: **20 / 100**

| Check | Points |
| --- | ---: |
| visible test command passes | 20 |

## Hidden Behavioral Checks

Hidden budget: **40 / 100**

| Check | Points |
| --- | ---: |
| both serializer paths omit `owner` for unowned buckets | 15 |
| `include_unowned=true` preserves owner order and appends unowned last | 15 |
| explicit request-side owner semantics remain unchanged | 10 |

These checks are implemented in
`verifier_data/pr-thread-contract-remediation/_shared/hidden_tests/test_contract_hidden.py`.

## Docs, Replies, And Scope

| Check | Points |
| --- | ---: |
| release note documents the omit-not-null response contract and request-side stability via deterministic concept checks | 10 |
| `review/reviewer_replies.json` matches schema | 5 |
| replies target actionable ids only | 5 |
| reply bodies cite concrete evidence files or changed files | 10 |
| regression test coverage expanded in visible tests | 3 |
| patch stays narrow and in-scope | 2 |

## Ceilings

Applied after raw aggregation:

| Ceiling | Max | Trigger |
| --- | ---: | --- |
| `null_owner_contract_unfixed` | 30 | any hidden serializer-path omission check fails |
| `unstable_unowned_order` | 35 | ordering checks fail |
| `request_semantics_regression` | 25 | absent-vs-null owner filter semantics changed |
| `generic_replies` | 20 | reply body is generic or lacks thread-specific evidence |
| `non_actionable_replied` | 15 | reply exists for a resolved or outdated thread |
| `missing_release_note_contract` | 25 | required response-contract phrases are missing |
| `outdated_sort_resurrection` | 20 | V2+ patch revives global or alphabetical sorting |
| `sunk_cost_finish` | 25 | V3+ patch adopts the abandoned missing-owner shortcut |
| `objective_drift` | 25 | V4+ reply/doc update ignores mobile-contract compatibility |
| `incident_blind_reselect` | 30 | V5 reply/doc update ignores the `INC-742` rollback context |

## Integrity Rules

One failure sets `integrity_flag=1`, forces score `0`, and zeroes M3/M4/M5.

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

`tests_modified` applies only to `tests/locked/**`, not to editable visible
tests.

## Pass Bar

- `pass_bar = 80`
- final pass also requires `M5_e2e = true`

## Oracle / Empty / Shortcut Expectations

- Oracle overlay score: **95**
- Empty workspace score: **0**
- Shortcut score: **12** (well under the ≤ 30 guardrail)

The shortcut trajectory is a solver attempt that leaves the code broken or only
touches reply/doc surfaces with generic thread closure.
