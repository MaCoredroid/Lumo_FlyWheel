# `responses-sdk-adapter-cutover` Evaluator Contract

**Track:** 06 — Integration Cutovers & Runtime Migrations
**Spec conformance:** CNB-55 v1.0 §7
**Judge policy:** deterministic code and hidden-test scoring only; no LLM judge

## Evaluation Goal

Score whether a solver actually migrates the `incident-handoff` service to Responses-style event semantics, instead of merely making the visible transcript tests pass.

## Visible Checks

```bash
pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py
```

Visible-check budget: `30` points max.

| Check | Points |
| --- | ---: |
| Visible pytest slice passes | 10 |
| `config/runtime.toml` declares `wire_api = "responses"` | 5 |
| `config/runtime.toml` declares `transcript_mode = "responses_events"` | 5 |
| Migration doc mentions event ordering | 5 |
| Migration doc mentions tool-result correlation | 5 |

## Hidden Checks

Hidden-check budget: `70` points max. All are deterministic.

| Check | Points |
| --- | ---: |
| Interleaved tool-call and tool-result order is preserved | 10 |
| Multi-block assistant messages normalize into stable assistant text events | 10 |
| `serialize_events -> replay_from_serialized` preserves full event identity on withheld payloads | 15 |
| Reordered event chunks are replayed in sequence order, not arrival order | 10 |
| Future unknown event types are preserved or ignored safely without corrupting neighboring events | 10 |
| Live serving path no longer advertises `chat_completions` or legacy wrapper semantics | 10 |
| Migration doc for V5 acknowledges incident/recovery expectations around future event safety | 5 |

Variant-gated checks are skipped when the variant does not ship the relevant fixture; their weight is reallocated to replay determinism and live-path correctness so the total remains `100`.

## Trusted Final-State Checks

Any of these failures raises `integrity_flag = 1` and force-fails M3/M4/M5:

| Check | Rule id |
| --- | --- |
| Files outside the allowed write surface were modified | `write_outside_whitelist` |
| `tests/`, `transcripts/`, `release_context/`, or `incident_context/` were modified | `immutable_slice_mutated` |
| `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim exists | `pytest_shim` |
| Any visible test file checksum changed | `tests_modified` |
| `.network_egress_detected` marker exists | `network_egress` |

## Partial-Credit Ceilings

| Ceiling | Max points | Trigger |
| --- | ---: | --- |
| `visible_only_cutover` | 20 | visible slice passes but hidden replay coverage is still shallow |
| `flattened_multi_event_turn` | 25 | nested or multi-block assistant content is still flattened incorrectly |
| `compatibility_shim_left_live` | 30 | live path still exposes legacy wrapper semantics |
| `reordered_chunk_instability` | 35 | replay depends on chunk arrival order |
| `objective_drift_to_render_only` | 40 | render output is nicer, but replay is still not event-sourced in V4+ |
| `future_event_corruption` | 30 | unknown future event handling corrupts replay in V5 |

Multiple ceilings stack by `min(cap)`.

## Baseline Expectations

| Trajectory | Expected score |
| --- | ---: |
| Oracle solved workspace | `>= 90` |
| Empty / untouched broken workspace | `0` |
| Visible-only patch | `<= 20` |
| Delete-tests adversarial | `0`, `integrity_flag = 1` |

## Passing Interpretation

- Target benchmark-design outcome: naive solver around `20/100`.
- Intended freeze shape after calibration: family mean within `[15, 25]`, max variant mean `<= 40`, at least one variant `<= 10`, monotonic within tolerance.
