# `codex-provider-rollover` Evaluator Contract

**Track:** 03 — Refactor Modernization
**Spec conformance:** CNB-55 v1.0 §7
**Judge policy:** deterministic code and hidden-test scoring only; no LLM judge

## Evaluation Goal

Score whether a solver actually repairs the concrete maintenance profile on the
Responses proxy path, instead of making a cosmetic config flip or a one-turn
smoke patch.

## Visible Checks

```bash
bin/run-visible-tests
```

Visible-check budget: `30` points max.

| Check | Points |
| --- | ---: |
| Visible test slice passes | 25 |
| Docs mention exact `previous_response_id` follow-up semantics | 5 |

## Hidden Checks

Hidden-check budget: `70` points max. All checks are deterministic.

| Check | Points |
| --- | ---: |
| Selected provider id is the canonical proxy-backed Responses profile | 10 |
| Selected provider stanza has the expected proxy URL, `wire_api = "responses"`, and `store = true` | 10 |
| Good two-turn smoke passes with exact continuity | 15 |
| Hidden negative fixtures for wrong provider / wrong previous-response id / missing store are rejected | 10 |
| Local tuning block remains byte-for-byte identical | 15 |
| Provider docs and smoke docs agree on provider id and proxy route | 5 |
| Variant-specific release / incident keywords are acknowledged | 5 |

## Trusted Final-State Checks

Any of these failures raises `integrity_flag = 1` and force-fails M3/M4/M5:

| Check | Rule id |
| --- | --- |
| Files outside the allowed write surface were modified | `write_outside_whitelist` |
| `tests/`, fixtures, `release_context/`, or `incident_context/` were modified | `immutable_slice_mutated` |
| `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim exists | `pytest_shim` |
| Any visible test file checksum changed | `tests_modified` |
| `.network_egress_detected` marker exists | `network_egress` |

## Partial-Credit Ceilings

| Ceiling | Max points | Trigger |
| --- | ---: | --- |
| `no_visible_green` | 25 | visible slice still fails |
| `legacy_default_selected` | 20 | selected provider remains legacy or canary |
| `proxy_route_incorrect` | 30 | selected stanza is not the canonical proxy-backed route with `store = true` |
| `continuity_not_exact` | 20 | smoke does not require exact `previous_response_id` chaining |
| `config_rewritten` | 30 | local tuning block was not preserved byte-for-byte |
| `docs_unaligned` | 40 | docs disagree with config or omit continuity semantics |
| `canary_anchoring` | 35 | V2 fix anchors on the distractor canary profile |
| `template_overwrite` | 30 | V3 uses the template path and loses local tuning fidelity |
| `maintenance_profile_drift` | 45 | V4 ignores maintenance-profile release objective |
| `rollback_blind` | 35 | V5 ignores rollback evidence around `store = true` |

Multiple ceilings stack by `min(cap)`.

## Baseline Expectations

| Trajectory | Expected score |
| --- | ---: |
| Oracle solved workspace | `>= 90` |
| Empty / untouched broken workspace | `0` |
| Pick-the-legacy-provider shortcut | `<= 20` |
| Delete-tests adversarial | `0`, `integrity_flag = 1` |

## Passing Interpretation

- Target benchmark-design outcome: naive solver around `20/100`.
- Intended freeze shape after calibration: family mean within `[15, 25]`, max
  variant mean `<= 40`, at least one variant `<= 10`, monotonic within
  tolerance.
