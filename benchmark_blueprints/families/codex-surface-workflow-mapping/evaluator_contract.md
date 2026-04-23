# `codex-surface-workflow-mapping` Evaluator Contract

**Track:** 02 — Codebase Understanding
**Spec conformance:** CNB-55 v1.0 §7
**Scorer:** deterministic stdlib-only Python

## Evaluation Goal

Score whether the agent can identify the live Codex-facing workflow in a noisy repo and map it consistently across multiple artifacts without using stale drafts, deprecated helpers, or outdated schedule claims.

## Visible Checks (30 points max)

```
pytest -q tests/test_workflow_map.py
```

| Check | Points |
| --- | ---: |
| `artifacts/workflow_map.json` exists | 4 |
| canonical JSON parses and matches `cnb55.workflow_mapping.v1` | 6 |
| rendered artifacts exist and are non-empty | 4 |
| skill / toml / automation sections are present | 4 |
| mapping note covers all four artifacts | 4 |
| at least one rejected candidate is recorded | 4 |
| automation proposal separates Task and Schedule sections | 4 |

## Hidden Checks (70 points)

| Check | Points |
| --- | ---: |
| `skill.entrypoint_command_literal == "make codex-daily-triage"` | 10 |
| `toml.entrypoint_command_literal == "make codex-daily-triage"` | 10 |
| `automation.command_literal == "make codex-daily-triage"` | 5 |
| `automation.schedule_literal == "0 9 * * 1-5"` | 10 |
| `automation.task_prompt` keeps schedule semantics out of the task body | 5 |
| mapping decisions ground commands and schedule in cited repo files | 15 |
| stale / dirty / incident candidates are rejected correctly for the variant | 10 |
| workflow scope is variant-correct and repo-specific | 5 |

## Trusted Final-State Checks

These are pass/fail integrity gates:

- no writes outside `artifacts/*` and `workflow_input.json`
- no modification of `Makefile`, `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, `skills/`, `drafts/`, `release_context/`, `incident_context/`, `tests/`, `.scenario_variant`, `AGENTS.md`, `Dockerfile`, or `bin/`
- no `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim
- no `.network_egress_detected` marker

Any integrity violation forces `shortcut_detected=true` and caps the score at 0.

## Partial-Credit Ceilings

| Ceiling | Max points | Trigger |
| --- | ---: | --- |
| `no_submission` | 0 | canonical JSON missing |
| `malformed_submission` | 10 | invalid JSON or wrong schema / variant |
| `inconsistent_entrypoints` | 0 | skill / toml / automation point at different entrypoints |
| `stale_entrypoint` | 20 | any primary artifact uses the deprecated helper path |
| `wrong_schedule` | 25 | schedule literal does not match `0 9 * * 1-5` |
| `schedule_in_prompt` | 30 | automation task prompt embeds schedule semantics |
| `ungrounded_mapping` | 25 | commands or schedule are not grounded in cited files |
| `generic_mapping` | 25 | source coverage too thin or scope wording is repo-agnostic |
| `dirty_state_reuse` | 35 | V3+ abandoned draft not rejected |
| `objective_drift` | 30 | V4+ blocker-first release context ignored |
| `incident_blind_reuse` | 25 | V5 rollback context ignored or helper path reintroduced |

## Baselines

- Oracle submission: expected `100`
- Empty workspace submission: expected `0`
- Stale-helper shortcut submission: expected `20`

## Dual-Band Emission

- `P_benchmark`: 0-100 benchmark score
- `M_training`: normalized deterministic score in `[0, 1]`
- No LLM-judge checks are active in this family; all awarded points remain in the deterministic training band.
