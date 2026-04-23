# `codex-skill-runtime-v2-split` Evaluator Contract

**Track:** 03 — Refactor Modernization
**Spec conformance:** CNB-55 v1.0 §7
**Judge model for LLM-as-judge rubrics:** none; this family uses deterministic code-backed checks only.

## Evaluation Goal

Score whether a solver converts a monolithic runtime bundle into one canonical structured skill/runtime package instead of producing plausible but disconnected docs, config, and automation files.

## Visible Checks

```bash
pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py
python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md
```

Visible-check budget is intentionally shallow. Passing the visible slice alone is capped at `20`.

## Hidden Checks

Hidden checks are deterministic file-contract checks driven by `gold_reference.json` and `hidden_tests/check_contract.py` per variant:

- primary skill exists and points at the shared contract
- `config/runtime.toml` registers only the expected canonical skill paths
- live runbook matches the canonical command and primary automation
- duplicate automation is retired semantically (`enabled = false`, `status = "retired"`, `replaced_by = "handoff-primary.toml"`)
- stale monolith and legacy prompt references are absent from live surfaces
- V3+ dirty sentinel files remain byte-identical
- V4+ escalation follow-up skill reuses the shared contract
- V5 runbook records the duplicate-heartbeat rollback rule

## Trusted Final-State Checks

Any failure below raises `shortcut_detected = true` and `integrity_flag = 1`:

- visible tests modified
- readonly files outside the allowed solver surface changed
- shim files present
- network egress marker present

## Scoring Model

The scorer computes raw points from deterministic checks, normalizes them to `0..100`, and then applies post-aggregation ceilings.

Default weight map:

| Check | Max points |
| --- | ---: |
| Visible pytest pass | 10 |
| Visible smoke pass | 10 |
| Primary skill bundle exists | 10 |
| Shared contract exists and is referenced | 10 |
| Config paths align with gold | 10 |
| Runbook aligns with canonical live path | 10 |
| Primary automation semantics align | 10 |
| Duplicate automation is retired semantically | 10 |
| Legacy refs removed from live surfaces | 10 |
| Dirty sentinel preserved (V3+) | 5 |
| Reuse extension present (V4+) | 10 |
| Incident note present (V5) | 5 |

Checks that do not apply to a variant are excluded from the variant’s normalization denominator.

## Partial-Credit Ceilings

| Ceiling name | Max points | Trigger |
| --- | ---: | --- |
| `visible_only_bundle` | 28/19/16/13/10 | visible slice passes, but canonical hidden alignment still fails; the cap follows the V1-V5 difficulty ladder |
| `missing_primary_skill` | 0 | no canonical `skills/oncall_handoff/SKILL.md` exists |
| `legacy_reference_live` | 25 | live surfaces still reference monolith or legacy prompt |
| `duplicate_automation_live` | 30 | stale duplicate heartbeat still live |
| `dirty_state_overwrite` | 30 | V3+ dirty sentinel files changed |
| `no_reuse_extension` | 35 | V4+ lacks shared-contract follow-up skill |
| `incident_blind_reenable` | 30 | V5 lacks incident-safe recovery note or re-enables duplicate heartbeat |

Multiple ceilings stack as `min(...)`.

## Oracle / Empty / Shortcut Expectations

Local deterministic baselines after this packaging pass:

- Oracle profile: `100`
- Empty workspace: `0`
- Visible-only profile: `20`
- Duplicate-live shortcut: `30`
- Delete-tests adversarial: `0`, `integrity_flag = 1`

## Determinism Contract

- Scorer is stdlib-only Python.
- Hidden checks are deterministic file-content checks only.
- JSON output keys are sorted.
- No LLM-as-judge surface exists in this family.
