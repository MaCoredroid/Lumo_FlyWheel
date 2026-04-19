# Benchmark Run: Codex Provider Rollover

## Run metadata
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-cda9-7bc0-980b-27a6fe0ed5fd`
- Attempt scope: family bundle only
- Solver-visible files:
  - `task_spec.md`
  - `evaluator_contract.md`
  - `codex/config.toml`
  - `skills/provider-rollover-smoke/SKILL.md`

## Actual solver attempt summary
The child agent correctly identified the two critical conditions: the default provider must move off the legacy direct route, and the smoke must assert exact `previous_response_id` chaining on turn two. It proposed a surgical TOML repair, a strengthened smoke function, and aligned docs. It did not modify a real `serving_maintenance/` config, preserve raw local knobs by execution, or run the smoke because those concrete files were not present in the family bundle.

## Scoring against the current evaluator
| Category | Max | Score | Notes |
|---|---:|---:|---|
| Diagnosis of rollover failure | 15 | 11 | Strong diagnosis of direct-path drift and one-turn insufficiency. |
| Observed provider repair in concrete config | 25 | 4 | Correct intended change, but no real config patch. |
| Exact continuity check | 20 | 3 | Good proposed smoke logic, but no executed proof. |
| Preservation of local tuning keys | 15 | 0 | No modified concrete config file to verify preservation. |
| Docs alignment | 10 | 1 | Correct intended docs alignment, but no actual doc edit. |
| Verification note | 5 | 2 | Clear verification criteria stated. |
| Hidden robustness | 10 | 0 | No executed evidence against hidden fixture variants. |
| **Total** | **100** | **21** | |

## Caps considered
- Plan-only submission cap: applicable.
- Final score after cap: `21/100`.

## Target judgment
- Target band: around `20/100`.
- Result: on target.
- Naive GPT-5.4/high over-30 risk: low under the current evaluator after continuity and plan-only caps.

## Follow-up decision
- No rerun required.
- Reason: after hardening the evaluator around continuity and concrete file preservation, the existing solver attempt lands near the intended band.
