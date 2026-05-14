# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is the primary live flag that directly controls the runtime branch. When set truthy, it sets shadow_enabled=true and effective_mode=shadow, causing preview_runtime_branch to return shadow_preview_path. The flag has a parser hit and a corresponding runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. It has parser presence and normalizes to ENABLE_SHADOW_PREVIEW, reaching the same shadow_preview_path. The runbook explicitly states it is accepted by the env parser but normalizes to the same shadow path and is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has parser presence but no runtime branch. The runbook states the live service does not branch on it anymore. It is left in reporting and operator notes only. The test confirms force_legacy_seen is tracked but the branch remains legacy_preview_path regardless.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with active runtime branching. Essential for preview rollout control. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Live alias with parser support but no standalone branch. Deprecate after legacy manifests migrate. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator notes dependency | Dead flag with parser presence but no runtime branch. Remove parser after confirming telemetry and notes no longer depend on it. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed live in runtime.py |
| observed | ENABLE_PREVIEW_V2 alias behavior | Alias behavior confirmed in config.py |
| missing | External deploy manifest references | Cannot verify legacy manifest usage outside this workspace copy |

ENABLE_SHADOW_PREVIEW is the primary flag controlling live preview routing. ENABLE_PREVIEW_V2 is a legacy alias normalizing to it. PREVIEW_FORCE_LEGACY is dead with no runtime impact.
