# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. Parser sets shadow_enabled=true and effective_mode=shadow. Runtime branches on config.shadow_enabled to shadow_preview_path. Test confirms branch changes. Runbook confirms this is the live preview router branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. Parser accepts it for legacy deploy manifests and normalizes to the same shadow path as ENABLE_SHADOW_PREVIEW. Runbook explicitly states it is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is dead - parser present but no runtime branching. Runbook explicitly states 'The live service does not branch on it anymore.' Test name 'test_force_legacy_flag_is_tracked_for_reporting_only' confirms it only sets force_legacy_seen for reporting/telemetry purposes. Runtime never checks this flag.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling preview rollout. Essential for runtime branching logic. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Live alias for backward compatibility. Deprecate after legacy manifests migrate to ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | operator notes, reporting dashboards | Dead flag with no runtime effect. Remove after migrating operator notes and telemetry dashboards to stop referencing it. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW live status | Confirmed via config.py parser and runtime.py branch |
| observed | ENABLE_PREVIEW_V2 alias status | Confirmed via runbook and test naming |
| observed | PREVIEW_FORCE_LEGACY dead status | Runbook states live service does not branch on it |
| missing | deploy manifest inventory | No inventory of legacy manifests using ENABLE_PREVIEW_V2 |

PREVIEW_FORCE_LEGACY is a dead flag with parser presence but no runtime reachability. It may cause confusion for operators who see the flag in config but it has no effect on runtime behavior. ENABLE_PREVIEW_V2 is a live alias that should be documented clearly to avoid being mistaken as a standalone flag.
