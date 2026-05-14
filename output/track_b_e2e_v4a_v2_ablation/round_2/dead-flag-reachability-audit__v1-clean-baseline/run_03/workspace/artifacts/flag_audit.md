# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live flag with both parser presence and runtime reachability. The config.py parser sets shadow_enabled=true and effective_mode=shadow when this flag is truthy. The runtime.py preview_runtime_branch function returns 'preview_runtime_branch:shadow_preview_path' when config.shadow_enabled is true. The test test_shadow_preview_live.py validates this end-to-end behavior.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a partial flag - it has parser presence but is NOT a standalone runtime branch. The docs explicitly state it 'normalizes to the same shadow path and is not a standalone runtime branch'. It is an alias for ENABLE_SHADOW_PREVIEW, accepted only for legacy deploy manifests. The parser symbol shows the aliasing relationship with the arrow notation.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is a dead flag - it has parser presence but NO runtime reachability. The docs state 'The live service does not branch on it anymore'. The test name 'test_force_legacy_flag_is_tracked_for_reporting_only' confirms it is only tracked for reporting. The legacy.py module only provides a label function (legacy_force_label) that does not affect runtime branching. The runtime.py preview_runtime_branch function does not check force_legacy_seen at all.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag for the preview feature. It has full parser and runtime support. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | This is an alias for ENABLE_SHADOW_PREVIEW kept only for legacy deploy manifests. Should be deprecated with migration path. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator notes dependency | This flag is dead for runtime but still used in reporting and operator notes. Collect telemetry before removal. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser_vs_runtime | Parser presence does not guarantee runtime reachability |
| observed | alias_distinction | ENABLE_PREVIEW_V2 is an alias, not a standalone flag |
| missing | operator_migration | No timeline for PREVIEW_FORCE_LEGACY operator migration |

ENABLE_SHADOW_PREVIEW is the primary live flag controlling the preview runtime branch. ENABLE_PREVIEW_V2 is a deprecated alias that should be migrated away from. PREVIEW_FORCE_LEGACY is dead code that can be safely be removed from the parser after telemetry migration.
