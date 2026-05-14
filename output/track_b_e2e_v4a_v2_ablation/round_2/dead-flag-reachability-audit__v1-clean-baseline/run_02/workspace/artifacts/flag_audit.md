# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag that directly enables shadow mode and causes effective_mode to become shadow and runtime branch to shadow_preview_path. Parser presence confirmed in config.py, runtime reachability confirmed in runtime.py, default value documented in defaults.toml.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is accepted by parser for legacy deploy manifests but normalizes to ENABLE_SHADOW_PREVIEW. It is not a standalone runtime branch - it is a live alias that maps to the same shadow path as the primary flag.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `config/defaults.toml`

PREVIEW_FORCE_LEGACY is only tracked for reporting purposes. Parser presence exists but runtime branch remains legacy_preview_path regardless of flag value. The flag sets force_legacy_seen but does not alter runtime branching behavior.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag for shadow preview rollout. Must remain for operational control. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias for ENABLE_SHADOW_PREVIEW. Can be removed after legacy deploy manifests migrate to use primary flag directly. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | telemetry consumers, operator notes | Dead flag with no runtime effect. Remove after telemetry consumers and operator documentation updated to not expect this flag. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW live path | Confirmed in config.py and runtime.py |
| observed | ENABLE_PREVIEW_V2 alias behavior | Normalizes to shadow |
| missing | external telemetry consumers | Unknown external systems that may read PREVIEW_FORCE_LEGACY |

PREVIEW_FORCE_LEGACY is dead code that only affects telemetry labels. Removing it requires updating legacy.py and removing parser hit tracking in config.py. ENABLE_PREVIEW_V2 alias can be deprecated after legacy deploy manifests are migrated.
