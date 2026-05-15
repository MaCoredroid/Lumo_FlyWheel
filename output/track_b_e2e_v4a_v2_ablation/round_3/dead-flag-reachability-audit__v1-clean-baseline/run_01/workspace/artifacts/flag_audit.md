# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is the primary live flag that controls the shadow preview feature. The parser reads it in load_preview_env, sets shadow_enabled to true, and the runtime branches to shadow_preview_path when enabled. This is confirmed by the test and runbook documentation.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a legacy alias that normalizes to ENABLE_SHADOW_PREVIEW in the parser. It does not have a standalone runtime branch - it maps to the same shadow_preview_path. The runbook confirms it is accepted for legacy deploy manifests but is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is parsed and tracked for reporting only. The parser sets force_legacy_seen but the runtime does not branch on it. The runbook confirms the live service does not branch on it anymore - it is left in reporting and operator notes only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag - do not remove |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Legacy alias still needed for backward compatibility but should be deprecated in favor of ENABLE_SHADOW_PREVIEW |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | none | Parser symbol can be removed once reporting is migrated |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser_vs_runtime | Parser presence does not imply runtime reachability |
| observed | alias_behavior | ENABLE_PREVIEW_V2 normalizes to ENABLE_SHADOW_PREVIEW |
| missing | migration_timeline | No timeline for removing legacy alias |

ENABLE_SHADOW_PREVIEW is the primary control for the live shadow preview feature. ENABLE_PREVIEW_V2 is a deprecated alias. PREVIEW_FORCE_LEGACY is dead - parser present but no runtime reachability.
