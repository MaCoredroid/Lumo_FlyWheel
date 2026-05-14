# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag. The parser reads it in load_preview_env and sets shadow_enabled=true. The runtime branch in preview_runtime_branch() checks config.shadow_enabled and returns shadow_preview_path. The runbook confirms this is the live preview router branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser accepts by env parser for legacy deploy manifests but normalizes to ENABLE_SHADOW_PREVIEW. It reaches the same shadow_preview_path runtime branch but only as an alias, not as an independent control.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is a dead flag. The parser reads it and sets force_legacy_seen for telemetry/reporting only. The live service does NOT branch on it - preview_runtime_branch() ignores force_legacy_seen entirely. The runbook confirms it is left in reporting and operator notes only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live control for shadow preview routing. Essential for feature rollout operations. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference it | Alias flag that should be deprecated after legacy manifests migrate to ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator notes may reference it | Remove from active config parsing but keep telemetry collection for migration tracking. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser_presence | All three flags have parser symbols in config.py |
| observed | runtime_branching | Only shadow_enabled affects runtime branch selection |
| missing | external_service_integration | No evidence of PREVIEW_FORCE_LEGACY being read by live service |

PREVIEW_FORCE_LEGACY presents the highest operational risk because operators may expect it to force legacy mode but it has no runtime effect. Dead flag confusion can lead to misdiagnosis during incidents.
