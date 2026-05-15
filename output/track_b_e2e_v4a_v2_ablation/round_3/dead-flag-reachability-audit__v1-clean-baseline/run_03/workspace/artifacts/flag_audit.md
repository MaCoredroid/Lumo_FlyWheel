# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`

ENABLE_SHADOW_PREVIEW is the primary live flag. The parser in config.py sets shadow_enabled=True and effective_mode='shadow'. The runtime.py branches on config.shadow_enabled to shadow_preview_path. The runbook confirms this is the live preview router branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is parsed but is an alias that normalizes to ENABLE_SHADOW_PREVIEW. The runbook states it is accepted for legacy deploy manifests but is not a standalone runtime branch. It shares the same runtime branch as ENABLE_SHADOW_PREVIEW.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed and sets force_legacy_seen=True but does not affect runtime branching. The runtime.py preview_runtime_branch function only checks config.shadow_enabled, ignoring force_legacy_seen. The runbook confirms it is left in reporting and operator notes only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling the preview router. Essential for runtime branching. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Can be deprecated once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator notes may reference this flag | Flag is dead for runtime but may be referenced in operator documentation. Clean up docs before removing parser. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | runtime_branching | runtime.py only branches on shadow_enabled |
| observed | alias_behavior | ENABLE_PREVIEW_V2 maps to same path as ENABLE_SHADOW_PREVIEW |
| missing | external_deploy_manifests | Cannot verify legacy deploy manifest usage in production |

PREVIEW_FORCE_LEGACY presents the highest risk because operators may believe it forces legacy mode when it only affects reporting labels. The flag appears functional but has no runtime effect on the live service path.
