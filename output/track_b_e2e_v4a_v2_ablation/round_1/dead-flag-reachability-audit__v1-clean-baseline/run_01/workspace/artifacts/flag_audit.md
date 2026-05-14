# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag that directly controls the runtime branch. When set to truthy, it enables shadow mode and causes preview_runtime_branch to return the shadow_preview_path. The runbook documents this as the primary control.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a partial flag that acts as an alias for ENABLE_SHADOW_PREVIEW. The parser accepts it for legacy deploy manifests but normalizes it to the same shadow path. It is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is a dead flag. The parser tracks it for reporting purposes only, but the live service does not branch on it. The test confirms it only affects legacy_force_label reporting, not the runtime branch.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag that controls preview routing. Must remain active for production use. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | This is an alias for ENABLE_SHADOW_PREVIEW kept for legacy compatibility. Deprecate once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | none | This flag is dead and only used for reporting. Remove from active configs and update documentation to reflect its removal. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW | Direct runtime branch control confirmed |
| observed | ENABLE_PREVIEW_V2 | Alias normalization to shadow path |
| observed | PREVIEW_FORCE_LEGACY | Parser present but no runtime branch |
| missing | external deploy manifests | No external evidence of legacy manifest usage |

ENABLE_SHADOW_PREVIEW is the primary live flag controlling preview routing. ENABLE_PREVIEW_V2 is a legacy alias. PREVIEW_FORCE_LEGACY is dead and should be removed from active configs.
