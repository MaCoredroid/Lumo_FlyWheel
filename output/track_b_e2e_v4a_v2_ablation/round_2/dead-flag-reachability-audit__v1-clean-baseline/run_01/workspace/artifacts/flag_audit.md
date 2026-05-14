# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag with both parser presence and runtime reachability. When enabled, it sets shadow_enabled=true and effective_mode=shadow, causing preview_runtime_branch to return 'preview_runtime_branch:shadow_preview_path'. The test test_shadow_preview_live.py confirms this changes the runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a partial flag - it has parser presence but is only an alias that normalizes to ENABLE_SHADOW_PREVIEW. It is not a standalone runtime branch. The docs explicitly state it is accepted for legacy deploy manifests but normalizes to the same shadow path. Test confirms it maps to shadow path but is not independent.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is a dead flag - it has parser presence but NO runtime reachability. The docs explicitly state 'The live service does not branch on it anymore'. The test confirms that when PREVIEW_FORCE_LEGACY is set, the branch remains 'preview_runtime_branch:legacy_preview_path' (the default). It only affects legacy_force_label for reporting purposes.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Live standalone flag with full runtime control of the shadow preview path. No blockers. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference it | Alias that normalizes to ENABLE_SHADOW_PREVIEW. Should be deprecated after legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | operator notes may reference it, reporting systems may log it | Dead flag with no runtime effect. Remove after operators and reporting systems stop referencing it. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed via config.py and runtime.py |
| observed | ENABLE_PREVIEW_V2 alias behavior | Confirmed via config.py and docs |
| observed | PREVIEW_FORCE_LEGACY dead status | Confirmed via docs and test |
| missing | migration timeline for legacy deploy manifests | No timeline found for ENABLE_PREVIEW_V2 removal |

PREVIEW_FORCE_LEGACY is the highest risk as it is a dead flag with parser presence but no runtime effect, which could mislead operators into thinking it controls legacy behavior when it only affects reporting labels.
