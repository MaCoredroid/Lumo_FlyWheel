# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `docs/preview_rollout_runbook.md`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is the canonical live flag. The parser in config.py sets shadow_enabled and effective_mode, and runtime.py branches on shadow_enabled to return the shadow preview path. The runbook confirms this is the active runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser accepts it for legacy deploy manifests but normalizes it to the same shadow path as ENABLE_SHADOW_PREVIEW. It has no independent runtime branch in runtime.py, confirmed by the runbook.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is dead. The parser records it but the live service does not branch on it. It only sets force_legacy_seen which is consumed by legacy_force_label for reporting. The runbook explicitly states it is left in reporting and operator notes only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the canonical live flag with active runtime branching. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias that normalizes to ENABLE_SHADOW_PREVIEW. Deprecate after confirming all deploy manifests use ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | operator notes and reporting pipelines may still reference force_legacy_seen | Dead flag with no runtime branch. Remove parser branch, dataclass field, legacy.py helper, and test after confirming reporting pipelines no longer depend on force_legacy_seen. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW is the only flag with a live runtime branch | Confirmed by inspecting preview_runtime_branch in runtime.py which only checks config.shadow_enabled |
| observed | ENABLE_PREVIEW_V2 normalizes to ENABLE_SHADOW_PREVIEW | Parser sets the same shadow_enabled flag and effective_mode shadow, confirmed by runbook |
| missing | Downstream consumers of force_legacy_seen outside this workspace | Cannot verify whether external reporting pipelines still read force_legacy_seen labels from this service |

PREVIEW_FORCE_LEGACY is dead code that should be removed after migration. ENABLE_PREVIEW_V2 is a safe alias that can be deprecated after deploy manifests migrate to ENABLE_SHADOW_PREVIEW.
