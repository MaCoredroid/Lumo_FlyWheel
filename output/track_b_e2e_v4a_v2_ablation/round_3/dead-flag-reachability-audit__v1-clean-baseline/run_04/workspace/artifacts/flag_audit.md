# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag. The parser in config.py sets shadow_enabled=true and effective_mode=shadow when this env var is truthy. The runtime branch in runtime.py checks config.shadow_enabled and returns the shadow_preview_path. This is the primary flag controlling preview mode.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser accepts it for legacy deploy manifests and normalizes it to the same shadow path as ENABLE_SHADOW_PREVIEW. The runbook explicitly states it is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is a dead flag. While the parser tracks it in force_legacy_seen, the runtime branch in runtime.py does not check this value at all. The runbook confirms the live service does not branch on it anymore - it is left only for reporting and operator notes.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling preview mode. Essential for runtime branching. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag for backward compatibility. Can be deprecated once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | none | Dead flag with no runtime effect. Remove parser code and update documentation to prevent operator confusion. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime path | Verified shadow branch is active in runtime.py |
| observed | ENABLE_PREVIEW_V2 alias behavior | Confirmed alias maps to same shadow path |
| missing | PREVIEW_FORCE_LEGACY removal timeline | No migration schedule found for dead flag removal |

PREVIEW_FORCE_LEGACY presents the highest risk because it is a dead flag that still has parser presence, which could mislead operators into thinking it controls runtime behavior when it does not.
