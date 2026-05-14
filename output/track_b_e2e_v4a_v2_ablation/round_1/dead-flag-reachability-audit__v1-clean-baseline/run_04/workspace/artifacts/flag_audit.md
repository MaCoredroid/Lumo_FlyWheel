# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag that controls the shadow preview mode. When set, it sets shadow_enabled=true and effective_mode=shadow in config, which causes preview_runtime_branch to return shadow_preview_path. This is the canonical flag with direct parser presence and runtime reachability.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a legacy alias that maps to the same shadow path as ENABLE_SHADOW_PREVIEW. It has parser presence (config.parser_hits records it) but is not a standalone runtime flag—it normalizes to ENABLE_SHADOW_PREVIEW behavior. The runbook confirms it's kept for legacy deploy manifests only.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has parser presence but no runtime reachability. The parser sets force_legacy_seen=true, but preview_runtime_branch never branches on this flag—it always returns legacy_preview_path when shadow is disabled. The flag only affects legacy_force_label for reporting/telemetry purposes. The runbook explicitly states the live service does not branch on it anymore.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with full parser and runtime reachability. Essential for shadow preview functionality. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference this flag | Alias flag kept for backward compatibility. Should be deprecated with migration path to ENABLE_SHADOW_PREVIEW once legacy manifests are updated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator notes reference this flag, telemetry reports may still include force_legacy_seen | Dead flag with no runtime effect. Remove from parser after updating operator documentation and telemetry to avoid confusion about its purpose. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime path | Verified in runtime.py preview_runtime_branch function |
| observed | ENABLE_PREVIEW_V2 alias mapping | Parser maps V2 to shadow path, confirmed in config.py |
| observed | PREVIEW_FORCE_LEGACY no runtime branch | runtime.py does not reference force_legacy_seen |
| missing | integration tests for PREVIEW_FORCE_LEGACY removal | No integration tests found for this flag removal |

PREVIEW_FORCE_LEGACY is a dead flag with parser presence but no runtime branching effect. Operators may mistakenly believe it controls runtime behavior when it only affects telemetry labels. This creates operational confusion and should be deprecated with clear migration guidance.
