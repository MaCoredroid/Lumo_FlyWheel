# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is the primary live flag that directly controls the shadow preview mode. When set to a truthy value, it sets shadow_enabled=True and effective_mode='shadow', causing preview_runtime_branch to return 'preview_runtime_branch:shadow_preview_path'. This is the canonical runtime branch for the preview feature.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a legacy alias that normalizes to ENABLE_SHADOW_PREVIEW. It has parser presence and reaches the same runtime branch, but it is not a standalone flag—it is explicitly documented as an alias for legacy deploy manifests. The parser_hit shows the normalization chain 'ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW'.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY has parser presence (sets force_legacy_seen=True) but no runtime reachability. The runtime branch remains 'preview_runtime_branch:legacy_preview_path' regardless of this flag. It is used only for reporting via legacy_force_label() and operator notes. The test explicitly verifies the branch is legacy_preview_path even when the flag is set.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with direct runtime control. Essential for preview feature operation. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag for backward compatibility. Should be deprecated once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | reporting systems using legacy_force_label | Dead flag with no runtime effect. Remove after reporting migration. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Directly controls shadow_preview_path |
| observed | ENABLE_PREVIEW_V2 alias normalization | Normalizes to ENABLE_SHADOW_PREVIEW |
| missing | PREVIEW_FORCE_LEGACY historical usage data | No usage metrics available for dead flag |

ENABLE_SHADOW_PREVIEW is the only standalone live flag controlling the preview runtime branch. ENABLE_PREVIEW_V2 is a partial alias for backward compatibility. PREVIEW_FORCE_LEGACY is dead—parser present but no runtime effect.
