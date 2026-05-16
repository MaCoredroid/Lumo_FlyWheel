# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag with both parser presence and runtime reachability. The config parser sets shadow_enabled and effective_mode, and the runtime branches to shadow_preview_path when enabled.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a partial flag - it has parser presence but is an alias that normalizes to ENABLE_SHADOW_PREVIEW. It is not a standalone runtime branch; it reuses the same shadow path as ENABLE_SHADOW_PREVIEW.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is a dead flag - it has parser presence but no runtime branch. The live service does not branch on it anymore; it is only used for reporting and telemetry via legacy_force_label.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Live flag with active runtime branching; core feature control |
| `ENABLE_PREVIEW_V2` | `docs_cleanup` | legacy deploy manifests | Alias flag for backward compatibility; document as deprecated alias |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator notes reference | Dead flag used only for telemetry; collect usage data before removal |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Verified in runtime.py and tests |
| observed | ENABLE_PREVIEW_V2 alias behavior | Confirmed normalizes to shadow path |
| missing | External deploy manifest usage | Cannot verify legacy manifest frequency |

PREVIEW_FORCE_LEGACY is dead code that should be removed to reduce maintenance burden. ENABLE_PREVIEW_V2 should be documented as an alias to avoid confusion.
