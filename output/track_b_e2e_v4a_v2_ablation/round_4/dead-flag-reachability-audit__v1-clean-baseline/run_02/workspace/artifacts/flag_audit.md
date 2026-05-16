# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. It has both parser presence in load_preview_env and a direct runtime branch in preview_runtime_branch that returns shadow_preview_path when shadow_enabled is true. The test test_shadow_preview_live.py confirms end-to-end behavior.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser accepts it for legacy deploy manifests and normalizes it to ENABLE_SHADOW_PREVIEW. The runtime branch it triggers (shadow_preview_path) is the same as ENABLE_SHADOW_PREVIEW. The parser_hits tracking shows the alias relationship explicitly.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has parser presence but no runtime branch. The config sets force_legacy_seen flag which is only used by legacy_force_label for reporting/telemetry. The runtime branch in preview_runtime_branch does not check force_legacy_seen at all. The test name 'test_force_legacy_flag_is_tracked_for_reporting_only' confirms this is dead code for operational purposes.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with active runtime branch. Must remain for production preview routing. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag for backward compatibility. Can be deprecated once legacy manifests migrate to ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator notes dependency | Dead flag with no runtime impact. Remove parser after confirming telemetry and operator notes no longer depend on it. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW parser | Present in config.py load_preview_env |
| observed | ENABLE_SHADOW_PREVIEW runtime | Branches in runtime.py preview_runtime_branch |
| observed | ENABLE_PREVIEW_V2 alias | Parser normalizes to shadow path |
| observed | PREVIEW_FORCE_LEGACY dead | No runtime branch, only telemetry |
| missing | external deploy manifests | Cannot verify legacy manifest usage outside workspace |

ENABLE_SHADOW_PREVIEW is the only standalone live flag controlling the preview runtime branch. ENABLE_PREVIEW_V2 is a legacy alias. PREVIEW_FORCE_LEGACY is dead code retained only for telemetry.
