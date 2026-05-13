# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag with both parser presence and runtime reachability. The parser sets shadow_enabled and effective_mode, and runtime.py branches on config.shadow_enabled to return shadow_preview_path.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `—` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. Parser accepts it for legacy deploy manifests and normalizes to shadow path, but there is no independent runtime branch symbol - it maps to ENABLE_SHADOW_PREVIEW's path.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is dead - parser presence does not prove runtime reachability. The flag only sets force_legacy_seen for reporting purposes; the live service does not branch on it anymore per the runbook documentation.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Active live flag with both parser and runtime reachability; core control for preview shadow mode. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag maintained for backward compatibility; deprecate once legacy manifests migrate completes. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator notes reference | Dead flag with parser presence; collect telemetry on usage before removal to avoid breaking operator workflows. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser_hits_tracking | parser_hits dict tracks which env vars were read |
| observed | runtime_branching | runtime.py only branches on shadow_enabled, not force_legacy_seen |
| missing | external_deploy_manifests | No deploy manifest files found in this workspace copy |

PREVIEW_FORCE_LEGACY presents highest risk due to misleading parser presence suggesting active control; dead flag may cause operator confusion if removed without telemetry-first deprecation.
