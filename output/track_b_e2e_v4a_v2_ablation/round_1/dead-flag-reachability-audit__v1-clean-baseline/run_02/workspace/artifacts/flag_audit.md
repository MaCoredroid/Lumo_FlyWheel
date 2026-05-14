# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is the primary live flag. Parser in config.py sets shadow_enabled and effective_mode. Runtime in runtime.py branches on shadow_enabled to return shadow_preview_path. Test confirms end-to-end path from parser to runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a legacy alias that normalizes to ENABLE_SHADOW_PREVIEW. Parser hit exists but it maps to the same shadow path. Not a standalone runtime branch per runbook docs. Live as alias only, not independent.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY has parser presence but no runtime reachability. Parser sets force_legacy_seen for reporting only. The legacy_force_label function is for telemetry, not branching. Runbook confirms live service does not branch on it anymore.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with full runtime reachability. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `docs_cleanup` | legacy deploy manifests | Alias flag for backward compatibility. Document that it normalizes to ENABLE_SHADOW_PREVIEW. Remove when legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator reporting workflows | Dead flag kept for telemetry. Monitor usage before removal. Update operator notes to clarify it has no runtime effect. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed via runtime.py and test |
| observed | ENABLE_PREVIEW_V2 alias behavior | Confirmed via config.py normalization |
| observed | PREVIEW_FORCE_LEGACY dead status | No runtime branch found in runtime.py |
| missing | External migration status for legacy manifests | Not yet verified in this workspace copy |

Dead flag PREVIEW_FORCE_LEGACY remains in parser for backward compatibility but has no runtime effect. Risk of confusion if operators enable it expecting behavior change. Recommend docs_cleanup to clarify it is reporting-only.
