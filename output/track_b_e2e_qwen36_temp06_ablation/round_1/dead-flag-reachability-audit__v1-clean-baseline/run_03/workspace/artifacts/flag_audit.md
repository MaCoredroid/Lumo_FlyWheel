# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_shadow_preview_live.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag. The parser in config.py sets shadow_enabled and effective_mode, and the runtime branch in runtime.py checks shadow_enabled to return the shadow preview path. The test confirms the full end-to-end path from env var to runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW. The parser accepts it and normalizes to the same shadow path, but it does not have its own standalone runtime branch. The runbook explicitly states it is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is dead for runtime branching. The parser still reads it and sets force_legacy_seen for reporting, but the runtime branch in runtime.py does not check force_legacy_seen. The runbook confirms the live service does not branch on it anymore.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling the preview shadow path. No cleanup needed at this time. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias flag with no standalone runtime branch. Deprecate in favor of ENABLE_SHADOW_PREVIEW once legacy deploy manifests are updated. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | Operator reporting pipelines may depend on force_legacy_seen telemetry | Dead for runtime branching but still parsed for reporting. Collect telemetry on force_legacy_seen usage before removal. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | Parser presence | All three flags are parsed by load_preview_env in config.py |
| observed | Runtime branching | Only shadow_enabled is checked in preview_runtime_branch |
| missing | Production deployment manifests | No CI or deployment config found in workspace to verify live flag values |

ENABLE_SHADOW_PREVIEW is the only flag with full live parser and runtime reachability. ENABLE_PREVIEW_V2 is a deprecated alias. PREVIEW_FORCE_LEGACY is dead for branching but retained only for reporting.
