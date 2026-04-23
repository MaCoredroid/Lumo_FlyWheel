# Flag Audit

- variant: `v5-recovery-in-thread`
- highest_operational_risk: `ENABLE_PREVIEW_V2`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `src/preview/runtime.py`, `src/preview/service.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is live because the env parser records it and the service reaches the live shadow runtime branch when it is enabled.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `src/preview/service.py` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `src/preview/service.py`, `tests/test_preview_v2_alias.py`, `release_context/release_freeze_guidance.md`, `incident_context/preview_alias_cleanup_rollback.md`

ENABLE_PREVIEW_V2 is partial: the parser still accepts it, but only as a legacy alias that maps into ENABLE_SHADOW_PREVIEW rather than a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `repo_evidence/in_progress_patch/preview_force_legacy.patch` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`, `repo_evidence/in_progress_patch/preview_force_legacy.patch`, `incident_context/recovery_thread.md`

PREVIEW_FORCE_LEGACY is dead in the live runtime: the parser records the flag for reporting, but the service never branches on it and the only extra references are stale helpers or abandoned patches.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Keep the live rollout control because the runtime still branches on the shadow preview path. |
| `ENABLE_PREVIEW_V2` | `remove_after_migration` | rollback incident requires explicit migration proof | Do not remove the alias until manifest migration is verified post-incident; removal is only safe after migration evidence exists. |
| `PREVIEW_FORCE_LEGACY` | `do_not_remove_now` | incident follow-up still open | Keep the stale references documented until the rollback follow-up is complete; avoid another high-noise cleanup while the incident thread is open. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | live shadow branch | The runtime branch symbol is explicit in src/preview/runtime.py. |
| to_verify | deploy manifest migration completion | Alias removal depends on confirming deploy manifests no longer use ENABLE_PREVIEW_V2. |
| missing | operator telemetry on legacy alias usage | The bundle does not include current rollout telemetry proving alias usage is fully gone. |

The rollback incident means cleanup must explicitly wait for migration completion. Missing that incident is an operational judgment failure even if the classifications are otherwise close.
