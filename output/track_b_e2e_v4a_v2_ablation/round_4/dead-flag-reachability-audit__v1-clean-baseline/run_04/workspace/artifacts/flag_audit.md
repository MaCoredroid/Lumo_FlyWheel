# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is a live standalone flag. The parser reads it in config.py, sets shadow_enabled=true and effective_mode=shadow, and the runtime branch in runtime.py returns 'preview_runtime_branch:shadow_preview_path' when shadow_enabled is true. The test confirms this produces the shadow preview path.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser accepts it for legacy deploy manifests but normalizes it to ENABLE_SHADOW_PREVIEW behavior. It triggers the same shadow path but is explicitly documented as an alias that does not have its own runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is a dead flag. While the parser reads it and sets force_legacy_seen=true for reporting purposes, the runtime branch in runtime.py does not check this flag at all. The test confirms that even with PREVIEW_FORCE_LEGACY=1, the branch remains 'preview_runtime_branch:legacy_preview_path' - the default. The runbook explicitly states 'The live service does not branch on it anymore.'

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag controlling the preview feature. It has both parser presence and runtime reachability. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | This is an alias for ENABLE_SHADOW_PREVIEW kept for backward compatibility. Should be deprecated once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | none | This flag has no runtime effect and should be removed from documentation and operator notes. The parser can remain for telemetry but should be clearly marked as deprecated. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser_vs_runtime | Parser presence does not equal runtime reachability |
| observed | alias_semantics | ENABLE_PREVIEW_V2 normalizes to ENABLE_SHADOW_PREVIEW |
| missing | external_deploy_manifests | No access to actual deploy manifests in this workspace |

PREVIEW_FORCE_LEGACY presents the highest risk because operators may believe it forces legacy mode when it actually has no effect on runtime branching. The flag is parsed and tracked for reporting but does not influence the live service branch decision.
