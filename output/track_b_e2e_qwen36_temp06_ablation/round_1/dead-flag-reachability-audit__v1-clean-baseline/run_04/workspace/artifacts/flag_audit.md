# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_shadow_preview_live.py` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in load_preview_env, sets shadow_enabled and effective_mode, and the runtime branch in preview_runtime_branch checks config.shadow_enabled to return the shadow_preview_path. The dedicated test confirms the end-to-end path from env var through parser to runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW. The parser accepts it and normalizes it to the same shadow_enabled=True and shadow path. It has no independent runtime branch. The test confirms it maps to the shadow path, and the runbook explicitly states it is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `config/defaults.toml`, `src/preview/config.py`, `src/preview/runtime.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is dead at runtime. The parser records force_legacy_seen=True but preview_runtime_branch never checks it. The branch decision depends solely on shadow_enabled. The test explicitly asserts the branch remains legacy_preview_path when this flag is set, and legacy_force_label is a reporting-only helper. The runbook confirms the live service does not branch on it anymore.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling the preview runtime branch. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias only. Normalize all deploy manifests to ENABLE_SHADOW_PREVIEW, then remove the elif branch in load_preview_env. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator notes and reporting dashboards may still display force_legacy_seen in build_preview_plan output | Parser branch and legacy_force_label are reporting-only. Remove from config parsing once operator dashboards are updated, and clean up legacy.py. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branching | Confirmed by config.py, runtime.py, and test_shadow_preview_live.py |
| observed | ENABLE_PREVIEW_V2 alias normalization | Confirmed by config.py elif branch and test_preview_v2_alias.py |
| missing | External feature-flag service integration | No evidence of remote flag service or feature-flag SDK; env vars appear to be the sole mechanism |

Three flags analyzed: ENABLE_SHADOW_PREVIEW is the sole live runtime branch controlling preview routing. ENABLE_PREVIEW_V2 is a legacy alias normalizing to the same path. PREVIEW_FORCE_LEGACY is dead at runtime with parser present but no branching.
