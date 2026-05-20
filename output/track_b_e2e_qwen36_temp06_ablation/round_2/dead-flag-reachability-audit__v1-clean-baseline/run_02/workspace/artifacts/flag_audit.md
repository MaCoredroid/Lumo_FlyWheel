# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`, `tests/test_shadow_preview_live.py`

ENABLE_SHADOW_PREVIEW is parsed in load_preview_env (sets shadow_enabled=True and effective_mode=shadow) and the runtime branches on it in preview_runtime_branch returning shadow_preview_path. This is the only flag with both parser presence AND a dedicated runtime branch, making it live.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is accepted by the parser for legacy deploy manifests but normalizes to ENABLE_SHADOW_PREVIEW. It sets the same shadow_enabled flag with no standalone runtime branch. The runbook confirms it is not a standalone runtime branch, making it partial.

| `PREVIEW_FORCE_LEGACY` | partial | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is parsed in load_preview_env (sets force_legacy_seen=True) but the live service does not branch on it. The runbook confirms it is left in reporting and operator notes only. The legacy_force_label helper uses it for reporting, not runtime control, making it partial.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Live and actively used for shadow preview routing. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias normalizing to ENABLE_SHADOW_PREVIEW; deprecate in favor of canonical flag. |
| `PREVIEW_FORCE_LEGACY` | `do_not_remove_now` | operator tooling may depend on the legacy-forced reporting label, no migration path defined for force_legacy_seen consumers | Parser present but no runtime branch; safe to keep for now as reporting-only. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | Parser presence vs runtime reachability | Parser hits in load_preview_env do not guarantee runtime branching; confirmed by PREVIEW_FORCE_LEGACY having a parser hit but no runtime branch. |
| observed | ENABLE_PREVIEW_V2 alias normalization | ENABLE_PREVIEW_V2 normalizes to ENABLE_SHADOW_PREVIEW in the parser; no standalone runtime branch exists. |
| missing | Deployment rollout percentages | No evidence of feature flag rollout percentages or gradual rollout configuration in this codebase. |

ENABLE_SHADOW_PREVIEW is the only live flag with a dedicated runtime branch controlling the shadow preview path. It represents the highest operational risk as it directly affects runtime behavior.
