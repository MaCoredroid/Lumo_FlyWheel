# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in load_preview_env (config.py), sets shadow_enabled=True and effective_mode=shadow, and the runtime branch in runtime.py checks config.shadow_enabled to return the shadow_preview_path. The test test_shadow_preview_live.py confirms end-to-end behavior. The runbook documents it as the live router branch.

| `ENABLE_PREVIEW_V2` | live | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW. The parser in config.py accepts it in the elif branch and normalizes it to the same shadow_enabled=True / effective_mode=shadow config, producing the identical parser hit suffix ->ENABLE_SHADOW_PREVIEW. The runtime branch symbol is the same shadow_preview_path. The runbook explicitly states it is accepted for legacy deploy manifests but is not a standalone runtime branch. The test confirms the alias maps to the shadow path.

| `PREVIEW_FORCE_LEGACY` | partial | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed in load_preview_env and sets force_legacy_seen=True, but it does NOT affect the runtime branch decision. The runtime.py preview_runtime_branch function only checks shadow_enabled, so PREVIEW_FORCE_LEGACY has no runtime_branch_symbol. The runbook confirms the live service does not branch on it anymore. The legacy.py legacy_force_label function uses force_legacy_seen only for a reporting label string, not for any control-flow decision. The test confirms the branch remains legacy_preview_path and the label is legacy-forced, proving it is reporting-only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag controlling the shadow preview path. It must be kept as-is since it is the sole runtime branch driver. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | This is a live alias that normalizes to ENABLE_SHADOW_PREVIEW. It should be deprecated in docs and eventually removed once all deploy manifests migrate to the canonical flag name. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operators may rely on the reporting label in legacy.py | This flag is parsed but has no runtime branch effect. Before removing it, telemetry should confirm no operators depend on the force_legacy_seen reporting label. The parser hit and legacy label function should be removed together. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed by runtime.py and test_shadow_preview_live.py |
| observed | ENABLE_PREVIEW_V2 alias normalization | Confirmed by config.py elif branch and test_preview_v2_alias.py |
| missing | PREVIEW_FORCE_LEGACY downstream consumers | No evidence of force_legacy_seen being consumed outside legacy.py reporting label function |

PREVIEW_FORCE_LEGACY is the highest operational risk because it is a partial flag with parser presence but no runtime branch, making it a candidate for dead code that could silently mislead operators who believe the flag forces legacy mode when it only produces a reporting label.
