# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is a live standalone flag. The parser in src/preview/config.py reads the env var and sets shadow_enabled=True and effective_mode=shadow. The runtime in src/preview/runtime.py branches on config.shadow_enabled, returning the shadow_preview_path. The test confirms the parser hit and the branch outcome. The runbook documents this as the primary shadow control. The false-positive note that parser presence does not prove a runtime branch is disproved here because we can trace the full path from env var through config.shadow_enabled to the runtime branch.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW, not a standalone flag. The parser in src/preview/config.py accepts it and normalizes to the same shadow_enabled=True and shadow effective_mode, but it does not introduce a new runtime branch. The runbook explicitly states it is not a standalone runtime branch. The test confirms the alias mapping to shadow_preview_path. The false-positive note that parser presence does not prove a runtime branch applies here: the parser hit exists but there is no distinct runtime branch for this flag — it aliases are not standalone flags.

| `PREVIEW_FORCE_LEGACY` | partial | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY has parser presence but no runtime reachability. The parser in src/preview/config.py reads the env var and sets force_legacy_seen=True, but the runtime in src/preview/runtime.py never reads force_legacy_seen — it only checks shadow_enabled. The flag is only used by legacy.py for reporting labels. The runbook explicitly states the live service does not branch on it anymore. The test confirms the branch remains legacy_preview_path even when the flag is set. The false-positive note that parser presence does not prove a runtime branch is directly demonstrated here: the parser hit exists but there is no runtime branch.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag controlling the shadow preview runtime path. It must remain until the shadow path is permanently retired. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | This flag is an alias with no standalone behavior. It should be deprecated in favor of ENABLE_SHADOW_PREVIEW once legacy deploy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | Reporting pipeline depends on force_legacy_seen for operator labels | This flag is parsed but has no runtime branch. Before removal, telemetry should confirm no operator tooling depends on the reporting-only label. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed in src/preview/runtime.py branching on shadow_enabled |
| observed | ENABLE_PREVIEW_V2 alias normalization | Confirmed in src/preview/config.py normalizing to shadow_enabled |
| missing | PREVIEW_FORCE_LEGACY downstream consumers | No evidence of force_legacy_seen being read outside legacy.py in this workspace |

ENABLE_SHADOW_PREVIEW is the sole live standalone flag controlling the shadow preview runtime path. ENABLE_PREVIEW_V2 is a legacy alias that normalizes to the same path. PREVIEW_FORCE_LEGACY is parsed for reporting but has no runtime branch in the live service.
