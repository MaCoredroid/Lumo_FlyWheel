# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `docs/preview_rollout_runbook.md`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is parsed in load_preview_env, sets shadow_enabled to true, and the runtime branch in preview_runtime_branch checks config.shadow_enabled to return the shadow preview path. This flag is fully live with parser and runtime reachability confirmed by both code and tests.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is parsed by the env parser and normalizes to the same shadow preview path as ENABLE_SHADOW_PREVIEW, but it is not a standalone runtime branch. The runbook explicitly states it is accepted for legacy deploy manifests only. It is a live alias, not a live standalone flag.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`, `config/defaults.toml`

PREVIEW_FORCE_LEGACY is parsed and sets force_legacy_seen to true, but the live service runtime does not branch on it. The runbook confirms it is left in reporting and operator notes only. The test name itself states it is for reporting only. Parser presence does not prove runtime reachability, as noted in false_positive_notes.md.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the primary live flag controlling the shadow preview path. Removing it would break the active rollout mechanism. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | This flag is a live alias to ENABLE_SHADOW_PREVIEW. It should be deprecated with a migration path for legacy deploy manifests that still reference it. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator reporting depends on force_legacy_seen label | This flag is dead at runtime but still parsed for reporting. Before removal, telemetry should confirm no operators depend on the force_legacy_seen reporting label. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime reachability | Confirmed by runtime.py branching on config.shadow_enabled |
| observed | ENABLE_PREVIEW_V2 alias normalization | Config.py maps V2 to the same shadow path as shadow preview |
| missing | PREVIEW_FORCE_LEGACY operator dependency | No evidence of live operator dashboards consuming force_legacy_seen label |

ENABLE_SHADOW_PREVIEW is the sole live standalone flag driving the shadow preview runtime branch. ENABLE_PREVIEW_V2 is a legacy alias normalizing to the same path. PREVIEW_FORCE_LEGACY is dead at runtime and only tracked for reporting.
