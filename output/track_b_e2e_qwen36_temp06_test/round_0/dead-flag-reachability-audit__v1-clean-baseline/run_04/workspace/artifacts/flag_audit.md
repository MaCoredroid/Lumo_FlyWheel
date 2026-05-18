# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in config.py (sets shadow_enabled and effective_mode), and the runtime branch in runtime.py explicitly checks config.shadow_enabled to return the shadow_preview_path. The test in test_shadow_preview_live.py confirms end-to-end parser-to-runtime reachability. This flag is fully live with both parser presence and runtime branching.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is parsed by the env parser in config.py but immediately normalizes to the same shadow path as ENABLE_SHADOW_PREVIEW. It is an alias, not a standalone flag. The runtime branch symbol it reaches is identical to ENABLE_SHADOW_PREVIEW, confirming it shares the same runtime path. The runbook explicitly states it is not a standalone runtime branch. Parser presence does not equal independent runtime reachability per false_positive_notes.md.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed in config.py (sets force_legacy_seen) but the live runtime branch in runtime.py does not inspect force_legacy_seen at all. The only consumer is legacy_force_label in legacy.py, which is a reporting-only label function, not a runtime branch. The runbook confirms the live service no longer branches on it. Per false_positive_notes.md, a parser hit and a test helper do not prove the live service reads the flag. This flag is dead at runtime.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with full parser-to-runtime reachability. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias flag with no standalone runtime branch. Can be deprecated once legacy deploy manifests are migrated to ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator reporting tools may still read force_legacy_seen | Dead at runtime — parser sets the flag but no live service branches on it. Only legacy reporting labels consume it. Safe to remove from parser after confirming reporting tools no longer depend on force_legacy_seen. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PARSER runtime branch | Confirmed in runtime.py: shadow_enabled check returns shadow_preview_path |
| observed | ENABLE_PREVIEW_V2 alias behavior | Config.py elif branch normalizes V2 to shadow_enabled True |
| missing | PREVIEW_FORCE_LEGACY production telemetry | No telemetry or metrics data available to confirm whether force_legacy_seen is still read by downstream reporting pipelines |

ENABLE_SHADOW_PREVIEW is the sole live flag controlling the shadow preview runtime path. It carries the highest operational risk because changes to its parser or runtime branch directly affect the live preview router behavior. ENABLE_PREVIEW_V2 is a safe alias with no independent path. PREVIEW_FORCE_LEGACY is dead at runtime and only surfaces in reporting labels.
