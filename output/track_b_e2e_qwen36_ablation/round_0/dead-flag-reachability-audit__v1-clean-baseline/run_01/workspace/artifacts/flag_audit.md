# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the canonical live flag. The env parser reads it in load_preview_env and sets shadow_enabled which drives the runtime branch to shadow_preview_path. The runbook confirms this is the active runtime branch for the live preview router. Parser hit plus runtime reachability both confirmed.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW. The parser accepts it for legacy deploy manifests and normalizes it to the same shadow path. It is not a standalone runtime branch — there is no independent code path that only V2 can trigger. The runbook explicitly states it normalizes and is not standalone. Parser present, runtime reachable only via alias resolution.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has a parser hit in load_preview_env but no runtime branch in the live service. The flag only sets force_legacy_seen which is consumed by legacy_force_label for reporting only. The runbook states the live service does not branch on it anymore. The test confirms plan branch remains legacy_preview_path — the flag does not alter control flow. Parser presence does not prove runtime reachability.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Canonical live flag with confirmed parser and runtime branch. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still reference this env var | Live alias with no independent runtime branch. Should be deprecated in favor of ENABLE_SHADOW_PREVIEW once legacy manifests are migrated. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | Operator notes and reporting still reference the flag | Dead flag with only reporting-level usage. Collect telemetry on deploy manifests that still set this before removing from parser and defaults. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PARSER reachability | Confirmed parser hit and runtime branch in config.py and runtime.py |
| observed | ENABLE_PREVIEW_V2 alias behavior | Confirmed alias normalization to shadow path in config.py and runbook |
| missing | Production deploy manifest inventory | No evidence of current deploy manifests using these flags found in workspace |

PREVIEW_FORCE_LEGACY is a dead flag with parser presence but no runtime branch. ENABLE_PREVIEW_V2 is a live alias that normalizes to ENABLE_SHADOW_PREVIEW. Only ENABLE_SHADOW_PREVIEW has full live parser-to-runtime reachability.
