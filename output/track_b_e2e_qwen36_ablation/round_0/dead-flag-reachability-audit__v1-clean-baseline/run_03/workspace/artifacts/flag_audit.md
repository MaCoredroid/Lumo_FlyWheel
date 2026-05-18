# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env` | `preview_runtime_branch` | `tests/test_shadow_preview_live.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is a live standalone flag. The parser in config.py reads the env var and sets shadow_enabled to true, and runtime.py branches on config.shadow_enabled to return the shadow preview path. The test confirms the branch actually changes at runtime, disproving the false-positive hypothesis that parser hits alone prove reachability.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env` | `—` | `tests/test_preview_v2_alias.py` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW, not a standalone flag. The parser in config.py accepts the env var but normalizes it to the same shadow_enabled path with no unique runtime branch. The runbook confirms this mapping. The test disproves the false positive that a separate V2 branch exists — the plan shows shadow_preview_path, identical to shadow preview.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is dead at runtime. The parser in config.py sets force_legacy_seen and records a parser hit, but runtime.py never branches on that field — it only checks shadow_enabled. The legacy.py module only uses it for a reporting label. The test disproves the false positive by showing that setting the flag still returns legacy_preview_path with no branch change, confirming it is reporting-only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Active standalone flag that controls the shadow preview runtime path; removing it would break the preview rollout control plane. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias that normalizes to ENABLE_SHADOW_PREVIEW with no unique runtime effect. Safe to remove from parser once all legacy deploy manifests are migrated to use ENABLE_SHADOW_PREVIEW directly. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | Operator dashboards may surface the force_legacy_seen field for reporting | Dead flag with no runtime branching — only sets a reporting label in legacy.py. Parser hit in config.py can be removed after confirming no operator tooling depends on the force_legacy_seen telemetry field. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | Shadow preview is the sole active runtime branch | Confirmed by runtime.py which only checks config.shadow_enabled and ignores force_legacy_seen for branching decisions. |
| observed | ENABLE_PREVIEW_V2 normalizes to shadow path | Config parser maps ENABLE_PREVIEW_V2 to shadow_enabled with parser hit recording the alias chain. |
| missing | Operator telemetry consumption of force_legacy_seen | No evidence found in this workspace copy of any dashboard or pipeline that reads the force_legacy_seen field; cannot verify whether removing it breaks external tooling. |

ENABLE_SHADOW_PREVIEW is the sole live standalone flag controlling the shadow preview runtime branch. ENABLE_PREVIEW_V2 and PREVIEW_FORCE_LEGACY are safe to clean up as they carry no independent runtime effect.
