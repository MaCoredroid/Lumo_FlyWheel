# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in load_preview_env (config.py), sets shadow_enabled=True which is directly checked in preview_runtime_branch (runtime.py) to return the shadow_preview_path branch. The full chain from env var through parser to runtime branch is exercised in test_shadow_preview_live.py and documented in the runbook.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser in config.py recognizes it and normalizes it to the same shadow_preview_path as ENABLE_SHADOW_PREVIEW. It has no independent runtime branch since it maps to the same path as the primary flag. The runbook explicitly states it normalizes to the same shadow path and is not a standalone runtime branch.

| `PREVIEW_FORCE_LEGACY` | partial | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`, `config/defaults.toml`

PREVIEW_FORCE_LEGACY is parsed by load_preview_env in config.py and sets force_legacy_seen=True, but the live runtime branch in runtime.py does not check this field. The only consumer is legacy_force_label in legacy.py which produces a reporting label. The runbook confirms the live service does not branch on it anymore and it is reporting and operator notes only.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with direct runtime branch. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Should be deprecated in docs and parser warnings added before eventual removal to give deploy manifests time to migrate. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | Reporting pipeline depends on force_legacy_seen for operator labels | Parser present but no runtime branch. Before removal, verify that the reporting-only label from legacy_force_label is not consumed by any downstream telemetry or alerting system. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed: runtime.py branches on shadow_enabled which is set by ENABLE_SHADOW_PREVIEW. |
| observed | ENABLE_PREVIEW_V2 alias normalization | Confirmed: config.py maps ENABLE_PREVIEW_V2 to the same shadow path as ENABLE_SHADOW_PREVIEW. |
| observed | PREVIEW_FORCE_LEGACY reporting-only | Confirmed: runtime.py does not check force_legacy_seen; only legacy.py uses it for labels. |
| missing | Downstream consumers of legacy_force_label output | Not visible in this workspace copy; cannot confirm whether the label output feeds telemetry. |

ENABLE_SHADOW_PREVIEW is the sole flag controlling the shadow preview runtime path. ENABLE_PREVIEW_V2 is a deprecated alias that normalizes to the same path, and PREVIEW_FORCE_LEGACY is reporting-only with no runtime effect. The highest operational risk is ENABLE_SHADOW_PREVIEW since it is the only flag with a direct runtime branch, and misconfiguration could silently route all traffic to the preview path.
