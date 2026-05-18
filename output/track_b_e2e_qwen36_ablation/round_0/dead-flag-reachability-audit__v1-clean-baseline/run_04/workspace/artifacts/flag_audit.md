# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in load_preview_env (config.py), sets shadow_enabled=True which drives the runtime branch in preview_runtime_branch (runtime.py) to shadow_preview_path. The test test_shadow_preview_live.py confirms the end-to-end path from env var through plan. Defaults.toml declares it as a config key.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is a live alias, not a standalone flag. The parser in config.py catches it in the elif branch and normalizes it to the same shadow_enabled=True state as ENABLE_SHADOW_PREVIEW, producing the same runtime branch symbol. The runbook doc explicitly states it is not a standalone runtime branch. The alias mapping is recorded in parser_hits with the arrow notation indicating aliasing.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed and tracked (parser_hits records it, force_legacy_seen is set) but has no runtime branch. The config.py parser sets force_legacy_seen=True, but runtime.py never reads this field. The runbook explicitly states the live service does not branch on it anymore. The test confirms branch remains legacy_preview_path — the flag is reporting/telemetry only with no behavioral effect on the live router.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag driving shadow preview mode. No reason to remove. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference it | Alias of ENABLE_SHADOW_PREVIEW. Keep parser hit for backward compat but plan removal after migration. |
| `PREVIEW_FORCE_LEGACY` | `telemetry_first` | operator dashboards may display the force_legacy_seen label | Dead flag with no runtime branch. Verify telemetry consumers no longer depend on force_legacy_seen before removing parser code. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW parser and runtime branch | Confirmed in config.py and runtime.py source |
| observed | ENABLE_PREVIEW_V2 alias behavior | Parser maps to same shadow path as ENABLE_SHADOW_PREVIEW |
| missing | External deploy manifest inventory for PREVIEW_FORCE_LEGACY | No evidence outside workspace of which deploys still set PREVIEW_FORCE_LEGACY |

PREVIEW_FORCE_LEGACY is a dead flag that is still parsed but has no runtime effect. It should be removed from config defaults and the parser to avoid operator confusion. ENABLE_PREVIEW_V2 is a benign alias but should be documented or deprecated.
