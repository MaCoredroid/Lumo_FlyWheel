# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag. The parser in config.py sets shadow_enabled=True, and runtime.py branches on config.shadow_enabled to return the shadow_preview_path. The test confirms the full path from env through parser to runtime branch. The false positive that parser presence alone proves reachability is disproved here by the actual runtime branch in runtime.py.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `docs/preview_rollout_runbook.md`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`

ENABLE_PREVIEW_V2 is a live alias of ENABLE_SHADOW_PREVIEW, not a standalone flag. The parser in config.py normalizes it to the same shadow path via the elif branch. The runbook explicitly states it is not a standalone runtime branch. The false positive that this is an independent flag is disproved by the runbook documentation and the parser normalization code.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `src/preview/legacy.py`, `docs/preview_rollout_runbook.md`, `tests/test_force_legacy_reporting_only.py`

PREVIEW_FORCE_LEGACY is a dead flag with no runtime branch. The parser in config.py sets force_legacy_seen=True, but runtime.py never reads force_legacy_seen; it only checks shadow_enabled. The flag is only used in legacy.py for a reporting label. The runbook confirms the live service does not branch on it. The false positive that a parser hit proves a runtime branch is disproved by inspecting runtime.py which has no reference to force_legacy.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag actively branching the runtime. No cleanup needed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference ENABLE_PREVIEW_V2 | Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Can be deprecated once deploy manifests migrate to the canonical flag name. |
| `PREVIEW_FORCE_LEGACY` | `do_not_remove_now` | reporting pipeline may still consume force_legacy_seen telemetry | Dead flag with no runtime branch but still parsed for reporting. Removing the parser would break legacy_force_label reporting. Keep until reporting migration is complete. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | Parser single source | config.py load_preview_env is the sole env parser |
| to_verify | Deploy manifest references | Whether external deploy manifests still set ENABLE_PREVIEW_V2 |
| missing | External telemetry consumers | No visibility into whether force_legacy_seen is consumed by external monitoring |

PREVIEW_FORCE_LEGACY is the highest risk because it is a dead flag still parsed and tracked, creating operator confusion without affecting runtime behavior. ENABLE_PREVIEW_V2 is a safe alias. ENABLE_SHADOW_PREVIEW is the only actively branching flag.
