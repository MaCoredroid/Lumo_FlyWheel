# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env` | `preview_runtime_branch` | `config/defaults.toml` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

ENABLE_SHADOW_PREVIEW is the primary live flag. The parser in config.py sets shadow_enabled and effective_mode when the env var is truthy. The runtime in runtime.py branches on config.shadow_enabled to return the shadow preview path. The defaults.toml listing it as false is only a static default, not proof of dead code.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env` | `preview_runtime_branch` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is parsed by the env parser and normalizes to the same shadow path as ENABLE_SHADOW_PREVIEW, but it is not a standalone runtime branch. The runbook confirms it is a legacy alias kept only for old deploy manifests. The test shows it maps to shadow path, which proves parser presence but not independent liveness.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed by the env parser and sets force_legacy_seen, but the live runtime never reads that field. The runtime branch in runtime.py only checks shadow_enabled. The flag only affects a reporting label in legacy.py. The test asserts the branch stays on legacy_preview_path, disproving any runtime effect.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag — parser and runtime branch both active and tested. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests may still set ENABLE_PREVIEW_V2 | Alias flag normalizing to ENABLE_SHADOW_PREVIEW; safe to remove after legacy deploy manifests are updated. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | Confirm no external tooling depends on force_legacy_seen reporting | Dead flag — parser sets a reporting-only field that no runtime branch reads. Safe to remove after confirming no external tooling reads force_legacy_seen. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed in runtime.py preview_runtime_branch function |
| observed | ENABLE_PREVIEW_V2 alias normalization | Confirmed in config.py elif branch and runbook documentation |
| missing | External tooling consuming force_legacy_seen | No evidence of external consumers found in this workspace copy |

PREVIEW_FORCE_LEGACY is dead — parser present but no runtime branch reads it. ENABLE_PREVIEW_V2 is a live alias normalizing to ENABLE_SHADOW_PREVIEW. Only ENABLE_SHADOW_PREVIEW is a fully live standalone flag.
