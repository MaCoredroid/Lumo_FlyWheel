# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the sole live flag. Parser reads the env var and sets shadow_enabled which is consumed by preview_runtime_branch to return the shadow path. Test and docs confirm end-to-end reachability.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is parsed by the env parser as an elif branch but normalizes to the same shadow_enabled flag. There is no independent runtime branch for V2. The runbook explicitly states it is not a standalone runtime branch. Parser hit does not prove runtime reachability.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed and sets force_legacy_seen but the live service does not branch on it. The only consumer is legacy_force_label which is a reporting-only label. The test asserts the runtime branch remains legacy. The runbook confirms the live service does not branch on it anymore.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Sole live flag. Removing it would break the preview router. Must be kept as the canonical shadow preview toggle. |
| `ENABLE_PREVIEW_V2` | `deprecate` | docs/preview_rollout_runbook.md | Alias with no independent runtime branch. Deprecate and redirect operators to ENABLE_SHADOW_PREVIEW. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | tests/test_force_legacy_reporting_only.py | Dead flag with no runtime effect. Can be removed after confirming no external deploy manifests still reference it. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed via config.py parser and runtime.py branch logic. |
| observed | ENABLE_PREVIEW_V2 alias normalization | Parser normalizes V2 to shadow_enabled; no standalone branch exists. |
| missing | External deploy manifest references to PREVIEW_FORCE_LEGACY | Cannot verify from workspace alone whether external manifests still set this flag. |

ENABLE_SHADOW_PREVIEW is the only live flag controlling the preview router. ENABLE_PREVIEW_V2 is a dead alias and PREVIEW_FORCE_LEGACY is dead. Removing the dead/partial flags reduces config surface without runtime impact.
