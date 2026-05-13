# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. The parser in config.py sets shadow_enabled=True and effective_mode='shadow' when this env var is truthy. The runtime.py branches on config.shadow_enabled to return 'preview_runtime_branch:shadow_preview_path'. The test confirms this branch is taken. This is a standalone live flag with full parser and runtime reachability.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a legacy alias accepted by the parser for backward compatibility with old deploy manifests. It normalizes to the same shadow path as ENABLE_SHADOW_PREVIEW but is not a standalone runtime branch. The runbook explicitly states it is 'not a standalone runtime branch'. Parser presence does not equal runtime reachability here since it does affect the branch, but only as an alias.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has parser presence (sets force_legacy_seen=True) but no runtime branching. The runbook states 'The live service does not branch on it anymore'. The test confirms the branch remains 'preview_runtime_branch:legacy_preview_path' even when the flag is set. The legacy_force_label function only produces a label for reporting purposes. Parser presence does not equal runtime reachability - this flag is dead for branching purposes.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag controlling preview rollout. Must be retained for operational control. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference this flag | Alias flag with no standalone value. Deprecate after confirming no active deployments depend on it. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | need to update operator runbooks and remove references to this flag | Flag is dead for runtime branching but may be referenced in operator docs and runbooks. Clean up documentation and telemetry labels. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | parser vs runtime distinction | Parser hits in config.py do not guarantee runtime branching |
| observed | alias semantics | ENABLE_PREVIEW_V2 aliases to ENABLE_SHADOW_PREVIEW for legacy support |
| missing | active deploy manifest inventory | Cannot verify if any live deployments still use ENABLE_PREVIEW_V2 |

PREVIEW_FORCE_LEGACY poses highest risk because it appears functional (has parser, has test, has legacy.py helper) but produces no runtime effect. Operators may believe it forces legacy mode when it only updates a reporting label.
