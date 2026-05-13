# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag with both parser presence in load_preview_env and runtime reachability via preview_runtime_branch. It directly controls the shadow_preview_path branch in the live service router.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a legacy alias accepted by the env parser for backward compatibility with old deploy manifests. It normalizes to ENABLE_SHADOW_PREVIEW and shares the same runtime branch, but is not a standalone flag with its own runtime path.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY has parser presence and sets force_legacy_seen, but the live service does not branch on it anymore. It is retained only for reporting and operator notes via legacy_force_label. The test confirms the branch remains legacy_preview_path regardless of this flag.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Primary live flag with full runtime reachability. Must be retained for active preview rollout control. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests | Alias flag for backward compatibility. Should be deprecated with migration path to ENABLE_SHADOW_PREVIEW once legacy manifests are updated. |
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | operator notes reference | Dead flag with no runtime effect. Remove from env parser and update documentation to clarify it only affects reporting labels. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed in runtime.py and tests |
| observed | ENABLE_PREVIEW_V2 is an alias | Confirmed in config.py and runbook docs |
| observed | PREVIEW_FORCE_LEGACY has no runtime branch | Confirmed in test_force_legacy_reporting_only.py |
| missing | external deploy manifest inventory | Unknown how many legacy manifests still use ENABLE_PREVIEW_V2 |

PREVIEW_FORCE_LEGACY is a dead flag that should be deprecated. It has parser presence but no runtime reachability, which could mislead operators into thinking it controls behavior when it only affects only reporting labels.
