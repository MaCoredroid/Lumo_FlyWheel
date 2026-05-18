# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

ENABLE_SHADOW_PREVIEW is the primary live flag. It is parsed in config.py (sets shadow_enabled and effective_mode), and the runtime branch in runtime.py explicitly checks config.shadow_enabled to return the shadow_preview_path. The test and runbook both confirm this is the active runtime branch. Parser hit alone does not prove reachability, but here the runtime code does branch on it.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is accepted by the env parser but is not a standalone flag. It normalizes to ENABLE_SHADOW_PREVIEW (the elif branch in config.py sets the same shadow_enabled flag). The runbook explicitly states it is not a standalone runtime branch. It is a live alias, not a live standalone flag. The test confirms it maps to the shadow path via the alias.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/legacy.py`, `tests/test_force_legacy_reporting_only.py`, `config/defaults.toml`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is parsed and tracked (sets force_legacy_seen) but the live service does not branch on it. The runtime.py preview_runtime_branch function only checks shadow_enabled, never force_legacy_seen. The runbook states the flag is left in reporting and operator notes only. The test confirms the branch remains legacy_preview_path regardless. The legacy_force_label helper is a reporting-only label, not a runtime branch.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | Active flag with live runtime branch. Removing it would break the shadow preview path. Keep as the canonical flag. |
| `ENABLE_PREVIEW_V2` | `deprecate` | legacy deploy manifests may still reference it | Live alias with no independent runtime branch. Deprecate and migrate deploy manifests to ENABLE_SHADOW_PREVIEW, then remove the parser elif branch. |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | reporting and operator notes may reference the label | Dead at runtime — parser present but no runtime branch. Remove after migrating any reporting pipelines that depend on force_legacy_seen or legacy_force_label. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | ENABLE_SHADOW_PREVIEW runtime branch | Confirmed by runtime.py shadow_enabled check and test_shadow_preview_live |
| observed | ENABLE_PREVIEW_V2 alias normalization | Config.py elif branch maps V2 to same shadow_enabled flag |
| missing | PREVIEW_FORCE_LEGACY reporting consumers | No evidence of live service consuming force_legacy_seen beyond the label helper |

ENABLE_SHADOW_PREVIEW is the only flag with a live, independent runtime branch. ENABLE_PREVIEW_V2 is a live alias normalizing to it. PREVIEW_FORCE_LEGACY is dead at runtime — parser present but no runtime branch. The highest operational risk is removing ENABLE_SHADOW_PREVIEW since it directly gates the shadow preview path.
