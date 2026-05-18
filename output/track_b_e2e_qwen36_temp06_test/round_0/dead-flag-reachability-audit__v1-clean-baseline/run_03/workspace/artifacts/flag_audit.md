# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `PREVIEW_FORCE_LEGACY`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/false_positive_notes.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_shadow_preview_live.py`, `config/defaults.toml`

This is the primary live flag. It is parsed in load_preview_env, sets shadow_enabled on PreviewConfig, and the runtime branch function preview_runtime_branch explicitly checks config.shadow_enabled to return the shadow preview path. The flag is fully reachable from env var through parser to runtime decision.

| `ENABLE_PREVIEW_V2` | live | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `docs/preview_rollout_runbook.md` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_preview_v2_alias.py`, `docs/preview_rollout_runbook.md`

ENABLE_PREVIEW_V2 is a live alias for ENABLE_SHADOW_PREVIEW. The parser accepts it for backward compatibility with legacy deploy manifests, but it normalizes to the same shadow_enabled flag and effective_mode=shadow. It does not have its own standalone runtime branch. Classification is live because it does affect runtime behavior, but it is an alias not a standalone flag.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `tests/test_force_legacy_reporting_only.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`, `tests/test_force_legacy_reporting_only.py`, `docs/preview_rollout_runbook.md`

PREVIEW_FORCE_LEGACY is classified as dead because although the parser recognizes it and sets force_legacy_seen on PreviewConfig, the runtime branch function preview_runtime_branch does not inspect force_legacy_seen. The flag has no effect on which code path executes. It only populates a reporting label via legacy_force_label. The parser presence creates a false impression of reachability, but the runtime is unaffected.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | Verify no external deploy manifests still set PREVIEW_FORCE_LEGACY and expect the legacy-forced label, Coordinate removal of config/defaults.toml preview_force_legacy entry | The flag is dead — parser present but no runtime branch. Remove the parser branch in load_preview_env and the force_legacy_seen field from PreviewConfig. The legacy_force_label function becomes unused and can also be removed. |
| `ENABLE_PREVIEW_V2` | `deprecate` | Legacy deploy manifests still reference ENABLE_PREVIEW_V2, No migration timeline established for operators to switch to ENABLE_SHADOW_PREVIEW | The alias is live and needed for backward compatibility with legacy deploy manifests. Add a deprecation warning in the parser branch so operators know to migrate to ENABLE_SHADOW_PREVIEW. Maintain compatibility while signaling future removal. |
| `ENABLE_SHADOW_PREVIEW` | `keep` | none | This is the sole live standalone flag driving the shadow preview runtime branch. No cleanup needed. |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| observed | Parser presence does not imply runtime reachability | docs/false_positive_notes.md states parser hit does not prove runtime branch; confirmed by PREVIEW_FORCE_LEGACY |
| observed | ENABLE_PREVIEW_V2 is an alias not a standalone flag | Runbook confirms V2 normalizes to shadow path and is not a standalone runtime branch |
| missing | No migration timeline for ENABLE_PREVIEW_V2 alias removal | No evidence found of a planned deprecation schedule or operator migration path for V2 to shadow preview |

PREVIEW_FORCE_LEGACY is dead at runtime but still parsed, creating operator confusion. ENABLE_PREVIEW_V2 is a live alias for ENABLE_SHADOW_PREVIEW and should be deprecated with a migration path. ENABLE_SHADOW_PREVIEW is the sole live standalone flag driving the shadow preview runtime branch.
