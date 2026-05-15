# Flag Audit

- variant: `v1-clean-baseline`
- highest_operational_risk: `ENABLE_SHADOW_PREVIEW`

## Flag Status

| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |
|---|---|---|---|---|---|
| `ENABLE_SHADOW_PREVIEW` | live | — | `load_preview_env:ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`

ENABLE_SHADOW_PREVIEW is the primary live flag controlling preview runtime. When enabled, it sets shadow_enabled=true causing preview_runtime_branch to return shadow_preview_path.

| `ENABLE_PREVIEW_V2` | partial | ENABLE_SHADOW_PREVIEW | `load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW` | `preview_runtime_branch:shadow_preview_path` | `src/preview/config.py` |

Evidence: `src/preview/config.py`, `tests/test_preview_v2_alias.py`

ENABLE_PREVIEW_V2 is a legacy alias that normalizes to ENABLE_SHADOW_PREVIEW. It has parser presence and routes to shadow_preview_path but is not a standalone flag.

| `PREVIEW_FORCE_LEGACY` | dead | — | `load_preview_env:PREVIEW_FORCE_LEGACY` | `—` | `src/preview/runtime.py` |

Evidence: `src/preview/config.py`, `src/preview/runtime.py`

PREVIEW_FORCE_LEGACY is dead code. It is parsed and sets force_legacy_seen for reporting only. The runtime branch in preview_runtime_branch does not check this flag at all.

## Cleanup Plan

| flag | action | blockers | rationale |
|---|---|---|---|
| `PREVIEW_FORCE_LEGACY` | `docs_cleanup` | none | Flag is dead, parser presence does not provide runtime value |
| `PREVIEW_FORCE_LEGACY` | `remove_after_migration` | none | force_legacy_seen field is unused by runtime |

## Assumption Ledger

| status | topic | note |
|---|---|---|
| missing | PREVIEW_FORCE_LEGACY runtime usage | No evidence of PREVIEW_FORCE_LEGACY being used in runtime branching |
| observed | parser presence | Parser presence does not imply runtime reachability |
| observed | ENABLE_PREVIEW_V2 | ENABLE_PREVIEW_V2 classified as partial due to alias nature |

ENABLE_SHADOW_PREVIEW is the only standalone live flag. ENABLE_PREVIEW_V2 is a legacy alias. PREVIEW_FORCE_LEGACY is dead and can be safely removed.
