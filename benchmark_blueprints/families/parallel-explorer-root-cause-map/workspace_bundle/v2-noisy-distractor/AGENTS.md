# Agent Instructions — `parallel-explorer-root-cause-map`

## Task

Investigate why `release_readiness` over-reports blocked owners after the
scheduler refactor. Produce a unified root-cause brief plus one bounded
remediation plan. Do not patch the repo. Your only writable outputs are
`brief_input.json` and the generated files under `brief/`.

## Required workflow

1. Read at least two different surfaces before deciding: one adapter / aggregation path and one renderer / artifact path.
2. Author `brief_input.json` at the workspace root.
3. Run `./bin/cnb55-brief validate brief_input.json`.
4. Run `./bin/cnb55-brief submit brief_input.json`.

## Inputs

- `src/release_readiness/adapters/fs_source.py`
- `src/release_readiness/adapters/env_source.py`
- `src/release_readiness/core/aggregation.py`
- `src/release_readiness/renderers/markdown_renderer.py`
- `tests/test_root_cause_map.py`
- `docs/triage-policy.md`
- `artifacts/logs/`
- `artifacts/review/incident_thread.md`

## Suspects To Rank

- `S1-fs-alias-normalization`
- `S2-aggregation-grouping`
- `S3-renderer-duplicate-headings`
- `S4-env-watchlist-parser`

## What Good Looks Like

- the accepted suspect identifies the true source-normalization defect
- the aggregation layer is named as the place where the defect becomes visible in totals
- the renderer is ruled out as downstream-only
- the exact failing assertion is quoted
- at least one contradictory artifact is explicitly explained away
- the remediation plan names the smallest safe patch target and explicit non-goals

## Variant-specific notes

- An operator handoff note leans hard on the renderer. It may be stale or downstream-only.

## Rules

- Do not modify `src/`, `tests/`, `docs/`, `artifacts/`, `release_context/`, or `incident_context/`.
- Do not add shim files such as `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Do not use the network.
