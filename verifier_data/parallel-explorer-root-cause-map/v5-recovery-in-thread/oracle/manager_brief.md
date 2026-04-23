# Root Cause Brief

- Variant: `v5-recovery-in-thread`
- Accepted suspect: `S1-fs-alias-normalization`

## Ranked Suspects
### 1. S1-fs-alias-normalization
- Role: `primary`
- File: `src/release_readiness/adapters/fs_source.py`
- Symbol: `normalize_fs_owner_alias`
- Summary: The file-backed scheduler path emits space-preserving owner keys like `team ops`, while the env path emits hyphenated keys like `team-ops`; aggregation trusts `owner_key` verbatim, so aliases split before totals are counted.
- Evidence: src/release_readiness/adapters/fs_source.py, src/release_readiness/core/aggregation.py, tests/test_root_cause_map.py, artifacts/logs/runtime_snapshot_2026_04_14.json

### 2. S2-aggregation-grouping
- Role: `amplifier`
- File: `src/release_readiness/core/aggregation.py`
- Symbol: `merge_blocked_owner_rows`
- Summary: Aggregation is where the bug becomes visible because it keys buckets on the already-split `owner_key`, but it is consuming bad upstream normalization rather than inventing new aliases itself.
- Evidence: src/release_readiness/core/aggregation.py, artifacts/logs/runtime_snapshot_2026_04_14.json, tests/test_root_cause_map.py

### 3. S4-env-watchlist-parser
- Role: `secondary`
- File: `src/release_readiness/adapters/env_source.py`
- Symbol: `normalize_env_owner_token`
- Summary: The env path is consistent and hyphen-normalized; it matters only because it disagrees with the file-backed path after the refactor.
- Evidence: src/release_readiness/adapters/env_source.py, artifacts/logs/runtime_snapshot_2026_04_14.json

### 4. S3-renderer-duplicate-headings
- Role: `downstream-only`
- File: `src/release_readiness/renderers/markdown_renderer.py`
- Symbol: `render_blocked_owner_section`
- Summary: The renderer faithfully prints the already-duplicated owner buckets; it explains the visible symptom but not the inflated `blocked_owner_total`.
- Evidence: src/release_readiness/renderers/markdown_renderer.py, artifacts/logs/rendered_report_excerpt.md, artifacts/logs/runtime_snapshot_2026_04_14.json

## Investigation Threads
- Question: Where do owner aliases diverge before the count is computed?
  Files: src/release_readiness/adapters/fs_source.py, src/release_readiness/adapters/env_source.py, src/release_readiness/core/aggregation.py
  Finding: The scheduler file path preserves space-separated owner keys, the env path hyphenates them, and aggregation buckets on the mismatched keys without re-canonicalizing.
- Question: Is the renderer a cause or just a visible symptom?
  Files: src/release_readiness/renderers/markdown_renderer.py, artifacts/logs/rendered_report_excerpt.md, incident_context/INC-447-renderer-rollback.md
  Finding: The renderer only reflects the split buckets it is handed; contradictory artifacts over-emphasize the symptom, but the runtime snapshot shows the duplication existed before rendering.

## Evidence Table

- Claim: The file-backed adapter emits a space-preserving owner key that disagrees with the env path.
  File: `src/release_readiness/adapters/fs_source.py`
  Symbol: `normalize_fs_owner_alias`
  Artifact: `artifacts/logs/runtime_snapshot_2026_04_14.json`
  Why misleading: Visible markdown duplication happens later and should not be mistaken for the first divergence.
- Claim: Aggregation trusts `owner_key` verbatim, so the upstream mismatch becomes duplicate blocked owners.
  File: `src/release_readiness/core/aggregation.py`
  Symbol: `merge_blocked_owner_rows`
  Artifact: `tests/test_root_cause_map.py::test_scheduler_aliases_collapse_before_rendering`
  Why misleading: A dedupe in aggregation would mask the symptom but leaves the root normalization bug in place.
- Claim: The renderer prints what it is given and does not create new owner buckets.
  File: `src/release_readiness/renderers/markdown_renderer.py`
  Symbol: `render_blocked_owner_section`
  Artifact: `artifacts/logs/rendered_report_excerpt.md`
  Why misleading: The duplicated headings are downstream-visible and can trick a shallow read into blaming formatting.
- Claim: The failing test is asserting a unique-owner invariant that breaks before rendering.
  File: `tests/test_root_cause_map.py`
  Symbol: `test_scheduler_aliases_collapse_before_rendering`
  Artifact: `assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"`
  Why misleading: A renderer-only patch cannot make this assertion true because the count is wrong upstream.

## Remediation Plan
- Patch target: `src/release_readiness/adapters/fs_source.py` :: `normalize_fs_owner_alias`
- Why smallest safe patch: Repairing file-backed owner canonicalization keeps the aggregation contract stable and removes the duplicate buckets before they can inflate totals or render duplicate headings.
- Validation steps: pytest tests/test_root_cause_map.py -q, confirm runtime snapshot keys collapse to `team-ops` and `platform-infra` before rendering
- Non-goals: do not patch the renderer first, do not broad-rewrite aggregation during the hotfix window

## Verification Note
- Failing assertion: assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"
- Contradictory artifact: `incident_context/INC-447-renderer-rollback.md`
- Resolution: This rollback note is essential context: it proves a renderer-only fix was already tried and failed because the upstream owner keys remained split.

