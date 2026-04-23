# Evaluator Contract: `parallel-explorer-root-cause-map`

## Purpose

Score a read-heavy orchestration task where the solver must decompose investigation work across non-overlapping explorer questions, synthesize the results, and identify the true causal path without shipping speculative code.

## Submission Artifacts

Required:

- ranked suspect list covering all four seeded suspects
- at least two investigation threads
- evidence table with at least four rows
- bounded remediation plan
- verification note

## Score Breakdown

Visible deterministic band (`M_training`, 90 points total):

- 25 points: structural completeness
- 36 points: correct primary cause + symbol-level causal trace
- 14 points: renderer disproof + contradictory-artifact handling
- 15 points: bounded remediation target + non-goals

Probe-only deterministic band (`P_only`, 10 points total):

- 4 points: suspect-ranking richness
- 3 points: cross-thread synthesis quality
- 3 points: bounded-plan clarity

Total `P_benchmark = M_band + P_only`.

## Visible Checks

### Structure Gate — 25 points

- 6 points: ranked suspect list is complete and contiguous
- 6 points: at least two non-overlapping investigation threads
- 8 points: evidence table has at least four rows with file + symbol references
- 5 points: verification note and remediation non-goals are present

### Causal Accuracy Gate — 36 points

- 10 points: accepted suspect is `S1-fs-alias-normalization`
- 10 points: top-ranked file + symbol are `src/release_readiness/adapters/fs_source.py` + `normalize_fs_owner_alias`
- 8 points: aggregation suspect is present with `src/release_readiness/core/aggregation.py` + `merge_blocked_owner_rows`
- 8 points: exact failing assertion is quoted and tied back to the normalization → aggregation path

### Disproof Gate — 14 points

- 8 points: renderer suspect is explicitly ruled downstream-only with `src/release_readiness/renderers/markdown_renderer.py` + `render_blocked_owner_section`
- 6 points: one contradictory artifact is explicitly named and explained away as stale, misleading, or downstream-only

### Remediation Gate — 15 points

- 10 points: patch target stays on `src/release_readiness/adapters/fs_source.py` + `normalize_fs_owner_alias`
- 5 points: non-goals explicitly rule out renderer-first or broad aggregation rewrite work

## Probe-Only Checks

- suspect summaries are specific rather than generic
- the two investigation threads are genuinely different surfaces
- the remediation plan names a smallest safe patch and validation sequence

## Score Caps

- missing brief file: max `0`
- malformed structured brief: max `10`
- wrong primary suspect: max `20`
- no symbol-level causal trace: max `30`
- renderer selected as primary cause: max `25`
- fewer than two non-overlapping investigation threads: max `30`
- missing contradictory-artifact disproof: max `35`
- renderer-first remediation target: max `35`
- V3 sunk-cost renderer patch selected as fix: max `30`
- V4 aggregation-rewrite plan despite hotfix constraints: max `40`
- V5 no incident acknowledgment: max `15`

## Hidden Checks

- hidden scorer verifies exact seeded suspect ids, not free-form prose guesses
- hidden scorer verifies one adapter symbol, one aggregation symbol, and one renderer symbol with the correct causal roles
- hidden scorer verifies the exact failing assertion string
- hidden scorer verifies the contradictory artifact path for each variant
- hidden scorer verifies the patch target remains on source normalization rather than renderer formatting

## Baselines

- oracle brief must score `>= 90`
- empty brief must score `0`
- shortcut brief that picks the renderer as primary must score `<= 25`

## Hardness Judgment

Expected failure modes for naive probes:

- blame the renderer because the visible output is duplicated
- stop after one thread and never reconcile the conflicting artifacts
- propose a broad dedupe in aggregation instead of fixing source normalization
- treat V3’s abandoned renderer patch as momentum instead of noise
- ignore V5’s rollback note and re-select the previously reverted renderer fix
