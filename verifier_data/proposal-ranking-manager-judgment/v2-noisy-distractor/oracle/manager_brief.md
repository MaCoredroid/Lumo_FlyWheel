# Manager Brief — v2-noisy-distractor

*Machine-readable source of truth: `brief/manager_brief.json` (schema cnb55.manager_brief.v2).*

## Decision

Accept **P4**. Rankings, citations, and assumptions are listed below.

## Ranking

- **P4** (rank 1, **ACCEPTED**) — Surgical schema-compile warm-start pays down the dominant hot-path segment and is staffable in Q3 with Ravi covering during Priya's 40% parental leave ramp-down.
  - Citations: `proposals/P4.md`, `repo_evidence/perf/schema_compile_bench.md`, `repo_evidence/meta/staffing.md`, `repo_evidence/meta/roster_memo.md`, `repo_evidence/code/normalizer_excerpt.py`, `repo_evidence/perf/hot_path_flame_2026_04_02.txt`
  - Constraint tags: staffing, rollout, perf
- **P2** (rank 2, rejected) — L2 response cache is a close-call trade-off vs P4; real win but coupled to watermark without staged rollout.
  - Citations: `proposals/P2.md`, `repo_evidence/perf/l2_cache_shadow_replay.md`, `repo_evidence/meta/rollout_history.md`
  - Constraint tags: rollout, regression
- **P5** (rank 3, rejected) — Validator microservice perf numbers are stale (January 2026 bench / jan_bench, outdated and superseded pre-streaming) and ownership is ambiguous after Kenji handoff.
  - Citations: `proposals/P5.md`, `repo_evidence/perf/validator_service_jan_bench.md`, `repo_evidence/meta/staffing.md`
  - Constraint tags: staffing, regression
- **P1** (rank 4, rejected) — Dropping structural validation is perf-positive but regressed SDK contracts twice in Q2 rollout history.
  - Citations: `proposals/P1.md`, `repo_evidence/perf/hot_path_flame_2026_04_02.txt`, `repo_evidence/meta/rollout_history.md`
  - Constraint tags: regression, rollout
- **P3** (rank 5, rejected) — Rust rewrite is staffing-blocked this quarter (Diego at 0%); headline delta is unstaffable primary work.
  - Citations: `proposals/P3.md`, `repo_evidence/perf/rust_prototype_notes.md`, `repo_evidence/meta/staffing.md`
  - Constraint tags: staffing, regression

## Primary Risk and Mitigations

**Risk.** Schema-cache warm-up cost on cold process start can regress p99 start-up latency for freshly scheduled replicas until the compiled cache fills, with Priya at only 40% Q3 (parental leave handoff) Ravi must cover the ramp-down reviewer slot.

**Mitigations.**
- staged rollout behind a feature flag at 1%/10%/50%/100%
- pre-warm the cache via shadow traffic replay before real traffic
- kill switch reverts to the non-compiled path on SLO breach
- canary observability on start-up p99 for 24h before ramping
- Ravi cover during Priya ramp-down (mid-quarter handoff) so rollout gates are not blocked on a single 40% reviewer

## Assumption Ledger

| Status | Topic | Note |
| --- | --- | --- |
| missing | Whether the stale Jan bench has been re-run post handoff | validator_service_jan_bench.md is outdated; no fresh bench is on file to supersede it. |
| to_verify | Schema-compile p99 start-up regression | Must measure cold-start p99 on canary before ramp. |
| missing | L2 cache / watermark coupling failure enumeration | Shadow replay flags the risk but does not enumerate the failure paths; would want a deeper review before re-entering. |
| to_verify | Diego Q4 return date | Staffing doc lists 0% Q3 but no explicit Q4 allocation. |
| to_verify | Ravi cover bandwidth during Priya 40% ramp-down | Mid-quarter staffing update pre-approves Ravi cover but does not name a backup if Ravi is also over-subscribed. |
