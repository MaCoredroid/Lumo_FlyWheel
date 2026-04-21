# Manager Brief — v4-multi-corpus-objective

*Machine-readable source of truth: `brief/manager_brief.json` (schema cnb55.manager_brief.v2).*

## Decision

Accept **P2**. Rankings, citations, and assumptions are listed below.

## Ranking

- **P2** (rank 1, **ACCEPTED**) — L2 cache now pairs latency with streaming reliability isolation; matches the new objective (INC-2411/2419/2427).
  - Citations: `proposals/P2.md`, `repo_evidence/perf/l2_cache_shadow_replay.md`, `release_context/release_notes_2026_03.md`
  - Constraint tags: rollout, release, perf
- **P5** (rank 2, rejected) — Validator microservice delivers reliability isolation under the new streaming objective; January 2026 jan_bench perf is stale and superseded pre-streaming — flagged here.
  - Citations: `proposals/P5.md`, `repo_evidence/perf/validator_service_jan_bench.md`, `release_context/release_notes_2026_03.md`
  - Constraint tags: release, regression
- **P4** (rank 3, rejected) — Surgical warm-start is strong latency win but does NOT advance streaming-reliability — mismatch vs current objective.
  - Citations: `proposals/P4.md`, `repo_evidence/perf/schema_compile_bench.md`, `release_context/release_notes_2026_03.md`
  - Constraint tags: release, perf
- **P1** (rank 4, rejected) — Fast-ship validation drop; does not advance streaming reliability and keeps SDK regression risk.
  - Citations: `proposals/P1.md`, `repo_evidence/meta/rollout_history.md`, `release_context/release_notes_2026_03.md`
  - Constraint tags: release, regression
- **P3** (rank 5, rejected) — Rust rewrite; sunk-cost abandoned patch; staffing-blocked (Diego 0%); does not advance streaming reliability.
  - Citations: `proposals/P3.md`, `repo_evidence/perf/rust_prototype_notes.md`, `repo_evidence/meta/staffing.md`
  - Constraint tags: staffing, regression

## Primary Risk and Mitigations

**Risk.** L2 cache invalidation coupling with the streaming watermark module can regress truncation guarantees under load, and watermark state changes landed AFTER the shadow replay numbers were collected (INC-2427) — so the replay baseline is a conservative lower-bound, not a ship-ready SLO.

**Mitigations.**
- feature flag gated 1%/10% canary
- kill switch back to no-cache path on watermark lag
- shadow replay comparing watermark boundaries with/without cache
- SLO observability on truncation rate during rollout
- staged rollout halts at 10% if the INC-2427 watermark regression reappears; Mei pre-warm reviewer for the watermark coupling gate

## Assumption Ledger

| Status | Topic | Note |
| --- | --- | --- |
| missing | Streaming-reliability SLO numerical target for INC-2411/2419/2427 | Release notes describe truncation incidents but do not set an explicit SLO; would need PM confirmation. |
| missing | Whether stale Jan validator bench has been refreshed | validator_service_jan_bench.md is outdated; no fresh bench found. |
| to_verify | Whether P2 shadow-replay results still hold after watermark changes | Shadow replay predates INC-2427 watermark changes. |
