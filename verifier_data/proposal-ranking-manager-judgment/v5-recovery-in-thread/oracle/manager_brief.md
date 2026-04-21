# Manager Brief — v5-recovery-in-thread

*Machine-readable source of truth: `brief/manager_brief.json` (schema cnb55.manager_brief.v2).*

## Decision

Accept **P5**. Rankings, citations, and assumptions are listed below.

## Ranking

- **P5** (rank 1, **ACCEPTED**) — Validator microservice delivers reliability isolation and does NOT touch the watermark module that caused INC-2481; January 2026 jan_bench perf is stale / superseded pre-streaming and flagged accordingly.
  - Citations: `proposals/P5.md`, `incident_context/incident_2026_04_P2_rollback.md`, `release_context/release_notes_2026_03.md`, `repo_evidence/perf/validator_service_jan_bench.md`
  - Constraint tags: incident, release, rollout
- **P4** (rank 2, rejected) — Surgical warm-start; does not touch the watermark that caused INC-2481; safe latency work as a fast-follow.
  - Citations: `proposals/P4.md`, `repo_evidence/perf/schema_compile_bench.md`, `incident_context/incident_2026_04_P2_rollback.md`
  - Constraint tags: incident, perf
- **P2** (rank 3, rejected) — L2 cache — was the ACCEPTED pick in prior cycle and was ROLLED BACK via INC-2481; cannot lead until watermark redesign.
  - Citations: `proposals/P2.md`, `incident_context/incident_2026_04_P2_rollback.md`, `incident_context/watermark_bug_notes.md`, `incident_context/prior_ranking.md`
  - Constraint tags: incident, regression
- **P1** (rank 4, rejected) — Structural-validation drop; SDK contract risk unchanged; does not advance streaming-reliability post INC-2481.
  - Citations: `proposals/P1.md`, `repo_evidence/meta/rollout_history.md`, `release_context/release_notes_2026_03.md`
  - Constraint tags: release, regression
- **P3** (rank 5, rejected) — Rust rewrite; staffing-blocked (Diego 0%) AND depends on watermark redesign from INC-2481 — sunk cost, abandoned skeleton.
  - Citations: `proposals/P3.md`, `repo_evidence/perf/rust_prototype_notes.md`, `repo_evidence/meta/staffing.md`, `incident_context/watermark_bug_notes.md`
  - Constraint tags: staffing, incident

## Primary Risk and Mitigations

**Risk.** Validator cold-start cost combined with the new RPC boundary becoming a new control-flow surface that must be observed under load post-INC-2481, while the watermark redesign timeline remains unknown.

**Mitigations.**
- feature flag staged rollout 1%/10%/50%
- shadow replay validator responses against in-proc baseline
- kill switch to revert to in-proc validator on SLO breach
- canary SLO observability on RPC latency + error rate

## Assumption Ledger

| Status | Topic | Note |
| --- | --- | --- |
| missing | Timeline for watermark redesign referenced in INC-2481 | Incident doc notes a redesign is needed but no target quarter is attached; P2 re-entry depends on it. |
| to_verify | Validator RPC observability SLO | Need PM sign-off on the RPC SLO before ramping P5. |
| missing | Whether stale Jan validator bench has been refreshed | validator_service_jan_bench.md is outdated. |
