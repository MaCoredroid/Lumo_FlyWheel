# Codex-Bench — Low-Level Design Index

> Derived from HLD Spec v2.3 · April 2026

This document lists the 13 LLDs required to implement Codex-Bench. Each entry covers what the component handles, how it connects to other LLDs, and the recommended design-then-implement sprint sequence. LLDs are grouped by architectural layer matching the sprint phases in the HLD.

---

## Change Summary (v1.3 → v2.3 derivation)

The HLD underwent substantial structural changes between v1.3 and v2.3. The LLD index reflects these:

- **Bench-Train (350 SWE-bench tasks) removed entirely.** Replaced by the Codex-Long track as the primary training data source.
- **Codex-Long track added** with 55+ scenario families, family-disjoint splits (Train-Long / Val-Long / Test-Long / Public-Dev), state-based verifiers, and an integrity protocol. This introduces a new foundational LLD (**LLD-13**).
- **Bench-Control (~50 SWE-bench tasks)** added as a small in-domain design control arm (not a primary data source).
- **Sprint 0b** added as a dedicated 4–6 week benchmark authoring phase, producing the Codex-Long scenarios and verifiers that LLD-03 and LLD-05 build against.
- **Gates 1b (streamed tool-call event semantics) and 1c (frontier API compatibility)** added to Sprint 0.
- **Gate 4 completely redesigned**: now a family-aware Codex-Long pilot measuring wall-clock by scenario type, matched families, and structural diversity — not a simple SWE-bench pilot.
- **Contribution B formally split** into B1 (harness specificity on Test-Long, family-clustered bootstrap) and B2 (transfer to SWE Final-Test).
- **Smaller-v1 escape hatch** pre-registered with Rule 1 (Test-Long family floor) and Rule 2 (Gate 4 thresholds). B2-only proceed rule added for the 35-family path.
- **Comparison slot reframed**: practical default is a second Qwen variant; GLM-4.7 and Step 3.5 Flash are aspirational add-ons pending Gates 1b + 6.
- **LLD-02 scope fundamentally changed** (new pool structure). **LLD-03, LLD-05, LLD-06, LLD-07, LLD-09, LLD-10, LLD-12** all require moderate-to-significant updates.

---

## 🔵 Foundation Layer — Design & Implement First (Sprint 0 / Sprint 0b)

### LLD-01 · vLLM Serving Layer

**Covers:** Model loading (FP8 / NVFP4 quantization), prefix caching config, chunked prefill, KV cache dtype, CUDA graph settings, Codex CLI endpoint wiring (`wire_api = "responses"` only), LoRA adapter serving for Sprint 3, model switching protocol, `/metrics` observability contract.

**Connects to:** Root dependency — every other LLD that touches a model targets this endpoint. LLD-03 (Orchestrator) calls it per task; LLD-08 (Logprob Proxy) sits in front of it; LLD-07 (Benchmark Runner) triggers model switches; LLD-09 (mini-SWE-Agent) calls `/v1/chat/completions` directly.

**v2.3 impact:** Minor update — not a rewrite. Model registry §3 needs the comparison slot reframed (second Qwen fallback, GLM/Step aspirational); 27B sprint column updated from "Bench-Train" to "Bench-Control + Codex-Long collection"; Sprint 0 checklist §14 should reference Gate 1b proto-fixtures; §17 Sprint 3 arithmetic updated from 1000 to 1100 B2 core runs. Core serving design (quantization, prefix caching, launch config, metrics formulas, model switching, LoRA serving) is unchanged.

**Sprint:** Design S0 → Implement S0

---

### LLD-13 · Codex-Long Scenario Framework *(NEW)*

**Covers:** The entire Codex-Long benchmark infrastructure that Sprint 0b produces. Specifically: scenario family spec format (repo pattern, injected breakage class, grading invariant, milestone definitions, shortcut-resistance notes); Docker environment factory (per-variant container images pinned to base image digests); state-based verifier implementation and post-run injection protocol; milestone check scripts (injected post-run, never visible to the agent); oracle solution isolation (used for solvability proof during authoring, never distributed); the integrity protocol (verifier/oracle isolation, test resource injection timing, shortcut prevention); the two-tier benchmark audit protocol (Tier 1 type coverage + Tier 2 adversarial audit); and the family-to-split assignment freeze procedure.

**Connects to:** Foundational dependency for all official Codex-Long collection and evaluation. LLD-02 (Data Pool Manager) consumes the frozen family-to-split assignment table. LLD-03 (Task Orchestrator) launches scenarios using the Docker env factory and invokes verifiers post-run. LLD-05 (Evaluator) uses the state-based verifier for grading (not patch-based). LLD-07 (Benchmark Runner) pulls Codex-Long task lists. The smaller-v1 escape hatch (35-family path) and Rule 1 (Test-Long family floor) are enforced at the freeze point specified here.

**Proto-scenario exception (Gate 3, Sprint 0):** The HLD requires one Codex-Long pilot scenario through Codex as part of Gate 3 (RL target feasibility). This single proto-scenario is authored in Sprint 0 before Sprint 0b begins — analogous to how Gate 1b uses proto-fixtures rather than the full Sprint 0b fixture set. It does not require the full LLD-13 framework (no family-to-split assignment, no integrity audit, no verifier injection protocol). It must be a representative Codex-Long-style scenario (multi-turn, tool-heavy) sufficient to measure 27B rollout wall-clock. The full LLD-13 framework governs all subsequent Codex-Long execution from Gate 4 onward.

**Sprint:** Design S0b → Implement S0b (must complete before Sprint 1 begins; Gate 3 proto-scenario authored in Sprint 0)

---

### LLD-02 · Data Pool Manager

**Covers:** Two separate track structures with strictly separated roles.

*Track 1 — SWE-bench Verified:* Three disjoint pools: Dev-Bench (50 tasks, Contribution A leaderboard), Bench-Control (~50 tasks, in-domain SFT control arm), Final-Test (100 tasks, sealed until Sprint 3). Seed assignment, run-state tracking, deduplication guards, Final-Test seal enforcement.

*Track 2 — Codex-Long:* Four family-disjoint splits consumed from LLD-13's frozen assignment table: Train-Long (~30 families / ~150–240 envs), Val-Long (~10 families / ~30–50 envs), Test-Long (~10 families / ~40–60 envs), Public-Dev (~5 families / ~15–20 envs). Family-level metadata (scenario type, variant list, split assignment). Test-Long seal enforcement. Pre-declared reduced-split geometry for the 35-family path (Train-Long ~20, Val-Long ~7, Test-Long ~6, Public-Dev ~2).

*Cross-track:* Enforces Bench-Control + Train-Long-only access for LLD-10 (SFT training) — Final-Test and Test-Long are never used for training. Tracks run state per pool × model × seed.

**Connects to:** Consumes frozen split tables from LLD-13. Feeds task/env lists into LLD-07 (Benchmark Runner); labels outputs in LLD-06 (Trajectory Parser) by pool and family; enforces training-set-only access for LLD-10. Must exist before any run starts.

**v2.3 impact:** Full rewrite required. The old three-pool SWE-bench structure (Dev-Bench / Bench-Train 350 / Final-Test) is gone. Bench-Train is replaced by Codex-Long Train-Long. Bench-Control is new. Family-level tracking is new.

**Sprint:** Design S1 → Implement S1 early

---

## 🟡 Core Pipeline Layer — Design Sprint 1, Implement Sprint 1

### LLD-03 · Task Orchestrator

**Covers:** End-to-end per-task execution loop for **two distinct task types**:

*SWE-bench path:* SWE-bench repo checkout → Docker container spin-up → `codex exec --yolo --json` invocation → stdout capture → container teardown → patch extraction (feeds LLD-05 SWE-bench evaluator).

*Codex-Long path:* Codex-Long scenario env launch (Docker image from LLD-13 env factory) → `codex exec --yolo --json` invocation → stdout capture → container teardown → **post-run verifier injection and state-based grading** (verifier scripts and test resources injected into a separate post-run container per LLD-13 integrity protocol — never available to the agent during execution). Milestone checks also injected post-run.

*Shared:* Prefix-cache flush between tasks (`POST /reset_prefix_cache`); persistent sessions during collection (no `--ephemeral`); per-task structured event stream capture; health-check before dispatch.

**Connects to:** Central coordinator. Consumes LLD-01 (model endpoint), LLD-02 (task/env lists and seed assignments), and LLD-13 (Codex-Long Docker envs and verifiers). Emits raw JSONL into LLD-04 (Latency Capture) and LLD-06 (Trajectory Parser). Drives LLD-05 (Patch Converter for SWE-bench; verifier invocation for Codex-Long) on session close.

**v2.3 impact:** Significant rewrite. Must handle dual-track execution (SWE-bench patches vs Codex-Long state-based grading). Codex-Long integrity protocol (verifier injection timing, oracle isolation) is architecturally new.

**Sprint:** Design S1 → Implement S1

---

### LLD-04 · Latency Telemetry Capture

**Covers:** Real-time delta-sampling of vLLM `/metrics` to extract TTFT, prefill throughput (using `request_prefill_kv_computed_tokens` — newly computed tokens only), decode throughput, and prefix cache hit rate — per turn and per task. Stores 4-metric profiles per model × task run. Applies to both SWE-bench and Codex-Long runs identically.

**Connects to:** Tightly coupled to LLD-03 (coordinates snapshot timing around each task). LLD-01 §9.2 defines the metric formulas and delta-sampling protocol. Output feeds LLD-12 (Results & Artifact Generator) for the Contribution A latency anatomy tables.

**v2.3 impact:** Minor update only. The metrics and formulas are unchanged. Now captures telemetry for Codex-Long runs in addition to SWE-bench, but the capture mechanism is identical.

**Sprint:** Design S1 → Implement S1 (with LLD-03)

---

### LLD-05 · Evaluator (Dual-Track)

**Covers:** Two distinct grading paths:

*SWE-bench path:* Extracts unified diffs from completed Codex sessions, converts to SWE-bench `predictions.jsonl` format, invokes the Docker-based SWE-bench grader. Handles batch submission and result ingestion. Used for Dev-Bench, Bench-Control, and Final-Test.

*Codex-Long path:* Invokes the state-based verifier from LLD-13 in a separate post-run container. Collects binary pass/fail plus milestone partial-credit signals. Verifier scripts and test resources are injected after the agent session terminates (per LLD-13 integrity protocol).

*Shared:* Pass/fail results update LLD-02 (run state) and feed LLD-12 (leaderboard/ablation tables).

**Connects to:** Consumes LLD-03 session output. SWE-bench grading depends on patch extraction; Codex-Long grading depends on LLD-13 verifiers. Pass/fail results update LLD-02 (run state) and feed LLD-12. Gate 2 (base solve rate on SWE-bench) and Gate 4 (Codex-Long pilot solve rates) depend directly on this.

**v2.3 impact:** Significant rewrite. Now handles two grading modalities (patch-based + state-based). Codex-Long state-based grading with milestone checks is architecturally new.

**Sprint:** Design S1 → Implement S1

---

### LLD-06 · Trajectory Parser

**Covers:** Converts raw Codex JSONL event streams into SFT-ready training records: role masking (assistant tokens only), filtering to successful traces, and emitting multiple training splits:

- **Codex-SFT-all**: All successful Codex-native Train-Long traces (B2 headline training set)
- **Codex-SFT-matched**: Codex traces from Train-Long scenario IDs also solved by SWE-Agent (B1 2×2 arm)
- **SWE-Agent-SFT-matched**: SWE-Agent traces from matched Train-Long scenario IDs (B1 reference arm)
- **SWE-Bench-Control-SFT**: Bench-Control SWE-bench traces (diagnostic appendix only)

Family-level and scenario-ID metadata preserved for matched-ID splitting. Logprob-aware variant for Phase 2b.

**Connects to:** Consumes LLD-03 output (both SWE-bench and Codex-Long trajectories). Consumes LLD-02 pool labels and LLD-13 family/scenario metadata for matched-ID determination. Directly feeds LLD-10 (SFT Training Pipeline). Gate 5 (logprob availability) determines whether Phase 2b data is also produced here.

**v2.3 impact:** Moderate rewrite. Matched-ID logic now operates on Codex-Long scenario IDs and family structure (not SWE-bench task IDs). Must handle both SWE-bench and Codex-Long trace formats. New SWE-Bench-Control-SFT variant.

**Sprint:** Design S1 → Implement S1

---

### LLD-07 · Benchmark Runner

**Covers:** Multi-model, multi-seed scheduled execution across all campaign types:

- **Dev-Bench**: 6 models × 50 SWE-bench tasks × 1 seed (~300 runs, baseline contingent on Gates 1b/1c)
- **Bench-Control**: 27B × ~50 SWE-bench tasks × 1–2 seeds (~50–100 runs)
- **Codex-Long Train-Long collection**: 27B × ~200 envs × 1–2 seeds (wall-clock TBD from Gate 4)
- **Codex-Long Val-Long collection**: 27B × ~40 envs × 1 seed
- **Codex-Long SWE-Agent arm**: 27B × ~200 Train-Long envs × 1 seed (via LLD-09)
- **Sprint 3 B1 eval**: 6 model variants × ~50 Test-Long envs × 1 seed (~300 runs)
- **Sprint 3 B2 eval**: 8 evaluation lines × Final-Test 100 tasks (1100 core runs)

Job queue, retry logic, model switching between vLLM restarts (via LLD-01), progress checkpointing, wall-time accounting. Gate 4 pilot execution as a sub-campaign before committing full Train-Long collection.

**Connects to:** Wraps LLD-03 (task execution); coordinates model switching via LLD-01; pulls task/env lists from LLD-02; Codex-Long envs from LLD-13; SWE-Agent arm dispatched through LLD-09.

**v2.3 impact:** Significant rewrite. Campaign structure completely different (no Bench-Train 350; new Codex-Long collection campaigns; B1+B2 eval campaigns with distinct run rosters; Gate 4 pilot as a managed sub-campaign).

**Sprint:** Design S1 → Implement S1–S2

---

## 🟠 Parallel / Supporting Layer — Design Sprint 1, Implement Sprint 1–2

### LLD-08 · Logprob Capture Proxy

**Covers:** FastAPI shim between Codex and vLLM. Intercepts completions to capture per-token logprobs transparently and passes them through. Handles the Responses API path (the only supported `wire_api` since Feb 2026). Documents the known vLLM feature-request noting Responses API logprobs were empty and closed as "not planned."

**Connects to:** Proxies to LLD-01 (serving layer). Output consumed exclusively by LLD-11 (DAPO RL Pipeline). Required for Gate 5 validation; only strictly needed for Phase 2b. Phase 2b is pre-killed until Gate 5 passes.

**v2.3 impact:** Minor update only. The DAPO fragility warning and Responses API path are already documented in LLD-01 v0.3. No structural changes.

**Sprint:** Sprint 0 validation spike (Gate 5: minimal end-to-end logprob capture proof — does vLLM return per-token logprobs on the Responses path at all?) → Full design S1 → Implement S1. The HLD places Gate 5 in Sprint 0. This does not require the full proxy to be built — a throwaway script that hits vLLM and checks for logprob fields is sufficient. The full FastAPI shim is Sprint 1 work, gated on Gate 5 having passed.

---

### LLD-09 · mini-SWE-Agent

**Covers:** Lightweight bash-only agentic harness used for:

1. **Codex-Long SWE-Agent data collection arm**: ~200 Train-Long envs × 27B × 1 seed — produces SWE-Agent-SFT-matched traces for the B1 2×2 reference arm
2. **Gate 4 pilot SWE-Agent arm**: 5 pilot families × all variants × 1 seed — provides matched-ID comparison data
3. **B1 eval through SWE-Agent**: Test-Long × {Base, Codex-SFT-matched, SWE-Agent-SFT-matched} × 1 seed
4. **B2 diagnostics through SWE-Agent**: Final-Test × {Base, Codex-SFT-all, SWE-Agent-SFT-matched} × 1 seed
5. **Phase 2b ablation** (stretch): evaluating the RL'd model through a non-Codex harness to prove harness specificity

**Connects to:** Uses LLD-01 (model endpoint, `/v1/chat/completions` directly — no Codex layer), pulls tasks/envs from LLD-02 and LLD-13, sends successful traces to LLD-06 (SWE-Agent-SFT-matched variant). Results feed LLD-12 (B1 2×2 ablation, B2 diagnostics).

**v2.3 impact:** Moderate rewrite. Now operates on Codex-Long scenarios (not just SWE-bench tasks). Must handle Codex-Long Docker envs and state-based verifier grading. Role expanded significantly for B1 and B2 eval.

**Sprint:** Design S1 → Implement S1–S2

---

## 🔴 Training & Evaluation Layer — Design Sprint 2, Implement Sprint 3

### LLD-10 · SFT Training Pipeline

**Covers:** QLoRA fine-tuning of Qwen3.5-27B on harness-native traces. Handles four model variants:

1. **Codex-SFT-all**: All successful Codex-native Train-Long traces (~120–160 at 2 seeds). B2 headline model.
2. **Codex-SFT-matched**: Codex traces from matched Train-Long scenario IDs only. B1 2×2 cell A/B.
3. **SWE-Agent-SFT-matched**: SWE-Agent traces from matched Train-Long scenario IDs. B1 2×2 cell C/D.
4. **SWE-Bench-Control-SFT** *(appendix only)*: Bench-Control SWE-bench traces (~15–20 at 2 seeds). Diagnostic control, not a peer arm.

Includes hyperparameter sweeps, checkpoint management, training throughput logging. SWE-smith-SFT is a separate reference comparison (not trained here — uses published SWE-smith data).

**Connects to:** Consumes LLD-06 training records. Emits LoRA adapter checkpoints that feed LLD-11 (DAPO warm-start) and are evaluated via LLD-07 (Benchmark Runner) — B1 on Test-Long, B2 on SWE Final-Test. Adapters served through LLD-01 (LoRA serving, §17). Results go to LLD-12.

**v2.3 impact:** Moderate rewrite. Training data source changed from SWE-bench Bench-Train traces to Codex-Long Train-Long traces. Matched-ID logic now operates on scenario IDs/families. SWE-Bench-Control-SFT variant is new. Explicit "appendix only" framing for control arm.

**Sprint:** Design S2 → Implement S3

---

### LLD-11 · DAPO RL Pipeline *(stretch — Phase 2b only, pre-killed until Gate 5)*

**Covers:** Group rollout management (G = 4–8 per prompt), DAPO loss with Clip-Higher asymmetric clipping, token-level loss weighting, action masking on assistant tokens only, no KL penalty. VeRL or OpenRLHF integration. Requires on-policy rollouts through Codex on Codex-Long Train-Long scenarios (20–80% solve rate range). Val-Long for early stopping.

**Connects to:** Depends on LLD-08 (logprobs), LLD-10 (warm-start checkpoint), LLD-03 (rollout execution through Codex-Long envs), LLD-02 (task sampling within Train-Long), and LLD-13 (Codex-Long verifier for reward signal). Design only if Gate 5 passes and Phase 2a shows signal.

**v2.3 impact:** Minor update. RL training data source is Codex-Long Train-Long (not SWE-bench Bench-Train). Reward signal comes from LLD-13 state-based verifiers (not patch-based SWE-bench grading). Phase 2b is explicitly pre-killed in all planning.

**Sprint:** Design S2 (if Gate 5 passes) → Implement S3

---

### LLD-12 · Results & Artifact Generator

**Covers:** Two distinct result packages:

*Contribution A:* Bootstrap CI computation, Dev-Bench leaderboard (up to 6 models × 4 latency metrics × solve rate × tool-call reliability rate), failure-mode taxonomy.

*Contribution B:*
- **B1** (Test-Long): Family-clustered bootstrap CIs for the 2×2 harness-specificity matrix. Environment-level bootstrap reported alongside but family-level is the primary claim. Explicitly framed as directional pilot evidence.
- **B2** (SWE Final-Test): Environment-level bootstrap CIs. Codex-SFT-all headline, comparison arms, harness-specificity diagnostic (ΔCodex vs ΔSWE). SWE-smith reference comparison (optional).

Publication artifact packaging: data release, leaderboard, reproduction instructions, blog tables, Codex-Long Public-Dev split release.

**Connects to:** Terminal node — no other LLD depends on it. Aggregates from LLD-04 (latency), LLD-05 (solve rates for both SWE-bench and Codex-Long), LLD-06 (tool reliability, trajectory statistics), and LLD-10 (trained model scores).

**v2.3 impact:** Significant update. B1/B2 split requires two distinct statistical pipelines. Family-clustered bootstrap for B1 is new. Harness-specificity diagnostic (ΔCodex / ΔSWE) formalization is new. Public-Dev release packaging is new.

**Sprint:** Design S2 → Implement S2–S3

---

## Sequencing Summary

| # | LLD | Component | Design | Implement | v2.3 Change Level |
|---|-----|-----------|--------|-----------|-------------------|
| 1 | LLD-01 | vLLM Serving Layer | S0 | S0 | Minor update |
| 2 | LLD-13 | Codex-Long Scenario Framework | S0 proto-scenario + S0b | S0b | **NEW** |
| 3 | LLD-02 | Data Pool Manager | S1 | S1 early | **Full rewrite** |
| 4 | LLD-03 | Task Orchestrator | S1 | S1 | Significant rewrite |
| 5 | LLD-04 | Latency Telemetry Capture | S1 | S1 (with LLD-03) | Minor update |
| 6 | LLD-05 | Evaluator (Dual-Track) | S1 | S1 | Significant rewrite |
| 7 | LLD-06 | Trajectory Parser | S1 | S1 | Moderate rewrite |
| 8 | LLD-07 | Benchmark Runner | S1 | S1–S2 | Significant rewrite |
| 9 | LLD-08 | Logprob Capture Proxy | S0 spike (Gate 5) → S1 | S1 (if Gate 5 passed) | Minor update |
| 10 | LLD-09 | mini-SWE-Agent | S1 | S1–S2 | Moderate rewrite |
| 11 | LLD-10 | SFT Training Pipeline | S2 | S3 | Moderate rewrite |
| 12 | LLD-12 | Results & Artifact Generator | S2 | S2–S3 | Significant update |
| 13 | LLD-11 | DAPO RL Pipeline (stretch) | S2 if Gate 5 | S3 | Minor update |

---

## Critical Dependency Paths

**Contribution B critical path (guaranteed — Phase 2a SFT):** `LLD-01 → LLD-13 → LLD-03 → LLD-06 → LLD-10 → LLD-12`

LLD-13 is now on the critical path — no full Codex-Long collection or evaluation can begin until Sprint 0b delivers the scenario framework. (Exception: Gate 3 uses a single proto-scenario authored in Sprint 0 — see LLD-13 note.)

**Contribution B stretch branch (Phase 2b DAPO — pre-killed until Gate 5):** `LLD-08 → LLD-11`, branching off after LLD-10. This is not on the guaranteed critical path. Phase 2b is dead weight in all planning until Gate 5 passes; do not sequence other LLDs around it.

**Contribution A critical path:** `LLD-01 → LLD-07 → LLD-05 → LLD-12`

Unchanged structurally, though LLD-07 and LLD-05 have broader scope.

**Gate 4 critical path:** `LLD-01 → LLD-13 → LLD-03 → LLD-05 (Codex-Long verifier) → LLD-07 (pilot sub-campaign)`

Gate 4 is a new critical-path gate that must complete before any full Train-Long collection is committed.

---

*Document version: 2.1*
*Derived from HLD Spec v2.3 · April 2026*
*v2.1 fixes: (1) LLD-11 removed from Contribution B critical path — Phase 2a SFT is the guaranteed path, DAPO is a stretch branch off LLD-10. (2) LLD-08 Gate 5 sequencing resolved: Sprint 0 validation spike (does vLLM return logprobs on Responses path?), full proxy implementation Sprint 1. (3) Gate 3 proto-scenario exception added to LLD-13 — one Codex-Long-style scenario authored in Sprint 0 for 27B rollout wall-clock measurement, before the full Sprint 0b framework.*
