# Variant progression (V1 → V5)

The five-variant pattern is the family's **discrimination curve**. Each variant adds exactly one new dimension of judgment pressure on top of its predecessor. The agent should score cleanly on V1 (≥ pass_bar) and partially or not at all on V5; the curve between them is the signal.

Do not rearrange the order. V1-V5 correspond one-to-one with §10.1's monotonicity gate (`V1 ≥ V2 ≥ V3 ≥ V4 ≥ V5 ± tolerance`), which is how the freeze gate detects a family where the variants aren't actually progressing in difficulty.

## V1 — Clean baseline

**Goal.** Confirm the agent can produce a structurally valid brief against minimal evidence.

**Evidence.** The minimum needed to rank N proposals: the proposal files themselves, one perf note, one staffing memo, one rollout-history doc. No stale data, no distractors, no dirty state.

**Ceilings that can fire.** None new. V1 is the canonical-task floor — if `missed_staffing_update` or `objective_drift` fires on V1, the evidence is leaking V2+ content.

**τ threshold.** ≥ 0.67 (N=4 proposals admits discrete τ in {1.0, 0.67, 0.33, …}; this allows at most one inversion).

**Expected oracle ≥ 90; expected capable-model mean ≈ 85-95.** V1 is a floor-check, not a discriminator.

## V2 — Noisy distractor

**Goal.** Penalize the agent for anchoring on stale or irrelevant evidence that looks superficially reasonable.

**Evidence delta vs V1.**
- Add a stale perf marker (e.g., `validator_service_jan_bench.md` labeled with the quarter *before* a known regression). Register its markers in `gold_ranking.json.stale_perf_markers`.
- Optionally add a departing-engineer row to `staffing.md` with a handoff-plan footnote. This creates a legitimate constraint the agent must acknowledge but doesn't change the gold ranking.
- Add one extra proposal (typically the validator-microservice P5) so the ranking is N=5.

**New ceilings.**
- `ignored_stale_perf = 25`. Fires when `accepted_proposal.summary` or its citations reference any marker in `stale_perf_markers` without a simultaneous acknowledgment of staleness in `primary_risk`, the accepted entry, or the assumption ledger.

**τ threshold.** ≥ 0.80 (N=5 admits τ in {1.0, 0.80, 0.60, …}; this allows at most one inversion).

**Expected capable-model mean.** Similar to V1 if the agent handles the distractor correctly; ceiling fire drops to ≤ 25 if it doesn't.

## V3 — Dirty state

**Goal.** Penalize sunk-cost thinking. The workspace contains the residue of an abandoned effort; the agent must not treat it as a head start.

**Evidence delta vs V2.**
- Add `repo_evidence/code/in_progress_patch/` (and optionally a second scaffolding directory for a different proposal) with a README that reads like work was interrupted mid-flight.
- Optionally add a half-complete `brief/manager_brief.md` from "the previous session" that picks the sunk-cost proposal. The agent must ignore or overwrite this.

**New ceilings.**
- `sunk_cost_finish = 30`. Fires when the accepted proposal corresponds to the abandoned-patch owner (usually P3) — i.e., the agent "completed" the sunk-cost work instead of ranking it last.

**Expected capable-model mean.** Similar to V2. Ceiling fires on agents that confuse "there's existing code" with "this is the right pick".

## V4 — Multi-corpus / objective drift

**Goal.** Force the agent to re-weight proposals when the objective changes between cycles. The right accepted pick is no longer the V1-V3 pick — it's the one that advances the *new* objective.

**Evidence delta vs V3.**
- Add `release_context/release_notes_*.md` describing a recent incident thread (e.g., INC-2411, INC-2419, INC-2427) that shifts the team's priority from raw latency to streaming reliability.
- Re-point `gold_ranking.accepted` to the proposal that matches the new objective (commonly P2 for reliability-isolation or P5 for validator-service).
- Keep V1-V3's evidence intact so the agent must actively reason "the old answer was P4, but the objective changed, so now P2".

**New ceilings.**
- `objective_drift = 45`. Fires when the agent picks the V1-V3 accepted proposal instead of the new objective's pick — i.e., they didn't re-read the corpus and noticed nothing changed. This is the workhorse ceiling: it reliably cuts capable-model scores in half on V4.

**Expected capable-model mean.** ≈ 40-50. Most capable models drift-blind to V4; the ones that notice the release_context shift jump to ≥ 90.

## V5 — Recovery in thread

**Goal.** Force incident-aware decision-making. The accepted pick from the prior cycle was rolled back; the agent must explicitly acknowledge the incident and either re-select the prior pick with a concrete reason the rollback is resolved, or pick a successor.

**Evidence delta vs V4.**
- Add `incident_context/incident_2026_04_P2_rollback.md` (the incident that rolled back the V4 accepted pick), `incident_context/watermark_bug_notes.md` (root-cause notes), and `incident_context/prior_ranking.md` (the ranking that was accepted then rolled back).
- Re-point `gold_ranking.accepted` to the incident-safe successor (commonly P5, which doesn't touch the module that caused the rollback).

**New ceilings.**
- `incident_blind_reselect = 30`. Fires when `accepted == prior_ranking[0]` (the rolled-back pick) AND no citation in the brief references any file under `incident_context/`. A manager who re-selects the rolled-back pick without citing the incident is blind to the history.
- `ignored_stale_perf = 25` (carried forward) still applies; V5 evidence typically carries the same jan_bench markers as V2-V4.

**Expected capable-model mean.** Highly variable. When the model catches both the incident and the stale perf, it scores ≥ 90; when it misses either, it drops to 25-30. V5's high variance is the "honest frontier signal" — do not try to flatten it, it is the judgment being measured.

## Per-variant `gold_ranking.json` fields

Minimum fields every variant needs:

```json
{
  "accepted": "P4",
  "ranking": [{"proposal_id": "P4", "rank": 1, "reason": "…"}, …],
  "primary_risk": "schema-cache warmup cost on cold process start",
  "staffing_blocked_proposal": "P3",
  "staffing_update_file": "repo_evidence/meta/roster_memo.md",
  "staffing_update_owner_proposal": "P4",
  "pass_bar": 40,
  "readonly_tree_hashes": { /* filled by regen script */ },
  "test_ranking_brief_sha256": "…",
  "variant_id": "v1-clean-baseline"
}
```

V2+ adds: `stale_perf_markers: [...]`.
V3+ adds: `sunk_cost_trap_proposal: "P3"`.
V4+ adds: (no new top-level field — the objective_drift ceiling reads from `accepted` vs V1-V3's `accepted`).
V5+ adds: (no new top-level field — `incident_blind_reselect` reads from `brief.ranking[0].proposal_id` and citations).

## Cross-variant rule: keep AGENTS.md identical

All five variants ship the same `AGENTS.md`. Differences in the agent's visible instructions across variants would teach the agent what changed, which is the disclosure paradox. The variant difference must live only in the evidence and in the gold, never in what the agent is told.
