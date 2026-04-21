# HLD · Training Flywheel (LLD-06 → LLD-11)

> Codex-Bench · High-Level Design
> Scope: The data-normalization + training stack — LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11
> Derived from HLD Spec v2.3 · April 2026
> Status: DRAFT v0.1 — framing doc before LLD-06/07/08/09/10/11 are drafted in detail

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial framing. Establishes the "one raw event store, three derived training views, one evaluation boundary" architecture that LLD-06 through LLD-11 jointly implement. Locks the no-human-labels stance, the SFT → auto-preference → RLVR training sequence, and the verifier-first integrity posture. |

---

## 1. Purpose

LLD-06 through LLD-11 together own every step between *"the agent finished a run"* and *"we have a trained model to evaluate."* They are six adjacent documents, but they are not six independent designs. They are one flywheel. This doc pins the architectural decisions that the six downstream LLDs must be coherent with, so that each LLD can be written without re-litigating the cross-cutting choices.

**The six components in scope:**

| LLD | Role in the flywheel |
|---|---|
| LLD-06 Trajectory Parser | Normalizes raw agent runs into the canonical event store and emits the derived training views |
| LLD-07 Benchmark Runner | Drives the rollout/collection campaigns that feed the event store, and drives the evaluation campaigns that consume trained checkpoints |
| LLD-08 Logprob Capture Proxy | Gate-5-contingent shim that makes Phase 2b (RL) viable by exposing per-token logprobs on the Responses API path |
| LLD-09 mini-SWE-Agent | Second harness used both as a *data source* (SWE-Agent-SFT-matched) and as an *evaluation harness* (B1 reference arm, B2 diagnostic arm) |
| LLD-10 SFT Training Pipeline | QLoRA fine-tuning on the SFT view; produces the B1 2×2 arms and the B2 headline model |
| LLD-11 DAPO RL Pipeline | Stretch-only, pre-killed until Gate 5 passes; consumes the RL prompt pool and the verifier reward |

The rest of this document is the architectural contract that holds across all six.

---

## 2. Core Design Decisions

Three decisions drive every downstream choice in these six LLDs. Each is load-bearing and should be revisited only if the underlying assumption changes.

### Decision 1 — One canonical event store; three derived training views

Do **not** split raw runs at collection time into "this is for SFT, this is for DPO, this is for RL." Store every run once, in a canonical form, and derive the training-method-specific views as projections.

**Why:** SFT, preference/ranking, and RL want *different shapes* of the same underlying data. Pre-committing a run to one method at collection time is wasted optionality — the same run may enter the SFT view as a successful trajectory, contribute to the preference view as the winner of a same-task pair, and its prompt may seed the RL pool. If we shard the raw store by intended use, we lose that compounding.

**How to apply:** LLD-06 is the normalization + projection layer. It owns the canonical schema and the three derived tables. LLD-07 drives the collection, but does not decide "this is SFT data" at collection time — it just runs rollouts and lets LLD-06 project.

### Decision 2 — Train sequentially: SFT → auto-preference → RLVR

Run the three training methods in order, not in parallel, and gate each on the previous one showing signal.

1. **SFT first.** Lowest risk, behavior-cloning from successful traces. Already the guaranteed Phase 2a path in the HLD.
2. **Auto-derived preference/ranking second.** Offline DPO-style objective on pairs constructed *programmatically* from same-task attempts. Cheap incremental win on top of SFT.
3. **Policy-gradient RL (DAPO) third.** Only after (a) the grader/replay path is solid, (b) Gate 5 has passed, and (c) SFT has shown signal. Phase 2b is pre-killed in all planning.

**Why:** RL is the most expensive and the most likely to game a weak verifier. Walking through SFT and auto-preference first means we (i) have a warm-start checkpoint RL actually needs, (ii) have a grader that has survived SFT-era scrutiny, and (iii) know which prompts land in the 20–80% solve band RL requires.

**How to apply:** LLD-10 lands first with the SFT arms. Auto-preference data is *emitted* by LLD-06 from the start (so we're not regretting missing metadata later) but *training on it* is a Phase 2a.5 add-on, not a blocker. LLD-11 is the stretch branch.

### Decision 3 — No human labeling; verifier-first flywheel

The grader is the label. We do not staff a human preference-labeling pipeline, and the SFT view does not depend on human-written gold traces.

**Why:** Coding and agentic tasks are *verifiable*. The benchmark already exists: hidden tests + state-based verifiers + milestone vectors + run metadata (turns, tool calls, crash flags, forbidden-edit flags). Current SOTA training pipelines for this regime — SWE-smith's execution-backed filtering, Agent-RLVR's environment-derived rewards, ReVeal's iterative self-verification, Self-play SWE-RL's test-free bug-inject/repair loop — all demonstrate that verifier-first beats human-labeled on cost/coverage in this domain. Humans are valuable for rubric design and reward-hack audits, *not* for per-run labels.

**How to apply:** Any budget that would otherwise go to labeling goes into **verifier hardening** instead — hidden tests, anti-shortcut checks, verifier isolation from the agent, family-disjoint evaluation, and the LLD-13 integrity protocol. The risk we are accepting is *reward hacking*, and the mitigation is grader quality + adversarial audits, not human review of every trajectory.

---

## 3. The Flywheel Architecture

```text
                 ┌──────────────────────────────────────────┐
                 │           RAW EVENT STORE                │
                 │  (canonical, one row per run,            │
                 │   lossless wrt tokens/tools/reward)      │
                 │                                          │
                 │  LLD-06 is the writer.                   │
                 │  LLD-03/05 are the upstream producers.   │
                 │  LLD-02 owns pool/family metadata.       │
                 └──────────────┬───────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
 ┌───────────────┐     ┌────────────────┐     ┌──────────────────┐
 │ SFT view      │     │ Preference     │     │ RL prompt pool   │
 │ (per-turn +   │     │ view           │     │ (tasks + grader  │
 │  per-traj)    │     │ (pairwise /    │     │  ref + reward    │
 │               │     │  listwise)     │     │  config)         │
 │ successful    │     │ same-task      │     │ Train-Long only, │
 │ Codex traces  │     │ attempts,      │     │ 20–80% band      │
 │ + matched-ID  │     │ auto-labeled   │     │                  │
 │ SWE-Agent     │     │ by outcome     │     │                  │
 │ variants      │     │ rank           │     │                  │
 └───────┬───────┘     └────────┬───────┘     └─────────┬────────┘
         │                      │                       │
         ▼                      ▼                       ▼
  ┌──────────────┐      ┌───────────────┐      ┌─────────────────┐
  │ LLD-10 SFT   │      │ DPO (Phase    │      │ LLD-11 DAPO RL  │
  │ QLoRA        │      │ 2a.5 add-on)  │      │ (stretch;       │
  │              │      │               │      │  Gate 5 only)   │
  └──────┬───────┘      └───────┬───────┘      └────────┬────────┘
         │                      │                       │
         └──────────┬───────────┴───────────────────────┘
                    ▼
           ┌────────────────────┐
           │ Evaluation         │
           │ (LLD-07 + LLD-05 + │
           │  LLD-09 harness    │
           │  comparison)       │
           └────────────────────┘
```

### 3.1 The canonical event store

Every run — SWE-bench Dev-Bench, Bench-Control, Codex-Long Train-Long, Val-Long, SWE-Agent collection arm, Gate 4 pilot, final eval — writes *one row* to a single store. The schema is lossless with respect to the things all three training views need, and enriched with the metadata that makes auto-labeling possible.

```yaml
# Canonical run record (LLD-06 writer contract)
task_id:              # scenario ID (Codex-Long) or instance ID (SWE-bench)
split:                # dev_bench | bench_control | train_long | val_long | test_long | final_test | public_dev
family_id:            # Codex-Long family or null for SWE-bench
harness:              # codex | swe_agent
model_id:             # resolves against model_registry.yaml
seed:                 # integer
prompt_state_0:       # initial prompt + repo snapshot hash
trajectory_jsonl_ref: # pointer to full event stream
turn_records:         # [{turn_idx, state_text, assistant_tokens, tool_calls, tool_outputs}]
final_artifact_ref:   # patch (SWE-bench) or state digest (Codex-Long)
outcome:              # pass | fail | timeout | crash | no_patch
cl_pass:              # Codex-Long verifier bool, null for SWE-bench
milestone_vector:     # [{milestone_id, passed_bool}] or null
failure_mode:         # enum from LLD-05
wall_time_seconds:    # float
token_count:          # {prompt, completion, total}
tool_call_count:      # int
crash_flags:          # list of infrastructure/tool errors
forbidden_edit_flags: # list, empty if clean
grading_manifest_ver: # for reproducibility of cl_pass
```

This is the authoritative shape. It matches what LLD-05 v0.3 already normalizes for grading (`outcome`, `cl_pass`, `milestone_json`, `grading_manifest_ver`), plus the trajectory-level fields LLD-06 needs to project into training data. LLD-06 does not re-grade — it reads LLD-05's normalized fields as the source of truth for `outcome` / `cl_pass`.

### 3.2 Derived view A — SFT

Projection filters:

- `outcome == pass` (SWE-bench) or `cl_pass == true` (Codex-Long)
- `split in {bench_control, train_long}` — never Final-Test, never Test-Long
- optional: `forbidden_edit_flags == []` to exclude obvious reward hacks from the training corpus

Emission variants (the four LLD-10 training arms):

| Variant | Source | Role |
|---|---|---|
| Codex-SFT-all | All successful Codex-native Train-Long traces | B2 headline model |
| Codex-SFT-matched | Codex successes on scenario IDs also solved by SWE-Agent | B1 2×2 cells A/B |
| SWE-Agent-SFT-matched | SWE-Agent successes on the same matched scenario IDs | B1 2×2 cells C/D |
| SWE-Bench-Control-SFT | Successful Bench-Control SWE-bench traces | Diagnostic appendix only |

Row shape is turn-level with assistant-token-only masking:

```yaml
# SFT training row
task_id:
turn_index:
state_text_or_messages:  # prompt + compact history + latest tool outputs
assistant_target_tokens:
mask:                    # assistant-only
weight:                  # default 1.0; allows down-weighting e.g. long tool churn
quality_flags:           # e.g. ["clean_patch", "low_tool_churn"]
```

### 3.3 Derived view B — Auto-preference

Projection is a join over the event store grouped by `(task_id, family_id)` with at least two attempts.

Automatic ranking rules (strict, programmatic — no human label):

1. **Outcome rank:** `pass > milestone-rich fail > bare fail > timeout > crash`.
2. **Tie on pass:** fewer turns wins; ties on turns → fewer tool calls; ties → no `forbidden_edit_flags` wins; final tie → arbitrary stable ordering.
3. **Tie on fail:** more milestones passed wins; ties → fewer `crash_flags` wins; final tie → dropped (not emitted as a preference pair).

Row shape:

```yaml
# Preference training row
task_id:
state_id:               # shared prompt state the two responses branched from
chosen_response:        # trajectory ref + assistant-token view
rejected_response:
preference_reason_code: # enum: OUTCOME_RANK | TURN_COUNT | TOOL_COUNT | MILESTONE_MARGIN | CLEANLINESS
margin_score:           # numerical delta (e.g. milestone count diff, turn diff)
human_override:         # null in v1 (no-human-labels stance)
```

This view is **emitted from day one** (LLD-06 responsibility) even though training against it is a Phase 2a.5 add-on. The cost of producing pairs during collection is negligible; the cost of re-deriving them later if the metadata was never captured is high.

### 3.4 Derived view C — RL prompt pool

RL does not consume a fixed trajectory set. It consumes `(prompt, grader, reward-config)` triples and generates fresh rollouts. So the "view" is really a filtered prompt catalog:

```yaml
# RL prompt entry
task_id:
initial_state:          # deterministic seed state
grader_ref:             # LLD-13 verifier handle
milestone_config_ref:   # partial-credit config
reward_weights:         # {pass: 1.0, milestone_partial: ..., penalties: {...}}
difficulty_bucket:      # inferred from SFT-model solve rate: too_easy | in_band | too_hard
allowed_for_online_rl:  # bool — Train-Long only, in-band only
```

Difficulty bucketing is derived from the SFT-model's observed solve rate on each prompt; only the 20–80% band flows to LLD-11. This is a hard operational constraint of DAPO rollout groups being informative.

### 3.5 The evaluation boundary

The same canonical store that feeds training also feeds evaluation, but with *strict* split enforcement:

- Training views can only project over `{bench_control, train_long}`.
- Test-Long is sealed for B1 evaluation; Final-Test is sealed for B2 evaluation.
- LLD-02 enforces access control; LLD-06 honors it; LLD-10 cannot read a row outside the training-allowed set.

This is not a new policy — it is the HLD v2.3 evaluation boundary, restated here because the flywheel architecture would silently violate it if the projection filters were wrong.

---

## 4. Component Roles

### 4.1 LLD-06 Trajectory Parser — the writer

LLD-06 is the single writer for the canonical event store *and* the three derived views. Its responsibilities expand beyond the v1 LLD-06 scope:

- **Normalization:** consume raw Codex JSONL (from LLD-03) plus SWE-Agent traces (from LLD-09) and produce canonical run records. Use LLD-05's normalized outcome fields — do not re-grade.
- **SFT view:** emit all four SFT variants with turn-level rows and assistant-only masking.
- **Preference view:** emit pairs from same-task attempt groups using the deterministic ranking rules in §3.3.
- **RL prompt view:** emit the prompt catalog with difficulty buckets populated after the first SFT checkpoint exists.
- **Metadata preservation:** family IDs, scenario IDs, forbidden-edit flags, milestone vectors — all must survive the normalization.

### 4.2 LLD-07 Benchmark Runner — the driver

LLD-07 drives both sides of the flywheel:

- **Collection campaigns** populate the event store: Train-Long 27B collection, Val-Long collection, Bench-Control, Codex-Long SWE-Agent arm (delegated to LLD-09).
- **Evaluation campaigns** consume trained checkpoints: B1 on Test-Long, B2 on Final-Test, Gate 4 pilot sub-campaign.
- **Gate enforcement:** LLD-07 is where Gate 4 blocks full Train-Long commitment, and where the RL prompt pool's difficulty bucketing gets populated (by running the SFT checkpoint over Train-Long prompts).

LLD-07 does **not** know anything about training-view projection. It only knows campaigns, rollouts, and model switching.

### 4.3 LLD-08 Logprob Capture Proxy — the enabler

Exists to make LLD-11 viable. Not on the Phase 2a critical path. Gate 5 sequencing is unchanged from the v2.1 index:

1. Sprint 0 spike — throwaway script proves vLLM returns per-token logprobs on the Responses API path.
2. If Gate 5 passes: Sprint 1 full FastAPI shim.
3. If Gate 5 fails: Phase 2b stays dead; LLD-11 never designs in detail; the flywheel ends at Phase 2a.5.

### 4.4 LLD-09 mini-SWE-Agent — the second harness

LLD-09 plays two roles in the flywheel:

- **Data producer:** runs ~200 Train-Long envs to generate SWE-Agent-SFT-matched traces. Its outputs feed LLD-06 just like Codex outputs do.
- **Evaluation harness:** for B1 (proving harness specificity: does a Codex-trained model fall over when evaluated through SWE-Agent?) and B2 diagnostics.

The flywheel architecture does not care which harness produced the trace — the canonical store's `harness` field discriminates, and LLD-06's projection filters use that field to emit the correct variant.

### 4.5 LLD-10 SFT Training Pipeline — the guaranteed path

LLD-10 consumes the SFT view and trains the four arms listed in §3.2. Training output is LoRA adapter checkpoints that feed back to LLD-01 (serving) and LLD-07 (evaluation).

The **most important checkpoint** from LLD-10 is the Codex-SFT-all adapter, because:

- It is the B2 headline result.
- It is the warm-start for LLD-11 (if Gate 5 passes).
- It is what we run over Train-Long prompts to bucket RL difficulty (if Gate 5 passes).

So LLD-10 is the hinge: Phase 2a stands on its own as a publishable result, *and* it is the upstream of every Phase 2b step.

### 4.6 LLD-11 DAPO RL Pipeline — the stretch branch

Pre-killed until Gate 5. When designed, it consumes:

- **Prompt pool:** LLD-06's RL view, filtered to `allowed_for_online_rl == true` and `difficulty_bucket == in_band`.
- **Reward:** LLD-13 state-based verifier + milestone partial credit + penalties (timeout, crash, forbidden-edit, malformed tool output).
- **Warm start:** the Codex-SFT-all LoRA from LLD-10.
- **Rollout execution:** through LLD-03 with LLD-08 proxying for logprob capture.

The training loop itself (DAPO, group size, Clip-Higher, action masking) is unchanged from the HLD v2.3 spec; the flywheel contribution is just that the prompt pool and reward config come from one coherent place.

---

## 5. Training Sequence

| Phase | Trigger | What trains | Data view | Gating |
|---|---|---|---|---|
| 2a.0 | After Gate 4 + Train-Long collection complete | LLD-10 Codex-SFT-all + matched arms + Bench-Control appendix | SFT view | Hard-required for B2 |
| 2a.5 | After Phase 2a shows signal, optional | DPO on LLD-06 preference view (off-policy) | Preference view | Soft — skip if Phase 2a is already at the publication bar |
| 2b | After Gate 5 + Phase 2a signal | LLD-11 DAPO RL on Train-Long prompts | RL prompt pool | Pre-killed; never assumed in planning |

**The rule:** no phase starts before the previous phase has produced a checkpoint that beats the previous best on Val-Long by a bootstrap-CI-separated margin. Gate-style, not calendar-based.

---

## 6. No-Human-Labels Stance — What We're Buying and Paying For

### What we buy

- **Speed.** No labeling pipeline to staff, no interface to build, no reviewer calibration.
- **Coverage.** Every run produces usable signal, not just the ones a human annotator has time for.
- **Reproducibility.** The label is a function of the verifier; re-running the verifier reproduces the label deterministically.
- **Alignment with current SOTA.** SWE-smith (execution-backed filtering, 50k synthesized instances from 128 repos), Agent-RLVR (9.4% → 22.4% on SWE-bench Verified with env-derived rewards), ReVeal (turn-level credit via iterative verification), Self-play SWE-RL (no human issues or tests), Agent Lightning (decouple execution from training) — the trajectory of the field is toward verifier-first, not labeler-heavy.

### What we pay

- **Reward-hack risk.** A weak verifier becomes the ceiling: RL will find the shortest path to pass it, not the shortest path to good engineering. The recent RLVR-gaming literature is unambiguous on this.
- **Quality-vs-outcome conflation.** Two traces that both pass the verifier are marked equivalent, even if one is maintainable and one is a hack. Automatic preference rules partially address this (cleanliness, turn count) but do not substitute for human judgment on ambiguous calls.
- **Narrow optimization target.** The grader defines "good." If the benchmark's verifiers have systematic blindspots, we train into them.

### How we pay it — the mitigations

1. **Verifier hardening as a first-class investment.** Hidden tests (never visible to the agent), anti-shortcut checks (forbidden-file detection, test-tampering detection), verifier isolation (run in a separate post-run container per LLD-13's integrity protocol), family-disjoint evaluation (Test-Long families never seen in training). These are already in LLD-13; this doc reaffirms them as *load-bearing for the training stack*, not just the benchmark.
2. **Adversarial audit of the grader.** Before Sprint 3, an explicit pass where we try to construct traces that exploit the verifier. Anything we find, we patch *before* RL touches the prompt.
3. **Milestone partial credit.** Forces the model to make real progress, not just pass the final check. Dense reward is cheaper than human preference.
4. **v1.5 escape hatch for humans.** We reserve the option to add a small, targeted human pairwise review set later — specifically on (a) both-pass comparisons, (b) close-call failures, (c) cases where automatic rules declare a winner but a spot-check disagrees. This is not in v1 scope. It exists as a contingency if the flywheel plateaus.

---

## 7. Research Grounding

This architecture is not invented from scratch. It composes patterns that have shipped in public research on agentic/coding RL. The relevant references, and what each contributes:

| Paper / system | Pattern we inherit |
|---|---|
| **SWE-smith** | Automatic task generation + execution-backed filtering as the data engine. Confirms that 50k-scale trajectory corpora from ~100 repos are feasible without human issue labels. |
| **Agent-RLVR** | Environment-derived reward (unit tests as verifiable signal) is strong enough to move multi-turn agents on SWE-bench Verified from ~9% to ~22% Pass@1. Their guidance layer is a good template if our sparse reward stalls. |
| **ReVeal** | Verifier-inside-the-loop with turn-level credit assignment. If our reward is too sparse late in Phase 2b, this is the upgrade path — denser verifier structure inside the trajectory, not more human labels. |
| **Self-play SWE-RL (SSR)** | Most aggressive test-free version. Not something we adopt directly (we *have* a benchmark), but a useful backstop: if our task pool gets exhausted, self-play bug-inject/repair is the continuation. |
| **Agent Lightning** | Decouple execution from training; turn runs into transition-level records. This is literally the "one raw event store" decision in §3. |
| **SWE-RM** | Learned reward model from execution outcomes (no human prefs) as a densifier when raw pass/fail is too coarse. Optional v2 upgrade, not v1 scope. |

The v1 flywheel is, approximately: SWE-smith's data-engine philosophy + Agent Lightning's record architecture + Agent-RLVR's reward source, stopping at SFT and auto-preference for Phase 2a and only branching to RLVR in Phase 2b.

---

## 8. Risks And Open Questions

Risks the downstream LLDs must address:

- **Verifier gaming.** Mitigated by §6. LLD-13 owns the integrity protocol; LLD-10/11 honor the forbidden-edit flags in the training view filter.
- **Preference-pair sparsity.** If we only run 1 seed per task for most campaigns, we get very few same-task pairs. LLD-07 should plan for ≥2 seeds on Train-Long at least for the Codex arm, so that LLD-06 has pairs to emit.
- **Difficulty-bucket miscalibration.** The RL prompt pool depends on SFT-model solve rates being meaningful. If the SFT model is far from the target, the 20–80% band collapses. LLD-07 should re-bucket after each SFT checkpoint.
- **Forbidden-edit detection coverage.** This is the single most important anti-reward-hack signal in the SFT view filter. LLD-13 owns the detection; LLD-06 owns the filter; both must move together.
- **Harness-specific trace leakage.** The B1 2×2 claim depends on matched-ID splits being clean. LLD-06's matched-ID logic operates on Codex-Long scenario IDs and family structure, not on SWE-bench task IDs. This is a v2.3 change; LLD-06 must get it right.

Open questions for the individual LLDs to resolve:

- **LLD-06:** exact turn-segmentation rule for SFT rows when a Codex turn contains multiple tool calls — do we emit one row per tool call or one row per assistant turn? (Default recommendation: one row per assistant turn; tool-call-level is an optimization.)
- **LLD-07:** seed-count policy for Train-Long Codex collection. Default recommendation: 2 seeds, to enable auto-preference pairs.
- **LLD-08:** whether to ship the proxy as a sidecar process or an in-process shim. Gate-5-dependent.
- **LLD-09:** whether SWE-Agent-SFT-matched requires its own "control for harness differences" arm, or whether the B1 2×2 design already covers it. Default: the 2×2 covers it.
- **LLD-10:** whether to train the auto-preference DPO arm as a separate adapter or as a post-SFT continuation of the Codex-SFT-all adapter. Default recommendation: continuation.
- **LLD-11:** whether to run DAPO against a frozen reference policy (SFT-all) or bootstrap from the current policy. HLD v2.3 says no KL penalty; this doc does not override that.

---

## 9. Sequencing Across Sprints

```text
Sprint 0   → LLD-08 Gate 5 spike (pass/fail binary decision)
Sprint 0b  → LLD-13 complete (unblocks everything else)
Sprint 1   → LLD-06 design+impl      (canonical store + 3 views)
           → LLD-07 design+impl      (collection + eval campaigns)
           → LLD-09 design+impl      (SWE-Agent harness)
           → LLD-08 full proxy impl  (IFF Gate 5 passed)
Sprint 2   → LLD-10 design           (SFT pipeline)
           → LLD-11 design           (IFF Gate 5 + Phase 2a signal)
Sprint 3   → LLD-10 implement + train 4 arms
           → LLD-11 implement + DAPO run (IFF still viable)
           → LLD-12 package results
```

Nothing in this sequencing contradicts the v2.1 LLD index — it just makes the "why these six go together" case explicit.

---

## 10. What This Doc Locks Down For The Downstream LLDs

When LLD-06 through LLD-11 are drafted in detail, they **must** be coherent with:

1. One canonical event store, written by LLD-06, read by the training LLDs via projection. No training LLD reads raw Codex JSONL directly.
2. The four SFT variants, the auto-preference view, and the RL prompt pool are the three — and only three — derived training views.
3. No human labels in v1. Any LLD that wants to add a human-labeling step must justify it against the §6 mitigations and must not block Phase 2a.
4. Training sequence is SFT → (optional auto-DPO) → RL, gated on signal, not calendar.
5. The evaluation boundary (Bench-Control + Train-Long only; Test-Long and Final-Test sealed) is enforced by LLD-02 access control, honored by LLD-06's projection filters, and trusted by LLD-10 and LLD-11.
6. Phase 2b is pre-killed in all planning. LLD-11 is stretch. LLD-08 is only built if Gate 5 passes. No other LLD schedules against Phase 2b.

Deviations from any of the six require a Change Summary entry in this document first.

---

*Document version: 0.1*
*Scope: LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11*
*Supersedes: nothing — new framing doc*
*Does not supersede: HLD Spec v2.3, the v2.1 LLD index, or LLD-13 v0.6. This doc is subordinate to all three.*
