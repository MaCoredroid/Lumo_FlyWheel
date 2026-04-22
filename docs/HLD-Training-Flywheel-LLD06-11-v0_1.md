# HLD · Training Flywheel (LLD-06 → LLD-11)

> Codex-Bench · High-Level Design
> Scope: The data-normalization + training stack — LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11
> Derived from HLD Spec v2.3 · April 2026
> Status: DRAFT v0.8 — family-level DONE spec split out into `HLD-Family-Test-Requirements-v0_1.md` (sibling doc in `docs/`). §15 reduced to a stub pointer; all outbound cross-references rewritten from §15.X → `Family-Test-Requirements §N`. No rule changes vs v0.7.

---

## Changelog

| Version | Change |
|---|---|
| v0.8 | **Family-level DONE spec split out** into its own sibling document `HLD-Family-Test-Requirements-v0_1.md` (v0.1). Motivation: §15 had grown to ≈ 230 lines with an audience (family authors finalizing CNB-55 artifacts) distinct from the rest of this HLD (LLD-06 → LLD-11 implementers). The split-out doc owns §1 Why, §2 Two layers of DONE, §3 Layer A (freeze gate), §4 Layer B (14-item checklist), §5 Required verification matrix, §6 Precedent decisions, §7 Blocking vs non-blocking, §8 Handoff template. §15 in this HLD is now a short pointer section summarising the two-layer DONE model and listing the split-out doc's table of contents. All outbound cross-references in §12, §16, and §17 rewritten from `§15.X` → `Family-Test-Requirements §N` (10 references touched: §12 Terminal-Bench row; §17.1, §17.5, §17.6, §17.7, §17.8, §17.10, §17.12 Updates/Rule/Authoring lines). **No rule changes** — every authoring obligation, verification-matrix row, blocking-vs-non-blocking decision, and worked `proposal-ranking-manager-judgment` example carries over verbatim into the new doc. The changelog entries for v0.4 through v0.7 retain their original `§15.X` citations as historical references. |
| v0.7 | Second red-team pass. Five hardenings, all concentrated in §17 and §3.1: (a) **Recovery-trace loss upweight scoped to corrective segments only** (§17.3). The v0.6 rule cloned the model harder on the bad detour plus the correction — directly contra Rewarding Progress. v0.7 restricts the 2× multiplier to turns *after* the recovery pivot and with `state_delta_score > 0`, i.e. the correction, not the valley. (b) **`task-invalid` families get an explicit invalidation protocol** (§17.1). Flagging alone was governance-theater. v0.7 specifies: future training sets exclude data from the family since flag time; prior checkpoints marked non-comparable across the flag boundary (leaderboard versioned); the family cannot serve as a preference anchor once flagged; data is quarantined (not deleted) in case re-validation succeeds. (c) **Guidance-rollout provenance contract** (§17.4). v0.6 described "teacher trace injected into context" without locking down the source. v0.7 enumerates allowed provenance classes (prior autonomous successes — free; current-policy search — free; stronger-model trace — restricted + flagged; oracle trace — forbidden for training; benchmark-author trace — forbidden), adds a mandatory `guidance_source` field to every RL rollout, and routes guidance-conditioned rollouts through a separate accounting path that cannot mix with autonomous trajectories in SFT/preference without explicit policy. (d) **RAWR miner extended** (§17.12). The v0.6 criterion `P ≥ 0.8 AND G ≤ 0.4` catches "obvious bad process," missing the worse class: high $P$ with high but *shortcut-shaped* $G$ on a family under §17.6 $G$-watch. v0.7 adds a second criterion that fires on `P ≥ 0.8 AND (family under G-watch OR required-capability-tags under-exercised)`. (e) **§3.1 canonical schema updated** to reflect §17 additions. `raw_observation_blob`, `source_family_status`, `guidance_source`, and `canonicalized_action_form` are now in the writer contract, not just prose in §17. |
| v0.6 | Red-team pass folded in as a new **§17 Post-Review Hardening** (13 items). Three substantive pushbacks kept: (a) SFT view stays success-only — recovery learning comes from a new "recovery-trace" sub-view and from preference/RL, not from SFT-on-failures (§17.3); (b) LLD-11's conceptual anchor is reframed as Agent-RLVR + Rewarding Progress + guidance, with DAPO as token-level optimizer only — not as the conceptual basis (§4.6, §17.4); (c) $P_{\text{benchmark}}$ / $M_{\text{training}}$ divergence is deliberate and kept, but now both bands must calibrate (§17.7). Substantive additions: monthly ABC-style validity audit loop (§17.1), byte-wise raw-evidence retention (§17.2), family-specific capability sub-tags (§17.5), $G$-gaming diagnostic doctrine (§17.6), variance-triggered seed escalation to 4 / 8 (§17.8), moving-frontier curriculum refresh per training round (§17.9), per-family saturation / renewal plan as new Layer-B item 14 (§17.10), harness-canonicalization + cross-harness held-out eval (§17.11), multi-mode RAWR verification rows (§17.12), explicit precedence "honesty > learnability" (§17.13). §15.4 item 10 tightened to require dual-band calibration; §15.4 item 14 added for saturation plan. Citations added in §12. |
| v0.5 | §15 rewritten as a **general authoring checklist applicable to all 55 families**. Each rule is stated in family-agnostic form first; `proposal-ranking-manager-judgment` is retained only as the canonical worked example inside labeled callouts. No substantive rules changed from v0.4 — the two-layer DONE model, 13-item Layer B checklist, verification-matrix requirement, Decision A (LLM-judge quarantine) and Decision B (JSON-deliverable state-delta) precedents, and blocking-vs-non-blocking table all carry over. What changed: the framing, the structure, and the authoring-handoff checklist (§15.8) — now any family can fill it in. |
| v0.4 | Added §15 **Family-Level DONE Criteria** (initial draft, structured around `proposal-ranking-manager-judgment`). The CNB-55 §10.1 freeze gate (oracle ≥ 90, empty = 0, red-team ceilings, deterministic scorer, probe-in-range) remains necessary but is no longer sufficient. A family is DONE for the flywheel when it additionally: (a) declares its milestones into the §7.5 5-slot template with M3 anti-shortcut present and non-trivial; (b) registers its capability-tag declarations (required / recommended / forbidden) per §8.3; (c) supplies an integrity-flag detector and a state-delta rule; (d) quarantines LLM-judge signals out of $P$ / $M$ per the §7.8 escape-valve rule; (e) produces a calibration matrix (oracle / empty / right-no-grounding / adversarial) × ($P$, $M$, $G$, $R$, $S_{\text{TTC}}$) that demonstrates signal separation; (f) supports ≥2 deterministic seeds so LLD-06 can emit preference pairs per §9.4. Old §15 renumbered to §16. |
| v0.3 | Added §8 **Middle-Step (Turn-Level) Grading**. Defines a three-layer turn-level rubric (harness compliance ~15%, intent / evidence ~25%, state progress ~60%), a stable capability-tag vocabulary (`localize`, `inspect`, `modify`, `verify`, `respect_invariants`) that each family maps raw tool calls into, a per-turn scoring schema (`raw_tool_call`, `capability_tags`, `evidence_score`, `state_delta_score`, `integrity_flag`, `compliance_flag`), and a trajectory-level aggregate $G \in [0,1]$. $G$ is **shaping signal only** — it is a tie-break term in $S_{\text{TTC}}$, an optional per-turn component in $R$ (ReVeal-style), and a preference margin refinement; it never enters benchmark truth. Grounded in Agent-RLVR, ReVeal (TAPO), Rewarding Progress, and AutoTool. Also: (a) canonical event schema in §3.1 now carries `turn_scoring[]`; (b) preference lex ordering in §7.3 inserts $G$ as a tie-breaker before "drop pair"; (c) §R and $S_{\text{TTC}}$ in §7 updated to reference the shaping channel; (d) §15 lock-downs extended. |
| v0.2 | Folded in: concrete loss functions (masked NLL / DPO / DAPO with formulas); one worked Codex JSONL example showing how a single run produces three different labels; milestone rubric (shared 5-slot template with weights 0.10/0.20/0.20/0.20/0.30, per-scenario-type adaptation, 3-level implementation ladder); scoring-model separation ($S_{\text{TTC}}$ for search vs $R$ for learning); adaptive TTC policy; the "20 solved is bootstrap, not saturation" decision. |
| v0.1 | Initial framing. Established the "one raw event store, three derived training views, one evaluation boundary" architecture; locked the no-human-labels stance and the SFT → auto-preference → RLVR training sequence. |

---

## 1. Purpose

LLD-06 through LLD-11 together own every step between *"the agent finished a run"* and *"we have a trained model to evaluate."* They are six adjacent documents, but they are not six independent designs. They are one flywheel. This doc pins the architectural decisions that the six downstream LLDs must be coherent with, so each LLD can be written without re-litigating the cross-cutting choices.

**The six components in scope:**

| LLD | Role in the flywheel |
|---|---|
| LLD-06 Trajectory Parser | Normalizes raw agent runs into the canonical event store; emits the three derived training views and the per-turn middle-step scores |
| LLD-07 Benchmark Runner | Drives the rollout / collection campaigns that feed the event store, and drives the evaluation campaigns that consume trained checkpoints |
| LLD-08 Logprob Capture Proxy | Gate-5-contingent shim that makes Phase 2b (RL) viable by exposing per-token logprobs on the Responses API path |
| LLD-09 mini-SWE-Agent | Second harness used both as a *data source* (SWE-Agent-SFT-matched) and as an *evaluation harness* (B1 reference arm, B2 diagnostic arm) |
| LLD-10 SFT Training Pipeline | QLoRA fine-tuning on the SFT view; produces the B1 2×2 arms and the B2 headline model |
| LLD-11 DAPO RL Pipeline | Stretch-only, pre-killed until Gate 5 passes; consumes the RL prompt pool and the verifier reward |

The rest of this document is the architectural contract that holds across all six.

---

## 2. Core Design Decisions

Four decisions drive every downstream choice. Each is load-bearing and should be revisited only if the underlying assumption changes.

### Decision 1 — One canonical event store; three derived training views

Do **not** split raw runs at collection time into "this is for SFT, this is for DPO, this is for RL." Store every run once, in a canonical form, and derive method-specific views as projections.

### Decision 2 — Train sequentially: SFT → auto-preference → RLVR

SFT first (behavior cloning from successes), auto-preference second (DPO on programmatically-constructed pairs), policy-gradient RL third (only after Gate 5 and SFT signal). Phase 2b is pre-killed in all planning.

### Decision 3 — No human labeling; verifier-first flywheel

The grader is the label. Labeling budget goes into verifier hardening instead. The risk we accept is reward hacking; the mitigation is grader quality + adversarial audits, not per-trajectory review.

### Decision 4 — Successful-trace SFT is a bootstrap, not a saturation point

If the base model is 20 / 100 on the benchmark, the 20 successes teach "how a win looks" but not "how to search, recover, or choose between near-misses." The failed 80 are the main engine for preference, RL, and test-time reranking. Metadata fidelity on failures must equal that on successes.

---

## 3. The Flywheel Architecture

```text
                 ┌──────────────────────────────────────────┐
                 │           RAW EVENT STORE                │
                 │  (canonical, one row per run,            │
                 │   lossless wrt tokens/tools/reward,      │
                 │   includes per-turn middle-step scores)  │
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
 │ (per-turn)    │     │ view           │     │ (tasks + grader  │
 │               │     │ (pairwise)     │     │  ref + reward    │
 │ successful    │     │ same-task      │     │  config)         │
 │ Codex traces  │     │ attempts,      │     │ Train-Long only, │
 │ + matched-ID  │     │ lex-ordered    │     │ 20–80% solve     │
 │ SWE-Agent     │     │ on P, M, H,    │     │ band             │
 │ variants      │     │ turns, G       │     │                  │
 └───────┬───────┘     └────────┬───────┘     └─────────┬────────┘
         │                      │                       │
         ▼                      ▼                       ▼
  ┌──────────────┐      ┌───────────────┐      ┌─────────────────┐
  │ LLD-10 SFT   │      │ DPO (Phase    │      │ LLD-11 DAPO RL  │
  │ masked NLL   │      │ 2a.5 add-on)  │      │ token-level     │
  │              │      │ — see §6.2    │      │ clipped PG      │
  │              │      │               │      │ — see §6.3      │
  │              │      │               │      │ + §8 shaping    │
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

```yaml
# Canonical run record (LLD-06 writer contract) — v0.7
task_id:              # scenario ID (Codex-Long) or instance ID (SWE-bench)
split:                # dev_bench | bench_control | train_long | val_long | test_long | final_test | public_dev
family_id:            # Codex-Long family or null for SWE-bench
family_status_at_write_time:
                      # enum: ok | watch | task_invalid | retired_saturated
                      # Snapshotted at write-time from §17.1 validity audit log.
                      # `task_invalid` at write-time → excluded from §17.1 post-flag training sets;
                      # retained at-rest for re-validation.
harness:              # codex | swe_agent
canonicalized_action_form:
                      # canonical string form of every tool call, normalized per §17.11.
                      # SFT view reads from this; raw harness form remains in raw_observation_blob.
model_id:             # resolves against model_registry.yaml
seed:                 # integer
prompt_state_0:       # initial prompt + repo snapshot hash
trajectory_jsonl_ref: # pointer to full event stream
raw_observation_blob: # byte-range reference into append-only raw log per §17.2.
                      # Stable; never rewritten. All derived fields carry `derived_from` pointers here.
turn_records:         # [{turn_idx, state_text, assistant_tokens, tool_calls, tool_outputs,
                      #   derived_from: <raw_observation_blob range>}]
turn_scoring:         # [{turn_idx, capability_tags, evidence_score, state_delta_score,
                      #   integrity_flag, compliance_flag,
                      #   corrective_turn_bool,    # §17.3: true iff this turn is part of the
                      #                            # corrective suffix, i.e. after the recovery pivot
                      #                            # AND state_delta_score > 0
                      #   derived_from: <raw_observation_blob range>}] — see §8
G:                    # trajectory-level aggregate from §8.4, in [0,1]
recovery_trace:       # §17.3: bool; true iff G dropped ≥ 0.3 mid-trace and trajectory recovered to success
guidance_source:      # §17.4: enum for RL rollouts only.
                      # one of: autonomous_prior_success | autonomous_current_policy_search |
                      #         stronger_model_flagged | oracle_trace_forbidden | authoring_only_debug
                      # Non-null only on RL-pool rollouts. Autonomous trajectories set this to null.
                      # Rows with `stronger_model_flagged` flow through the §17.4 separate accounting path.
                      # `oracle_trace_forbidden` and `authoring_only_debug` MUST NOT appear in training splits.
final_artifact_ref:   # patch (SWE-bench) or state digest (Codex-Long)
outcome:              # pass | fail | timeout | crash | no_patch
cl_pass:              # Codex-Long verifier bool (P in §7), null for SWE-bench
P_benchmark:          # §17.7: full 100-pt normalized total incl. LLM-judge pts
M_training:           # §17.7: symbolic-only normalized to [0,1]; read by LLD-06 into training views
milestone_vector:     # [{milestone_id, passed_bool}] — m_i in §7
failure_mode:         # enum from LLD-05
wall_time_seconds:    # float
token_count:          # {prompt, completion, total}
tool_call_count:      # int
crash_flags:          # list; C in §7 is derived bool
timeout_flag:         # T in §7
shortcut_flags:       # forbidden-edit, test-tamper, mock-bypass; H in §7
loopiness_flags:      # malformed tools, repeated no-op turns; L in §7
grading_manifest_ver:
schema_version:       # integer; additive-only per §17.2
```

LLD-06 does not re-grade — it reads LLD-05's normalized `outcome` / `cl_pass` as the source of truth for pass/fail, reads LLD-13's milestone manifest for `milestone_vector`, and *computes* `turn_scoring[]`, $G$, `recovery_trace`, and `corrective_turn_bool` from the raw trajectory post-hoc (agent never sees these during the run). `family_status_at_write_time` and `guidance_source` are snapshotted from LLD-13 (validity audit log) and LLD-07 (orchestrator provenance), respectively. `schema_version` bumps are **additive-only** — pre-$X$ raw must always be replayable into v$X$ derived views per §17.2.

### 3.2 Derived view A — SFT

Projection filters:

- `outcome == pass` (SWE-bench) or `cl_pass == true` (Codex-Long)
- `split in {bench_control, train_long}`
- `shortcut_flags == []`

Emission variants: Codex-SFT-all, Codex-SFT-matched, SWE-Agent-SFT-matched, SWE-Bench-Control-SFT (appendix only). Row shape is turn-level with assistant-token-only masking.

### 3.3 Derived view B — Auto-preference

Grouped over `(task_id, family_id)` with ≥2 attempts; ordered by the lex rule in §7.3 which now incorporates $G$. Pairs that remain tied after all criteria are dropped.

### 3.4 Derived view C — RL prompt pool

Prompt catalog filtered to Train-Long with `difficulty_bucket == in_band` and `allowed_for_online_rl == true`. Reward config references §7.2 and, optionally, the per-turn shaping channel in §8.5.

### 3.5 The evaluation boundary

Training views only project over `{bench_control, train_long}`. Test-Long and Final-Test are sealed. LLD-02 enforces; LLD-06 honors; LLD-10 and LLD-11 trust.

---

## 4. Component Roles

### 4.1 LLD-06 Trajectory Parser — the writer

- Normalizes raw Codex JSONL (LLD-03) and SWE-Agent traces (LLD-09) into canonical records.
- Computes `turn_scoring[]` and $G$ per §8 (capability-tag mapping, per-turn deltas).
- Emits all three derived views.
- Preserves metadata on failed runs at the same fidelity as successes.

### 4.2 LLD-07 Benchmark Runner — the driver

- Collection campaigns populate the event store (≥2 seeds on Train-Long Codex per §9.4).
- Evaluation campaigns consume trained checkpoints.
- Adaptive TTC orchestration per §9.2 using $S_{\text{TTC}}$ (which includes $G$ as tie-break).
- Re-buckets RL difficulty after each SFT checkpoint.

### 4.3 LLD-08 Logprob Capture Proxy — the enabler

Gate-5-contingent. Sprint 0 spike → full proxy Sprint 1 iff Gate 5 passes.

### 4.4 LLD-09 mini-SWE-Agent — the second harness

Both a data producer (SWE-Agent-SFT-matched) and an evaluation harness (B1, B2 diagnostics). Its outputs flow through LLD-06 identically to Codex outputs; `harness` is a column in the canonical store.

### 4.5 LLD-10 SFT Training Pipeline — the guaranteed path

Consumes the SFT view; emits four adapters. Codex-SFT-all is the hinge checkpoint (B2 headline, LLD-11 warm start, RL-difficulty-bucketing source).

### 4.6 LLD-11 Agent-RLVR RL Pipeline (DAPO-optimized) — the stretch branch

Consumes the RL prompt pool; reward from §7.2 with optional per-turn shaping from §8.5. Warm-started from Codex-SFT-all; rollouts via LLD-03 + LLD-08.

**Conceptual anchor (not the optimizer).** The framing for LLD-11 is **Agent-RLVR** (verifier-first rewards) + **Rewarding Progress** (per-turn shaping, used carefully per §8) + **guidance rollouts** on hard tasks (teacher-traced context injection for tasks in the <20% band, rather than exclusion). DAPO is the **token-level policy-gradient algorithm** that runs inside this framing — chosen for Clip-Higher stability on long-tail token losses — but DAPO is not the intellectual basis of LLD-11. It is the optimizer. See §17.4 for the full reframe and citations; papers listed in §12.

---

## 5. Worked Example — One Codex JSONL Run Produces Three Labels

Scenario: `alert-dedupe-investigation/inventory-oncall`. Inspect `logs/failure.log`, contract doc, and failing tests, then repair parser + dedupe behavior without deleting tests.

**Raw trajectory (illustrative):**

```json
{"type":"run_metadata","task_id":"alert-dedupe-investigation/inventory-oncall","codex_flags":"--yolo --json"}
{"type":"response.created"}
{"type":"response.output_text.delta","delta":{"text":"I'll inspect the failing log and tests first, then patch the parser and dedupe flow."}}
{"type":"tool_call","function":{"name":"shell","arguments":"cd /workspace && sed -n '1,200p' logs/failure.log && pytest -q tests/test_contract.py tests/test_integration.py"}}
{"type":"tool_result","output":"AssertionError: expected inventory_scope to remain stable ..."}
{"type":"response.output_text.delta","delta":{"text":"The regression is in alias normalization and dedupe key selection. I'll patch parser.py and dedupe.py, then rerun tests."}}
{"type":"tool_call","function":{"name":"shell","arguments":"python - <<'PY'\n# edit files\nPY\npytest -q tests/test_contract.py tests/test_integration.py"}}
{"type":"tool_result","output":"2 passed"}
{"type":"response.output_text.delta","delta":{"text":"Done. I normalized environment aliases to prod, bucketed observed_at by minute, stabilized inventory_scope, preserved occurrence_count/first_seen_at/last_seen_at, and preferred dedupe_hint while keeping the earliest display title human-readable."}}
```

**Verifier outputs:** $P = 1$, milestones $[m_1..m_5] = [T,T,T,T,T]$.

### 5.1 SFT rows

Three rows, one per assistant turn, assistant-token-mask only. Admitted to the SFT view because $P = 1$ and $H = 0$.

### 5.2 Preference pair

Only materializes when paired with another attempt on the same task. If a second attempt achieved $P=0$, $[T,F,F,F,F]$, LLD-06 emits a pair with `preference_reason_code = OUTCOME_RANK`.

### 5.3 RL pool entry

The *task* joins the pool; the policy generates fresh rollouts scored by §7.2.

### 5.4 Middle-step scores (new in v0.3)

The same trajectory produces a `turn_scoring[]` — see the worked capability-tag example in §8.7.

---

## 6. Loss Functions

### 6.1 SFT — masked next-token NLL

$$
L_{\text{SFT}} = - \sum_t m_t \log \pi_\theta(y_t \mid x, y_{<t})
$$

`m_t = 1` on assistant tokens, `0` elsewhere. QLoRA on Qwen3.5-27B per LLD-10.

### 6.2 Preference — DPO

$$
L_{\text{DPO}} = -\log \sigma \Big( \beta \big[ \log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) - \log \pi_{\text{ref}}(y^+|x) + \log \pi_{\text{ref}}(y^-|x) \big] \Big)
$$

Policy init: best Codex-SFT-all checkpoint. Reference: frozen SFT-all by default.

### 6.3 RL — DAPO token-level clipped PG

$$
r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

$$
L_{\text{RL}} = - \sum_t m_t \min \big( r_t \hat{A},\ \text{clip\_higher}(r_t) \hat{A} \big)
$$

`m_t` masks to assistant tokens. $\hat{A}$ derived from §7.2 reward, optionally with §8.5 per-turn shaping. Clip-Higher asymmetric per HLD v2.3. No KL penalty.

---

## 7. Milestone Rubric And Scoring Model

### 7.1 Variables

- $P \in \{0,1\}$: final verifier pass. **Benchmark truth.** Sole authority: `verify.sh` from LLD-13.
- $m_i \in \{0,1\}$: milestone $i$ boolean.
- $w_i$: milestone weight, $\sum_i w_i = 1.0$.
- $M = \sum_i w_i m_i \in [0,1]$: milestone progress score.
- $C, T, H, L \in \{0,1\}$: crash, timeout, shortcut/integrity-violation, loopiness/malformed-tool-use flags.
- $G \in [0,1]$: **trajectory middle-step score** (defined in §8). Shaping signal only, never benchmark truth.

Milestones and penalty flags are post-run helpers, invisible to the agent, injected into a separate container per LLD-13's integrity protocol.

### 7.2 Reward for RL learning (`R`)

$$
R = \begin{cases}
1.0 - 0.02 L, & P = 1 \\
0.6 M - 0.2 C - 0.1 T - 0.6 H - 0.02 L, & P = 0
\end{cases}
$$

Any solved run ≥ 0.98; any unsolved ≤ 0.6. Gap is intentional and non-negotiable. Optional per-turn shaping from §8.5 may be *added* on top — strictly ≤ 0.2 total magnitude — never used as a replacement.

### 7.3 Ordering for preference pairs (lexicographic, v0.3 adds $G$)

1. Higher $P$.
2. If tied on $P$: higher $M$.
3. If tied on $M$: fewer shortcut / crash / timeout flags.
4. If tied: fewer loopiness / malformed-tool flags and fewer turns.
5. **If still tied: higher $G$** (middle-step quality: better tool-intent and state-progress).
6. If still tied: **drop the pair**.

No human tie-breaks in v1.

### 7.4 Score for test-time compute (`S_TTC`)

$$
S_{\text{TTC}} = 1000 P + 100 M - 40 C - 20 T - 100 H - 5 L + 10 G
$$

The $10 G$ term is the v0.3 addition: when candidates tie on $P$ and $M$, $G$ resolves which one to surface. Cap ensures $G$ never outranks $P$ or dominates $M$.

### 7.5 Shared milestone template — 5 slots across all 55 families

| Slot | Name | Weight | What it checks |
|---|---|---|---|
| M1 | Localization | 0.10 | Correct files / subsystem / logs / tests identified |
| M2 | Primary fix | 0.20 | Main state change implemented |
| M3 | **Invariants preserved / anti-shortcut** | 0.20 | Tests not deleted, forbidden files untouched, no mock-bypass. **Mandatory.** |
| M4 | Functional checks | 0.20 | Visible tests / build / lint / contract checks pass |
| M5 | End-to-end integration state | 0.30 | Output shape, contract stability, cross-layer consistency |

### 7.6 Per-scenario-type reinterpretation

| Type | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| Feature evolution | Interfaces touched | Feature logic added | Old behavior preserved | Tests/docs pass | E2E feature correct |
| Migration / refactor | Dep surface identified | Call sites migrated | No stale API | Tests/build pass | Integration semantics preserved |
| Build / CI breakage | Root cause localized | Config / tooling fixed | No bypass | Build passes | Repo stable and runnable |
| Investigate-then-fix | Failure localized | Root-cause patch | Invariant preserved | Regression tests pass | Full behavior stabilized |
| Cross-layer changes | All surfaces touched | Primary contract change | Cross-layer invariants aligned | Per-layer checks pass | Full integration state correct |

### 7.7 Milestone-authoring hard rules

1. Milestones are boolean. No fuzzy 0–5 scores.
2. Later milestones should be worth ≥ earlier ones.
3. If later milestone logically depends on earlier one, gate it.
4. `verify.sh` is the **sole authority** for $P$.
5. $H = 1$ zeros out or heavily penalizes M3/M4/M5 regardless of surface appearance.
6. Milestones and penalty flags are never visible to the agent during execution.

### 7.8 The 3-level milestone implementation ladder

| Level | Tool | Good for | Examples |
|---|---|---|---|
| L1 | Regex / grep / JSON-field checks | Local file-state invariants | Dep version, forbidden string, config key |
| L2 | Structured analysis | Syntax / semantic surface | AST, import graph, schema validation, lint / typecheck, checksum anti-shortcut |
| L3 | Executable / learned | Real behavioral progress | Functional commands; targeted integration scripts; self-generated tests; reward model for residual quality (shaping only) |

**Escape-valve rule:** if a milestone requires an LLM judge to define, it does not belong in the symbolic milestone layer. Move it to reranking or a reward model.

---

## 8. Middle-Step (Turn-Level) Grading

### 8.1 Why turn-level, and why not exact tool-call matching

Post-run milestones (§7) give us a dense-ish summary of "did the final state reach these properties?" They say nothing about the *path*. RL and TTC both benefit from per-turn signal, and the literature has moved decisively in that direction:

- **Agent-RLVR** shows plain RLVR is too sparse in agentic environments; it adds agent guidance, dynamic error feedback, and environment interactions, with unit tests validating trajectories.
- **ReVeal** turns code generation into iterative generation-verification turns with explicit turn-level credit assignment (TAPO) — this is literally "grade the middle of the trajectory."
- **Rewarding Progress** gives the cleanest theoretical rule: a good process reward measures whether a step increases the chance of future correctness, not whether the step *looks* correct in isolation.
- **AutoTool** is the closest public example of explicit multi-step tool-selection supervision via structured rationales.

The design discipline the strongest of these papers converge on: **grade progress through tool use, not exact tool choreography.** Hardcoding one CLI sequence across 55 families is brittle, punishes creative-but-correct search, overfits to Codex's current habits, and turns the benchmark into a workflow-imitation task. We explicitly reject that direction.

### 8.2 Three layers

Per turn, score across three layers. Weights reflect what actually predicts future success:

| Layer | Weight | What it grades | Signal source |
|---|---|---|---|
| Harness compliance | **15%** ($w_H$) | Tool-call schema valid, allowed tool only, arguments parse, no malformed repetition, no premature finalization | Schema checks on raw Codex JSONL events |
| Intent / evidence | **25%** ($w_I$) | Did the tool use reflect the right subgoal? (inspected failing log before editing, read dep manifest before migrating, ran visible check before finalizing) | Capability-tag mapping + family-declared evidence rules |
| State progress | **60%** ($w_S$) | Did the turn move the repo / test state closer to pass? | State snapshots before/after turn, per-turn milestone-vector delta |

Weights sum to 1.0 and are the defaults; a family may deviate within ±0.10 on any single layer with a Change Summary entry.

### 8.3 Capability tags — the intent vocabulary

One small stable vocabulary across all 55 families. Raw tool calls map to tags by rule (regex + structured argument inspection); no LLM judge in v1.

| Tag | What counts |
|---|---|
| `localize` | Reading logs, failing tests, or relevant source files to find the bug surface |
| `inspect` | Reading implementation files, dependency manifests, or configuration before changing them |
| `modify` | Edits applied to source / config / deps |
| `verify` | Running visible tests, build, lint, or contract checks |
| `respect_invariants` | Absence of test-tampering, forbidden-file touches, mock-bypasses |

Per family, LLD-13 declares:

- **Required capabilities:** at least one turn must carry this tag (e.g. investigate-then-fix requires `localize`).
- **Recommended capabilities:** positive evidence if present (e.g. `verify` before finalizing).
- **Forbidden capabilities:** any occurrence raises `integrity_flag` (e.g. editing `tests/` files for a no-test-modification family).

A single turn can carry multiple tags (a `shell` call that both reads logs and runs tests is `[localize, verify]`).

### 8.4 Per-turn scoring schema and trajectory aggregate

Per assistant turn, LLD-06 emits:

```yaml
turn_scoring:
  turn_index:
  raw_tool_call:        # verbatim function_call from JSONL (or null for pure-text turns)
  capability_tags:      # list from §8.3
  evidence_score:       # [0,1] — did the turn's tags match declared requirements at this point in the trajectory?
  state_delta_score:    # [0,1] — signed delta in visible state (test pass rate, milestone-vector partial, file-diff sanity)
  integrity_flag:       # bool — raises H for the run if true
  compliance_flag:      # bool — schema valid, allowed tool, args parse, no malformed repetition
```

Trajectory-level aggregate:

$$
G = w_H \cdot \overline{\text{compliance}} + w_I \cdot \overline{\text{evidence}} + w_S \cdot \overline{\text{state\_delta}}
$$

Defaults: $w_H = 0.15$, $w_I = 0.25$, $w_S = 0.60$.

### 8.5 How $G$ feeds the flywheel

$G$ is **shaping signal only**, never benchmark truth:

- **$S_{\text{TTC}}$ tie-break** (§7.4): the $+10 G$ term lets the reranker prefer a higher-quality path when two candidates tie on $P$ and $M$.
- **Preference margin refinement** (§7.3 step 5): $G$ resolves otherwise-tied pairs before they would be dropped.
- **Optional per-turn RL shaping (ReVeal-style):** LLD-11 may add a per-turn reward term proportional to `state_delta_score` on the turn, capped so the aggregate shaping contribution over a trajectory never exceeds 0.2 in magnitude. Binary $P$ and milestone $M$ remain the dominant signal.

$G$ does **not**:

- enter `verify.sh` or determine $P$
- replace $M$
- become a training target on its own

### 8.6 What NOT to do

Explicitly forbidden even when tempting:

1. **No single-path tool-call matching.** A family never declares "the correct solve sequence is A → B → C." Many valid paths exist; rewarding imitation punishes competence.
2. **No LLM judge in v1 middle-step grading.** If a family's evidence rule is so semantic it would need one, apply §7.8's escape-valve rule — downgrade the signal to a reward-model input for Phase 2b, don't inject it into the symbolic turn-scoring layer.
3. **No $G$-dominance.** $G$ is capped at 0.2 total magnitude in $R$ and at $+10$ absolute in $S_{\text{TTC}}$. A trajectory with high $G$ and $P=0$ is still a failure.

### 8.7 Worked example — capability-tag mapping

Continuing §5's `alert-dedupe-investigation/inventory-oncall` successful run. LLD-13 declares this family requires `localize`, recommends `verify` (pre- and post-edit), and forbids touching `tests/` (violation → `integrity_flag`).

| Turn | Raw tool call | Tags | Evidence | Compliance | State delta |
|---|---|---|---|---|---|
| 1 | `sed -n '1,200p' logs/failure.log && pytest ...` | `[localize, verify]` | 1.0 (localize required met; verify recommended met) | 1 | 0.10 (tests confirmed failing; no edit yet) |
| 2 | `python ... edit ...; pytest ...` | `[modify, verify]` | 1.0 (modify required; verify recommended met post-edit) | 1 | 0.90 (2 passed after edit) |
| 3 | (final text only, no tool call) | `[]` | — | 1 | 0 |

Aggregate:

$$
G = 0.15 \cdot 1.00 + 0.25 \cdot 1.00 + 0.60 \cdot 0.50 = 0.70
$$

A weaker run that edited before inspecting would score evidence ≈ 0.3 at turn 2 (modify-without-localize) and $G$ would differentiate it from this run *automatically, with no humans*, even if both eventually passed.

---

## 9. Test-Time Compute Policy

### 9.1 Why TTC matters for the failed 80

If the base model is 20 / 100 at pass@1, many of the other 80 are solvable with another attempt, deeper search, or better candidate ranking. Precedent: Agent-RLVR lifts 22.4% → 27.8% Pass@1 via reward-model reranking over 32 patches at test time; o1 reported consensus-over-64 and reranker-over-1000 gains on AIME.

### 9.2 Adaptive budget policy

Adaptive, not flat-N:

1. Run **8 attempts** initially.
2. Score each with $S_{\text{TTC}}$ from §7.4 (includes $G$).
3. If any attempt has $P = 1$: stop; take highest-$S_{\text{TTC}}$ solved attempt.
4. If no solve but best $M \geq 0.70$: expand to **16**, then **32**.
5. If all attempts have $M < 0.20$: stop early.
6. If best $M$ is high but $m_3 = 0$ (anti-shortcut invariant broken): do *not* expand aggressively — likely a reward-hack basin.
7. Optional: second-stage selector / reward model on top 2–4 candidates.

### 9.3 $S_{\text{TTC}}$ and $R$ are different functions

| Function | Purpose | Lives in |
|---|---|---|
| $S_{\text{TTC}}$ | Rank candidates at inference / collection time | LLD-07 (best-of-N), LLD-06 (preference margins) |
| $R$ | Update model parameters | LLD-11 |

The $1000 \cdot P$ term in $S_{\text{TTC}}$ is fine for ranking; it would dominate gradient updates. LLD-11 reward must not reuse $S_{\text{TTC}}$ as-is.

### 9.4 Collection-side implication: ≥2 seeds on Train-Long Codex

TTC and auto-preference both depend on multiple attempts per task. LLD-07 must plan ≥2 seeds on Train-Long Codex; 3 seeds on top Gate-4 families if budget allows.

---

## 10. Training Sequence

| Phase | Trigger | What trains | Data view | Loss | Reward / target | Gating |
|---|---|---|---|---|---|---|
| 2a.0 | After Gate 4 + Train-Long collection complete | LLD-10 Codex-SFT-all + matched arms + Bench-Control appendix | SFT view | $L_{\text{SFT}}$ | Assistant tokens | Hard-required for B2 |
| 2a.5 | After Phase 2a shows signal, optional | DPO on LLD-06 preference view (off-policy) | Preference view | $L_{\text{DPO}}$ | Auto pairs per §7.3 (incl. $G$) | Soft |
| 2b | After Gate 5 + Phase 2a signal | LLD-11 DAPO RL on Train-Long prompts | RL prompt pool | $L_{\text{RL}}$ | $R$ (§7.2) + optional §8.5 shaping | Pre-killed; never assumed |

Rule: no phase starts before the previous phase produces a checkpoint that beats the previous best on Val-Long by a bootstrap-CI-separated margin.

---

## 11. No-Human-Labels Stance

### What we buy

Speed, coverage, reproducibility, SOTA alignment (SWE-smith, Agent-RLVR, ReVeal, SSR, Agent Lightning, SWE-RM, AutoTool).

### What we pay

Reward-hack risk; quality-vs-outcome conflation; narrow optimization target.

### How we pay it

1. Verifier hardening as a first-class investment (hidden tests, $H$ flag detection, verifier isolation, family-disjoint evaluation, LLD-13 integrity protocol).
2. Adversarial grader audit before Sprint 3.
3. M3 mandatory and $H=1$ zeros out later milestones.
4. Middle-step grading (§8) keeps exact tool-call mimicry from becoming the shortcut: the agent can't game $G$ by imitating one scripted path because no such path exists.
5. v1.5 human pairwise review as contingency only.

---

## 12. Research Grounding

| Paper / system | Pattern we inherit |
|---|---|
| SWE-smith | Automatic task generation + execution-backed filtering as the data engine |
| Agent-RLVR | Environment-derived reward lifts SWE-bench Verified 9% → 22%; reward-model reranking adds another 5 pts |
| ReVeal | Turn-level credit assignment (TAPO) — the template for §8.5 per-turn shaping |
| Rewarding Progress | The theoretical rule behind §8: grade whether a step raises future success probability, not whether it *looks* correct |
| AutoTool | Explicit multi-step tool-selection supervision via structured rationales — template for the capability-tag vocabulary in §8.3 |
| Self-play SWE-RL (SSR) | Most aggressive test-free version; fallback if the task pool gets exhausted |
| Agent Lightning | Decouple execution from training; transition-level records — the §3 architecture |
| SWE-RM | Learned reward model from execution outcomes as a densifier for the L3 escape valve in §7.8 and the capped shaping in §8.5 |
| ABC — Agentic Benchmark Checklist (2507.02825) | Task-validity flaws materially shift measured performance; motivates §17.1 monthly validity audit loop and §17.12 systematic RAWR generators |
| Terminal-Bench | Benchmarks saturate; versioned renewal is required — motivates Family-Test-Requirements §4 item 14 and §17.10 per-family saturation plan |
| SWE-Gym (2412.21139) | Failure-trajectory signal is essential; success-only SFT plateaus — motivates §17.3 recovery-trace SFT view and the preference/RL-on-failures design |
| Training Long-Context SWE Agents with RL (2508.03501) | Long-horizon agent RL needs retry structure and environment-aware densification, not just a stronger optimizer — reinforces §17.4 Agent-RLVR reframe |
| DAPO (2503.14476) | Token-level clipped PG with Clip-Higher — optimizer choice inside LLD-11, not the conceptual anchor |
| Multi-SWE-bench (2504.02605) | Cross-harness / cross-language generalization as a leakage check — motivates §17.11 cross-harness held-out eval |

---

## 13. Risks And Open Questions

Risks:

- **Verifier gaming.** Mitigated by §11. LLD-13 owns integrity; LLD-10/11 honor `shortcut_flags`.
- **Preference-pair sparsity.** Enforced by §9.4 — ≥2 seeds on Train-Long Codex.
- **Difficulty-bucket miscalibration.** LLD-07 re-buckets after each SFT checkpoint.
- **Shortcut detection coverage.** $H$ flag is the single most important anti-reward-hack signal. LLD-13 owns detection; LLD-06 owns the filter.
- **Harness-specific trace leakage.** B1 2×2 depends on matched-ID splits being clean on Codex-Long scenario IDs and family structure.
- **Capability-tag mapping drift.** If LLD-13 families evolve independently, tag definitions in §8.3 can drift. Mapping rules live in a single registry owned by LLD-06, not per-family.
- **Middle-step over-weighting.** $G$ must never dominate $P$ or $M$ in $R$ or preference. Caps in §7.2, §7.4, §8.6.
- **LLM-judge creep.** Apply §7.8 escape-valve rule aggressively.

Open questions:

- **LLD-06:** turn-segmentation — one row per assistant turn vs per tool call. Default: per assistant turn.
- **LLD-06:** how to compute `state_delta_score` cheaply. Default recommendation: visible-test pass-rate delta + `milestone_vector` delta + file-diff sanity check.
- **LLD-07:** seed policy on Train-Long. Default: 2 seeds Codex, 1 seed SWE-Agent, 3 seeds on top Gate-4 families if budget allows.
- **LLD-08:** sidecar vs in-process. Gate-5-dependent.
- **LLD-09:** whether SWE-Agent-SFT-matched needs its own harness-control arm beyond B1 2×2. Default: 2×2 covers it.
- **LLD-10:** DPO arm as separate adapter or post-SFT continuation. Default: continuation. Reference policy: frozen SFT-all.
- **LLD-11:** whether to turn on §8.5 per-turn shaping from rollout 1, or only if sparse-reward plateau appears. Default: off at v1, on as a gated experiment.
- **LLD-13:** per-family capability-tag declarations (required / recommended / forbidden) — new authoring obligation introduced by §8.

---

## 14. Sequencing Across Sprints

```text
Sprint 0   → LLD-08 Gate 5 spike
Sprint 0b  → LLD-13 complete (55-family milestone rubric + capability-tag declarations)
Sprint 1   → LLD-06 design+impl      (canonical store + 3 views + §7 scoring + §8 turn-scoring)
           → LLD-07 design+impl      (collection + eval + adaptive TTC using S_TTC incl. G)
           → LLD-09 design+impl      (SWE-Agent harness)
           → LLD-08 full proxy impl  (IFF Gate 5 passed)
Sprint 2   → LLD-10 design           (SFT pipeline, losses per §6.1 + §6.2)
           → LLD-11 design           (IFF Gate 5 + Phase 2a signal; §6.3 + §7.2 + optional §8.5)
Sprint 3   → LLD-10 implement + train 4 SFT arms → (optional) DPO arm
           → LLD-11 implement + DAPO run (IFF still viable)
           → LLD-12 package results
```

---

## 15. Family-Level DONE Criteria — Pointer

The full family-level DONE specification — a general authoring checklist that applies to all 55 families — has been split out into its own standalone document as of v0.8:

**→ `HLD-Family-Test-Requirements-v0_1.md`** (sibling file in this `docs/` directory)

That document owns:

- §1 **Why this spec exists** — why the CNB-55 §10.1 freeze gate is necessary but no longer sufficient once the family enters the flywheel.
- §2 **Two layers of DONE** — Layer A (benchmark honesty under CNB-55) + Layer B (flywheel readiness under this HLD).
- §3 **Layer A** — the pre-existing CNB-55 freeze-gate obligations (oracle ≥ 90, empty = 0, red-team ceilings, deterministic scorer, probe-in-band, immutable-slice checksums).
- §4 **Layer B** — the 14-item flywheel readiness checklist (milestones, capability tags, tag overrides, state-delta rules, integrity flags, LLM-judge quarantine with dual-band calibration, ≥ 2 seeds, variance-triggered seed escalation, saturation / renewal plan).
- §5 **Required verification matrix** — the 6-trajectory × 5-metric grid every family must populate; multi-row RAWR requirement per §17.12.
- §6 **Precedent decisions** — Decision A (LLM-judge quarantine) and Decision B (state-delta via JSON-deliverable transitions), both first settled on `proposal-ranking-manager-judgment`.
- §7 **Blocking vs non-blocking** — which items gate admission to the RL prompt pool vs which are recommended.
- §8 **Per-family authoring handoff template** — the 9-box sign-off sheet each family owner submits when declaring "flywheel-ready."

**Two-layer DONE, in one sentence:** a family is DONE for the flywheel when it is both benchmark-honest under CNB-55 (Layer A) *and* flywheel-ready under this HLD (Layer B, owned by the split-out spec). Passing Layer A alone is necessary but no longer sufficient — it lets the family be probed, but not trained on.

**Cross-reference convention.** References elsewhere in this HLD that previously pointed into §15.X now point into the split-out document using the form `Family-Test-Requirements §N` (e.g. `Family-Test-Requirements §4 item 10` replaces the old `§15.4 item 10`). Every reference in §12 (research grounding), §16 (downstream-LLD lock-down), and §17 (post-review hardening) was updated in v0.8; no rules changed. Any future evolution to the checklist itself — new Layer-B items, new RAWR modes, tightened calibration — happens in the split-out doc and is versioned independently.

**Rationale for the split.** §15 had grown to roughly 230 lines, about 20% of this HLD, and its audience (family authors finalizing CNB-55 artifacts and running the `Family-Test-Requirements §5` verification matrix) is distinct from the audience of the rest of the HLD (LLD-06 → LLD-11 implementers). Splitting keeps each document focused, lets the family-test spec evolve on its own cadence, and gives the family-authoring team a single file to cite in their own `benchmark_run.md` entries.

---

## 16. What This Doc Locks Down For The Downstream LLDs

LLD-06 through LLD-11 drafts **must** be coherent with:

1. One canonical event store, written by LLD-06, read by the training LLDs via projection. No training LLD reads raw Codex JSONL directly.
2. Four SFT variants, one auto-preference view, one RL prompt pool — the three, and only three, derived training views.
3. No human labels in v1.
4. Training sequence is SFT → (optional auto-DPO) → RL, gated on signal.
5. Losses are §6.1 (masked NLL), §6.2 (DPO), §6.3 (DAPO).
6. Reward $R$ is §7.2; $S_{\text{TTC}}$ is §7.4; they are different functions.
7. Every family implements the 5-slot milestone template from §7.5 with weights 0.10 / 0.20 / 0.20 / 0.20 / 0.30, per-type reinterpretations from §7.6, authoring rules from §7.7.
8. Milestone implementation obeys the 3-level ladder in §7.8; semantic-only milestones go to reranking, not the benchmark's symbolic layer.
9. **Every turn is scored per §8.4** (`raw_tool_call`, `capability_tags`, `evidence_score`, `state_delta_score`, `integrity_flag`, `compliance_flag`), using the capability-tag vocabulary in §8.3 and the three-layer aggregate in §8.4. LLD-13 families declare required / recommended / forbidden capabilities.
10. **$G$ is shaping only.** It enters $S_{\text{TTC}}$ as a capped tie-break (+10), preference ordering as a late tie-breaker, and optionally $R$ as per-turn shaping capped at 0.2 trajectory magnitude. $G$ never determines $P$ or replaces $M$.
11. No single-path tool-call matching. Capability-tag evidence is the only accepted form of tool-sequence supervision in the benchmark layer.
12. TTC policy is adaptive per §9.2, not flat-N. Collection runs ≥2 seeds on Train-Long Codex (§9.4).
13. Evaluation boundary (Bench-Control + Train-Long only; Test-Long and Final-Test sealed) enforced by LLD-02 and honored by LLD-06 projection filters.
14. Phase 2b is pre-killed in all planning. LLD-11 is stretch. LLD-08 is built only if Gate 5 passes.

Deviations from any of these require a Change Summary entry in this document first.

**v0.6 additions to the lock-down list:**

15. **Monthly benchmark validity audit** (§17.1). Determinism is not sufficient; ABC-style task-validity review is required on a recurring cadence post-freeze.
16. **Raw evidence retention** (§17.2). The canonical event store retains byte-wise raw tool observations; derived/compressed fields are always recomputable from raw. No information-lossy compression at write-time.
17. **Dual-band calibration** (§17.7). Families with `P_benchmark ≠ M_training` must calibrate *both* into the §7.8 probe band.
18. **Variance-triggered seed escalation** (§17.8). ≥2 seeds is a floor; the RL/preference stack escalates to 4 or 8 on unstable families.
19. **Moving-frontier curriculum** (§17.9). The 20–80% band is re-bucketed every training round; saturated families trigger the §17.10 renewal plan.
20. **Harness canonicalization + cross-harness held-out eval** (§17.11). SFT view strips harness-specific action syntax; every N rounds a family seen in one harness is evaluated on the other.
21. **Systematic RAWR rows** (§17.12). Verification matrix is extended from a single "right-answer-no-grounding" row to ≥1 row per declared RAWR mode.
22. **Honesty > learnability precedence** (§17.13). Where benchmark honesty conflicts with cleaner learnable signal, honesty wins. All v0.6 additions are specific cases of this precedence.

---

## 17. Post-Review Hardening (v0.6)

This section folds in red-team pressure points raised against v0.5. Each subsection states the rule, the pushback where I disagreed with the reviewer's framing, the mechanism, and which existing section it updates. External anchors: ABC (2507.02825), Terminal-Bench, SWE-Gym (2412.21139), Training Long-Context SWE Agents with RL (2508.03501), Agent-RLVR (2506.11425), Rewarding Progress (2410.08146), DAPO (2503.14476), Multi-SWE-bench (2504.02605), SWE-bench Verified.

### 17.1 Benchmark validity audit loop (agrees with red-team point 1)

**Rule.** CNB-55 §10.1 freeze-gate determinism is necessary but not sufficient. Every family is subject to a recurring **task-validity audit** after freeze, not just at freeze. ABC documents that outcome/task-validity flaws can move measured performance significantly while determinism remains perfect — the family passes every scorer invariant yet measures the wrong thing.

**Mechanism.**

- **Calendar cadence.** Monthly. One rotating reviewer plus the family author.
- **Anomaly trigger.** Any family where the probe model's score moves ≥ 5 points across two adjacent monthly checkpoints *without* a scorer change triggers an out-of-cycle audit.
- **Procedure.** Reviewer samples one oracle trace, one mid-score trace, one failure trace; runs the ABC checklist (goal specified, success criteria observable, environment stable, shortcut coverage, leakage, difficulty calibration, judge reliability where applicable); flags fall into {OK, watch, task-invalid, retire}.
- **Remediation + invalidation protocol (v0.7 hardening).** When a family is flagged `task-invalid`:
  1. **Future training sets.** Every training run initiated after the flag timestamp **excludes** data where `family_id` matches the flagged family **AND** `family_status_at_write_time ∈ {task_invalid, retired_saturated}`. This is a hard projection filter in LLD-06, not a best-effort heuristic. Data written *before* the flag but from the same family is admitted only if its write-time snapshot shows `family_status_at_write_time = ok`; in that case the data retains a `provisional_admit` tag and is subject to exclusion if a later audit concludes the invalidity was present at write-time.
  2. **Prior checkpoint comparability.** Checkpoints produced from training sets that included the now-invalid family are **marked non-comparable** across the flag boundary. The leaderboard carries a `checkpoint_validity_version` counter that bumps on each invalidation; comparisons across bumps require a prose justification in `benchmark_run.md`.
  3. **Preference anchors.** A family in `task_invalid` cannot serve as a preference anchor for new pair construction in the §3.3 auto-preference view from the flag timestamp onward. Pairs already written are retained under the same `provisional_admit` / exclusion rule as (1).
  4. **RL prompt pool.** The family is removed from the RL prompt pool immediately on flag; rollouts in-flight are terminated; partial trajectories from those rollouts are written to the event store with `family_status_at_write_time = task_invalid` and are excluded from training views but retained for re-validation.
  5. **Quarantine, not deletion.** Data from `task_invalid` families is **not deleted**. If a later audit downgrades the flag (e.g. the validity concern was author-fixable rather than structural), the data becomes eligible for projection again under the family's new status. This is why write-time snapshot semantics matter: the field records what was believed at the moment of writing, not a retroactively-updated judgement.
  6. **Naming in leaderboards and reports.** Any leaderboard number that was ever computed using a now-`task_invalid` family must be annotated with the invalidation in any reprint; it cannot be silently updated or silently quoted. This is the honesty > learnability precedence (§17.13) applied to public metrics.
- **Artifact.** Each audit appends a dated row to `benchmark_run.md` under a new "validity audit log" heading. The row records: audit date, reviewer, sampled traces, ABC checklist outcome, flag verdict, and — if `task_invalid` — the `checkpoint_validity_version` bump.

**Updates:** Family-Test-Requirements §3 Layer A (adds audit-loop obligation as a post-freeze standing requirement); §3.1 canonical schema (`family_status_at_write_time` field); §3.2 SFT view + §3.3 auto-preference view + §3.4 RL prompt pool (projection filters honor `family_status_at_write_time`).

### 17.2 Raw evidence retention (agrees with red-team point 2)

**Rule.** The canonical event store (§3.1) retains **byte-wise raw tool observations and raw stdout/stderr** in full. Every derived or compressed field — `capability_tags`, `evidence_score`, `state_delta_score`, `turn_scoring`, preference margins — carries a `derived_from` pointer back to the raw slice it was computed over. No information-lossy compression at write-time.

**Why.** If a year from now we want to recompute richer process rewards (new reward model, new judge, new metric), we must be able to re-derive them from the same raw substrate we trained on originally. If the store only keeps `evidence_score: 0.6` without the raw observation text that produced it, we are locked into today's evaluator.

**Mechanism.**

- Schema bump: `turn[].raw_observation_blob` is an opaque byte-range reference into an append-only log file per run. No per-turn inline blob; the reference is stable.
- Schema evolution is additive-only. Derived fields may be added over time; the raw never changes.
- `benchmark_run.md` migration-plan section must state, for any schema change: "can v$X$ derived views be recomputed from pre-$X$ raw?" Answer must be yes.

**Updates:** §3.1 canonical store (adds `raw_observation_blob` field + derivation-pointer rule). LLD-06 writer contract inherits this directly.

### 17.3 Recovery-trace SFT sub-view (partial pushback on red-team point 3)

**Pushback I am keeping.** The red team argues SFT on successes alone undertrains recovery. I disagree with the naive conclusion "therefore SFT should include failures." SFT on failures teaches failure distributions. Recovery is not a failure; recovery is a **within-success phenomenon** — a trace that stumbles, self-corrects, and finishes cleanly. That trace is a success under $P$ and belongs in the SFT view. A trace that never recovers is a failure and belongs in the preference and RL views where the loss function is designed to handle it.

**Fold-in.** Decision 4 already says "20/100 is a bootstrap, not saturation" — the system expects failure-side signal to flow through preference + RL. v0.6 introduced a recovery-trace sub-view of the SFT view; **v0.7 fixes where the upweight lands**. The v0.6 rule cloned the model harder on the whole bad-detour-plus-correction span, which is directly contra the [Rewarding Progress](https://arxiv.org/abs/2410.08146) principle "reward steps that raise future success probability, not steps that merely *appear* in a successful trace." The bad detour is not a step that raised success probability — it lowered it; the *correction* raised success probability back.

**The v0.7 rule:**

- **Trace labeling (unchanged).** A successful trace is labeled `recovery_trace=true` if $G$ drops by ≥ 0.3 at any point before recovering, OR if the trace contains at least one turn with `integrity_flag=0` but `state_delta_score ≤ 0` (visible thrash that does not tamper).
- **Pivot definition (new in v0.7).** For each recovery trace, LLD-06 identifies the **recovery pivot turn $t^*$**: the smallest turn index $t$ such that (a) $G$ is monotonically non-decreasing over $[t, T]$ for the remainder of the trace, and (b) $\text{state\_delta\_score}_t > 0$. $t^*$ is the first turn where the trajectory has visibly committed to correcting.
- **Per-turn corrective flag (new in v0.7).** For every turn $t$ in a recovery trace, LLD-06 emits `corrective_turn_bool = True` iff $t \geq t^*$ **and** `state_delta_score_t > 0` **and** `integrity_flag_t = 0`. All other turns (including the entire pre-pivot bad detour) emit `corrective_turn_bool = False`.
- **Loss weighting (v0.7 scope).** The **2× multiplier in masked-NLL loss applies only to turns with `corrective_turn_bool = True`**. Detour turns ($t < t^*$ and/or `state_delta_score ≤ 0`) receive the standard 1× weight — the model sees that they happened in a successful trace, but is not cloned harder on them.
- **Target mix (unchanged).** 30–40% of the SFT view by row count should be recovery traces. If fewer than 30% of TTC-collected successes are recovery traces on a given family, LLD-07 runs more attempts on that family before the next SFT refresh.

**Why this is the correct reading of Rewarding Progress.** The principle is to upweight steps whose *advantage* — expected future success conditioned on taking this step vs not — is positive. In a recovery trace, the detour has *negative* advantage (it set the trajectory back); the corrective suffix has positive advantage (it re-established the path to success). Upweighting the whole span averages these together and loses the signal we care about. Upweighting only `corrective_turn_bool = True` turns targets the positive-advantage segment directly.

**What this does not do.** It does not add failures to the SFT view. Failures continue to feed preference + RL (their proper home). Papers: SWE-Gym (2412.21139), Training Long-Context SWE Agents with RL (2508.03501) — both use failure signal via preference / RL, not via SFT on failures.

**Updates:** §3.1 canonical schema (adds `recovery_trace`, `turn_scoring[].corrective_turn_bool`); §3.2 SFT view (sub-view definition); §6.1 SFT loss (2× multiplier gated on `corrective_turn_bool`, not on span membership).

### 17.4 LLD-11 reframed: Agent-RLVR + guidance; DAPO as optimizer (partial pushback on red-team point 4)

**Pushback I am keeping.** DAPO is not a conceptual basis for agent RL. It is a token-level PG algorithm with Clip-Higher, developed for math. The red team's sharpest version of this point is right: you cannot name LLD-11 after its optimizer any more than you can name LLD-10 "the AdamW pipeline."

**Fold-in.** LLD-11's conceptual anchor is now explicitly **Agent-RLVR** (verifier-first rewards — Agent-RLVR 2506.11425 reports 9% → 22% on SWE-bench Verified from environment-derived rewards plus 5 more from reward-model reranking) **+ Rewarding Progress** (2410.08146 — per-turn progress shaping, consumed per §8 with the capping discipline in place) **+ guidance rollouts** (teacher-traced context injection for tasks that land in the <20% band on the current-model probe, instead of excluding them from the prompt pool).

DAPO remains the inside-choice token-level optimizer — chosen for Clip-Higher stability on the long-tail of token-level losses in multi-turn rollouts — but it is not the framing. §4.6 title updated to "LLD-11 Agent-RLVR RL Pipeline (DAPO-optimized)." §12 research grounding reorganized to put Agent-RLVR and Rewarding Progress above DAPO.

**Concrete mechanism changes vs v0.5:**

- RL prompt pool (§3.4) now distinguishes three buckets: in-band (20–80%), saturation-watch (>80%), needs-guidance (<20%). The needs-guidance bucket participates in training via guidance rollouts (teacher trace injected into the context, policy learns to continue from the guided state) rather than exclusion.
- LLD-11 draft must specify the guidance-rollout mechanism before consuming the prompt pool. Not optional.

**Guidance-rollout provenance contract (v0.7 hardening).** "Teacher trace injected into context" is a leakage channel unless the trace source is bounded. v0.7 locks down provenance with an enumerated `guidance_source` field (schema in §3.1) and a separate accounting path. Allowed classes, ranked by permissiveness:

| `guidance_source` | Allowed in training? | Provenance rule |
|---|---|---|
| `autonomous_prior_success` | **Yes, freely.** Treated as regular RL rollout data. | The teacher trace is a prior autonomous solve of the *same task* by a checkpoint of the system itself (current policy, prior checkpoint, or an earlier SFT checkpoint). No external model; no oracle; no human. |
| `autonomous_current_policy_search` | **Yes, freely.** | Teacher trace is produced by current-policy best-of-$N$ search on the same task. Equivalent to expanded TTC under §9.2. |
| `stronger_model_flagged` | **Yes, but routed through the §17.4 separate accounting path** — the resulting rollout carries `guidance_source = stronger_model_flagged` and cannot be mixed with autonomous data in the same SFT or preference batch without an explicit mixing policy ratified by LLD-10 + LLD-11 owners. Held-out eval reports these rollouts as a separate line item so their contribution to metrics is legible. | Teacher trace is from a strictly stronger model (e.g. a larger Codex variant, a frontier model) used only on needs-guidance tasks. Per-family cap on the fraction of training that carries this tag. Tracked in `benchmark_run.md`. |
| `oracle_trace_forbidden` | **No. Blocking projection filter.** | Teacher trace from the family's reference `oracle/` input. Using this in training silently memorizes the gold answer; that is the evaluation boundary being breached from the other side. LLD-06 rejects any record with this value from all training views. |
| `authoring_only_debug` | **No. Blocking projection filter.** | Teacher trace from the family author during authoring or debugging. Retained for audit; never trained on. |

**Why the separate accounting path for `stronger_model_flagged`.** [Agent-RLVR](https://arxiv.org/abs/2506.11425) supports guidance in sparse agentic environments; it does not say stronger-model traces are equivalent to verifier-first RL data. They carry the stronger model's biases, its harness habits, and (if repeated) its specific solution templates. Mixing them into the same loss update as autonomous rollouts confounds the attribution of any held-out improvement — did the policy learn from its own experience, or did it distill the stronger model? The separate accounting path means:

1. Guidance rollouts live in a distinct prompt-pool bucket (`guidance_pool`) and a distinct SFT variant if they ever feed SFT (`Codex-SFT-guided`, appendix only — not the hinge checkpoint).
2. Loss updates on guidance rollouts are computed separately and scaled (default: 0.5× of autonomous updates; overridable per-family).
3. Evaluations on Bench-Control that benefit from `stronger_model_flagged` rollouts must attribute the delta; the `benchmark_run.md` line for any checkpoint trained with guidance must report both "autonomous-only $P_{\text{benchmark}}$" and "with-guidance $P_{\text{benchmark}}$."
4. Cap: per family, `stronger_model_flagged` rollouts may not exceed 15% of the family's total RL-pool consumption over any training round.

**LLD-07 / LLD-11 contract.** The orchestrator (LLD-07) is responsible for tagging `guidance_source` at rollout-emission time, not at training-ingest time. Ingest-time tagging is easy to skip. Emission-time tagging makes provenance audit-trailable and is required for the validity audit loop (§17.1) to verify that no `oracle_trace_forbidden` rollouts slipped into training views.

**Updates:** §4.6 (reframe); §3.1 canonical schema (`guidance_source` field + forbidden-value projection filter); §3.4 RL prompt pool (three-bucket + guidance_pool); §12 (reorder + citations); §17.9 moving-frontier curriculum uses the three-bucket structure; §17.13 — guidance contract is an explicit application of honesty > learnability.

### 17.5 Family-specific extended capability vocabulary (agrees with red-team point 5)

**Rule.** The core vocabulary `{localize, inspect, modify, verify, respect_invariants}` is unchanged — it is the cross-family aggregation vocabulary used by LLD-06 and by §8.3 scoring. However, families (especially Track 10 Strategic Management and Track 4 review-heavy families) may declare **extended sub-tags** that refine a core tag.

**Sub-tag schema.**

```yaml
capability_tags:
  core:
    required: [localize, inspect, modify, verify]
    forbidden: [modify:tests/, modify:immutable_slice]
  extended:
    - parent: inspect
      sub_tag: prioritize
      semantics: reading with intent to rank competing options
    - parent: inspect
      sub_tag: evidence_triage
      semantics: reading with intent to separate signal from noise
    - parent: modify
      sub_tag: policy_tradeoff
      semantics: writing a recommendation that explicitly names a tradeoff
    - parent: verify
      sub_tag: assumption_honesty
      semantics: running an assumption-ledger / ground-check
```

**Aggregation.** LLD-06 rolls sub-tags up into their parent core tag for cross-family aggregation. Within-family analysis (fine-grained middle-step scoring, reward-model training inputs) sees both levels. The raw event store retains both.

**Example.** For `proposal-ranking-manager-judgment`, sub-tags `inspect:prioritize`, `inspect:evidence_triage`, `modify:policy_tradeoff`, `verify:assumption_honesty` are all declared. A V5 trace that reads `incident_context/` qualifies for `inspect:evidence_triage`; a trace that writes a primary_risk explicit tradeoff qualifies for `modify:policy_tradeoff`.

**Updates:** §8.3 capability-tag vocabulary (core-plus-extended pattern); Family-Test-Requirements §4 item 5 (authoring obligation covers both levels).

### 17.6 $G$-gaming diagnostic doctrine (agrees with red-team point 6)

**Rule.** If $G$ rises for a family while $P_{\text{benchmark}}$ **and** $M_{\text{training}}$ stay flat for more than 3 training rounds, the default diagnosis is **not** "healthy latent progress." The default diagnosis is **"$G$ is being gamed for this family"** until proven otherwise.

**Why this direction.** $G$ is justified (§8) as a leading indicator of outcome-level improvement. A leading indicator that never leads anywhere is, by definition, not a leading indicator. The correct null hypothesis after 3 rounds of $G$-progress-without-outcome is that the shaping signal has developed a shortcut the outcome layer is correctly refusing to reward.

**Resolution procedure.**

1. Re-run the Family-Test-Requirements §5 verification matrix with the *current* model (not the probe model).
2. Inspect: do the family's known shortcut trajectories (pick-P3, finish-partial, right-answer-no-grounding, etc.) now show $G$ comparable to the oracle? If yes → $G$'s definition for this family has a shortcut.
3. Redefine $G$ for that family (usually by tightening `state_delta_score` rules or adding a sub-tag check) and re-run the matrix.
4. Only if the matrix still separates shortcuts from oracle cleanly AND $G$ still rises without outcome → accept the latent-progress interpretation, and schedule a re-probe in 2 more rounds.

**Tripwire.** Automated: LLD-06 emits a `G_progress_without_outcome` flag on any family meeting the condition; LLD-07 pauses sampling for that family until the audit completes.

**Updates:** §8.5 ($G$ feeds the flywheel — adds the gaming-diagnostic rule).

### 17.7 Dual-band calibration (partial pushback on red-team point 7)

**Pushback I am keeping.** The red team's concern is that $P_{\text{benchmark}}$ and $M_{\text{training}}$ can diverge materially, letting authors calibrate to a leaderboard number driven by signals the trainer never sees. I am not collapsing the two. The divergence is the §7.8 escape-valve rule applied honestly: Track 10 families need LLM-judge sub-metrics to grade their primary output, and those signals do not belong in training labels. Forcing $P_{\text{benchmark}} \equiv M_{\text{training}}$ would either (a) drop Track 10 entirely or (b) smuggle LLM-judge signal into training under another name.

**Fold-in.** The fix is not to collapse the divergence; it is to close the calibration loophole. v0.6 requires **dual-band calibration** at the §7.8 probe: both $P_{\text{benchmark}}$ and $M_{\text{training}}$ per-variant means must fall inside [15, 25] on the probe model. If $P_{\text{benchmark}}$ is in band but $M_{\text{training}}$ is out of band (too easy because all the hardness lives in the LLM-judge points trainer can't see; or too hard because the deterministic milestones are under-weighted) the family is not DONE.

**Reporting.** `benchmark_run.md` probe-log row adds two columns: `M_training_mean` per variant and `P_benchmark − M_training` per variant. Divergence > 0.25 triggers an authoring note in the same row explaining why.

**Updates:** §7.8 (probe-band rule extended to dual-band); Family-Test-Requirements §4 item 10 (dual-band calibration blocking).

### 17.8 Variance-triggered seed escalation (agrees with red-team point 8)

**Rule.** ≥2 seeds is a floor, not a target. Every family declares a per-seed variance threshold; exceeding it auto-escalates to more seeds.

**Thresholds (defaults, overridable per family):**

- Per-seed std. dev. on $M_{\text{training}}$ ≤ 0.10 at 2 seeds → stay at 2.
- 0.10–0.20 → escalate to 4 seeds.
- > 0.20 → escalate to 8 seeds.
- If at 8 seeds the std. dev. is still > 0.15, the family is flagged as "inherently high-variance" and the auto-preference view (§3.3) rejects all pairs where within-seed variance exceeds between-seed signal-to-noise ratio. Preference pairs over unstable families are easy to hallucinate from seed noise alone.

**Why this matters for preference construction.** Two seeds can produce a 0.2 margin on a noisy family purely from sampling. That margin becomes a preference pair that looks like learned signal but is noise. The escalation rule prevents seed noise from being laundered into preference signal.

**Updates:** §3.3 auto-preference view (adds noise-vs-signal rejection rule); §9.4 collection-side seeds (escalation policy replaces the flat "≥2" floor).

### 17.9 Moving-frontier curriculum refresh (agrees with red-team point 9)

**Rule.** The RL prompt pool's 20–80% current-model band is **recomputed every training round**, not fixed at family freeze.

**Mechanism.** Each training round ($N$ hours of RL, or after an SFT refresh):

1. Run the current checkpoint against every registered family, ≥ seed-escalated sample count, get current-model $P_{\text{benchmark}}$ per variant.
2. Re-bucket every variant into {in-band (20–80%), saturation-watch (>80%), needs-guidance (<20%)}.
3. In-band variants flow to the normal prompt pool.
4. Saturation-watch variants trigger §17.10 renewal review; they stay in the pool at reduced sampling weight until renewed or retired.
5. Needs-guidance variants flow to Agent-RLVR guidance rollouts (§17.4) rather than exclusion.

**Per-training-round artifact.** `benchmark_run.md` gets a dated "curriculum bucket map" row showing every family's variant-by-variant current bucket. Moves across buckets are the leading indicator of training progress *and* of saturation.

**Updates:** §3.4 RL prompt pool (moving-frontier bucketing); §9.2 adaptive TTC (reads current bucket map for attempt-count policy).

### 17.10 Per-family saturation / renewal plan (agrees with red-team point 10)

**Rule (now Layer-B item 14).** Every family declares, in `task_spec.md`: (a) a saturation threshold on current-model $P_{\text{benchmark}}$ (default: mean across variants > 80% for 2 consecutive training rounds); (b) a renewal mechanism (variant tightening / new trap / retirement).

**Why this is blocking.** Terminal-Bench's lesson is not "hard CLI tasks matter"; it is "benchmarks saturate, and without versioning, training optimizes toward author-specific artifacts rather than the intended capability." Without a declared renewal plan, a saturated family silently shifts from "this family teaches capability X" to "this family teaches the idiomatic quirks of this family's author." That drift is invisible in training loss curves.

**Mechanism.** On 2nd consecutive saturation-watch round, LLD-06 emits a `saturation_renewal_due` flag. The family author has one training-round cycle to land a renewal PR or flag the family for retirement. Failure to act moves the family to retired; training runs stop sampling from it; its already-collected data carries a `source_family_status=retired_saturated` column and is not used as a preference-pair anchor for new runs.

**Updates:** Family-Test-Requirements §4 item 14 (new blocking item); §11 no-human-labels stance (renewal PRs are author-only, no human labeling involved).

### 17.11 Harness-leakage mitigations (agrees with red-team point 11)

**Rule.** Mixing Codex and mini-SWE-Agent trajectories into one event store (§3.1) risks the model learning harness-specific action formatting, retry habits, or prompt priors instead of the intended capability. Three mitigations:

1. **Canonicalize at write-time.** The SFT view (§3.2) normalizes harness-specific action syntax to a canonical form (e.g. `shell("…")` → `TOOL.shell("…")`, Codex-style apply_patch → canonical diff-application). The original harness-form is retained only in `raw_observation_blob` (§17.2) and is never seen by the SFT loss.
2. **Harness-ID dropout.** During SFT and RL, with probability 0.2 per training example, strip the harness-ID context token and train on the canonicalized form alone. Forces the model to learn capability rather than harness idiom.
3. **Cross-harness held-out eval.** Every 4 training rounds, evaluate the current checkpoint on families it has seen in only one harness, using the *other* harness. If degradation on the held-out harness exceeds 0.1 on $P_{\text{benchmark}}$, training stack has learned harness priors; rebalance sampling and re-run.

**Paper reference.** Multi-SWE-bench (2504.02605) formalizes cross-language / cross-harness generalization as the anti-overfitting test.

**Updates:** §3.2 SFT view (canonicalization); §6 losses (dropout variable); §9 TTC policy (cross-harness eval cadence).

### 17.12 Systematic RAWR generators (agrees with red-team point 12)

**Rule.** The Family-Test-Requirements §5 verification matrix's single "right-answer-no-grounding" row is upgraded to **one row per declared RAWR mode per family**. RAWR = right-answer-wrong-reasons.

**Authoring obligation (new in Family-Test-Requirements §5).** Every family declares, in `task_spec.md`, its RAWR modes. Each is a named shortcut class: the agent produces the correct nominal output via a reasoning path that should not count as solving the task. For `proposal-ranking-manager-judgment`:

- `grounding_stripped`: correct `accepted` + `ranking` but `citations: []` on rejections (already present).
- `citation_fabricated`: correct `accepted` + `ranking` but citations point to real files while rejection rationale doesn't actually follow from the cited content (needs LLM-judge or structural heuristic).
- `constraint_named_not_respected`: accepted proposal cites the blocking constraint in summary but the ranking still ignores it.

Each mode gets its own matrix row with expected $P$, $M$, $G$, $R$, $S_{\text{TTC}}$ ranges. ABC's core lesson is that RAWR classes are persistent and benchmark-specific — one-off red-team coverage systematically under-represents them.

**Auto-generation (two criteria, v0.7).** LLD-06 monitors training trajectories and surfaces RAWR candidates under **either** of two criteria:

- **Criterion A — process-visibly-bad.** `P_benchmark ≥ 0.8` AND `G ≤ 0.4`. This is the v0.6 rule. It catches "right answer, obviously bad process" — traces where the outcome is correct but the shaping signal knows the reasoning path was wrong.
- **Criterion B — $G$-dominant shortcut (new in v0.7).** `P_benchmark ≥ 0.8` AND **any one of**:
  - The family is currently in `G_progress_without_outcome` watch per §17.6 (i.e. $G$ rising without $P$ / $M$ moving for > 3 rounds), OR
  - The trace's declared `required` capability tags (§17.5) are under-exercised — specifically, ≥ 1 required core tag has zero turns matching it in `turn_scoring[]`, OR
  - The trace's $G$ is in the top quartile for the family AND its `corrective_turn_bool` turn count is zero (high shaping score with zero visible correction on a family where recovery is expected).

Criterion B catches the failure class that poisons $G$ itself: the shaping metric looks healthy, the outcome looks healthy, but the trajectory reached the outcome via a path that does not exercise the capabilities the family is supposed to teach. That is precisely the class §17.6 says to suspect, and it is the class [ABC](https://arxiv.org/abs/2507.02825) flags as systematically under-represented by one-off red-team coverage.

**Why one criterion isn't enough.** Criterion A fires only when $G$ disagrees with $P$. If $G$ is itself flawed, it will agree with $P$ on shortcuts too — high $P$, high $G$, low honesty. Criterion B fires exactly in that regime: the family-level signal (G-watch, required-tag coverage) is what flags the trace, not the trace's own $G$. The two criteria triangulate: A checks that $G$ catches what $P$ misses; B checks that $G$ hasn't quietly started catching the wrong things.

**Reviewer flow.** For each candidate, the family author reviews and either (a) accepts it as a new declared RAWR mode and adds a matrix row in Family-Test-Requirements §5, (b) rejects it with justification in `benchmark_run.md`, or (c) — for Criterion B candidates — triggers the §17.6 $G$-redefinition procedure instead of adding a mode, because a consistent stream of Criterion B hits on one family is evidence $G$'s definition is the problem, not the trace.

**Updates:** Family-Test-Requirements §5 verification matrix (multi-row RAWR requirement + the capability-coverage check in Criterion B); §17.6 ($G$-redefinition is triggered by Criterion B pattern, not only by §17.6's own tripwire); LLD-06 (RAWR-candidate emission implements both criteria).

### 17.13 Honesty > learnability precedence (responds to the meta-question)

**The meta-question (reviewer's framing).** Is the HLD designing a flywheel for benchmark-honest agent improvement, or for maximally-learnable deterministic signal? The red team's read is that it's trying to do both, and most risk lives in the overlap.

**Answer — explicit precedence.** The HLD optimizes for **both**, with a strict precedence: **honesty is the hard constraint; learnability is the optimization target inside the constraint.** When a cleaner learnable signal would require softening a task-validity check, loosening an anti-shortcut detector, or bringing a non-deterministic judge into the training label, **honesty wins and learnability takes the hit.**

**Where this is already binding in the doc:**

- Dual-emit scorer (§7.8, §17.7): we lose 13 pts of training signal on `proposal-ranking-manager-judgment` rather than trust LLM-judge labels. Honesty > learnability.
- $G$ capped at 0.2 trajectory magnitude and 10 absolute $S_{\text{TTC}}$ (§7.2, §7.4): we accept a weaker shaping signal rather than let $G$ dominate. Honesty > learnability.
- Recovery-trace sub-view (§17.3) instead of SFT-on-failures: we accept the smaller SFT dataset rather than pollute SFT with failure distributions. Honesty > learnability.
- Saturation-renewal (§17.10): we retire saturated families rather than keep mining them for cheap gradient. Honesty > learnability.
- Cross-harness held-out eval (§17.11): we take a 4-round tax on training throughput rather than let harness priors go undetected. Honesty > learnability.

**Where the precedence binds in the future.**

- If LLD-11 rollouts produce a dense reward that correlates 0.9 with $G$ but only 0.3 with $P$, we use it only inside the cap. Honesty > learnability.
- If a future reward model trained on `P_benchmark` (which includes LLM-judge pts) proves measurably better than one trained on `M_training`, that is not an argument for merging the two. It is an argument for improving `M_training`'s deterministic coverage of what the LLM judge is catching. Honesty > learnability.

**What this does not promise.** The precedence does not guarantee the flywheel works. It guarantees that if the flywheel doesn't work, it will fail honestly (measurable $P$ / $M$ don't move) rather than dishonestly (a fake metric rises while the agent degrades). That asymmetry is the whole point of the verifier-first architecture; v0.6 makes it explicit.

**Updates:** This section is the doctrine the whole doc rests on; no lower-section edits required. Future authoring should cite §17.13 directly when a design decision trades learnable signal for honest measurement.

---

*Document version: 0.7*
*Scope: LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11*
*Does not supersede: HLD Spec v2.3, the v2.1 LLD index, or LLD-13 v0.6. This doc is subordinate to all three.*
