# HLD · Training Flywheel (LLD-06 → LLD-11)

> Codex-Bench · High-Level Design
> Scope: The data-normalization + training stack — LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11
> Derived from HLD Spec v2.3 · April 2026
> Status: DRAFT v0.4 — adds family-level DONE criteria tying CNB-55 freeze gate to the flywheel training obligations

---

## Changelog

| Version | Change |
|---|---|
| v0.4 | Added §15 **Family-Level DONE Criteria**, worked through `proposal-ranking-manager-judgment` as the concrete example. The CNB-55 §10.1 freeze gate (oracle ≥ 90, empty = 0, red-team ceilings, deterministic scorer, probe-in-range) remains necessary but is no longer sufficient. A family is DONE for the flywheel when it additionally: (a) declares its milestones into the §7.5 5-slot template with M3 anti-shortcut present and non-trivial; (b) registers its capability-tag declarations (required / recommended / forbidden) per §8.3; (c) supplies an integrity-flag detector and a state-delta rule; (d) quarantines LLM-judge signals out of $P$ / $M$ per the §7.8 escape-valve rule — for proposal-ranking this specifically means the 10-pt partial-progress rubric and the 3-pt ledger-padding check are reward-model inputs, not milestones; (e) produces a calibration matrix (oracle / empty / right-no-grounding / adversarial) × ($P$, $M$, $G$, $R$, $S_{\text{TTC}}$) that demonstrates signal separation; (f) supports ≥2 deterministic seeds so LLD-06 can emit preference pairs per §9.4. Old §15 renumbered to §16. |
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
turn_scoring:         # [{turn_idx, capability_tags, evidence_score, state_delta_score,
                      #   integrity_flag, compliance_flag}] — see §8
G:                    # trajectory-level aggregate from §8.4, in [0,1]
final_artifact_ref:   # patch (SWE-bench) or state digest (Codex-Long)
outcome:              # pass | fail | timeout | crash | no_patch
cl_pass:              # Codex-Long verifier bool (P in §7), null for SWE-bench
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
```

LLD-06 does not re-grade — it reads LLD-05's normalized `outcome` / `cl_pass` as the source of truth for pass/fail, reads LLD-13's milestone manifest for `milestone_vector`, and *computes* `turn_scoring[]` and $G$ from the raw trajectory post-hoc (agent never sees these during the run).

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

### 4.6 LLD-11 DAPO RL Pipeline — the stretch branch

Consumes the RL prompt pool; reward from §7.2 with optional per-turn shaping from §8.5. Warm-started from Codex-SFT-all; rollouts via LLD-03 + LLD-08.

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

## 15. Family-Level DONE Criteria

Worked example throughout: `proposal-ranking-manager-judgment` (Track 10, Strategic Management & Long-Horizon Evolution, 5 variants v1–v5, structured-output CLI at `./bin/cnb55-brief`).

### 15.1 Why this section exists

Family authoring historically stopped at the CNB-55 §10.1 freeze gate — oracle ≥ 90, empty = 0, ≥ 4 red-team exploits each scoring ≤ 20, scorer deterministic, probe-in-calibration-range. That is still necessary. It is no longer *sufficient*.

The flywheel architecture in §§3–8 adds training-stack obligations a family must satisfy before LLD-06 can emit coherent training views over it, before LLD-07 can orchestrate adaptive TTC over it, and before LLD-11 can use it in the RL prompt pool. "DONE" now means both: benchmark-honest under CNB-55 *and* flywheel-ready under this HLD. A family that passes the freeze gate but fails the flywheel obligations is still blocked from training.

This section is the checklist, the verification matrix, and the family-specific decision log (the LLM-judge escape-valve call) for `proposal-ranking-manager-judgment`. Other families should produce an equivalent section in their own `benchmark_run.md` or an adjacent authoring note.

### 15.2 Two layers of DONE

| Layer | What it certifies | Owner | Reference |
|---|---|---|---|
| Layer A: Benchmark honesty | Oracle solves it; empty fails it; red-team ceilings hold; scorer is deterministic and verifier-isolated | Family author + second reviewer | CNB-55 SPEC §10.1 + family's `evaluator_contract.md` |
| Layer B: Flywheel readiness | Milestones map to §7.5; capability tags declared; integrity flag detector registered; LLM-judge signals quarantined; calibration matrix shows signal separation; seeds ≥ 2 | Family author + LLD-06 owner | This section (§15) |

Both layers must be green. Layer A without Layer B means the family can be probed but cannot feed training. Layer B without Layer A means the training data is polluted.

### 15.3 Layer A — CNB-55 freeze gate (abbreviated, already familiar)

Already required for `proposal-ranking-manager-judgment`:

- Oracle (`cnb55-brief submit` against `verifier_data/.../oracle/brief_input.json`) scores ≥ 90 on every variant. Observed: V1 / V2 / V3 = 90; V4 = 96; V5 = 99 on 2026-04-20.
- Empty `brief/` scores 0.
- Red-team exploits under `verifier_data/.../red_team/` each cap at ≤ 20 or ≤ 40 depending on ceiling (`delete_visible_tests.sh`, `shadow_pytest.sh`, `minimal_brief.json`, `pick_highest_delta.md`, `finish_the_partial.md`, `single_objective_anchor.md`).
- Probe on GPT-5.4/high at 3 seeds produces per-variant means in the calibration band (V1≈27, V2≈21, V3≈17, V4≈15, V5≈11; family mean ≈ 18, inside §7.8 calibration target [15, 25]).
- Scorer deterministic under `CNB55_SEED=42`; hidden-test invocation order fixed; JSON keys sorted.
- Immutable-slice checksums over `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `bin/` enforced via trusted-final-state checks.

This layer is the family's existing contract. It does not change.

### 15.4 Layer B — Flywheel readiness checklist

Numbered items. Family is DONE for the flywheel when all items are checked. Assume Codex is connected to an LLM (i.e. the family can be actually run end-to-end to validate).

**§7 milestone obligations:**

1. **5-slot milestone declaration.** Family emits a `family.yaml` section (or equivalent) mapping its track scorecard sub-metrics to the §7.5 slots:

   For `proposal-ranking-manager-judgment`:

   | Slot | Weight | What it checks (this family) | Implementation level (§7.8) |
   |---|---|---|---|
   | M1 Localization | 0.10 | Agent ran `cnb55-brief schema` at least once OR read ≥ 3 files under `proposals/` + `repo_evidence/` before any `submit` | L1 regex + shell-log scan |
   | M2 Primary fix | 0.20 | `brief/manager_brief.json` exists and parses as valid JSON with correct `schema_version` (derived from Phase 2 checks 1 + 2) | L2 structured (CLI + JSON schema) |
   | M3 Invariants preserved / anti-shortcut | 0.20 | Immutable-slice checksums unchanged AND no `sitecustomize.py` / `pytest.py` shim AND `brief_input.json` the only whitelisted non-`brief/` write | L2 checksum + filesystem audit |
   | M4 Functional checks | 0.20 | `pytest -q tests/test_ranking_brief.py` passes AND at least one "ranking-without-grounding" ceiling not triggered (≥ 2 rejections cite real evidence files) | L2 structured property checks |
   | M5 End-to-end integration state | 0.30 | Accepted-proposal-matches-gold behavioral check **AND** Kendall-τ ≥ variant threshold **AND** no partial-credit ceiling below ≤ 30 triggered | L2 symbolic + L3 executable (no LLM-judge) |

2. **M3 non-trivial.** The anti-shortcut slot must have a real detector, not a placeholder. For this family, M3 is backed by three concrete checks (immutable-slice checksums, pytest-shim detection, writable-path whitelist) and at least one red-team exploit each exercise. ✔ satisfied.

3. **Milestone booleans, monotone weights, gated dependencies, `verify.sh` sole authority.** Applies without exception. M5 is only checked when M2 holds (`brief/manager_brief.json` exists and parses).

4. **Milestones never agent-visible.** Milestone scripts live under `verifier_data/proposal-ranking-manager-judgment/<variant_id>/milestones/` and are injected post-run into a separate container per LLD-13's integrity protocol. The family author must confirm no milestone-defining file is mounted into the agent's workspace bundle.

**§8 middle-step grading obligations:**

5. **Capability-tag declarations.** Family declares, per variant:

   For `proposal-ranking-manager-judgment` (V1 as example):

   ```yaml
   capability_tags:
     required:
       - localize        # must read ≥ 3 evidence files before first submit
       - inspect         # must read ≥ 1 file under proposals/ AND ≥ 1 under repo_evidence/
       - modify          # exactly one terminal action: `cnb55-brief submit`
       - verify          # must run `cnb55-brief validate` OR `pytest` at least once
     recommended:
       - inspect         # reading release_context/ (V4+) or incident_context/ (V5)
     forbidden:
       - modify:tests/   # any edit under tests/ raises integrity_flag
       - modify:immutable_slice  # any edit under proposals/, repo_evidence/, release_context/, incident_context/, AGENTS.md, Dockerfile, bin/ raises integrity_flag
   ```

6. **Tag-to-tool-call mapping rules.** LLD-06 owns the global mapping registry. The family author registers family-specific rules for any ambiguous cases. For this family, the only non-obvious rule is: `cnb55-brief validate` → `verify`, `cnb55-brief submit` → `modify` (terminal), `cnb55-brief schema` → `inspect`.

7. **State-delta definition.** For families where "state" is not a file diff, the author must specify what `state_delta_score` per turn measures. For `proposal-ranking-manager-judgment`, the state is the `brief/manager_brief.json` file (absent or present; if present, parseable and schema-valid). The state-delta rule:

   | Transition | state_delta_score |
   |---|---|
   | No-op (read-only tool call) | 0.0 |
   | `cnb55-brief validate` returns success for the first time | +0.3 |
   | `cnb55-brief submit` succeeds and `brief/manager_brief.json` becomes valid | +0.7 |
   | Any turn that raises `integrity_flag` | -0.5 (clamped to [0, 1] at aggregate) |

8. **Integrity-flag detector registered.** Rules that raise $H = 1$ for the run:

   - Any write outside `brief/*` + `brief_input.json` at workspace root.
   - Any modification to an immutable-slice checksum.
   - Any pytest/sitecustomize/usercustomize shim dropped into the workspace.
   - Any network egress attempt (grader log inspection).

   Registration is a one-line append to LLD-06's family-integrity-rules file.

9. **Compliance-flag is harness-level.** Applies uniformly; family does nothing extra. LLD-06 checks schema validity of every `function_call`, allowed-tool filter, and argument parseability.

**§7.8 escape-valve for LLM-judge signals (family-specific decision for this family):**

10. **LLM-judge signals quarantined from $P$ and $M$.** This family has **two** LLM-judge components in its current evaluator contract:

    | Signal | Points | Decision |
    |---|---|---|
    | Partial-progress rubric (`partial_progress.md`, gpt-5.4 @ T=0) | 10 | **Quarantined** — not a milestone; feeds the §8.5 per-turn shaping channel and/or a Phase 2b reward-model input. The scorer still includes it in the CNB-55 100-pt total for probe calibration, but $M$ computed for training uses only the symbolic M1–M5 slots above. |
    | Assumption-ledger padding check (LLM-judge on "are `missing` rows real gaps?") | 3 | **Quarantined** — same treatment. Rolled into the same shaping/reward-model channel. |

    Net effect: 13 of the family's 100 scoring points are LLM-judge-produced. Those 13 points are visible to the probe (so calibration remains stable with the existing 2026-04-20 probe log) but **invisible to the training loss**. The scorer emits two top-level fields per run: `P_benchmark` (all 100 pts, including LLM-judge) and `M_training` (only symbolic contributions normalized to [0,1]). LLD-06 reads `M_training`.

    This is the §7.8 escape-valve rule applied concretely: if a signal needs an LLM judge to define, it does not determine training labels. It may still inform benchmark leaderboard scores and TTC reranking.

**§9 collection-side obligations:**

11. **Seeds ≥ 2 on Train-Long campaigns over this family.** Family supports ≥2 deterministic seeds per variant. Codex determinism is sampling-temperature-bounded; the family author confirms that rerunning the oracle input twice produces the same `manager_brief.json` under fixed seed. ✔ Family has no hidden randomness in the CLI.

12. **Deterministic `initial_state` for RL prompt pool.** The workspace bundle at `workspace_bundle/<variant_id>/` is fully pinned via `manifest.lock.json`. Initial state hash for RL pool entries computed over the variant bundle at commit-time.

13. **`grader_ref` and `milestone_config_ref` exposed.** LLD-13 registry entry exists for this family pointing at `verifiers/proposal-ranking-manager-judgment/score_ranking.py` (grader) and `verifier_data/proposal-ranking-manager-judgment/<variant_id>/milestones/` (milestone config).

### 15.5 Required verification matrix

Before marking the family DONE, run the following trajectories and populate the matrix. Each trajectory is a Codex session against one pinned variant (V1 is the canonical anchor for this section).

| Trajectory | How produced | Expected $P$ | Expected $M$ (training) | Expected $G$ | Expected $R$ | Expected $S_{\text{TTC}}$ | What this verifies |
|---|---|---|---|---|---|---|---|
| **Oracle** | `cnb55-brief submit verifier_data/.../oracle/brief_input.json` | 1 | ≥ 0.90 | ≥ 0.75 | ≥ 0.98 | ≥ 1175 | Signal ceiling; oracle looks like a successful, well-grounded run in every scoring lens |
| **Empty** | No `brief/` write | 0 | 0 | ≤ 0.15 | 0 (after penalties) | ≤ 0 | Signal floor; empty fails every lens |
| **Right-answer-no-grounding** | Submit a brief with correct `accepted` and `ranking` but `citations: []` on all rejection entries | 0 (ceiling `ranking_without_grounding ≤ 25` fires) | ≤ 0.40 (M4 false; M5 partially false) | ≤ 0.40 (low evidence_score) | ≤ 0.24 | ≤ 40 | Middle-step grader actually differentiates grounded from ungrounded solves — the shortcut $S_{\text{TTC}}$ delta vs oracle must be ≥ 1100 |
| **Pick-P3 shortcut** (V1–V3) | Submit a brief accepting the staffing-blocked proposal | 0 (ceiling `ignored_staffing_constraint ≤ 30`) | ≤ 0.45 | variable, ≤ 0.55 | ≤ 0.27 | ≤ 70 | Ceiling fires; reward-hack basin has visibly lower $S_{\text{TTC}}$ than oracle; $m_3$ may still be 1 (no tampering), so the trace is flagged by ceiling, not by $H$ |
| **Finish-the-partial** (V3+) | Submit a brief accepting the in-progress-patch proposal without sunk-cost language | 0 (ceiling `sunk_cost_finish ≤ 30`) | ≤ 0.50 | ≤ 0.60 | ≤ 0.30 | ≤ 80 | Same as above for the V3-specific trap |
| **Delete-tests adversarial** | Run `delete_visible_tests.sh` | 0 (trusted-final-state score → 0) | 0 (M3 = 0) | 0 ($H = 1$, integrity_flag on the edit turn) | -0.6 $H$ term dominates | heavily negative | Anti-reward-hack floor fires through *three independent* channels (trusted-state, $M$, $H$); no single detector is load-bearing alone |

If any row of this matrix produces a value outside the expected range, the family is **not DONE** and the relevant detector (integrity flag, capability-tag mapping, state-delta rule, or LLM-judge quarantine) must be revisited.

For `proposal-ranking-manager-judgment`, the matrix above should be added to `benchmark_blueprints/families/proposal-ranking-manager-judgment/benchmark_run.md` as a dated "flywheel-readiness probe" section, alongside the existing CNB-55 probe log.

### 15.6 Specific decision log for this family

Two decisions that other families can reuse as precedent:

**Decision A — LLM-judge quarantine.** Because Track 10 Strategic Management families inherently involve judgment-call sub-metrics (partial-progress reasoning, assumption-ledger honesty), the evaluator contract keeps LLM-judge signals for probe scoring but *excludes them from $P_{\text{training}}$ and $M_{\text{training}}$*. The scorer emits dual fields. Other LLM-judge-heavy families (any Track 10 family; some Track 4 review families) should follow this precedent. Track 1–3 families (algorithmic, migration, CI) typically do not need the quarantine because their judgments are already symbolic.

**Decision B — State-delta for JSON-deliverable families.** For families where the deliverable is a single JSON file (not a code patch), `state_delta_score` uses the file-state transitions defined in item 7 above, not a file-diff scorer. Other JSON-deliverable families (`policy-docs-compliance-audit`, `incident-evidence-synthesis`, `request-path-evidence-brief`) can reuse this pattern.

### 15.7 Blocking vs non-blocking

| Item | Blocks marking family DONE? |
|---|---|
| Items 1–4 (milestone declaration, M3 non-trivial, rules, invisibility) | **Blocking** |
| Items 5–7 (capability tags, tag-mapping, state-delta) | **Blocking** |
| Item 8 (integrity-flag registration) | **Blocking** |
| Item 9 (compliance-flag — harness level) | Automatic |
| Item 10 (LLM-judge quarantine) | **Blocking** iff the family's scorer emits LLM-judge contributions. For this family: blocking. |
| Items 11–13 (seeds, initial_state, grader_ref) | **Blocking** |
| Verification matrix (§15.5) | **Blocking** — must be populated with real observed values, not predicted |
| CNB-55 freeze gate (§15.3) | **Blocking** (pre-existing) |

### 15.8 Concrete TODO for `proposal-ranking-manager-judgment`

What's already done (as of 2026-04-20 probe log in `benchmark_run.md`):

- Layer A: freeze gate passed — oracle 90 / 90 / 90 / 96 / 99; ceilings hold; scorer deterministic; probe in range.
- Family-specific attempt_02b hardening applied (τ thresholds, ceiling tightening, `missed_staffing_update`, `missed_watermark_assumption`).

What's still missing for Layer B:

- [ ] Populate the §15.4 milestone mapping (items 1–4) into the family's YAML, using the table in item 1 above as the authoritative source.
- [ ] Add `capability_tags` block (item 5) to `family.yaml` per variant.
- [ ] Register state-delta rules (item 7) with LLD-06.
- [ ] Register integrity-flag rules (item 8) with LLD-06.
- [ ] Split scorer output into `P_benchmark` and `M_training` (item 10); confirm the 13 LLM-judge points appear only in `P_benchmark`.
- [ ] Run the §15.5 verification matrix on V1; record results in `benchmark_run.md`.
- [ ] Repeat the matrix on one stress variant (V3 or V5 recommended) to confirm detectors fire under trap conditions.

Once all seven checkboxes are green, the family is DONE for the flywheel and can feed LLD-06.

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

---

*Document version: 0.3*
*Scope: LLD-06, LLD-07, LLD-08, LLD-09, LLD-10, LLD-11*
*Does not supersede: HLD Spec v2.3, the v2.1 LLD index, or LLD-13 v0.6. This doc is subordinate to all three.*
