# Serving Auto-Research — Hardening Plan

**Parent documents.** `docs/HLD-Serving-Backend-AutoResearch-v0_1.md` (parent HLD v0.1.1), `docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md` (agent sub-spec v0.1.11). This plan is additive to both — it does **not** supersede the v0.1.11 loop mechanics, and it does **not** scope-creep into v0.2 L0 / multi-family / multi-model work. It describes three workstreams that must complete before any further auto-research search (L0, L1-retry, or L2-enforced) can produce defensible artifacts.

**Driving evidence.** `reports/auto-research-round-2026-04-24-review.md` — the review of round `qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T072126Z` which produced bundle C003 and the in-flight Sprint-2 L2 run. The v0.1.11 loop mechanics held (commits clean, trailers correct, worktree stable, no CODEX_HOME fallout). What the round exposed is that the *measurement substrate* the loop sits on top of is not producing defensible answers — three orthogonal gaps, detailed below.

**Status at spec-authoring time.** bc05506 is the latest commit on `origin/main`. The Sprint-2 L2 round (`.../sprint-2-20260424T163930Z`) is in-flight with 10 result rows through C008 and no finalize. No action on the active round is assumed by this plan — see §6 sequencing for the recommended kill path.

---

## 0. Scope and non-goals

**v0.1 hardening scope — what this plan covers.** Three workstreams:
- **A.** Measurement hardening — widen n, make noise-floor defensible, fix the seed-trace thinking-token gap, make latency treatment explicit.
- **B.** L2 request-shaping enforcement — wire the four currently-unenforced fields into the inference proxy, OR explicitly descope them until the proxy catches up.
- **C.** Bundle-honesty metadata — add `round_provenance.confidence`, `improvement_over_baseline_ci_95`, `latency_above_slo`, `l2_enforcement_coverage` so downstream bundle consumers can tell a defensible production bundle from an exploratory artifact.

**Non-goals.** This plan does **not** touch:
- The agent loop mechanics (`run_round`, codex invocation, worktree discipline, trailer contract) — those are v0.1.11 and held up in the 2026-04-24 field test.
- The L0 kernel work — still deferred to v0.2 per parent HLD §7. The decision to actually start L0 is downstream of this hardening plan's output (§5 decision framework).
- The parent HLD's three-dim SLO definition. v0.1 remains throughput-primary; soft latency gating is added as metadata, not a feasibility rail.
- Generalization to other `(model, family)` pairs. Still v0.2.

**Relationship to v0.1.11 sub-spec.** This plan lands as an **addendum**, not a revision. The sub-spec sections this plan touches are enumerated in each workstream's "sub-spec changes" subsection. When the plan completes, the sub-spec bumps to v0.1.12 with those additions folded in; this doc becomes historical after v0.1.12 ships.

**Why this is one plan and not three separate docs.** The three workstreams are independent in implementation but sequentially dependent in validation:
- Workstream A must land before any round's results are trusted.
- Workstream B is only meaningful once A exists — otherwise "L2 enforcement works" is an untestable claim.
- Workstream C is the publication surface — without it, even a hardened-and-enforced round produces bundles that downstream consumers can't distinguish from exploratory ones.

A single plan keeps the dependencies explicit.

---

## 1. Context — why hardening before more search

The 2026-04-24 round review (bc05506) surfaced three structural gaps that the v0.1.11 loop cannot fix by running again:

**Gap 1 — Signal below noise.** Baselines baseline_a and baseline_b, configurationally identical, measured 0.008821 and 0.007583 req/s — a 16% spread. noise_floor = 2×|M₁−M₂| = 0.002476. The "winner" C003 beat the better baseline by 0.000954 req/s — below the noise floor by a factor of 2.6×. Rescreen-aware paired means tighten this further: C003 paired 0.008746 vs baseline_a paired 0.008482, margin 0.000264 (≈3% above baseline). With n=2 measurements per candidate on a workload whose baseline-to-baseline spread is 16%, a 3–10% "improvement" is not statistically distinguishable from variance. The v0.1.11 §4.4 rescreen mechanism is sound; it's starved of samples.

**Gap 2 — Thinking behavior untested.** `benchmark_blueprints/families/proposal-ranking-manager-judgment/seed_trace.jsonl` has four rows, all with `thinking_tokens: 0`. The holdout trace has three rows, all with `thinking_tokens: 0`. The harness reads and records correctly (v0.1.11 §9.2); the workload doesn't exercise thinking. So `reasoning_content_purity = 1.0` is true but vacuous — we can't leak thinking we're not generating. Every parent §4.1 thinking-contract gate (admission override-rejection, cache-salt per request, purity across reasoning_content / content) is unexercised by the current workload.

**Gap 3 — L2 enforcement is partial.** Sprint-2 traces record `request_shaping_enforcement.mode=substrate_measurement_only`, `real_proxy_enforcement=false`. Of the five L2 action-space fields — `concurrency_cap_eval`, `concurrency_cap_rollout`, `admission_queue_depth_max`, `per_request_kv_budget`, `priority_preemption` — **only `concurrency_cap_eval`** actually affects the harness (via `target_concurrency`). The other four are validated, written to the trace, and never reach the inference proxy. The Sprint-2 search is sampling metadata, not behavior.

Any additional round against the current substrate reproduces the same three gaps. Hardening first is cheaper than another round.

---

## 1.5 Workstream 0 — v0.1.12 loop-contract cleanup (prerequisite)

**Why this workstream exists (P1-GG fix).** The plan's §0 originally claimed v0.1.11 loop mechanics were clean. A follow-up review surfaced three live contract blockers that must land before A/B/C can build on top of a stable loop:

- **AR.2 and AR.13 scope.** v0.1.11 §9.3.AR.2 ("Every row is real-measured — generator starts with `RealMeasurementHarness`") and §9.3.AR.13 ("Synthetic-fixture rejection") predate the v0.1.10 `--harness synthetic` addition. Both verifier items fail-forever on a fixture round that legitimately uses `SyntheticMeasurementFixture`. **Fix:** scope both items to `--harness real` rounds explicitly, and add a symmetric AR for `--harness synthetic` rounds (every row must be `SyntheticMeasurementFixture`, `Fixture-Mode: true` trailer on every commit).
- **`ROUND_BUNDLE_READY` outcome value undefined.** The 2026-04-24 `round_result.json` carries `outcome: ROUND_BUNDLE_READY` but v0.1.11 §11.7 stopping_reason enum lists only `ROUND_PASSED | ROUND_INFEASIBLE | ROUND_BLOCKED | ROUND_BUNDLE_REJECTED`. **Fix:** decide whether `ROUND_BUNDLE_READY` is a fifth outcome (finalize succeeded but live gate was skipped/deferred), or an implementation typo for `ROUND_PASSED`. Add to enum or remove from runtime; either way document explicitly.
- **`run-round` exit semantics vs skill lifecycle.** v0.1.11 §8.8 returns 0 on `ROUND_PASSED`, non-zero on everything else. But `ROUND_INFEASIBLE` is a legitimate outcome per parent §5.7 rail 3 — treating it as a non-zero exit code conflates "honest-no-bundle" with "skill failed." **Fix:** `run-round` exits 0 on `ROUND_PASSED | ROUND_INFEASIBLE` (both are "honest terminal states"), non-zero on `ROUND_BLOCKED | ROUND_BUNDLE_REJECTED | ROUND_BUNDLE_READY-needing-live-gate`. Callers distinguish via the `outcome` field in the JSON report.

**Sub-spec changes (v0.1.12).** §11.7 stopping_reason enum gains `ROUND_BUNDLE_READY` (or removes it if typo). §8.8 `run-round` exit-code contract expanded. §9.3.AR.2 and §9.3.AR.13 rewritten to scope to `--harness real`; new §9.3.AR.2b and §9.3.AR.13b added for `--harness synthetic` symmetry.

**Effort.** ~1 day. Text-only sub-spec change + matching test assertions; no new implementation surface.

**Blocks:** Workstream A, B, C. Must land first.

---

## 2. Workstream A — Measurement hardening

### 2.1 Workload hardening — multi-family v5 sampling + thinking-mode diagnosis

**Problem.** Two orthogonal failures combine to make the current workload a poor signal:
- **Zero thinking tokens throughout** (see §1 Gap 2 and follow-up diagnosis). The existing `seed_trace.jsonl` has `thinking_tokens: 0` on every row; the harness reads-through, so measurement traces also show 0.
- **Single-family sampling is arbitrary.** The repo has **55 landed families** of which **28 have v5 (hardened) variants**. Picking just `proposal-ranking-manager-judgment` — a concrete task that's empirically not reasoning-heavy — is (a) likely overfitting to one family's easy prompts, (b) non-representative of the diversity of work the serving stack actually does in production, (c) the reason the current signal is both low-variance-bad (noise floor 25% of mean) and thinking-vacuous.

**Action — three subsections.**

#### 2.1.1 Pre-launch thinking probe (diagnostic, not capture)

Before any seed-trace work, confirm thinking fires at the serve layer. This is a **15-minute curl probe run manually or by a bootstrap precondition**, not a new long-running test.

**Case matrix.** Run two `/v1/responses` calls against the running vLLM:

```
# (A) Short concrete prompt, no override — matches today's codex path
curl -sX POST $BASE_URL/responses \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-27b","input":"Summarize: AI is useful.","max_output_tokens":2048}' \
  | jq '.usage'

# (B) Reasoning-heavy prompt + explicit thinking via chat_template_kwargs
curl -sX POST $BASE_URL/responses \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3.5-27b",
    "input":"Prove that sqrt(2) is irrational. Show every step.",
    "max_output_tokens":8192,
    "extra_body":{"chat_template_kwargs":{"enable_thinking":true}}
  }' | jq '.usage'
```

**Outcome matrix:**

| (A) reasoning_tokens | (B) reasoning_tokens | Diagnosis | Required fix before §2.1.2 |
|---|---|---|---|
| 0 | > 0 | Pipeline does not propagate thinking override; current codex path doesn't ask for it. | Capture script must send `extra_body.chat_template_kwargs.enable_thinking=true`; family-level codex configs must be audited. |
| 0 | 0 | **Blocker — thinking is genuinely off at serve layer.** Parser present, generation absent. Either chat template mismatched for 3.5, or vLLM shim strips the kwargs. | **Hardening plan pauses.** Investigate vLLM version, chat template path, and shim. Do not proceed until (B) produces non-zero. |
| > 0 | > 0 | Thinking fires when asked; current family is simply too easy to invoke it by default. | §2.1.2 multi-family v5 sampling includes reasoning-heavy families that naturally invoke thinking. |
| > 0 | 0 | Should not happen — (B) adds explicit override, can't be worse than (A). | Report as vLLM shim bug; investigate before proceeding. |

**Who runs this.** Day-0 operator task. Probe results recorded in `reports/thinking-probe-<yyyymmdd>.md` and referenced from the round's `round_spec.yaml.serving_thinking_probe`. A round cannot bootstrap until the probe has landed within the last 7 days and passed (case-row-1 or case-row-3).

**Implementation surface.** New `scripts/probe_serving_thinking.py` that wraps the two curl calls and writes the report. Phase A deliverable addition (item 10 of §5.1 impl brief in v0.1.12).

**Sub-spec changes (v0.1.12).** §11.1 production precondition gains item 10: "recent (< 7 days) thinking probe report exists at `reports/thinking-probe-*.md` and records outcome-row-1 or outcome-row-3." New verification item §9.3.AR.33 (§7 below).

#### 2.1.2 Multi-family v5 composite seed workload

**Rationale.** With 28 v5-landed families, sampling across them gives:
- Diversity of prompt complexity → measurement signal less family-overfit
- Natural coverage of thinking-heavy and non-thinking tasks → the aggregate seed trace has real thinking-token distribution without requiring a single-family rewrite
- Hardened (v5) content is the most representative of what production campaign traffic looks like, since v5 is the CNB-55 progression's final-variant state (see `CNB-55-authoring-spec.md` / `CNB55_FAMILY_CHECKLIST.md`)
- Throughput signal becomes "what does the serving stack do across the real workload portfolio," not "what does it do for proposal-ranking-brief specifically"

**Pool definition (P1-OO fix — eligibility on real artifacts that exist today).** A family is **eligible for the composite** iff all three checks pass against the v0.1.11-landed repo state:

1. `benchmark_blueprints/families/<family>/verification_matrix_v5*.md` exists on `main`. (Verified across 28 of 55 families at spec-authoring time.)
2. `benchmark_blueprints/families/<family>/codex/config.toml` exists and specifies `reasoning_effort` + a `wire_api` setting.
3. `benchmark_blueprints/families/<family>/family.yaml` marks the family as v5-landed (via one of the existing flywheel-ready indicators: `rawr_status: flywheel_ready`, or the presence of a `manifest.lock.json` alongside a live-probe record). **The plan does NOT require a `representative_trajectory_ref` field** — that field doesn't exist on any landed family and would be a chicken-and-egg blocker. The v5 oracle-trajectory data is **produced** by §2.1.3 per-family capture, not consumed as an input.

At spec-authoring time this enumerates **28 families**. The other 27 are v1–v4 or not-yet-hardened and stay out of the v0.1.12 pool; they'll join as they reach v5.

**Stratification math (P1-NN fix).** The composite must satisfy: every eligible family appears in **both** seed and holdout. Minimum-feasible defaults:

- Per-family capture: **≥ 4 turns** captured from each family's v5 oracle trajectory. (v5 trajectories typically have 5–20 turns; this is a minimum, not a cap.)
- Seed-vs-holdout split: **3 seed : 1 holdout per family** (75/25 per-family stratification).
- Pool of 28 families → seed total **84 rows minimum** (28 × 3), holdout total **28 rows minimum** (28 × 1).
- Aggregate holdout ratio ≈ 25% (not the 15% the prior draft claimed — 15% was arithmetically infeasible given stratification).

Families whose v5 oracle trajectory has fewer than 4 turns are either excluded from the pool for that capture run (recorded in `workload.yaml.pool_excluded_families` with the reason) or — operator's choice — resampled from the same trajectory's turns with replacement to reach 4. Default: exclude, require minimum. Prevents a short-trajectory family from dominating the composite by replication.

**Composite capture script (new).** `scripts/capture_multi_family_v5_workload.py`:

```
lumoserve serving-backend capture-multi-family-v5-workload \
  --pool-file benchmark_blueprints/workloads/multi-family-v5/pool.yaml \
  --samples-per-family <int, default 4> \
  --split-per-family "3:1"              # seed:holdout — default stratified 3:1 per family
  --split-seed <int, default 2026042401> \
  --min-trajectory-turns <int, default 4> \
  --require-thinking-probe <path to probe report>
```

Effects, in order:
1. Read `pool.yaml` — the enumerated list of eligible families produced by the §2.1.2 eligibility scan over `main`.
2. For each family in the pool, read the per-family seed trace at `benchmark_blueprints/families/<family>/seed_trace_v5.jsonl` (produced by §2.1.3 Phase 1 per-family captures). If the per-family trace has fewer than `min-trajectory-turns`, record the family in `pool_excluded_families` and skip.
3. Per eligible family, take `samples_per_family` rows from the family's trace (first N rows, or uniformly-sampled subject to `split_seed` reproducibility).
4. Apply per-family stratified split: first `split_per_family[0]` rows into seed, next `split_per_family[1]` rows into holdout. With the default `3:1`, every family contributes 3 rows to seed and 1 row to holdout.
5. Concatenate per-family contributions → `seed_trace.jsonl` + `holdout_trace.jsonl`. Shuffle within each file using `split_seed` so family identity isn't correlated with row order.
6. Write `benchmark_blueprints/workloads/multi-family-v5/workload.yaml` — the composite workload descriptor (see below).
7. Compute `workload_distribution_id` via the canonical non-circular procedure defined in parent HLD §5.9 (P1-YY): hash seed_trace.jsonl, hash holdout_trace.jsonl, hash workload.yaml **after setting `workload_distribution_id: null` in the yaml before canonicalization** (sort_keys + default_flow_style so the yaml hash is stable under comment/key-order edits), then hash the three hex-digest concatenation. Write the resulting id back to `workload.yaml.workload_distribution_id`. Verification re-runs the same procedure and compares.

**Composite workload descriptor.** New file `benchmark_blueprints/workloads/multi-family-v5/workload.yaml`:

```yaml
family_id: multi-family-v5              # P1-MM fix — composite family_id per parent §5.9/§6.4
workload_distribution_id: <sha256>
workload_distribution_id_hardening_version: v1-multi-family-v5-thinking-realistic
capture_date: <iso8601>
pool_size: <int>                        # 28 at v0.1.12
pool_families: [<family_id>, …]         # the eligible family_ids actually used
pool_excluded_families: [{family_id, reason}, …]   # e.g. "trajectory_turns_below_minimum"
samples_per_family: 4
split_per_family:
  seed_rows: 3
  holdout_rows: 1
total_seed_rows: 84                     # pool_size × split_per_family.seed_rows
total_holdout_rows: 28                  # pool_size × split_per_family.holdout_rows
thinking_probe_ref: reports/thinking-probe-<yyyymmdd>.md
thinking_probe_outcome: row-1 | row-3   # from §2.1.1
seed_trace_ref: seed_trace.jsonl
holdout_trace_ref: holdout_trace.jsonl
nominal_ttft_ms: 2000                   # advisory — not a feasibility gate (§2.3)
nominal_tpot_ms: 80                     # advisory
nominal_turn_ms: 30000                  # advisory
target_concurrency: 4                   # fixed per sub-spec v0.1.11 §9
```

**Target distribution in the composite seed + holdout traces** (minimums, auto-verified by §9.3.AR.26; P2-DDD fix — numbers match the 28-family × 3:1-per-family stratified split):
- ≥ 30% of rows with `thinking_tokens > 0` across the whole composite — should be satisfied automatically once at least ~9 of the 28 pool families are reasoning-heavy.
- ≥ 10% of rows with `thinking_tokens > response_tokens`.
- **Per-file family coverage** — `set(seed_families) == pool_families` AND `set(holdout_families) == pool_families` (strict equality per AR.26 P1-UU fix, not just union).
- **Per-family row counts** — every family contributes exactly `split_per_family.seed_rows = 3` seed rows and `split_per_family.holdout_rows = 1` holdout row by default (adjustable via `--split-per-family` if a pool family has extra trajectory turns; never fewer).
- **Aggregate counts** — seed row count **≥ 84** (28 families × 3 seed rows per family at v0.1.12); holdout row count **≥ 28** (28 × 1). These are strictly larger than the v0.1.1-plan draft's "≥40 / ≥8" — those numbers predated the stratification math and would have failed the per-file set-equality check.

**Archival.** Current `benchmark_blueprints/families/proposal-ranking-manager-judgment/seed_trace.jsonl` and `holdout_trace.jsonl` move to `seed_traces_archive/v1-single-family-no-thinking/`. They are not referenced by any round after v0.1.12 lands.

**Sub-spec changes (v0.1.12).** §0 scope — target tuple's `family_id` field keeps its name, but its value for hardened rounds is **`multi-family-v5`** — a composite-workload value that resolves under `benchmark_blueprints/workloads/multi-family-v5/workload.yaml` instead of `families/<name>/serving_workload.yaml`. Parent HLD §5.9 and §6.4 have been amended to explicitly permit composite `family_id` values; bundle storage path (`output/tuned_configs/<family_id>/…`), §6.4 hard-pin, admission-layer routing, and the entire bundle-validity contract stay verbatim. The value `multi-family-v5` is simply a composite family that lives under `workloads/` instead of `families/` — the runtime doesn't distinguish. §5.1 impl-brief precondition: composite workload artifacts exist on `main`. §11.1 production precondition: the **resolved workload descriptor** (per parent HLD §5.9 lookup order — `workloads/<family_id>/workload.yaml` preferred, `families/<family_id>/serving_workload.yaml` fallback — recorded in `round_spec.yaml.workload_descriptor_path`) has `workload_distribution_id_hardening_version == v1-multi-family-v5-thinking-realistic` for composite rounds or `v1-thinking-realistic` for legacy single-family rounds. P1-AAA fix: this paragraph previously hard-coded `serving_workload.yaml`, which would have reintroduced the composite-path bug fixed separately in §2.5.

**Effort.** ~3 days.
- Day 1: Author `capture_multi_family_v5_workload.py` with pool.yaml parser + per-family capture loop.
- Day 2: Run capture against real default-config serving against the 28-family pool. Realistic per-family capture time: ~2 min × 60 rows + serving startup = ~90 min. Cacheable if re-run.
- Day 3: Validate composite distribution against minimums; iterate pool weights if thinking coverage falls short.

#### 2.1.3 Per-family seed traces (prerequisite to §2.1.2)

Today only `proposal-ranking-manager-judgment/seed_trace.jsonl` exists. The other 27 v5 families don't have their own seed trace yet. §2.1.2's composite capture script CAN capture per-family on the fly, but it's slower and non-reusable. Better to split:

**Phase 1 (week 1):** `scripts/capture_seed_workload.py` is extended to `--family-id <name> --variant <v5>` and stores per-family traces at `benchmark_blueprints/families/<family>/seed_trace_v5.jsonl`. Run once per family in the pool.

**Phase 2 (week 2):** `capture_multi_family_v5_workload.py` reads per-family traces and composes them into the workload trace. Fast, reproducible, re-runnable without re-invoking serving.

This also means families that aren't yet v5-landed can be added to the pool at their own pace without re-capturing the whole composite.

**Per-family capture acceptance:** each family's `seed_trace_v5.jsonl` must have ≥ 4 rows and match the thinking-probe outcome (if case-row-1, capture uses `chat_template_kwargs` override; if case-row-3, no override needed).

**Sub-spec changes (v0.1.12).** §9.3.AR.26 covers per-family traces + composite + holdout (all three must meet the distribution target).

**Combined effort for §2.1.** ~4–5 days including per-family captures. Biggest risk: if the probe returns case-row-2 (thinking off entirely at serve layer), the whole plan blocks until that's fixed.

### 2.2 Statistical power — n≥5 baseline, n≥4 per rescreen candidate (profile-aware)

**Problem.** Two things:
- **Sample size.** v0.1.11 §4.4(a) computes noise_floor from n=2 baselines using `2 × |M₁ − M₂|`. With a 16% baseline-to-baseline spread and a sub-10% signal, 2 samples can't defensibly estimate σ. §4.4(b) rescreen uses n=2 per top-K candidate (1 screen + 1 rescreen). Same problem compounded.
- **Profile mismatch (P1-HH fix).** v0.1.11 runs baselines and main-loop candidates at **Screen profile (15 min)** but rescreens top-K at **Full profile (33 min)**. Different profile = different warmup length + different measurement window length → different numbers. Computing "candidate's `objective_mean` across 1 Screen + N Full" against "baseline at Screen only" mixes dimensions. Confidence derivation (§4.2) was referencing an undefined `baseline_rescreen_rows` as if baseline had been rescreened at Full too — it hadn't, and fixing the mismatch either means running baseline at Full (expensive) or keeping confidence derivation on Screen-only measurements.

**Action — Screen-profile-first confidence, Full-profile as a sanity cross-check only.**

**For baselines (Screen profile).** Replace double-baseline with **quintuple-baseline at Screen profile**. Run baseline 5 times, all configurationally identical (same yaml, different `candidate_uuid`, all at Screen profile). Compute:
- `baseline_mean_screen = mean(M₁..M₅)`
- `baseline_stddev_screen = stddev(M₁..M₅)`
- `noise_floor = 2 × baseline_stddev_screen` (approximates 95%-CI half-width conservatively; a proper t-distribution correction is `2.78 × stddev/√5` = ~1.24 × stddev, so 2× stddev is a deliberate-conservative choice)

Persist `baseline_mean_screen`, `baseline_stddev_screen`, `noise_floor` into `round_spec.yaml`.

**For candidates (Screen profile — same as baseline).** Top-K candidates are re-measured **3 additional times at Screen profile** (matching baseline), producing n=4 total Screen measurements per candidate (1 original main-loop + 3 Screen rescreens). Compute:
- `objective_mean_screen = mean(m₁..m₄)` for each candidate
- `objective_ci_95_screen = 2.78 × stddev/√4`
- `improvement_over_baseline_screen = objective_mean_screen − baseline_mean_screen`

Candidate confidence derives **exclusively from Screen-profile measurements**. Baseline and candidate are compared at the same profile; there's no profile mixing in the statistic (P1-HH fix).

**For candidates (Full profile — sanity cross-check only).** Each top-K candidate *also* gets **1 Full-profile rescreen** as a sanity measurement. The Full-profile measurement is recorded under `status: rescreened_full` (new sub-status), does not enter `objective_mean_screen`, and is used only to check:
- Does the Full-profile measurement fall inside `[objective_mean_screen ± 3 × stddev_screen]`? If not, flag the candidate `status: rescreened, notes: screen_full_divergence` — a warning that extending the window to 25 min changed the answer, which suggests the Screen profile is missing some steady-state effect worth investigating. Flagged candidates are not auto-removed from contention but are surfaced in the finalize report.

This keeps the primary statistic clean (profile-matched) while preserving the Full-profile "is this robust to a longer window?" check that §4.4(b) v0.1.11 was trying to get at.

**Flags (same as v0.1.11 plus one v0.1.12 addition):**
- `inconsistent_rescreen` — within-candidate Screen spread exceeds `noise_floor × 2`. Removed from winner contention.
- `high_variance_rescreen` — same as above for the Screen measurements; alias kept for v0.1.11 compatibility.
- `screen_full_divergence` (v0.1.12 new) — Full-profile measurement outside Screen CI, warning only.

**Budget math, properly done** (P1-II fix — single table, no double-count).

Previous budget (v0.1.11): 8h cap, ~6.5h active = 0.5h (baseline×2 Screen) + 3h (12×Screen main-loop) + 1.6h (3 × 1 × Full rescreen) + 0.5h (1 × Full holdout) + overhead.

**Hardened budget (v0.1.12) — one table, one recommendation.** Keep `iteration_cap = 12` (not reduced — that was the phantom Option A3 double-count). Baseline bumps to 5. Rescreen bumps to 3 Screen + 1 Full per candidate. K_rescreen = 3. Round wall-clock cap raised to **12 hours** (the honest choice).

| Phase | Count × Profile | Wall-clock |
|---|---|---|
| Baseline | 5 × Screen (15 min) | **1.25 h** |
| Main loop | 12 × Screen (15 min) | 3.0 h |
| Rescreen Screen (new primary) | 3 candidates × 3 measurements × Screen (15 min) | **2.25 h** |
| Rescreen Full (sanity cross-check) | 3 candidates × 1 measurement × Full (33 min) | 1.65 h |
| Holdout | 1 × Full (33 min) | 0.55 h |
| Overhead (bootstrap, finalize, gate) | — | ~1 h |
| **Total active** | — | **~9.7 h** |
| **Round cap (v0.1.12)** | — | **12 h** |

**No alternative budget options at v0.1.12.** The table above is the single hardened budget. If an operator needs a smaller round for CI, `run-round --harness synthetic` against the composite fixture is already the answer — no need to parameterize real-round budgets further.

**Sub-spec changes (v0.1.12).** §4.1 budget reconciliation table replaced with the above. §4.4(a) mechanism becomes "quintuple-baseline at Screen profile." §4.4(b) mechanism becomes "top-K rescreened at n=3 Screen + 1 Full, winner derived from Screen-only `objective_mean`." New `screen_full_divergence` flag. §8.4 `rescreen` CLI gains `--measurements-per-candidate-screen <int>` (default 3) and `--measurements-per-candidate-full <int>` (default 1). Verification items §9.3.AR.16 (noise floor) + §9.3.AR.17 (rescreen lineage) updated.

**Effort.** ~1 day sub-spec writing + ~half day of runner changes (adding the Screen/Full split in rescreen). Math is simple; wiring is just two nested loops in rescreen logic.

### 2.3 Latency-gate decision — soft gate, metadata-only

**Problem.** v0.1.11 §4.2 drops three-dim latency SLO entirely. C003's measured TTFT p95 was 109 s, turn p95 was 311 s. Both are wildly outside the family's nominal SLOs (2 s / 30 s). The round correctly marked C003 feasible under throughput-only semantics, but nothing in the bundle flags that the winner is user-unusable.

**Options considered:**
- **(a) Soft gate.** Record latency, don't refuse. Add `round_provenance.latency_above_slo: bool` so bundle consumers can see it.
- **(b) Hard gate at realistic ceilings.** Re-introduce latency gating at measured-reality ceilings (e.g. TTFT ≤ 300 s, turn ≤ 600 s — 10× nominal SLO, matches what default-config serves).
- **(c) Throughput-only, explicit doc.** Status quo, but add explicit doc that latency is not a feasibility dimension in v0.1.

**Recommendation: (a) soft gate.** Rationale:
- (b) sets a precedent for moving ceilings to match reality, which is the wrong direction — "SLOs" become meaningless once they're calibrated to current performance.
- (c) leaves bundle consumers without any latency signal in the bundle itself; they have to go read `measurement_trace.json` to know.
- (a) preserves throughput-primary optimization (v0.1.8 direction) but surfaces the latency reality as a first-class bundle field. Consumers that care about latency can gate on `round_provenance.latency_above_slo == false` at load time.

**Implementation.** `finalize-round` reads the winner's rescreen traces, computes `max(ttft_p95) > nominal_ttft_ms || max(turn_latency_p95) > nominal_turn_ms` (where the `nominal_*_ms` fields come from the **resolved workload descriptor** recorded in `round_spec.yaml.workload_descriptor_path` per §2.5 — which is `workloads/<family_id>/workload.yaml` for composite rounds, `families/<family_id>/serving_workload.yaml` for legacy), writes the result to `round_provenance.latency_above_slo`. The nominal ceilings are **advisory**, not gates.

**Sub-spec changes (v0.1.12).** §4.3 gains the soft-gate language. §5.9 parent `round_provenance` gets the `latency_above_slo` field. §8.6 `finalize-round` effects list gains the computation. New verification item §9.3.AR.27.

**Effort.** ~4 hours. Simple computation, field addition, test.

### 2.4 Workload descriptor schema — nominal ceilings as advisory (P2-BBB fix — resolved descriptor, not hard-coded `serving_workload.yaml`)

**Problem.** `serving_workload.yaml` at v0.1.8 removed `L_ttft_ms`, `L_tpot_ms`, `L_turn_ms`. §2.3 above needs them back, but as advisory (for soft gate), not mandatory gates. **Additionally, composite hardened rounds use `workloads/<family_id>/workload.yaml`, not `serving_workload.yaml`** — the ceilings must live in whichever descriptor the round resolved.

**Action.** Add three fields to the **resolved workload descriptor** (per parent HLD §5.9 lookup order — `workloads/<family_id>/workload.yaml` for composite family_ids, `families/<family_id>/serving_workload.yaml` for legacy single-family) as **nominal ceilings**:

```yaml
nominal_ttft_ms: 2000         # advisory — not a feasibility gate at v0.1
nominal_tpot_ms: 80           # advisory
nominal_turn_ms: 30000        # advisory
```

These fields are read by `finalize-round` from `round_spec.yaml.workload_descriptor_path` (populated by `bootstrap-round` per §2.5) for §2.3's `latency_above_slo` computation. The old `L_ttft_ms` / `L_tpot_ms` / `L_turn_ms` names are **not** reintroduced — they implied gates. The `nominal_` prefix makes advisory-nature explicit. Composite workload descriptors already include these fields in the §2.1.2 `workload.yaml` sample.

**Legacy single-family rounds** retain the same fields in `serving_workload.yaml`; the precondition and finalize reads are file-agnostic via the resolved-descriptor path.

**Sub-spec changes (v0.1.12).** §0 divergence note updated: SLO is still not a gate, but nominal ceilings re-appear as diagnostic advisories. §4.3 references them. §2.5 `workload_descriptor_path` is the authority on *where* they come from (composite vs legacy).

**Effort.** ~1 hour. Yaml edit + load-path in harness.

### 2.5 Precondition — workload id version (P1-VV fix — resolved descriptor, not always `serving_workload.yaml`)

**Problem.** If we change the seed trace (§2.1) but leave the workload yaml id unchanged, a round bootstrapped from an older yaml against the new trace produces silently-incorrect `workload_distribution_id` in its bundle.

**Resolved workload descriptor (v0.1.3-plan).** Per parent HLD §5.9's composite `family_id` amendment, the runtime resolves the workload descriptor in this order:

1. **Composite path:** `benchmark_blueprints/workloads/<family_id>/workload.yaml` (preferred; exists for composite family_ids like `multi-family-v5`).
2. **Single-family fallback:** `benchmark_blueprints/families/<family_id>/serving_workload.yaml` (for legacy single-family rounds; pre-hardening convention).

The precondition reads from **the resolved file**, not from `serving_workload.yaml` specifically. A composite round (`family_id: multi-family-v5`) reads the `workload_distribution_id_hardening_version` field from `workloads/multi-family-v5/workload.yaml`; a single-family round reads from `families/<name>/serving_workload.yaml`. Both files carry the same field name. Implementers must update whichever file their round points at — updating the wrong file will leave the precondition reading the stale version.

**Action.** Add **`workload_distribution_id_hardening_version`** (exact field name; P1-HHH fix — the prior text said "`version`" generically, which could have been read as a different or non-existent field) to the resolved workload descriptor (composite or single-family). `bootstrap-round` Phase A precondition: if `resolved_descriptor.workload_distribution_id_hardening_version != "v1-multi-family-v5-thinking-realistic"` (for composite) or `!= "v1-thinking-realistic"` (for single-family legacy) AND the round is not running under `--allow-legacy-workload`, refuse. Production rounds must run on the hardened workload.

**Sub-spec changes (v0.1.12).** §11.1 production preconditions gains this as item 10a with resolved-descriptor language. `--allow-legacy-workload` flag on `bootstrap-round` for operators who explicitly want to replay an old workload (e.g., regression reproducing). `bootstrap-round` internally resolves the workload descriptor per parent HLD §5.9 order and records `round_spec.yaml.workload_descriptor_path` so downstream verifiers know which file was read.

**Effort.** ~2 hours.

---

## 3. Workstream B — L2 request-shaping enforcement

### 3.1 Current state audit — what actually works

Per Sprint-2 trace inspection (`request_shaping_enforcement.mode=substrate_measurement_only`, `real_proxy_enforcement=false`):

| L2 field | Validated? | Recorded? | Actually enforced? |
|---|---|---|---|
| `concurrency_cap_eval` | yes | yes | **yes** — maps to `RealMeasurementHarness.target_concurrency` |
| `concurrency_cap_rollout` | yes | yes | no |
| `admission_queue_depth_max` | yes | yes | no |
| `per_request_kv_budget` | yes | yes | no |
| `priority_preemption` ∈ `{off, strict, graceful}` | yes | yes | no |

**Sprint-2 trace evidence.** C001 with `concurrency_cap_eval=3`: `sustained_concurrency=3.0`. C002–C008 with shaped queue/KV/preemption fields but `concurrency_cap_eval=4`: all measured `sustained_concurrency=4.0`. The four un-enforced fields have zero effect on measurement.

### 3.2 Enforcement plan per field

Each un-enforced field needs a specific inference-proxy change. Estimating effort and risk per field.

**`concurrency_cap_rollout` — medium effort.**
- Requires separating eval vs rollout traffic in the admission layer.
- Class labeling already exists in LLD-SB-02 telemetry (`class` label on Prometheus metrics); just needs to become an admission routing input.
- Separate semaphore per class, both semaphores accounted toward `max_num_seqs`.
- **v0.2 in-scope.** Effort ~3 days.

**`admission_queue_depth_max` — medium effort.**
- Bounded-queue admission with structured 429 rejection when full.
- LLD-04 already has an admission layer (parent §4.4); needs explicit depth tracking + bound.
- Overflow behavior: return HTTP 429 with `Retry-After` header and `error.code = queue_full`.
- **v0.2 in-scope.** Effort ~2 days.

**`per_request_kv_budget` — stays advisory at v0.2 (P1-KK fix).**
- Originally planned as "implement via `max_tokens` override" (Option B.1). The reviewer's correct objection: KV usage is `prompt_tokens + prefix_cache_tokens + response_tokens`, not just response length. Capping output tokens (`max_tokens`) bounds only the response portion — prompts of 4k tokens consume KV whether or not the response is capped. Marking `per_request_kv_budget` as "enforced" when we're really bounding output length would be a silent contract violation that mis-advertises what the proxy actually does.
- Real enforcement requires vLLM engine-level KV accounting (block-manager-aware admission that refuses requests whose projected KV footprint exceeds the per-request budget). That's v0.3+ work; potentially requires upstream vLLM changes.
- **v0.2: `per_request_kv_budget` stays as an *advisory* field — validated, recorded in the trace with `enforcement: advisory`, but the proxy does not act on it.** A v0.3 revision swaps it to enforced when real KV accounting lands.
- A separate output-only knob — call it `max_response_tokens` — is not added in v0.2 either; we have `max_model_len` + workload-trace per-turn `output_tokens` which together bound response length well enough for v0.2 purposes.

**`priority_preemption` ∈ `{off, strict, graceful}` — advisory at v0.2, v0.3 decision point.**
- Requires vLLM scheduler hooks. Upstream vLLM does not expose this as a runtime-configurable policy on our pinned version.
- `strict` means: preempt any in-flight rollout request when an eval request arrives.
- `graceful` means: let rollout requests finish their current step but don't schedule new rollout requests while eval traffic is active.
- Both require scheduler changes vLLM doesn't currently support.
- **v0.2: drop from the action space entirely and note `enforcement: advisory` if a caller sets it. v0.3: either patch vLLM upstream or accept permanent advisory status.**

### 3.3 Scope for v0.2 L2 re-run

Given §3.2 effort estimates and the "hardening first" theme:

**Enforced fields (3) — v0.2 L2 action space.** `concurrency_cap_eval` (already works), `concurrency_cap_rollout`, `admission_queue_depth_max`. The rescreen + finalize paths operate over these three; the bundle's L2 config block records these three as `enforced`.

**Advisory fields (2) — validated, recorded, not enforced.** `per_request_kv_budget` (awaiting real KV accounting per §3.2), `priority_preemption` (awaiting vLLM scheduler hooks). Callers may set them; the harness validates and records them with `enforcement: advisory` per field in the trace; the `rescreen` top-K selection ignores them; `finalize-round` includes them in the bundle's L2 config block under a separate `advisory_fields:` section so downstream consumers don't mistake recording for enforcement.

### 3.4 Acceptance criteria

Every L2 trace after the fix must satisfy:
- `request_shaping_enforcement.mode` ∈ `{enforced, enforced_minus_advisory}` (never `substrate_measurement_only`).
- `request_shaping_enforcement.real_proxy_enforcement: true`.
- `request_shaping_enforcement.enforced_fields` lists the three enforced fields: `concurrency_cap_eval`, `concurrency_cap_rollout`, `admission_queue_depth_max`.
- `request_shaping_enforcement.advisory_fields` lists the two advisory fields if they appear in the candidate: `per_request_kv_budget`, `priority_preemption` (each with the actual value recorded but marked `enforcement: advisory`).

Sub-spec changes (v0.1.12). §8.4 `rescreen` acceptance check. New verification item §9.3.AR.28 — every L2 trace has `real_proxy_enforcement: true` and the enforced-field set matches exactly the three listed. New §10.1 subsection describing L2 field coverage as three enforced + two advisory.

### 3.5 Effort total

~5 days (3 for `concurrency_cap_rollout` + 2 for `admission_queue_depth_max`) for workstream B. Plus ~2 days of integration testing (full L2 round against fixture + a real small L2 round to verify). **Call it 1.5 weeks.** P1-KK fix drops 4 days of `per_request_kv_budget` Option B.1 work from the original estimate — that knob is now advisory until real KV accounting lands in v0.3+.

### 3.6 What to do with the in-flight Sprint-2 round

`.round.lock` held at `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-2-20260424T163930Z/`. 10 result rows through C008, no finalize.

**Recommendation.** **Kill it now, preserve the artifacts, do not finalize.** Rationale:
- The round can't produce a usable bundle — every row has `real_proxy_enforcement: false`.
- Finalizing anyway would emit a bundle with `round_provenance.dry_run: false` but effectively-empty shaping fields, which is worse than no bundle (a consumer couldn't tell it was exploratory).
- The partial artifacts are valuable for post-mortem — keep them.

**Kill procedure.** One CLI command: `lumoserve auto-research status --round-id <id>` to confirm state, then delete `.round.lock` and write a `BLOCKED.md` with reason `manual_abort_pending_l2_enforcement_fix`. No new CLI is needed. The round branch is left where it is; a future post-mortem reads it. Operators never touch `main`.

This is operational, not a spec change.

---

## 4. Workstream C — Bundle honesty metadata

### 4.1 New `round_provenance` fields

Parent §5.9 bundle schema gains four fields inside `round_provenance`:

```yaml
round_provenance:
  # ... existing v0.1.8 fields (dry_run, round_id, round_branch, …) …
  confidence: defensible | within_noise_floor | exploratory | unknown  # v0.1.12
  improvement_over_baseline_req_per_s: <float>                          # v0.1.12 — Screen-only per §4.2
  improvement_over_baseline_ci_95: [<float_low>, <float_high>]          # v0.1.12 — Welch-t 95% CI
  latency_above_slo: <bool>                                             # v0.1.12 (§2.3)
  screen_full_consistency: consistent | divergent                       # v0.1.12 (§4.2 full-profile cross-check; P2-WW fix)
  l2_enforcement_coverage:                                              # v0.1.12 (§3.4)
    enforced_fields: [<field>, …]
    advisory_fields: [<field>, …]
    mode: enforced | enforced_minus_advisory | substrate_measurement_only | not_l2
  workload_descriptor_path: <path>                                      # v0.1.12 (§2.5 P1-VV resolved descriptor)
```

### 4.2 How `finalize-round` populates them

Computation rules at finalize time:

**`confidence` derivation (Screen-only per §2.2; P1-PP fix — no profile mixing).**

Inputs:
- `baseline_screen_measurements`: the 5 `eval_throughput` values from the 5 baseline rows (all Screen profile, per §9.3.AR.29).
- `winner_screen_measurements`: the 4 `eval_throughput` values for the winner candidate's 1 main-loop measurement + 3 Screen rescreens (per §9.3.AR.30). **Full-profile rescreen measurements are NOT included here** — they populate `screen_full_divergence` diagnostics only.
- `noise_floor`: from `round_spec.yaml`, equals `2 × stddev(baseline_screen_measurements)` per §9.3.AR.29.
- `winner_row`: the winner's main-loop row (for reading `notes` for `inconsistent_rescreen` / `high_variance_rescreen` / `screen_full_divergence` flags).

```python
def derive_confidence(
    baseline_screen_measurements: list[float],   # n=5
    winner_screen_measurements: list[float],     # n=4
    noise_floor: float,
    winner_row: ResultsRow,
) -> str:
    # Sample-size gate.
    if len(baseline_screen_measurements) < 5 or len(winner_screen_measurements) < 4:
        return "unknown"

    # Within-candidate variance gate (Screen-only spread).
    if winner_row.notes in {"inconsistent_rescreen", "high_variance_rescreen"}:
        return "exploratory"

    # Compute Screen-only paired comparison.
    baseline_mean_screen = mean(baseline_screen_measurements)
    objective_mean_screen = mean(winner_screen_measurements)
    delta_screen = objective_mean_screen - baseline_mean_screen

    # Welch-t 95% CI for difference of means (n_b=5, n_c=4).
    ci_low, ci_high = improvement_ci_95_welch(
        baseline_screen_measurements,
        winner_screen_measurements,
    )

    if ci_low > noise_floor:
        return "defensible"           # lower 95% bound beats noise floor
    if delta_screen > 0 and ci_low > 0:
        return "within_noise_floor"   # improvement is positive but CI crosses noise floor
    return "exploratory"              # no reliable positive improvement
```

**Full-profile cross-check, separate field.** After `derive_confidence` returns, `finalize-round` also computes `screen_full_consistency` using the 1 Full-profile rescreen per top-K candidate. If the Full measurement lies outside `[objective_mean_screen ± 3 × stddev_screen]`, set `round_provenance.screen_full_consistency: divergent` and add a note to `run_log.json`; otherwise `screen_full_consistency: consistent`. This is metadata — it does NOT override or modify `confidence`.

**`latency_above_slo`** — per §2.3. True if any of `max(ttft_p95_ms)`, `max(turn_latency_p95_ms)` across winner's rescreen rows exceeds the nominal ceiling from the resolved workload descriptor (per §2.4 / §2.5 — composite rounds read `workloads/<family_id>/workload.yaml`, legacy reads `families/<family_id>/serving_workload.yaml`).

**`l2_enforcement_coverage`** — populated only for L2 rounds. Read from the winner's trace's `request_shaping_enforcement` block.

### 4.3 How bundle-loader consumes them

The new `round_provenance` honesty fields (`confidence`, `improvement_over_baseline_*`, `latency_above_slo`, `screen_full_consistency`, `l2_enforcement_coverage`, `workload_descriptor_path`) are **operator-facing diagnostics** — not hard pins in parent §6.4. **Separately**, v0.1.3-plan (P1-SS) promoted `workload_distribution_id` to a hard pin and v0.1.4-plan (P1-YY) gave it a canonical non-circular definition — those ARE validity-rule changes. Readers should not conflate the two: honesty fields in `round_provenance` stay metadata; the one pre-existing-schema field that moved from metadata to hard pin is `workload_distribution_id`, not any of the v0.1.12 additions. (P2-III fix — the prior wording "parent §6.4 does not change" was stale after v0.1.3-plan shipped; this paragraph now distinguishes which additions are diagnostics and which §6.4 change actually happened.)

Three production-guardrail options for the loader, picked per deployment:

- **Strict.** `/admin/load_tuned_config` refuses bundles with `confidence != defensible`. Bluntest; safest.
- **Warn-and-log.** Loader logs a structured warning if `confidence != defensible` or `latency_above_slo == true`, but loads anyway. Operator-facing.
- **Pass-through.** Metadata is recorded in serving state but has no behavioral effect. Current-v0.1.11 behavior.

**Recommendation: Warn-and-log for v0.1.12.** `/admin/load_tuned_config` logs a structured warning on non-defensible bundles but loads them. Flip to Strict for v0.2 when confidence derivation has more data to support it. This stays a deployment-time choice via a new loader config `bundle_confidence_policy: strict|warn|passthrough` (default `warn`).

### 4.4 Effort

~2 days. Mostly mechanical — compute a few fields, write them into an existing yaml schema, thread a loader warning.

---

## 5. Decision framework — what to run after hardening

Workstreams A + B land. Then what?

**Step 1: Run one hardened re-measurement round, not a new search.**

Call it **Round H1**. Use the new `replay-round` CLI (§5.1), not `run-round` — H1 is a deterministic re-measurement of a known candidate, so it runs **without codex** and imports C003's config verbatim. Concretely (P1-TT fix — `--import-candidate` takes the original `candidate.yaml`, not the bundle):

```
lumoserve auto-research replay-round \
  --workload-file benchmark_blueprints/workloads/multi-family-v5/workload.yaml \
  --baselines 5 \
  --import-candidate output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T072126Z/candidates/003/candidate.yaml \
  --rescreens-screen 3 \
  --rescreens-full 1 \
  --holdout-rows 28 \
  --round-root output/auto_research
```

The `--import-candidate` flag expects a **candidate.yaml** (raw `max_num_seqs: …` / `gpt_memory_utilization: …` lines). AR.34 byte-compares `candidates/import_001/candidate.yaml` against the imported source, so the source must already be a candidate.yaml shape. Passing a bundle-yaml (which wraps the config inside `tuned_config_bundle.layer_1_vllm.…`) would fail AR.34's byte comparison. If you only have a bundle on hand, add a pre-step to extract its `vllm_config` block to a canonical candidate.yaml before calling `replay-round`, or (future work) introduce a `--import-bundle` variant of `replay-round` that does the extraction itself.

This produces:
- 5 baseline measurements at Screen profile on the composite multi-family-v5 workload → `baseline_mean_screen ± baseline_stddev_screen`.
- 1 Screen + 3 Screen rescreens of C003's imported config → `objective_mean_screen` at n=4.
- 1 Full rescreen of C003 → `screen_full_consistency` diagnostic (§4.2).
- Holdout validation across 28 stratified rows → pass/fail (§4.4-c semantics).
- Finalize → `round_provenance.confidence` derived per §4.2 Screen-only algorithm.

Every row is real-measured against the composite workload. No codex proposes anything; H1 is Python-driven from start to finish. This matches the replay-round contract in §5.1 and avoids the P1-JJ failure mode where `iteration_cap=1` wouldn't actually force C003 through the loop.

**Step 2: Branch on H1 outcome.**

**Case α — H1 shows C003 clearly beats baseline** (`confidence: defensible`): L1 had real signal; the v0.1.11 round was right but noisy. Next:
- Promote C003's config as the new Sprint-0 production bundle (replaces the 2026-04-24 exploratory bundle).
- Start workstream B validation — run a fresh L2 round (Sprint-2 v2) on top of C003 with the enforced shaping fields.
- L0 work stays v0.2+; L1+L2 is a real substrate worth polishing first.

**Case β — H1 shows C003 is within baseline noise** (`confidence: within_noise_floor` or `exploratory`): L1-only is tapped out on this family+hardware. Next:
- Publish C003 as reference-only; keep serving on default config.
- **L2 re-run is almost certainly pointless** — request shaping without latency or kernel improvement will produce similar sub-noise wins.
- The honest next step is **L0 substrate work** (parent §7.2 Sprint 1: DeltaNet + GatedAttn kernel selection + autotune). This is weeks of kernel effort but it's where the per-request TTFT of 109s actually becomes 10s.

**Case γ — H1 shows C003 is worse than baseline** (`improvement_over_baseline_req_per_s < 0` with `ci_95[1] < 0`): The 2026-04-24 round's C003 selection was noise in the wrong direction. Same v0.2 L0 path as case β, plus: treat the 2026-04-24 bundle as an anti-artifact and record it in §11's "what not to do" lessons.

### 5.1 How H1 runs — `replay-round` CLI, not a search round (P1-JJ fix)

Setting `iteration_cap=1` in `run-round` does **not** force C003 through the loop — the per-iteration codex agent writes its own `candidate.yaml` based on its own reading of `iteration_brief.md` + `results.tsv`, and it can propose anything. Making `run-round` honor "measure this exact pre-existing candidate" is a contract mismatch — the whole point of `run-round` is that codex proposes.

H1 therefore uses a **new, separate CLI subcommand** that skips codex entirely:

```
lumoserve auto-research replay-round \
  --workload-file benchmark_blueprints/workloads/multi-family-v5/workload.yaml \
  --baselines 5 \
  --import-candidate <path-to-C003/candidate.yaml> \
  --rescreens-screen 3 \
  --rescreens-full 1 \
  --holdout-rows 28 \
  --round-root output/auto_research \
  [--harness real|synthetic]
```

**`--holdout-rows` default** (P2-XX fix): at v0.1.12 the composite workload has `total_holdout_rows: 28` (28 families × 1 holdout row per family, per §2.1.2). `--holdout-rows` default is **28** and the CLI refuses a value below `pool_size × split_per_family.holdout_rows` (currently `28 × 1 = 28`) — going lower would violate the per-family stratification invariant that every pool family appears in holdout. An operator can pass a larger value to draw additional holdout rows if any pool family has more than 1 remaining trajectory turn after the main split.

**Effects, in order** (v0.1.12 addition):
1. Bootstrap a round with a new `round_id` and git worktree (same machinery as §8.1 `bootstrap-round`).
2. Materialize **N baseline candidates** (N = `--baselines`) with default-config yaml pre-written at `candidates/baseline_[a..e]/candidate.yaml`.
3. Copy the imported candidate verbatim to `candidates/import_001/candidate.yaml` (special iteration_id `import_NNN`, grammar addition in v0.1.12).
4. Python runs: 5 × Screen baseline `measure+commit-candidate` → 1 × Screen measure+commit of the imported candidate → 3 × Screen rescreens + 1 × Full rescreen of the imported candidate (per §2.2) → holdout validation (§8.5) → finalize-round.
5. **No `codex exec` is ever invoked.** No agent session transcripts, no `iteration_brief.md` substitution. Purely Python-driven.
6. The round's FINALIZE commit body records the imported-candidate source: **`imported_from_candidate: <path>`** (a `candidate.yaml` file, not a bundle yaml per P1-TT) and `imported_from_commit: <sha>` (the commit where the source candidate.yaml was authored). AR.34's byte-comparison runs against this path; a bundle-yaml path here would fail AR.34's shape check (see §7 AR.34 artifact note). P2-CCC rename: the prior `imported_from_bundle` name was a holdover from the v0.1.1-plan draft where H1 imported bundle paths — now that replay-round takes candidate.yaml, the field name reflects that.

**Why a separate subcommand and not a flag on `run-round`.** `run-round`'s contract is "Python outer loop spawns codex per iteration." A flag like `--skip-codex-force-candidate <path>` mutates that contract in a confusing way and compromises the §11.1 precondition set. `replay-round` has its own preconditions (harness + git + worktree), its own internal loop (five-baseline + N-rescreen), its own acceptance — it's a cousin of `run-round`, not a mode of it. Keeping them separate keeps v0.1.11's `run-round` contract clean.

**Sub-spec changes (v0.1.12).** New §8.10 `replay-round`. §7.2 iteration-id grammar extension: `^(\d{3}|baseline_[a-e]|rescreen_\d{2}|rescreen_full_\d{2}|import_\d{3})$`. §11.1 precondition: `replay-round` shares common substrate checks (items 1–6) with `run-round` but skips 7a/9a (it doesn't consume codex). New verification item §9.3.AR.34 (§7 below).

**Why H1 is only 1 imported candidate.** The point of H1 is not search. It's to answer one question: "is C003 actually better than default, under hardened measurement?" Search-over-L1 again without first answering that question is wasted compute — you'd be building more decisions on top of an ambiguous finding.

If H1 says "yes, C003 is real," THEN a second **hardened search round** — regular `run-round` with iteration_cap=12 on the multi-family composite workload — is worth running. If H1 says "no," then no further L1 search round is worth running and the path forward is L0 substrate (parent §7.2).

---

## 6. Sequencing

**Day 0 — Kill Sprint-2 + run thinking probe.** Kill per §3.6 (operational). Separately run the §2.1.1 thinking probe against the live vLLM; record `reports/thinking-probe-<yyyymmdd>.md`. If the probe returns case-row-2 (thinking genuinely off at serve layer), **the whole plan pauses** until that's investigated and fixed.

**Days 1–2 — Workstream 0 (loop-contract cleanup, §1.5).** Sub-spec-only changes: scope AR.2/AR.13 to `--harness real`, add AR.2b/AR.13b symmetric checks for `--harness synthetic`, define `ROUND_BUNDLE_READY` or remove it, expand `run-round` exit-code contract. All text + matching test assertions. Blocks A/B/C.

**Week 1 — Workstream A.1 (multi-family v5 composite workload).**
- Day 3–4: Author `scripts/probe_serving_thinking.py` (standalone probe-runner), `scripts/capture_seed_workload.py` extended with `--family-id` + `--variant v5`.
- Day 5: Per-family seed captures for the 28 v5-landed families. Budget ~2 min per family × 28 = ~1 h of serving time, plus ~5 min per-family setup overhead. Parallelizable if multiple vLLM instances.
- Day 6: Author `scripts/capture_multi_family_v5_workload.py` + `benchmark_blueprints/workloads/multi-family-v5/pool.yaml`.
- Day 7: Run composite capture; validate against §2.1.2 distribution minimums (thinking coverage, stratification, row counts); iterate pool weights if needed.

**Week 2 — Workstream A.2 + A.3 + Workstream C.**
- Day 6–7: Sub-spec v0.1.12 draft — §4.1 budget table, §4.4(a) quintuple-baseline, §4.4(b) n=4 rescreen, §4.3 soft gate, `round_provenance` fields, new AR verification items (§2.3, §2.5, §3.4).
- Day 8: Update `round_driver.py` implementation of the hardened phases.
- Day 9: Workstream C implementation (bundle fields + loader warn-and-log).
- Day 10: Integration test — full hardened round against synthetic fixture.

**Week 3 — Workstream B (P2-RR fix — `per_request_kv_budget` work dropped per §3.2).**
- Days 11–13: `concurrency_cap_rollout` in the inference proxy (class-routed admission + separate semaphore).
- Days 14–15: `admission_queue_depth_max` in the inference proxy (bounded queue + structured 429 rejection).
- Day 16: Integration test — small real L2 round against fixture to verify `real_proxy_enforcement: true` on every trace with the three enforced fields.

(`per_request_kv_budget` Option B.1 is **removed** from this plan per §3.2 — it stays advisory until real KV accounting lands in v0.3+. Saves ~4 days.)

**Week 4 — Round H1 via replay-round + decision.**
- Day 17: Run H1 via `replay-round` (§5.1) against hardened multi-family-v5 workload, real harness, real vLLM. ~12 h.
- Day 18: Review H1 results. Branch on §5 decision (case α / β / γ).

**Week 5+ — Either L2 re-run (case α) or L0 scoping (case β/γ).**

**Total: ~4 weeks from bc05506 to decision point** (was 5 before the P2-RR drop). This is still slow relative to "run another round next week," but it produces a defensible answer instead of another ambiguous one.

---

## 7. Verification — new items for v0.1.12 sub-spec

These extend the v0.1.11 §12 `9.3.AR.*` checklist. Each is pass/fail, artifact-backed.

- **9.3.AR.26 Thinking-realistic seed + holdout traces present, strict per-file family coverage (P2-LL + P1-UU fix).** **Both** `seed_trace_ref` AND `holdout_trace_ref` resolved files independently satisfy: ≥ 30% of rows with `thinking_tokens > 0`, ≥ 10% with `thinking_tokens > response_tokens`, at least one row with `thinking_tokens ≥ 4096`. Seed row count ≥ 84; holdout row count ≥ 28 (these are the §2.1.2 stratification minimums, not the v0.1.1-plan loose minimums). **Per-file family coverage (P1-UU strengthened)** — not a union check:
  - `set(row.family_id for row in seed) == set(workload.yaml.pool_families)` — every eligible family appears in seed independently.
  - `set(row.family_id for row in holdout) == set(workload.yaml.pool_families)` — every eligible family appears in holdout independently.
  - For every `family_id` in the pool: `count_in_seed(family_id) >= 3` AND `count_in_holdout(family_id) >= 1` (matches §2.1.2's 3:1 split-per-family default).
  - `set(workload.yaml.pool_excluded_families)` is disjoint from `set(workload.yaml.pool_families)` — no family is both in-pool and excluded.
  
  A broken split where family A is only in seed and family B is only in holdout fails both the first and second bullets above (A not in holdout set, B not in seed set). The prior "union covers pool" check could miss this; the new per-file assertion cannot. Artifact: `jq` summary + per-file family-set comparison against `workload.yaml.pool_families`.

- **9.3.AR.27 Soft latency gate computed.** The bundle's `round_provenance.latency_above_slo` field is present and matches a fresh computation over the winner's rescreen traces against the workload's nominal ceilings. Artifact: re-computation match.

- **9.3.AR.28 L2 enforcement coverage truthful (P1-KK fix).** For L2 rounds, every row's `measurement_trace.json.request_shaping_enforcement.enforced_fields` equals exactly `[concurrency_cap_eval, concurrency_cap_rollout, admission_queue_depth_max]` — the three v0.2-enforced fields. `advisory_fields`, if present, equals a subset of `[per_request_kv_budget, priority_preemption]`. `real_proxy_enforcement: true` on every row; `mode` ∈ `{enforced, enforced_minus_advisory}`. Artifact: `jq` extraction + proxy-config cross-check.

- **9.3.AR.29 Noise floor from n=5 baseline stddev (Screen-only).** `round_spec.yaml.noise_floor` equals `2 × stddev(M₁..M₅)` for the five **Screen-profile** baseline rows in `results.tsv` (P1-HH fix — same profile throughout the statistic). `results.tsv` has exactly 5 baseline rows per round, labeled `baseline_a` through `baseline_e`. No baseline rows at Full profile in v0.1.12. Artifact: computation match.

- **9.3.AR.30 Rescreen n=3 Screen + n=1 Full per top-K candidate (P1-HH fix).** For each `parent_candidate_uuid` that appears in rescreen rows, the count of `status: rescreened` Screen-profile rows with that parent is **exactly 3**, and the count of `status: rescreened_full` rows is **exactly 1**. Screen-profile rescreens enter `objective_mean_screen` (winner selection); Full-profile rescreens populate only the `screen_full_divergence` diagnostic flag (§2.2). Artifact: row-count per parent + profile-column check.

- **9.3.AR.31 `confidence` field correctly derived (Screen-only).** Bundle's `round_provenance.confidence` matches a re-computation using **Screen-profile baselines and Screen-profile candidate rescreens only** per the §4.2 algorithm. Full-profile measurements are not mixed into the confidence number. Artifact: re-computation match.

- **9.3.AR.32 `workload_distribution_id_hardening_version` pin (P1-VV fix — resolved descriptor).** Round's `round_spec.yaml.workload_descriptor_path` is populated with the actual file read at bootstrap — for composite rounds `benchmark_blueprints/workloads/<family_id>/workload.yaml`, for legacy single-family rounds `benchmark_blueprints/families/<family_id>/serving_workload.yaml`. That file's `workload_distribution_id_hardening_version` is `v1-multi-family-v5-thinking-realistic` (composite) or `v1-thinking-realistic` (legacy), unless the round was bootstrapped with `--allow-legacy-workload`. Artifact: round_spec inspection + read the named descriptor file and confirm the version field.

- **9.3.AR.33 Serving thinking probe recorded and current.** `round_spec.yaml.serving_thinking_probe` field is populated with: (a) a path to a probe report under `reports/thinking-probe-*.md`, (b) the report's `capture_date` is within 7 days of round bootstrap, (c) the outcome is `row-1` or `row-3` (thinking fires when asked). A `row-2` outcome blocks round bootstrap at §11.1 precondition time. Artifact: round_spec inspection + probe report existence + file-mtime check.

- **9.3.AR.34 `replay-round` runs without codex (P1-JJ fix).** For rounds whose `round_provenance.round_type == replay`, no `candidates/<iter>/agent_session.jsonl` files exist — the whole round was Python-driven. The imported candidate's directory is `candidates/import_<NNN>/` with a `candidate.yaml` that is byte-identical to the `--import-candidate` source file referenced in the FINALIZE commit. The source must be a candidate.yaml (not a bundle yaml); bundle paths fail AR.34 by design (P1-TT). Artifact: file-existence grep + byte-comparison against source.

- **9.3.AR.35 `screen_full_consistency` populated and aligned with Screen stats (P2-WW fix).** The bundle's `round_provenance.screen_full_consistency` field is present and equals `consistent` or `divergent`. Derivation matches §4.2: consistent iff the 1 Full-profile rescreen measurement for the winning candidate lies in `[objective_mean_screen ± 3 × stddev_screen]`; divergent otherwise. Divergent is allowed (advisory signal, not a gate) but must be flagged in `run_log.json.diagnostics.screen_full_divergence_note`. Artifact: re-computation using `screen_full_measurement` from the winner's Full rescreen trace + comparison against Screen stats in `round_spec.yaml`.

- **9.3.AR.36 Every bundle pins `workload_distribution_id` via the canonical procedure (P1-SS + P1-EEE fix).** The parent §6.4 validity rule now hard-pins `workload_distribution_id`. Every v0.1.12+ bundle records a `workload_distribution_id` that equals `compute_workload_distribution_id(descriptor_path)` per the parent §5.9 canonical procedure — **not** the naïve `sha256(seed + holdout + workload.yaml)` formulation from earlier plan drafts. The canonical procedure (a) nulls the `workload_distribution_id` field in the descriptor before yaml canonicalization (`sort_keys=True, default_flow_style=False`), (b) resolves `seed_trace_ref` / `holdout_trace_ref` relative to the descriptor path so both composite (`workloads/<family_id>/workload.yaml`) and legacy (`families/<family_id>/serving_workload.yaml`) layouts work, (c) hashes each of seed, holdout, and canonical yaml independently, then hashes the concatenation of hex digests. Loading a bundle into a serving stack whose resolved workload descriptor produces a different canonical hash triggers `bundle-validity: refused, mismatching_field: workload_distribution_id`. Artifact: (1) bundle inspection, (2) **re-run parent §5.9's `compute_workload_distribution_id()` reference implementation** against the resolved descriptor path and confirm byte-identical hex digest, (3) a synthetic refusal test (mutate any non-self-referential field in descriptor yaml, attempt load, expect refusal).

- **9.3.AR.37 Descriptor's `workload_distribution_id` is verified-not-minted at bootstrap (P1-GGG fix).** `bootstrap-round` MUST NOT write or mutate `workload.yaml.workload_distribution_id` (or `serving_workload.yaml.workload_distribution_id` for legacy). If the descriptor lacks a `workload_distribution_id`, bootstrap refuses with `descriptor_missing_workload_distribution_id`; if the descriptor's id differs from a freshly-computed canonical value, bootstrap refuses with `descriptor_workload_distribution_id_mismatch`. Artifact: (1) synthetic test — remove the id field from a fixture descriptor, attempt bootstrap, expect the first refusal; (2) synthetic test — mutate a trace byte after capture without re-computing, attempt bootstrap, expect the second refusal; (3) git-blame check — verify no code path in `bootstrap-round`'s implementation writes to the `workload_distribution_id` key.

Total v0.1.12 AR items: **37** (was 25 at v0.1.11; new items AR.26–AR.37 landed across hardening plan v0.1.0-plan through v0.1.5-plan).

---

## 8. Open questions

### 8.1 What if hardened baseline variance is still very high?

H1 might find that even with n=5 baselines and a thinking-realistic workload, baseline σ is still on the order of the mean — meaning the workload itself is fundamentally too noisy for per-candidate configurations to produce distinguishable signals.

Possible causes: thermal throttling on GB10, background processes on the host, vLLM-internal nondeterminism outside what `determinism_pass_rate` catches (e.g., request-ordering variance), workload has legitimately high turn-length variance.

If this happens, the honest next step is **workload characterization**, not more search — understand what's driving the variance before trying to optimize against it.

### 8.2 Should `priority_preemption` stay advisory forever?

If vLLM upstream doesn't gain scheduler hooks in a timeframe we care about, `priority_preemption` may stay advisory permanently. That's fine as long as the bundle metadata is honest about it. Open: whether v0.3 should drop the field entirely or keep it as a yaml annotation for downstream consumers who do their own scheduling.

### 8.3 Second family before L0?

If H1 produces case β (L1 is tapped out for this family), the decision path says "start L0." But another family on the same hardware might have different L1 headroom — e.g., a rollout-heavy workload might benefit much more from L1 tuning than an eval-heavy one. Open: commission a second family + hardened round before investing in L0, to confirm that L1-tapped-out is hardware-level, not family-level.

### 8.4 Strict bundle confidence policy at what timeframe?

§4.3 recommends `warn` at v0.1.12 and `strict` at v0.2. Open: what's the explicit trigger for flipping? Probably "after we've seen ≥ 3 hardened rounds all produce `defensible` confidence," which would validate that the hardening actually works. Needs a concrete success criterion before the flip.

---

## 9. Changelog

- **v0.1.0-plan (2026-04-24)** — Initial hardening plan. Three workstreams (measurement hardening, L2 enforcement, bundle honesty metadata), 7 new verification items, 5-week sequence from bc05506 to decision point. Drives a v0.1.12 sub-spec revision on completion, which folds the additions in and retires this plan.
- **v0.1.1-plan (2026-04-24)** — Eight findings from the first review round resolved, plus one scope change from the operator. **P1-GG (loop cleanup assumed done, isn't)**: added new §1.5 Workstream 0 — scope AR.2/AR.13 to `--harness real`, add symmetric AR.2b/AR.13b for synthetic mode, define `ROUND_BUNDLE_READY` (seen in the 2026-04-24 report but absent from v0.1.11 §11.7 enum), expand `run-round` exit-code contract (`ROUND_INFEASIBLE` is a 0 exit, not non-zero). Blocks A/B/C. **P1-HH (profile mismatch in statistics)**: §2.2 rewritten — baseline, main-loop, and primary rescreen all use Screen profile; Full-profile rescreen (n=1 per candidate) is a sanity cross-check, not part of confidence derivation. `derive_confidence` references Screen-only measurements. New `screen_full_divergence` flag surfaces cross-profile inconsistency without double-counting. **P1-II (budget double-count)**: §2.2 budget table made single-source-of-truth; `iteration_cap = 12` stays, no phantom Option A3, hardened budget is ~9.7 h active inside a 12 h round cap. **P1-JJ (H1 can't force C003 via iteration_cap=1)**: new `lumoserve auto-research replay-round` subcommand spec'd in §5.1. Python-driven, no codex exec, imports an existing `candidate.yaml` verbatim into `candidates/import_<NNN>/`. H1 uses replay-round, not iteration_cap=1. New iteration-id grammar form `import_\d{3}`. **P1-KK (`per_request_kv_budget` misleading as `max_tokens`)**: §3.2 + §3.3 reverted — `per_request_kv_budget` stays **advisory** at v0.2 (not enforced); v0.2 L2 action space is **3 enforced fields** (`concurrency_cap_eval`, `concurrency_cap_rollout`, `admission_queue_depth_max`) + 2 advisory (`per_request_kv_budget`, `priority_preemption`). Workstream B effort drops from 2 weeks to 1.5 weeks. **P2-LL (AR.26 only checks seed)**: AR.26 now covers both seed_trace_ref and holdout_trace_ref with identical minimums. **Folded-in from post-review dialog:** §2.1.1 pre-launch serving thinking probe (curl case-matrix diagnostic), new `scripts/probe_serving_thinking.py`, §11.1 precondition + §9.3.AR.33 verification. **Operator scope change:** §2.1 reframed from single-family seed capture to **multi-family v5 composite** — pool of 28 v5-landed families, per-family seed captures aggregate into `benchmark_blueprints/workloads/multi-family-v5/seed_trace.jsonl`. `workload_distribution_id_hardening_version: v1-multi-family-v5-thinking-realistic`. Round target tuple's `family_id` becomes `workload_id`; default is `multi-family-v5`, not a single family. Verification items now 9 total (AR.26–AR.34); §7 lists all. Total effort estimate: ~5 weeks bc05506 → decision, of which ~1 week is per-family seed captures (new workstream).
- **v0.1.2-plan (2026-04-24)** — Six findings from the second review round resolved; operator approved parent-HLD edit for the bundle-identity fix. **P1-MM (composite workload breaks parent §5.9/§6.4 bundle-identity contract — parent HLD edit approved)**: reverted `family_id → workload_id` rename. Parent HLD §5.9 + §6.4 gained a one-paragraph amendment each: `family_id` MAY name a composite workload (e.g. `multi-family-v5`) that resolves under `benchmark_blueprints/workloads/<name>/` instead of `families/<name>/`. The bundle storage path, validity hard-pin, admission-layer routing, and every other parent-HLD contract surface stays verbatim — `family_id: multi-family-v5` is just a composite-family value. Plan §2.1.2 updated to use `family_id` throughout; `workload.yaml` is the descriptor that `family_id` resolves to. **P1-NN (stratified split math infeasible)**: plan's original defaults (samples_per_family=2, total_rows=60, holdout_ratio=0.15, strict stratification across 28 families) were arithmetically impossible — holdout of 9 rows cannot include each of 28 families. Defaults rewritten: **samples_per_family=4 (3 seed : 1 holdout per family)**, minimum 4 turns per family, seed=84 rows, holdout=28 rows, overall split ~75/25. New `pool_excluded_families` field records any family whose v5 trajectory has <4 turns. **P1-OO (pool eligibility referenced absent metadata)**: dropped `representative_trajectory_ref` from eligibility (doesn't exist on any family). Pool eligibility now based on three artifacts that DO exist today: `verification_matrix_v5*.md`, `codex/config.toml`, and the v5-landed flywheel-ready marker on `family.yaml`. Per-family trajectory data is produced by §2.1.3 captures, not consumed as an input. **P1-PP (`derive_confidence` still used mixed-profile names)**: §4.2 algorithm rewritten with explicit Screen-only inputs (`baseline_screen_measurements: list[float]`, `winner_screen_measurements: list[float]`) + Welch-t CI. Full-profile rescreen is a separate `screen_full_consistency` field in `round_provenance`, not mixed into confidence. **P1-QQ (§5 H1 description contradicted §5.1 replay-round design)**: §5 Step 1 rewritten — H1 uses `replay-round` CLI explicitly, no codex-exec, with the exact command line shown inline. No more "iteration_cap=1 and codex proposes" confusion. **P2-RR (§6 sequencing Week 4 still scheduled dropped KV-budget work)**: Week 3 now lists only the two enforced-field changes (`concurrency_cap_rollout` 3d + `admission_queue_depth_max` 2d) + integration test. Week 4 becomes H1 round + decision. Total plan duration shrinks to ~4 weeks (was 5). Sub_spec_version notation unchanged; this is a plan revision, not a sub-spec revision.

---

- **v0.1.3-plan (2026-04-24)** — Six findings from the third review round resolved; one additional parent-HLD amendment. **P1-SS (composite validity ignored workload identity — parent HLD amendment)**: parent §6.4 `workload_distribution_id` promoted from metadata-only to **hard-pinned** for all families (composite and single). Bundle tuned against one composite pool cannot silently load after the pool / seed-trace / workload.yaml content changes. Parent §5.9 schema comment updated. Matching verification added as new §9.3.AR.36 (validity refusal on mutated workload.yaml). **P1-TT (H1 imported a bundle where replay-round wants a candidate)**: §5 Step 1 command now references `output/auto_research/.../candidates/003/candidate.yaml` (the original candidate.yaml), not the bundle yaml. AR.34 byte-compares that path, so bundle-shaped inputs fail by design. Noted a future `--import-bundle` variant that could extract a bundle's `vllm_config` block into a canonical candidate if needed. **P1-UU (AR.26 family-coverage check too weak)**: rewritten as strict **per-file** set-equality — `set(seed_families) == pool` AND `set(holdout_families) == pool`, plus per-family `seed_count ≥ 3 AND holdout_count ≥ 1`. Union-only check could miss A-in-seed-only / B-in-holdout-only splits; per-file equality cannot. Also added disjointness check between `pool_families` and `pool_excluded_families`. **P1-VV (composite preconditions read `serving_workload.yaml`, not the resolved descriptor)**: §2.5 precondition rewritten with parent HLD §5.9 resolution order — composite family_ids read from `workloads/<name>/workload.yaml`, legacy from `families/<name>/serving_workload.yaml`. `bootstrap-round` records `round_spec.yaml.workload_descriptor_path` so downstream verifiers know which file was read. AR.32 updated to read from the recorded path, not a hard-coded filename. **P2-WW (`screen_full_consistency` used but not in schema)**: §4.1 `round_provenance` schema extended with `screen_full_consistency`, `workload_descriptor_path`, and an alignment note on `improvement_over_baseline_*` being Screen-only. New verification item §9.3.AR.35 checks the field is populated and matches re-computation. **P2-XX (replay-round `--holdout-rows 12` stale)**: default bumped to **28** to match composite stratification. CLI refuses values below `pool_size × split_per_family.holdout_rows`; documented invariant that going lower violates per-family holdout coverage. Verification items now **11 total** (AR.26–AR.36); §7 lists all. Parent HLD now has **two** amendments from this plan (§5.9 composite family_id, §6.4 workload_distribution_id promotion) — both load-bearing and both documented inline in those sections with pointers back to this plan.

---

- **v0.1.4-plan (2026-04-24)** — Six findings from the fourth review round resolved; one additional parent-HLD amendment (3rd total from this plan). **P1-YY (self-referential workload hash)**: v0.1.3 defined `workload_distribution_id = sha256(seed + holdout + workload.yaml)` but workload.yaml itself contains that field — non-computable. Parent HLD §5.9 now defines a canonical non-circular procedure: null the `workload_distribution_id` field in yaml before canonicalization (`sort_keys=True`, `default_flow_style=False`), hash each of the three inputs separately, hash the concatenation of hex digests. Properties: (1) non-circular (fixed point under the procedure), (2) stable under comment/key-order edits, (3) sensitive to any content drift. Python reference implementation included in parent §5.9. Verification re-runs the same procedure at load time. **P1-ZZ (parent §5.9 validity summary omitted `workload_distribution_id`)**: summary paragraph in parent §5.9 updated to include `workload_distribution_id` alongside `environment_pin`, `serving_posture_pin`, `model_id`, `family_id` — matches the §6.4 hard-pin table. **P1-AAA (plan §2.1 sub-spec-change text hard-coded `serving_workload.yaml`)**: §2.1 Sub-spec-changes paragraph updated to say the resolved workload descriptor (per parent §5.9 lookup order, recorded in `round_spec.yaml.workload_descriptor_path`) is what §11.1 reads — `workloads/<family_id>/workload.yaml` for composite, `families/<family_id>/serving_workload.yaml` for legacy. Prevents the composite-path bug from sneaking back in during v0.1.12 sub-spec folding. **P2-BBB (nominal ceilings hard-coded in `serving_workload.yaml`)**: §2.3 + §2.4 rewritten to reference the resolved workload descriptor. Composite rounds read nominal_* ceilings from `workloads/<name>/workload.yaml`; legacy reads from `families/<name>/serving_workload.yaml`; finalize-round reads the path recorded in `round_spec.yaml.workload_descriptor_path`, so it's file-agnostic. **P2-CCC (replay FINALIZE `imported_from_bundle` field name stale)**: renamed to **`imported_from_candidate`** to match P1-TT's candidate.yaml-source contract. AR.34 byte-compares this path; bundle-yaml shapes fail loudly. **P2-DDD (target row counts ≥40/≥8 stale)**: bumped to ≥84 seed / ≥28 holdout to match the 28-family × 3:1 stratified split from §2.1.2. The old minima predated stratification and would have failed AR.26's strict per-file set-equality check. Verification items unchanged at 11 total (AR.26–AR.36). Parent HLD now has **three amendments** from this plan: §5.9 composite `family_id` (v0.1.2), §6.4 `workload_distribution_id` hard-pin (v0.1.3), §5.9 canonical hash computation (v0.1.4).

---

- **v0.1.5-plan (2026-04-24)** — Five findings from the fifth review round resolved. **P1-EEE (AR.36 still used old circular formula)**: AR.36 rewritten to call parent §5.9 canonical procedure explicitly — re-run `compute_workload_distribution_id(descriptor_path)` and byte-compare. Implementers can't implement the wrong algorithm; verifiers can't pass a bundle under the wrong hash. **P1-FFF (canonical hash was composite-layout-specific)**: parent §5.9 reference function rewritten to accept a `descriptor_path` (composite OR legacy) and read `seed_trace_ref` / `holdout_trace_ref` from the descriptor itself. New `_resolve_ref()` helper handles absolute vs descriptor-relative trace paths. Fourth property added to the canonical procedure's list: **layout-agnostic**. Legacy `serving_workload.yaml` descriptors that don't yet declare `seed_trace_ref` / `holdout_trace_ref` need those fields added — tracked as a v0.1.12 sub-spec migration item. **P1-GGG (bootstrap minted ids silently)**: parent §5.9 rewritten — the canonical id is written **exactly once, by the capture step**, never by `bootstrap-round`. Bootstrap has a new verification contract: missing id → refuse `descriptor_missing_workload_distribution_id`; mismatched canonical recomputation → refuse `descriptor_workload_distribution_id_mismatch`. This closes the "silent bless at launch" hazard. New verification item §9.3.AR.37 checks the no-mint invariant + both refusal paths. **P1-HHH (§2.5 precondition named "version" loosely)**: exact field name **`workload_distribution_id_hardening_version`** used throughout, not the generic "version". **P2-III (§4.3 said parent §6.4 doesn't change)**: rewritten to distinguish — `round_provenance` honesty fields (confidence, latency_above_slo, screen_full_consistency, l2_enforcement_coverage, etc.) stay as diagnostics; the one §6.4 change that did happen (v0.1.3-plan P1-SS `workload_distribution_id` hard-pin) is called out separately. Readers can't miss the actual validity-rule change. Verification items now **12 total** (AR.26–AR.37). Parent HLD now carries **four amendments** from this plan: §5.9 composite `family_id` (v0.1.2), §6.4 `workload_distribution_id` hard-pin (v0.1.3), §5.9 canonical hash computation (v0.1.4 — rewritten to layout-agnostic in v0.1.5), §5.9 capture-mints-bootstrap-verifies contract (v0.1.5).

---

*End of hardening plan v0.1.5. For reviewers: the composite-workload validity contract is now end-to-end sound across all layouts. Capture computes the canonical id once (composite or legacy); bootstrap-round verifies without ever mutating; `/admin/load_tuned_config` re-verifies at load time. The canonical procedure is layout-agnostic via the `descriptor_path + seed_trace_ref + holdout_trace_ref` interface, not the hard-coded filenames the v0.1.4 draft had. Plan remains ~4 weeks bc05506 → decision; no schedule change. The remaining open questions in §8 are unchanged.*
