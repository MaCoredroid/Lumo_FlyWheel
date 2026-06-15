# FR13 — The deployment-binding TEMP-0.6 distributional drift gate

CPU / read-only design. A speed-campaign GPU run is live — NO boot here. All vLLM citations are read
fresh from the pinned image (`scripts/vllm_src.sh`, sha `3dbe092e…`, never `/tmp`). int-view equality
reasoning for any byte check, NEVER atol; TV/KL for the distributional part.

Bug-class playbook rows in force (`FR13_BUG_CLASS_PLAYBOOK.md`):
- **#9 Silent fallback / vacuous instrument** — "a run passes while measuring nothing". The temp-0
  argmax gate passing at deployment temp-0.6 is a *category-#9 risk made subtle*: the instrument
  (temp-0 argmax flip) is not vacuous, it is just **measuring a strictly weaker event** than the one
  that decides deployment losslessness. Remedy = run the instrument that fires on the actual deployment
  event (sampled-token drift), with engagement asserts, before trusting "lossless at 0.6".
- **#11 Oracle identity / batch-composition sensitivity** — the oracle MUST be the actual deployment
  non-spec **recurrent** decode (`_forward_core_decode_non_spec`), pinned same BI-flag state on both
  arms; NEVER chunked-prefill / streamed-logprobs / serial-torch / a backend NAME. native-vs-native is
  the control; floors are measured, not assumed.
- **#12 Measurement traps** — raw counters only; capture-once pinned prompts; validated denominators;
  multi-sample p95 floors (the single-draw 0.0593 is superseded by the measured 0.1133, see §2).

---

## 1. FRAMING — CONFIRMED and sharpened

The user's framing is **CORRECT**. Sharpened with code-read + the banked artifact structure.

### 1a. argmax = temp-0 (CONFIRMED, code-read)
At `temperature=0.0` the served token is the **argmax** of the spec verify forward. A "flip" =
`served_argmax != recurrent_decode_argmax` — a **POINT measure**: it fires only when the **TOP**
token changes. It is blind to any reshaping of the sub-argmax tail.

### 1b. the 13% IS exactly this temp-0 argmax point-measure (CONFIRMED, artifact-read)
The big-denom rescore (`output/fr13_bigdenom_rescore/`) is a **temp-0 greedy** serve scored against the
recurrent oracle. **MEASURED from the banked artifacts (this task):**

| arm | positions | flips | clear-margin | clear% (Wilson95) |
|-----|-----------|-------|--------------|-------------------|
| cat9   | 8717 | 1782 | 1181 | **13.548%** [12.846, 14.283] |
| native | 8752 | 1830 | 1224 | **13.985%** [13.275, 14.728] |

CIs overlap, cat9 LOWER. The serve was `temperature: 0.0` on both arms (per
`FR13_CONFIRM_SPEC_VS_NONSPEC.md` Link 1, and the served streams are bare token ids — confirmed: the
record schema is `{served_token_ids}` only, `output/fr13_bigdenom_rescore/cat9_src.json:records`). The
`deviation_nat`/`clear_margin` are computed against the recurrent oracle argmax. **This is the temp-0
point-measure, confirmed spec-vs-non-spec (the four CODE-READ links in `FR13_CONFIRM_SPEC_VS_NONSPEC.md`
hold and are not reopened here).**

### 1c. temp-0.6 lossless is DISTRIBUTIONAL — the gate the argmax measure cannot reach (CONFIRMED)
At deployment `temperature=0.6` the served token is **SAMPLED via rejection sampling**. The lossless
property is the rejection-sampling theorem: the spec output **distribution** == the no-spec target
**distribution**. The binding object is `q = softmax(verify_logits/0.6)` vs `p = softmax(decode_logits/0.6)`,
NOT the argmax of either.

**The gap is real and one-directional:** a position can be **argmax-identical** (`argmax q == argmax p`,
so the temp-0 gate PASSES) yet have a **drifted sub-argmax tail** (`q != p` below the top), which changes
which token is **SAMPLED** at 0.6 → the temp-0 argmax gate PASSES while temp-0.6 sampling DRIFTS.
Concretely: rejection sampling accepts a draft token `x` with prob `min(1, p(x)/q(x))` and on reject
resamples from `norm(max(0, p-q))`; **every** channel of `q` and `p`, not just the argmaxes, enters the
realized output distribution. So argmax-flip-rate is a **lower bound proxy** for, not a measure of,
temp-0.6 distributional drift.

**Quantitative support that the tail carries real mass at 0.6 (MEASURED, this task, on the FLIP subset
where the recurrent oracle top-5 is banked):** at `T=0.6` the median recurrent-target `P(top1)` is
**0.742 (cat9)** / **0.739 (native)**, and **70.5% (cat9)** / **71.4% (native)** of flip positions have
`P(top1) < 0.9` at 0.6. So at the deployment temp there is substantial sub-argmax mass that sampling
will reach. **CAVEAT (do not over-read):** this is `p`-side concentration only — it shows the *target*
is not a delta at 0.6, hence temp-0.6 sampling is not trivially argmax-collapsed; it does NOT measure
`q`-vs-`p` drift (that needs `q`, which is not banked — §3). Its value here is to **refute** any claim
that "13% argmax-pass ⇒ done at 0.6": the temp-0.6 event is genuinely distributional.

**Conclusion:** the deployment-binding lossless gate is the **temp-0.6 distributional** one. It decides
the levers (L1/L4), not the temp-0 argmax rate. The 13% pass is necessary, not sufficient (class #9 — a
weaker event measured in place of the deployment event; class scalar-blindspot, `reference_scalar_metric_per_token_blindspot`).

---

## 2. The PROPER temp-0.6 drift gate (design)

Two candidate constructions. The gate uses **(a) as the binding lossless instrument** and **(b) as the
deployment-realism cross-check**, paired with a **per-position view** (blindspot rule), all on the
**recurrent oracle frame** with the **multi-sample native floor**.

### (a) Per-position distributional TV/KL at temp 0.6 — `q` vs `p` (BINDING)
Per served position:
- `q = softmax(verify_logits / 0.6)` from the **spec verify forward** (the dist the engine actually
  samples from at deployment).
- `p = softmax(decode_logits / 0.6)` from the **no-spec recurrent decode oracle**
  (`_forward_core_decode_non_spec` → `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`,
  vLLM `gdn_linear_attn.py` L806-818 / L1045-1097; `FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN, single-token
  roll). NOT chunked-prefill (that hits `chunk_gated_delta_rule` at `num_prefills>0`, the frame the
  oracle docstring warns against), NOT streamed logprobs, NOT serial-torch, NOT a backend name.
- metric: symmetric **TV(q,p)** and **KL(p‖q)** (KL in the direction the sampler "feels": target `p`
  re-weighted by proposal `q`). Per-position, then aggregate (mean + p95 + the per-position vector).

**Temp recovery from banked top-K is valid IN PRINCIPLE:** from a recorded top-K `log_softmax` at T=1,
`softmax(·/0.6)` over that **same support set** is recoverable because the per-position additive constant
cancels in the softmax. So **IF both `q` and `p` top-K were banked**, temp-0.6 TV would be computable
without a re-boot. **They are not both banked — see §3.** (Truncation to top-K introduces a small,
**bounded** tail error: the omitted mass is `1 - sum(exp(topk_logprobs))`; at K=20 this is typically
<1e-3 of the T=1 mass and shrinks further at T=0.6 where the head sharpens. The GPU capture should record
the truncated-mass per position so the TV is reported with an explicit tail-error bar, not silently.)

**Why (a) is the most deployment-faithful BINDING instrument:** it isolates **verify-vs-decode numerical
drift** (the thing the levers move) from path/RNG non-determinism, because `q` and `p` are both
conditioned on the **same fixed served prefix** (forced-decode / teacher-forced), so there is no sampled
divergence between them. It is the continuous quantity that *predicts* temp>0 acceptance
(`FR13_DRIFT_TRACKER_DESIGN` (2): "the within-floor continuous quantity that predicts temp>0 acceptance").

### (b) Realized-sample BAG-TV with the multi-sample native floor (DEPLOYMENT CROSS-CHECK)
Run K temp-0.6 **served streams** (multi-seed), compute **bag-TV** = token-multiset TV between the cat9
served bag and the native served bag, vs the **native-self floor** = p95 of C(N,2) native-vs-native
temp-0.6 bag-TV draws (N=6-8 native seeds). cat9 PASSES iff `cat9-vs-native bag-TV ≤ floor_p95`.

- This is the **most deployment-realistic** number (it is literally what the deployed sampler emits,
  including the irreducible co-residency reduction-order non-det that forced-decode cannot remove —
  `FR13_DRIFT_TRACKER_DESIGN` (2) HONEST CAVEAT), but it is **coarser**: it conflates verify-drift with
  path/RNG divergence, so it cannot *attribute* a failure to a lever. It needs **streams**, not logits.

**The native temp-0.6 FLOOR for (b) — MEASURED (single-draw, class #12 caveat):**
`FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md` — B=4, temp 0.6, top_p 0.95, seed 1313 vs 1313, FLASH_ATTN,
CUDA-captured: **bag-TV ≈ 0.1133, 139/256 raw positional token mismatches** (native is NOT same-seed
deterministic at this deployed shape). This is **ONE C(2,2) draw** → it is the floor's center, NOT its
p95; per class #12 it MUST be upgraded to the N=6-8 p95 before it is a hard threshold (the historical
single-draw `BAG_TV_FLOOR = 0.0593` at `fr13_corruption_gate.py:123` is **superseded** by this 0.1133).

### Which is soundest + pairing
- **Binding lossless verdict = (a)** TV/KL on the recurrent frame (attributable, isolates verify-drift,
  predicts acceptance). **Deployment-reality gate = (b)** bag-TV vs the multi-sample p95 floor.
- **Pair the scalar with a per-position view (blindspot rule, `reference_scalar_metric_per_token_blindspot`):**
  a single bag-TV scalar hid a real lossless+superset defect before. Emit, alongside the scalar:
  the per-position `TV(q,p)` vector + the count/list of positions with `TV > floor_per_position` (the
  per-position floor = the same TV computed native-vs-native on the matched prefix) — the same role the
  per-token argmax probe plays for the temp-0 gate.
- **The single comparable scalar** (reuse `fr13_drift_tracker.py` `D`): `D = excess temp-0.6 drift over
  the native floor`, with sub-channels {mean TV(q,p) excess, bag-TV excess, per-position-over-floor
  count}. `D ≤ 0` ⟺ cat9 indistinguishable from the native-vs-native temp-0.6 floor.

---

## 3. First-pass banked estimate — **NOT COMPUTABLE** (q is not banked)

**MEASURED this task (artifact scan):** the temp-0.6 TV(q,p) is **NOT computable on banked data**,
because the **spec verify distribution `q` is absent** in the binding-frame capture.

What IS banked (verified by reading every per-position record):
- `output/fr13_bigdenom_rescore/rescore_{cat9,native}.json`: per position records
  `{served_token_id, oracle_argmax_id, oracle_argmax_logprob, served_logprob_in_oracle, deviation_nat,
  flip, clear_margin}`. **The recurrent oracle top-K (`p`) is present ONLY on the 1782 flip positions
  and is truncated to top-5** (not top-20, despite `top_k:20`); the ~7000 non-flip positions carry
  argmax-only. **No spec verify dist `q` anywhere** — the served stream is greedy token ids only
  (`ANY spec-verify dist (q) banked: False`, scanned all 8717 positions). → temp-0.6 TV(q,p)
  **not computable** here (q missing; p partial).
- `output/fr13_gold_margin/{tree,native}_capture.json`: DOES carry the **spec verify forward top-20
  (`q` at temp-0)** per served position (`temperature:0.0, top_p:1.0, top_k:20, top_logprobs[]`). BUT
  its paired `*_teacher_force.json` (`p`) is produced by a **max_tokens=1 `/v1/completions` re-prefill**
  (`fr13_gold_margin_probe.py:363-430`, `prompt + shared_continuation_ids` → query_len>>1 →
  `num_prefills>0` → **`chunk_gated_delta_rule` CHUNKED** path) — the **WRONG oracle frame** per the
  compare-target rule (`feedback_fr13_lossless_compare_target`: NEVER chunked-prefill), AND it is
  recorded at only the **4 fork positions** (one per pinned prompt). 4 chunked-frame positions cannot
  estimate a temp-0.6 distributional gate.

So: the one artifact with `q` (gold_margin) has the wrong `p` (chunked, 4 pts); the artifact with the
right `p` (big-denom recurrent oracle) has no `q`. **There is no banked (q, p)-recurrent pair.**

**What the banked data DOES license (clearly labelled, partial):**
1. The temp-0 argmax point-measure (§1b): cat9 13.548% ≈ native 13.985%, CIs overlap, cat9 lower —
   a within-floor PASS **at temp-0** (computed/MEASURED).
2. A **`p`-side temp-0.6 peakedness** read on the flip subset (MEASURED, §1c): median `P(top1)@0.6` =
   0.742 cat9 / 0.739 native; 70.5%/71.4% of flips have `P(top1)@0.6 < 0.9`. This is **symmetric
   between arms and is NOT a drift estimate** — it only refutes "argmax-pass ⇒ trivially-lossless-at-0.6"
   by showing the target is not a delta at 0.6. **It is INFERRED context, not the temp-0.6 gate.**

**No temp-0.6 drift number (the cat9-vs-native excess TV) can be honestly produced from banked data.**
Reporting one would be a class-#9 vacuous instrument (computing TV against the chunked frame or against a
non-existent `q`). The estimate awaits the §4 GPU capture. Script
`scripts/fr13_temp06_drift_estimate.py` is included; run on banked data it **emits the temp-0
point-measure + the p-side peakedness and explicitly refuses the q-vs-p TV with the reason "q not banked"**.

---

## 4. The rigorous GPU capture plan (queues AFTER the speed campaign frees the GPU)

A fresh **TEMP-0.6 paired** capture, recurrent-oracle frame, multi-sample native floor. Two streams,
both required (binding instrument (a) + deployment cross-check (b)).

**Stream A — forced-decode (q, p) per position [for gate (a)]:**
1. Pick the pinned SWE stream (capture-once rule, `output/fr13_acceptance_ladder/prompts_swe4.json`;
   pin `request.json`/`prompt_token_ids`, `reference_capture_once_native_pin_prompt`).
2. **cat9 arm:** serve at temp-0.6, capture per served position the **spec verify forward** full top-20
   `log_softmax` = the engine's actual proposal (extend the gold_margin top_logprobs capture, which
   already records the verify forward, to the full stream; record the truncated tail mass per position).
   This is `q` at T=1; apply `/0.6` at reduce.
3. **Recurrent oracle (p):** feed the **identical served prefix** into the **no-spec recurrent decode
   oracle** (`fr13_recurrent_decode_oracle.py`, `FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN, single-token
   roll — the SAME script that produced the binding big-denom `p`, NOT chunked-prefill), capture the
   per-position top-20 `log_softmax` = `p` at T=1. Engagement asserts: `RECURRENT_PATH_ENGAGED:true`,
   recurrent-decode-call counter > 0 (class #9 fail-loud).
4. Do the SAME for the **native E5** arm (its own verify `q` + its own recurrent `p`).
5. **Reduce (CPU, free):** per position `TV(softmax(q/0.6), softmax(p/0.6))` + `KL`; cat9-arm and
   native-arm. The binding number = **cat9 mean/p95 TV(q,p) vs the native-arm mean/p95 TV(q,p) floor**
   (each arm vs its OWN recurrent oracle — the apples-to-apples rule). Plus the per-position vector +
   over-floor count.

**Stream B — realized multi-seed served streams [for gate (b) + the native p95 floor]:**
6. Run native at temp-0.6 with **N=6-8 seeds** (B=4 deployed shape) → C(N,2) native-vs-native bag-TV
   draws → **floor = p95** (upgrades the single-draw 0.1133 to a robust threshold, class #12).
7. Run cat9 at temp-0.6 with the same K seeds → cat9-vs-native bag-TV → PASS iff `≤ floor_p95`.
8. Wrap with `fr13_drift_tracker.py` (K-seed orchestrator over `fr13_corruption_gate.run_gate`) → the
   single scalar `D` + sub-vector; append a `FR13_LADDER_LOG.md` row.

**Regime:** final verdict tier = **B=4 + CUDA-captured** (co-residency changes at B=4; the floor itself
is a B=4 number). Stream A's forced-decode can run B=1/eager for the attributable per-position read;
the deployment verdict (b) is B=4. Multi-sample floor is mandatory on **both** streams.

**Cost:** N=6-8 native + K cat9 temp-0.6 boots (serialized; the real expense) + 2 forced-decode oracle
passes. CPU reduce is free. ~2 GPU-days serialized given host-memory recovery between boots; sequence it
behind the speed campaign.

---

## 5. Lever decision rule

Levers (`FR13_AMPLIFICATION_LEVERS.md`): **L1** = margin-aware commit-at-near-tie (don't hard-commit the
spec argmax when `q` is a near-tie — sample-faithfully at the boundary); **L4** = last-stage boundary
fp32 (do the final verify logit / lm-head boundary in fp32 to shrink the `q`-vs-`p` numerical gap).

Decision, on the §4 temp-0.6 measurement (NOT the temp-0 argmax rate):

- **cat9 temp-0.6 drift WITHIN the native temp-0.6 floor** — i.e. `D ≤ 0`: cat9 mean/p95 `TV(q,p)` ≤ the
  native-arm `TV(q,p)` floor **AND** cat9-vs-native bag-TV ≤ floor_p95 **AND** per-position over-floor
  count ≤ native-self over-floor count → **lossless is met at the deployment temp. Do NOT do the levers.**
  The levers would add cost/complexity for no measurable deployment gain (`feedback_speed_is_the_goal_cost_gate`).

- **cat9 ABOVE the floor by margin `M`** (`M = cat9 drift − native floor > 0`, CI-separated via paired
  bootstrap, `fr10_superset_gate` style) → **the levers are warranted.** Localize `M` first
  (per-position over-floor positions name where): if the excess concentrates at **near-tie positions**
  (small `|q_top1 − q_top2|`) → **L1**, predicted to remove the near-tie-attributable share of `M`
  (commit-at-near-tie makes those positions sample-faithful instead of argmax-committed). If the excess
  is **diffuse / boundary-numerical** (the `q`-vs-`p` gap tracks the fp8 verify/lm-head boundary, the
  "diffuse GDN accumulation" carrier, `reference_diffuse_gdn_accumulation_explained`) → **L4**, predicted
  to shrink the per-position `TV(q,p)` by the fp32-boundary share. Re-measure `D` after each lever; a
  lever ships only if it reduces `M` toward `≤ 0` at B=4/CUDA-captured **without regressing the temp-0
  argmax rate or accept/event** (lossless gate per change, `project_fr13_speed_first_lossless_gate`).

- **Guard (research-before-deadend):** if cat9 is above the floor, do NOT conclude "lossy at 0.6" before
  confirming the §4 capture isn't itself confounded (chunked `p`, unpinned prompt, single-draw floor,
  BI-flag mismatch — classes #9/#11/#12) and running the adversarial-verify pass. The temp-0 argmax PASS
  is genuine evidence the carrier is small; the temp-0.6 measurement decides whether the *residual sub-
  argmax* share crosses the floor.

WY remains PARKED (not revived). No reward-hack (no rerouting `q` or `p` through native machinery to pass
the metric — `feedback_no_reroute_reward_hacking`).

---

## MEASURED vs INFERRED ledger
- **MEASURED (this task, banked artifacts):** temp-0 argmax rates 13.548% / 13.985% (re-counted from
  `rescore_{cat9,native}.json`); `q` absent from big-denom (scanned all 8717 cat9 positions); big-denom
  `p` = recurrent oracle, top-5, flip-only; gold_margin has `q` (verify top-20) but chunked `p` at 4 pts;
  flip-subset `p`-side `P(top1)@0.6` medians 0.742/0.739. native temp-0.6 bag-TV floor 0.1133 (read from
  `FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md`, single-draw).
- **COMPUTED (this task):** softmax(top-K/0.6) recovery is valid (constant cancels); the per-arm
  peakedness numbers in §1c.
- **INFERRED / not yet measured:** the cat9-vs-native temp-0.6 TV(q,p) excess `M` (needs §4 GPU capture);
  the per-lever `M`-reduction predictions (§5); the N=6-8 p95 floor (single-draw 0.1133 is the only banked
  draw). No temp-0.6 drift number is claimed from banked data.
