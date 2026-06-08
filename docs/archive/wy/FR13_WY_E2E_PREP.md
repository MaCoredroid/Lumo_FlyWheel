# FR13 WY E2E Prep — branch-oracle recipe + e2e prediction (read-only, 2026-06-08)

Read-only adversarial synthesis. No GPU/docker/server touched. Every load-bearing
number below was recomputed from the captured fp32 logits and source file:line, not
copied from context. This document is the **prediction + the recipe**; the B=4
CUDA-graph SWE-4 e2e is the instrument. **It does NOT declare pass.**

Captured artifacts under test (the current redteam-fixed WY build, 1.221e-4 L1 floor):
- Tree logits: `output/fr13_wy_gateA_20260608T163915Z/tree/logs/tree_final_logits.pt`
  (10 rows = node ids 0..9, `[10, 248320]` fp32, schema `fr13.final_logit_capture.v1`,
  source `Qwen3_5ForCausalLMBase.compute_logits`).
- Native logits: `.../native/logs/native_final_logits.pt` (6 rows = MTP-5 spine, `[6, 248320]` fp32).
- Spine ladder gate: `.../gateA_spine_ladder.json` (`passed=False`, threshold `0.0`, `max_abs=3.3203125`).
- E5 baseline: `output/fr10_native_mtp5_same8_20260604T210257Z/quick_native_mtp5_same8.json`.

Topology (verified from `parent=[-1,0,1,1,2,2,4,4,6,6]`, consistent with the 10 logit rows
and `fr10_tree_depth_positions.jsonl` `mrope_depth_first_tree=[0,1,2,2,3,3,4,4,5,5]`):
children `{0:[1], 1:[2,3], 2:[4,5], 4:[6,7], 6:[8,9]}`; spine (path0) `[0,1,2,4,6,8]`;
branch nodes `[3,5,7,9]` (ALL leaves); all leaves `[3,5,7,8,9]`.

---

## 0. Provenance / discrepancy flags (reconcile BEFORE any verdict)

1. **E5 accept/event basis: 3.076 vs 3.21 vs 2.61.** Three legitimate numbers, three regimes —
   do not conflate.
   - **3.076171875** = saved E5 quick baseline, B=4, 8 prompts × 4 samples, `naive_mtp`,
     FLASH_ATTN, CUDA-graph (`quick_native_mtp5_same8.json`; also `FR13_LADDER_LOG.md:135`).
     This is the canonical superset reference for the quick-decode regime.
   - **3.2132796780684103** = a *fresh aligned native arm* run during the prior failed e2e
     (`FR13_LADDER_LOG.md:146`). This is the prompt's "3.21" — a different run, not a rounding of 3.076.
   - **2.61 / 2.69** = native E5 *self-noise* arms in the SWE-4 spp16 deliverable regime
     (`FR13_RESULTS.md:90-98`), the basis on which the **0.059326171875 bag-TV floor** was measured.
   The binding superset comparison must use whichever native arm is measured **in the same run /
   same regime** as the WY e2e. **Orchestrator: reconcile which E5 number the WY e2e is compared
   against before declaring superset.**
2. **The WY kernel e2e has NEVER been run.** All prior full e2e numbers are STALE for this build:
   - exp2/base-e tree (superseded, NOT WY): accept/event 0.92, TPS 4.80, bag-TV 0.558 (`FR13_RESULTS.md:100-109`).
   - pre-prefill-fix FA2 fork (NOT WY, predates `FR13_FA2_PREFILL_NATIVE`): accept/event **1.1134**,
     accept/token 0.124, bag-TV **0.5017**, TPS 2.67 (`FR13_LADDER_LOG.md:146-147`,
     `output/fr13_argmax_e2e_20260608T055851Z/`). This is a STRUCTURAL/prefill/mask-breakage
     signature (4× accept collapse, 0.50 bag-TV cannot come from a 1-ULP softmax-mass shift without
     en-masse argmax flips, which contradicts the argmax-lossless single-event finding).
   **Do NOT cite any of these as the WY verdict.**
3. **Decode-mode constants:** `src/lumo_flywheel_serving/fr10_decode_modes.py:20-23`:
   `NAIVE_MTP="naive_mtp"`, `NON_MTP="non_mtp"`, `TREE_MTP="tree_mtp"` (briefing's "L20 NON_MTP"
   is off by one line; the *value* `non_mtp` is correct). The oracle must be `non_mtp`.

---

## 1. BRANCH-ORACLE RECIPE (codex must add this to the e2e run)

### Why a SEPARATE oracle is mandatory
The captured native run (`native_final_logits.pt`, 6 rows) is **MTP-5 single-chain**, confirmed by
`native/logs/fr10_tree_draft_branch.jsonl`: `method=qwen3_5_mtp`, `num_speculative_tokens=5`,
`metadata_types=["FlashAttentionMetadata"]`, `tree_branch_seen=false`. MTP-5 is a caterpillar spine
`[0,1,2,4,6,8]`; off-spine branch nodes `3,5,7,9` have **no native counterpart**, so the spine
ladder (`gateA_spine_ladder.json` `logits.lhs_rows=[0,1,2,4,6,8]`, `rhs_rows=[0,1,2,3,4,5]`) never
validated them. Branch certification = running native independently on each branch's path-to-root
(SpecInfer Def 4.1 `TreeAttention(u)=Attention(S_u)`; STree Eq.4-6 ancestor-masked recurrence from
committed-prefix state == per-path output), per `reference_gdn_tree_branch_oracle_losslessness`.

### The four branch paths (root-to-leaf, from `parent=[-1,0,1,1,2,2,4,4,6,6]`; all 4 are LEAVES)
| branch node | parent chain | path-to-root | depth | oracle prefix (ancestors) | tree-row argmax | p_top(t1.0) |
|---:|---|---|---:|---|---:|---:|
| 3 | 1←0 | `[0,1,3]` | 2 | nodes `[0,1]` | 314 | 0.8502 |
| 5 | 2←1←0 | `[0,1,2,5]` | 3 | nodes `[0,1,2]` | 2972 | 0.6591 |
| 7 | 4←2←1←0 | `[0,1,2,4,7]` | 4 | nodes `[0,1,2,4]` | 5743 | 0.2228 (flattest — watch) |
| 9 | 6←4←2←1←0 | `[0,1,2,4,6,9]` | 5 | nodes `[0,1,2,4,6]` | 248068 | 0.3054 |

Each branch prefix is a **subset of the spine prefix** (`[0]⊂[0,1]⊂[0,1,2]⊂[0,1,2,4]⊂[0,1,2,4,6]`),
all already certified 6/6 argmax-lossless. So each branch oracle needs only the spine-prefix native KV
state plus the single branch token at the leaf — the divergence-of-interest is purely the leaf node's
logit. The branch-path tokens to feed = the committed per-node target tokens along each path, read from
`tree/logs/tree_path_lcp.jsonl` `self_target_ids` (verified: `tree_argmax[1:] == self_target_ids`
exactly — the captured logits faithfully reproduce the committer's per-node target tokens).

### Server config (the native-on-branch-path oracle) — same image/model/seed/dtype as the captured native run
- **decode mode = `non_mtp`** (`fr10_decode_modes.py:21`): plain autoregressive native, NO MTP drafting,
  so each branch path is verified as a real prefix. The oracle MUST be no-MTP — MTP-5 drifts ~6e-5
  (per `reference_fr10`) AND has no branch counterpart.
- **attention backend = FLASH_ATTN** for the 16 full-attn layers + native GDN
  (`fla_chunk_gated_delta_rule`) for the 32 linear-attn layers. NOT TREE_ATTN, NOT the WY kernel —
  this is the incumbent native path = the trial's reference.
- **temp 0.6, top_p 0.95** (committer regime); read the argmax directly from logits.
- **depth-based RoPE** consistent with the tree run's base contract (`fr10_tree_depth_positions.jsonl`:
  `state=num_computed_tokens_cpu-1, mrope=num_computed_tokens_cpu`). For each branch path the position
  ids are the linear path depths `0..len-1`.
- **same final-logit capture hook** (schema `fr13.final_logit_capture.v1`, source
  `Qwen3_5ForCausalLMBase.compute_logits`) so oracle logits are directly comparable to the tree tensor.

### What to run / capture
4 independent native `non_mtp` forward passes, one per branch path. Each feeds
`committed_prefix(prompt) ++ branch-path tokens`. The oracle = the **final-node next-token logit
vector** of each pass (full vocab 248320, fp32) + its argmax. Save as
`branch_oracle_node{3,5,7,9}.pt`.

### Exact comparison + PASS criterion (PRIMARY branch gate)
For each branch node `b ∈ {3,5,7,9}`: compare **PER-DEPTH ARGMAX** of the WY tree branch-row
(`tree_final_logits.pt` row `b`) against the native-branch-path-oracle final-node argmax. Validate the
**PER-NODE MARGINAL**, never path-joint (Traversal Verification arXiv:2505.12398 §3 — joint shows
spurious divergence). **Do NOT use max_abs** — branch raw logits legitimately differ from the spine via
the ancestry mask (the spine ladder already shows benign `max_abs` up to 3.32 on argmax-agreeing rows).
Optionally also report `1-TV(softmax_0.6)` per node as a distributional sanity number; the **binding
gate is argmax**.

> **PASS = argmax(WY_tree_row_b) == argmax(native_branch_path_oracle_b) for ALL FOUR branch nodes (4/4).**

That certifies each off-spine node is target-on-its-path (the committer's per-node `target_p` matches
native), so the `canonical_multidraft` rejection sampler
(`fr10_tree_rejection_sampler.py:170` `sample_deterministic_multidraft_rejection_step`,
`:206 accept_prob = min(1.0, float(p[token]/q_mix_token))`) is fed the correct target distribution at
every branch node → lossless. A single per-node argmax mismatch starting at the first branch row = a
mask/state-construction bug (cross-branch bleed), NOT a rounding residual — fix the WIRING, do not
re-tolerance.

### What I could pre-check WITHOUT the oracle (the captured native file has only the 6 spine rows)
- **Captured-tree branch-row internal consistency — ALL PASS** (recomputed from `tree_final_logits.pt`):
  | row | argmax | max | min | NaN/Inf | p_top(t1.0) | p_top(t0.6) | ties | top5 |
  |---:|---:|---:|---:|:--:|---:|---:|---:|---|
  | 3 | 314 | 20.000 | -7.094 | F/F | 0.8502 | 0.9857 | 1 | `[314,369,321,25,303]` |
  | 5 | 2972 | 21.375 | -4.688 | F/F | 0.6591 | 0.8530 | 1 | `[2972,5354,25,328,1510]` |
  | 7 | 5743 | 17.250 | -6.719 | F/F | 0.2228 | 0.4350 | 1 | `[5743,5354,25,65531,2972]` |
  | 9 | 248068 | 15.062 | -12.125 | F/F | 0.3054 | 0.7443 | 1 | `[248068,5423,760,71093,3710]` |
  No NaN/Inf, single argmax each, well-formed peaked distributions. (Matches briefing exactly.)
- **Cross-consistency:** tree-row argmax `[248068,3299,369,314,13,2972,248044,5743,198,248068]` ==
  `self_target_ids` offset-by-1 (verified equal) — branch argmaxes `(314,2972,5743,248068)` are exactly
  the `self_target_ids` the committer compares against.
- **Branch parent-row gate is on the certified spine:** every branch's parent is a spine node, and all
  4 parent rows argmax-match native (branch3 par1=3299, branch5 par2=369, branch7 par4=13,
  branch9 par6=248044 — all match). So branch *accept decisions* are driven by already-lossless spine
  distributions. The 4 branch rows are LEAVES — their own forward logits are never consumed to gate a
  downstream verify in this tree; they only seed the drafter's next proposal. **Therefore the ONLY
  branch-correctness risk is state-construction/mask bleed in the WY forward of rows 3,5,7,9, which the
  spine ladder cannot see — this is the genuine open gap and REQUIRES the oracle above.**

### Mask / cross-branch-bleed risk (the one thing to watch)
If the WY visible-mask folds a NON-ancestor node into a branch node's state accumulation, that branch
node is no longer target-on-path → it manifests as a per-node argmax MISMATCH vs the branch-path oracle
(NOT a rounding residual), typically a one-depth argmax LAG starting at the first branch row. The WY
kernel ancestry mask is `fr10_gdn_tree_kernel.py:516-521` (`m_strict`/`m_visible` load) and the
per-(i,j) visible gate at `:573-583` (`vis = tl.load(visible_mask + i*N_PAD + j)`). Detection = the 4/4
gate. If node3 (shallowest, fewest ancestors) fails → suspect deepest-spine-prefix mask wiring; if only
node7/node9 (deep) fail → suspect ancestry-mask depth indexing on the full-attn TREE_ATTN layers
(MRoPE/depth-position path, `mrope_depth_first_tree=[0,1,2,2,3,3,4,4,5,5]`). STree's diagonal shortcut
does NOT cover the non-diagonal `(I-βkkᵀ)` gated-delta, so there is no shared accumulator — each branch
needs its own ancestor-ordered WY/UT reflector product; a shared-state shortcut bug would surface here
as branch argmax lag. **node7 caveat:** p_top=0.2228 is intrinsically low-confidence; if 7 fails by a
near-tie, inspect the top-2 gap before calling it a mask bug.

---

## 2. PREDICTED E5-vs-TREE OUTCOME (cross-check from captured logits)

Overall confidence: **MEDIUM.** Strongest evidence is for the GREEDY selector (the captured run's active
policy: `independent_winner_trace.jsonl` `policy=greedy_tree_lcp_max`, `temperature=0.0`;
`tree_sampler_debug.jsonl` `all_greedy=true` on 12/12 decision events). The SWE-4 *deliverable* samples
at temp 0.6 / top_p 0.95, so the temp-0.6 path is extrapolated, not captured.

### 2a. accept/event — PREDICTED ≥ E5 (PASS), with a temp-0.6 caveat
**Grounding (recomputed):**
- **Spine 6/6 argmax-lossless.** Tree rows `[0,1,2,4,6,8]` vs native `[0..5]` → argmax
  `{248068,3299,369,13,248044,198}` match on all 6 (re-derived). Drift `max_abs` 1.44–3.32 lands on
  irrelevant vocab indices, never the argmax index; native top1−top2 margins dominate → no flip.
  Corroborates `docs/archive/wy/FR13_WY_RESIDUAL_CLOSURE.md` §3.
- **Greedy accept-direction is EQUAL to native** (not merely ≥). Replaying the 5 spine drafts
  `[760,3299,5354,13,248046]` against tree-argmax vs native-argmax at the parent rows gives accept/reject
  `(F,T,F,T,F)` **identical** tree==native on all 5. Under greedy LCP, spine accept/event is EQUAL to
  native, and the 4 leaf branches (top-2 siblings) can only ADD accepts (superset), never subtract
  (all are LEAVES). ⇒ predicted accept/event ≥ E5 **provided branch 4/4 holds**.
- Per-depth `tree_p(native_token)` = `0.995/0.986/0.684/0.643/0.702/1.000`
  (`1-TV` = `0.989/0.993/0.861/0.900/0.795/1.000`) — all assign healthy mass to the native target token.

**HONEST CAVEAT (greedy-capture vs temp-0.6 deliverable):** under the `non_mtp`/temp-0.6 rejection
sampler the per-token `accept_prob` uses the FULL target prob, which DOES shift bidirectionally:
depth2 draft 5354 `p_tree=0.0852` vs `p_native=0.1795` (**DOWN −0.094**), depth3 draft 13
`p_tree=0.6427` vs `0.5467` (**UP +0.096**), depth4 draft 248046 `p_tree=0.1195` vs `0.0626`
(**UP +0.057**). These ±0.1 shifts are small vs the argmax margin and bidirectional, so net direction on
accept/event is not obviously down and is dominated by the argmax-aligned greedy backbone — but it is
**NOT guaranteed ≥ E5 under the sampled path; must be MEASURED at B=4.**

### 2b. bag-TV — RISK (plausible PASS, NOT assertable from captured logits)
- **E5 self-noise floor = 0.059326171875** emitted-token bag-TV (`FR13_RESULTS.md:97`,
  `native_e5_self_compare.json`) — TWO independent native arms, IDENTICAL native target p, pure
  finite-sample multinomial noise, zero systematic marginal shift.
- The WY tree introduces a SYSTEMATIC per-slot marginal shift. Recomputed per-spine-row temp-0.6
  exact-marginal TV (tree-target vs native-target) = `[0.0109, 0.0068, 0.139, 0.100, 0.205, 0.0]`,
  **mean 0.077, max 0.205**. This is NOT 1-ULP — it is the GDN/WY 1.221e-4 L1 seam amplified ~27000× to
  final logits (`gateA_spine_ladder.json` `logits.max_abs=3.3203125`). Nucleus (top_p 0.95) SUPPORTS
  match exactly (`sym_diff=0` on every spine row) → no new tokens enter; the shift is mass-reallocation
  WITHIN the shared nucleus.
- **Why it can still pass:** the deliverable bag-TV is a GLOBAL multiset over ~8 prompts × 16 samples ×
  up to 64 positions = thousands of slots. Under the rejection sampler each emitted slot is
  target-distributed, so the systematic bag bias = `‖mean_i(p_tree_i − p_native_i)‖_1 / 2`, which
  benefits from heavy cancellation when drift sign varies across uncorrelated slots (here depth2 DOWN,
  depth3/4 UP). The realized global-bag systematic term is typically ≪ the mean per-slot TV (0.077).
  Combined with the same ~0.059 finite-sample noise, a pass is plausible.
- **Why RISK not PASS:** (a) single greedy event, 6 spine rows, B=1 eager — not the B=4 temp-0.6 regime;
  (b) mean per-slot TV 0.077 already EXCEEDS the 0.059 floor, so if cancellation is weak (correlated
  drift toward the GDN-amplified direction) the systematic term alone could push bag-TV over floor;
  (c) the prior FA2-fork e2e (pre-prefill-fix, NOT this WY build) hit bag-TV 0.5017 — structural-failure
  signature, excluded here, but it shows argmax-lossless-spine alone has not yet yielded a passing e2e.
  **Honest verdict: plausible PASS, not assertable from captured logits.**

### 2c. TPS — PASS likely (≥ native) on capturability/HBM grounds, but unmeasured
- Capturability independently confirmed: `TreeAttentionMetadataBuilder._cudagraph_support =
  AttentionCGSupport.UNIFORM_BATCH`, vLLM log `Capturing CUDA graphs (decode, FULL) 4/4` +
  `Graph capturing finished`, no `AttentionCGSupport.NEVER` downgrade (`FR13_RESULTS.md:15-34`).
  `FR13_FA2_PREFILL_NATIVE=1` + FULL confirmed in the closure doc. This removes the eager-mode and
  capture-fallback penalties that sank earlier tree arms.
- TPS ≥ native also rides on accept/event ≥ native (more accepts = fewer decode steps); since
  accept/event is predicted ≥ E5 (greedy), the TPS basis is favorable.
- **CAVEAT:** the only PRIOR full-e2e TPS numbers are from the SUPERSEDED exp2/base-e tree (4.80 TPS vs
  E5 16.5) and the pre-prefill-fix FA2 fork (2.67 TPS) — both structurally broken, NOT the WY+prefill-
  native build. **The WY kernel's TPS has NEVER been measured.** Capturability gates PASS; the absolute
  TPS must come from the B=4 timed e2e.

---

## 3. CRISP E2E CHECKLIST (preconditions → measurement → branch oracle)

**Preconditions (verify BEFORE the timed run; do NOT measure if any fail):**
- [ ] WY kernel CUDA-graph **FULL-captures + serves at B=4** (look for `Capturing CUDA graphs (decode,
      FULL) 4/4` + `Graph capturing finished`, NO `AttentionCGSupport.NEVER`). B=4 changes co-residency;
      re-confirm here, not from the B=1 capture.
- [ ] **Gate-2 hooks OFF** (no FR12 GDN diagnostic CPU-copy capture during the timed run; per
      `FR13_RESULTS.md:117-120` the capture-hooks path must be gated off for a clean number).
- [ ] **Prefill native-aligned:** `FR13_FA2_PREFILL_NATIVE=1` confirmed in the patched container
      `tree_attn.py` (the prior e2e fail was a pre-prefill-fix build). Backend = TREE_ATTN, forked FA2,
      splice OFF, `FR10_TREE_GDN_WY=1`.
- [ ] Re-confirm spine argmax 6/6 holds at B=4 (the §2a/§3-of-closure check is B=1 eager).

**Measurement (the binding deliverable verdict — vs E5, NOT vs a TREE_ATTN baseline):**
- [ ] Regime: **B=4 + CUDA-graph-captured + SWE-Verified-4**, `samples_per_prompt=16`, `max_tokens=64`,
      temp 0.6, top_p 0.95, seed pinned. Compare to a native E5 arm measured in the SAME regime.
- [ ] **LOSSLESS gate:** emitted-token bag-TV(WY-tree vs E5) ≤ E5 self-noise floor (~0.059326171875).
- [ ] **SUPERSET gate:** accepted_per_draft_event(WY-tree) ≥ E5 (reconcile which E5 number per §0.1).
- [ ] Report the full E5-vs-TREE table (accept/event, accept/token, bag-TV, token-count TV, first-token
      TV, exact-sequence match, warm decode TPS). **Do NOT self-declare PASS** — return the table to the
      user/orchestrator.

**Branch oracle (close the remaining verification gap — §1):**
- [ ] Run 4 native `non_mtp` forward passes on paths `[0,1,3]`, `[0,1,2,5]`, `[0,1,2,4,7]`,
      `[0,1,2,4,6,9]` (FLASH_ATTN full-attn + native GDN, depth RoPE, same capture hook), save
      `branch_oracle_node{3,5,7,9}.pt`.
- [ ] **PRIMARY branch gate:** per-node argmax 4/4 — argmax(`tree_final_logits` row b) ==
      argmax(native `non_mtp` branch-path-oracle final node) for b ∈ {3,5,7,9}. Per-node marginal, NOT
      max_abs, NOT path-joint. Report `1-TV(softmax_0.6)` per node as a sanity number.
- [ ] If any branch fails → it is a MASK/state-construction WIRING bug (`fr10_gdn_tree_kernel.py:516-521`
      / `:573-583`), fix the wiring; do NOT re-tolerance and do NOT use copy/splice/dense.

**Status of the DEV pre-checks (all already green from captured artifacts):** spine 6/6 argmax-lossless;
branch-row internal consistency 4/4 SANE; branch parents argmax-match native 4/4; capturability confirmed.
`gateA_spine_ladder.json` `passed=False` is on the WRONG basis (max_abs 3.32 vs threshold 0.0) — the
argmax gate passes 6/6. **These are DEV pre-checks; the binding verdict is the e2e + branch oracle above.**

---

## Bottom line
The captured evidence is **strong but not sufficient**: greedy spine accept/reject is provably identical
to native (6/6 argmax-lossless, identical accept-direction on all 5 drafts), the only-LEAF branch nodes
can only add accepts, and the residual is a 1-bf16-ULP floor that never reaches an argmax index. Predicted
e2e: **accept/event ≥ E5 (PASS), TPS ≥ native (PASS likely), bag-TV (RISK — plausible pass, mean per-slot
TV 0.077 > floor 0.059, relies on cross-slot cancellation).** Two things are genuinely unmeasured and must
be the e2e's job: (1) the temp-0.6 sampled-distribution behavior (captured run was greedy), and (2)
branch-row losslessness (spine ladder cannot see it — needs the `non_mtp` branch-path oracle). No
copy/splice/dense used; read-only throughout.
