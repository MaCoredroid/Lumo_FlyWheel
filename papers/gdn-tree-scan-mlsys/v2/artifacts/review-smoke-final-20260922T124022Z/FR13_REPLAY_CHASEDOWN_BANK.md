# FR-13 GDN Tree-Verify — Replay Chasedown BANK + SEQ-path proactive-fix list

Audience: the monitor + codex_fr19. Purpose: stop re-discovering already-solved GDN
tree-verify seams; port prior fixes directly; fix the next fronts ahead of the ladder.

- Repo root: `/home/mark/shared/lumoFlyWheel`
- HEAD at writing: `3cf98c21` (verified `git rev-parse HEAD`).
- **WORKING TREE IS DIRTY** (verified `git status --porcelain`): `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
  and `tests/test_fr10_phase4_sampled_committer_wiring.py` carry uncommitted edits.
  Those edits are the **NEW `FR13_TREE_MAMBA_INITIAL_SPINE_ROW`** seed-row work (see §B.0),
  **NOT** the conv write-back. **No gateA ladder number is trustworthy until they are committed
  and bound to HEAD** (verify-work-committed discipline).
- Patcher (monkeypatch) = `scripts/fr10_phase4_patch_vllm_tree_gdn.py`.
- Serving kernel = `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.
- VERDICT MECHANISM: every entry's resolution is confirmed by the **authoritative gateA per-layer
  ladder** (input→layer0→…→logits), NOT by self-declaration. "FIXED" below means a ladder
  number proved it; "AT-RISK"/"OPEN" means the ladder has not yet shown 0.0 there.

Honesty key on every line: **[PROVEN]** = confirmed against live code and/or committed ladder
docs in this repo (I verified the file:line and commit existence); **[NEEDS-CONFIRM]** = plausible
from history but the current gateA ladder has not yet shown it.

---

## A. THE BANK — catalog of replay / FR12-era GDN chasedowns (most load-bearing first)

Format: BUG / ROOT / FIX (commit:line) / CLASS (WIRING vs KERNEL vs MEASUREMENT).

### A1. GDN scan-out reduction order — 2-D tile vs 1-D `tl.sum` — **KERNEL** [PROVEN]
- BUG: sequential tree-scan `gdn_scan_out` diverged from native FLA by `1.19e-7` (2^-23) at
  depth-2, head43/dim31 (spine + row-2 nonzero).
- ROOT: the state-value lane used a different Triton reduction tree than native. FR12 fixed it
  (`beec984a`) with a 2-D `[BV, DIM_K]` axis=1 reduction; the SEQ rewrite `88212830`
  (register-resident single-walk) **regressed** it down to bare 1-D `tl.sum` / `BLOCK_V=1`.
- FIX: `e4a6a2f2` restored `BV=16` 2-D form. **LIVE in HEAD**:
  `fr10_gdn_tree_kernel.py:277` `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K))`,
  `:339` `b_v -= tl.sum(state_i * b_k[None, :], axis=1)`,
  `:341` `state_i += b_v[:, None] * b_k[None, :]`,
  `:342` `out_i = tl.sum(state_i * b_q[None, :], axis=1)`,
  `:547` `grid = (num_vh, triton.cdiv(dim_v, BV))`, `:572` `BLOCK_V=BV`, `:17` `BV=16`.
- CLASS: KERNEL (reduction order). CONSTRAINT: do NOT widen to BV=32 — `h_cache` 8KB→256KB
  spills past GB10's 255-reg cap (`6430339a` BV-spill verdict; `FR13_BV_SPILL_VERDICT.md`).

### A2. Verifier INPUT-hidden — MTP fc-fusion `prev_hidden` threaded wrong — **WIRING** [PROVEN]
- BUG: pre-layer-0 verifier input hidden differed at depths≥1 (input max_abs by depth
  `[0.0, 0.326, 0.387, 0.299, 0.107]`), propagating to ~25 final_norm drift. Tokens/RoPE matched.
- ROOT: Qwen3.5 MTP does EAGLE fusion `fc(cat(norm(embed(token)), norm(prev_hidden)))`; the
  depth-0-match/1-4-differ signature = stale `inputs_embeds` rows after tree scheduling rewrote
  the token layout.
- FIX: `e8a64eed` `_patch_gpu_model_runner_tree_verify_input_ids()` — rebuild target embeddings
  from the final scheduled `input_ids`. Sentinel `# FR13_TREE_VERIFY_INPUT_IDS`
  (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:5106,:5116,:5122`; wired `:7496`).
  Root doc `92077796` (`FR13_INPUT_HIDDEN_ROOT.md`).
- CLASS: WIRING (pre-layer-0 input buffer — the MEMORY "top-down ladder root").

### A3. Conv prior-window read column (HEAD-vs-TAIL) + accepted-len off-by-one — **WIRING** [PROVEN]
- BUG: tree conv `conv1d_out` diverged via carried history taps. FR10-era handoff `64.0`→`0.0`.
  FR13 L4 staircase: depth0 0.0556 → depth1 0.0176 → depth2 0.0039 → depth3-5 EXACTLY 0.0
  (only the 3 history columns differ; numeric ruled out by the clean depth-3 cutoff).
- ROOT (FR10): read root column 0 + an accepted-length temporal offset instead of the per-request
  accepted column. (FR13 L4 tail-read hypothesis was a MISDIAGNOSIS — see A3b.)
- FIX (FR10): `301f42fe` — gather prior conv bank row from
  `spec_state_indices[:, clamp(accepted_len-1)]`, window via `_fr10_prior_col_base`
  (HEAD cols [0,1,2]). LIVE: `scripts/fr10_phase4_patch_vllm_tree_gdn.py:918`
  `_fr10_prior_col_base = arange(width-1)`; read `:1015-1020`; `compact_head` tag `:1197`.
- CLASS: WIRING (column convention).
- **STILL OPEN** as the current GDN front — see §C.

### A3b. FR13 L4 tail-read "fix" — MISDIAGNOSIS, reverted — **WIRING (lesson)** [PROVEN]
- BUG: the L4 verdict `76eeb452`/`37a349f2` added `_fr10_use_rolled_tail_prior` (layer≥4) to read
  TAIL cols.
- ROOT of the misdiagnosis: TAIL cols [9,10,11] are EMPTY; data lives in HEAD [0,1,2]. Tail-read
  made L4 WORSE: baseline `0.0125732421875` → tail-read `0.0252` ; uniform write-back was byte-
  identical to baseline (a no-op).
- FIX: `52b8323e` reverted the tail-read (HEAD is back to `compact_head`); ROOT relabeled to the
  STORED conv-state CONTENT (write-back). `FR13_L4_CONV_NOOP_DIAGNOSIS.md`,
  `FR13_L4_CONV_VERDICT.md`. The new write-back candidate is §C (UNVALIDATED).
- CLASS: WIRING. LESSON: a band-aid that is byte-identical to baseline tells you the lever is
  elsewhere; do not ship a "fix" that the ladder does not move.

### A4. Conv tap-product dtype: fp32 vs native bf16×bf16 — **KERNEL** [PROVEN]
- BUG: tree conv computed fp32*fp32 tap products; native `causal_conv1d_update` does bf16*bf16
  → fp32 accumulate → fp32 SiLU. fp32 taps OVERSHOOT (conv 0.001→0.0625 when fp32-tapped).
- ROOT: Qwen3-Next GDN conv has `bias=False`, so the only seam is tap-product rounding.
- FIX: `ef32ab2d` `_fr11_conv_tap_product()` (`:938`), gated by `FR11_TREE_CONV_NATIVE_BF16_TAPS`
  default `"1"` (`:934`); applied at all conv-accumulate sites
  (`:1287,:1395,:1425,:1470,:1477,:1527`). FR12 `1aece020` confirmed bf16-tap = 0.0.
- CLASS: KERNEL (cast-boundary). NOTE: the SiLU ex2.approx replica (`f19b4da4`) REGRESSED L0 live
  (`80872723`/`7f950694`) and was abandoned; bf16-tap is the surviving lever.

### A5. Full-attn depth-RoPE / MRoPE position wiring — **WIRING** [PROVEN]
- BUG: tree rows got FLAT positions instead of caterpillar depth positions; flat MRoPE reached
  Qwen3Next full-attn RoPE → `q_after_rope` diverged 4.52.
- ROOT: stock vLLM builds positions as `num_computed_tokens + query_pos` (flat); caterpillar
  siblings must SHARE depth; plus a state-base vs MRoPE-base off-by-one.
- FIX: `bfc596d7` `_patch_gpu_model_runner_tree_depth_positions()` + `c5fc346d` separate
  `_fr10_mrope_depth_pos` with `_fr10_mrope_base = num_computed_tokens`
  (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:4829,:4848,:4853,:4859-4862`). Naive remap
  `7145c888` was REVERTED (`ac643736`/`3ab4a096`). Mixed-prefill guard `e6be5306`.
- CLASS: WIRING (position/layout).

### A6. Forked-FA2 TREE_ATTN prefill drift (missing descale / scheduler extras) — **WIRING** [PROVEN, fix NOT live]
- BUG: forked-FA2 TREE_ATTN prefill diverged from native FLASH_ATTN prefill at L7 `attn_out_raw`
  → GDN L8 `h0_state_in ≈ 7.2e-4` → contributes to 1.9 final-logit spine drift. Forked FA2 is
  byte-exact 0.0 in DECODE (Gate-2 `d2f1ba18`).
- ROOT: `TreeAttentionImpl` prefill routes through `unified_attention(...)`; native
  `FlashAttentionImpl` calls `flash_attn_varlen_func(...)` with extras tree drops
  (`scheduler_metadata`, `q_descale`/`k_descale`, `cu_seqlens`, `max_seqlen`, `fa_version`,
  `num_splits`). Live: `tree_attn.py:403-422` vs `flash_attn.py:806-863`.
- FIX: `patches/fr13_fa2_prefill_native.patch` (flag `FR13_FA2_PREFILL_NATIVE`, default OFF;
  `git apply --check` clean), committed via `ea7e4eb0`. **NOT applied to the live patcher** (would
  risk an in-flight e2e launch). Docs `FR13_PREFILL_DRIFT.md`, `9f349b6d`, `2f8ef1ea`.
- CLASS: WIRING (missing-extras helper, not kernel math — decode is 0.0).

### A7. Forked-FA2 vs Triton TREE_ATTN — irreducible MMA-grouping floor (NOT a bug) — **KERNEL (floor)** [PROVEN, accepted]
- BUG: TREE_ATTN (Triton, qq_bias mask) vs native FA2 (CUTLASS) `attn_out` ~0.00195/layer;
  byte-exact Triton→FA2 is provably impossible (fragment-fixed MMA + warp-reduction order).
- ROOT: two different kernels. On the SAME captured q/k/v with correct `prefix+ancestors+self`
  rows, TREE_ATTN is FA2-equivalent to bf16: 14/16 calls whole-tree 0.0, **2 single-bf16-ULP
  elements in ~1M** (`FR13_FULLATTN_OP_LOCALIZE_RESULTS.md`, `2264dd4b`).
- FIX (deliverable form): forked vllm-flash-attn FA2 csrc with an additive `tree_bias` tile added
  to `acc_s` after QK gemm before softmax (`FR13_FA2_TREE_BIAS_FORK.md`, `af870131`, `29d3c8bd`).
  User ACCEPTED the 2-ULP no-copy floor (`110d6c02`, `2264dd4b`).
- CLASS: KERNEL (irreducible floor — accepted; gate = within-E5-floor / argmax-lossless, not 0).

### A8. WY state-store built from bf16-tapped OUTPUT basis (over-rounded) — **KERNEL** [PROVEN — WY arm ARCHIVED]
- BUG: WY recurrent STATE drifted `1.66e-3` (output 1-ULP, state 13×) → compounded to 3.32 logits
  → 56% reject.
- ROOT: state-store was built from the bf16-tapped OUTPUT basis, not a raw-fp32 state track.
- FIX: `8a975837` separate raw-fp32 `_state` track → offline state `1.66e-3 → 6e-8`.
- CLASS: KERNEL (basis/rounding). WY whole-path FALSIFIED as bit-exact target (`30b7e2d7`: native
  verify dispatch is SEQUENTIAL rank-1, WY is prefill-only) → ARCHIVED off main, pivot to
  sequential (`20be68a5`). Kept for the recurring failure *pattern* (state from a rounded basis).

### A9. WY fp32-vs-native-FLA bf16 boundaries (l2norm / solve-T) — **KERNEL** [PROVEN — WY arm ARCHIVED]
- BUG: WY L1 `1.22e-4` (2 argmax flips); WY byte-exact to fp32 CPU oracle (4.19e-9) but 1 bf16
  ULP off live FLA — WY was MORE accurate than the incumbent.
- ROOT: native FLA rounds at 4+ boundaries WY didn't (l2norm store, solve-T, KKt input,
  state-update v/k).
- FIX: `1f0c7237` `FLA_BF16_BOUNDARIES` constexpr + `79838bb9` output tap → spine output
  1.22e-4→0.0.
- CLASS: KERNEL (cast-boundary). LESSON: bar = bit-exact-to-incumbent, not ℝ-correct.

### A10. GDN scan row-select made fp32 op-order N_PAD-dependent (batch-invariance) — **KERNEL** [PROVEN]
- BUG: a single spine row's scan output depended on N_PAD / co-resident rows (#42960 GDN-scan).
- ROOT: row-select-by-reduction over the padded node axis = batch-dependent fp32 reduction.
- FIX: `beec984a` — replay each node's ancestor path in native statement order with DIRECT row
  loads; compute g/beta from raw a/b/A_log/dt_bias INSIDE the kernel (added `raw_a/raw_b/A_log/
  dt_bias` to `launch_tree_gdn_prepared`). Scan → 0.0 and N-independent; residual state delta
  4.77e-7 fp32-internal. This is the direct-load substrate the SEQ kernel (A1) builds on.
- CLASS: KERNEL (batch-invariance / reduction independence).

### A11. fp8 GEMM (o_proj/in_proj) per-token-group quant batch-invariance — **KERNEL** [PROVEN]
- BUG: after conv=0.0 and scan~5e-8, last L0 residual `o_proj_out = 1.2e-4` (fp8 GEMM),
  compounding to a TV p90=1.0 tail.
- ROOT: tree's branched row-layout makes per-token-group fp8 scale round differently than native's
  linear chain (#42960). The "scan bf16 boundary" lever (`f1270afb`) was a WRONG guess — REJECTED
  (`64aa5809`: forcing bf16 made scan 0.0078, 4 orders worse).
- FIX: driven to 0.0 in the L0 hard gate (`3d3f30e2`/`62516997`): scan/gate/o_proj/conv/in_proj all
  0.0 spine AND branches via `VLLM_BATCH_INVARIANT`-class alignment
  (`LUMO_BATCH_INVARIANT_VLLM=1`, FLASH_ATTN backend).
- CLASS: KERNEL (fp8 batch-invariance). METHOD: drive scan→gate→o_proj to 0.0 BEFORE other work;
  o_proj is a SYMPTOM of upstream input drift.

### A12. Deep-layer GDN spine onset L24/L45 (compounding) — **KERNEL/state** [PROVEN — decomposed]
- BUG: with GDN sub-ops 0.0 at L0, the 64-layer ladder first diverged deep (branched L24
  hidden 0.035/residual 0.25 → 5.25 at L63; spine-only L45). NOT branch-aligned → not contamination.
- ROOT: an L0-class seam that is bit-exact for the first 23 GDN layers then surfaces — the scan
  reduction (A1) and conv content (A3), amplified ~32× by the gate (1/rms) and compounded.
- FIX: decomposed by the top-down ladder into A1 (cleared L0-3) and A3 (open L4). Docs
  `2ffa705a`, `06c9defc`, `FR13_GATEA_DEEP_DIVERGENCE.md`.
- CLASS: KERNEL/state. LESSON: a "deep onset" is usually L0-class drift compounding — localize
  with the per-layer ladder, don't grind the deep layer.

### A13. SSM cross-step h0 handoff — MEASUREMENT artifact (EXCLUDED) — **NEITHER** [PROVEN]
- BUG: apparent SSM cross-step failure; conv looked `64.0`.
- ROOT: row-space ORACLE bug (wrong bank row) AND comparing native `conv_state` AFTER
  `causal_conv1d_update` rolled it. PRE-update capture: tree prior rows match native bit-exact 0.0
  (`FR12_CONV_STATE_DIAGNOSTIC.md`; FR11 probe-beta `a747220c`: h0 accepted-row 0.0 →
  `MATCH_WRONG_INITIAL_STATE_EXCLUDED`, #40738-class). The conv `64.0` was a separate real
  handoff-column bug = A3, fixed `301f42fe`.
- FIX: no kernel fix — capture PRE-update; use the accepted-column row, not column 0. Graph-safe
  remap `d2e28e4e`/`launch_tree_state_linear_remap` (`fr10_gdn_tree_kernel.py:170-203`) copies
  accepted-path ROWS (does NOT roll conv columns — EXONERATED as corruptor).
- CLASS: MEASUREMENT/oracle. LESSON: always capture native recurrent/conv state PRE-update.

### Cross-cutting bank notes (reusable)
1. Capture native conv/recurrent state **PRE-`causal_conv1d_update`** (A3, A13).
2. Bar is **bit-exact-to-incumbent, not ℝ-correct** (A9 was 4.19e-9 correct yet 1 ULP off).
3. fp32 reduction order is **batch-dependent** unless made N-independent (A1, A10, A11; #42960).
4. `o_proj_out` drift is usually a **symptom of upstream input drift** — drive conv→scan→gate→
   o_proj to 0.0 top-down (A11).
5. A "deep-layer onset" is usually **L0-class drift compounding** (A12).
6. **Splices are oracle-only** (`FR12_TREE_*_NATIVE_SPINE` ignored unless
   `FR12_NATIVE_SPINE_ORACLE=1`); a green number from calling native on the spine is rejected.
7. **WY is archived off main** (sequential rank-1 is the only served GDN tree path); A8/A9 are kept
   for the failure *patterns*, not live code.

---

## B. PROACTIVE-FIX LIST for the current SEQ tree-scan — RANKED, most-impactful first

State of each cataloged pattern in the live sequential path (`use_wy=False`,
`_tree_gdn_kernel` `fr10_gdn_tree_kernel.py:207-353`; monkeypatch `scripts/...`).

### B.0 (rank 0 — DISCIPLINE BLOCKER) Uncommitted seed-row work not in HEAD [PROVEN]
The dirty working tree (`git status`) adds, to the patcher, a NEW seam:
`FR13_TREE_MAMBA_INITIAL_SPINE_ROW` — align-mode Mamba state-row geometry must use the native
causal SPINE, not the full tree. Two coupled edits:
- `_patch_gdn_attn()` discounts side-leaves from `seq_lens` before `spec_state_indices_tensor` is
  sliced (diff hunk at `scripts/...:329`).
- `_fr13_tree_mamba_initial_seed_tokens()` + its preprocess wiring in `collect_mamba_copy_meta`
  remap the initial seed-token count `tree_n → path0_n` (diff hunks at `:5167`, `:5286`; also adds
  `import ast`).
This extends pattern (5) "prefill/input seeding". **It is NOT committed.** Per verify-work-committed:
the monitor must require fr19 to `git add` + commit these (feature branch) and confirm
`git grep` finds them in HEAD before ANY gateA ladder number is trusted — otherwise a "fixed"
reading evaporates from HEAD (the A2 input-fix-lost precedent). RANK 0 because it gates the
trustworthiness of every number below it.

### B.1 (rank 1 — THE LIVE BLOCKER) Conv-state write-back CONTENT — A3, RE-OPENED, AT-RISK [PROVEN open; candidate NEEDS-CONFIRM]
- STATUS: this is the current stuck L4 front. The READ is correct/reverted (compact HEAD cols
  `:918`,`:1015-1020`). The OPEN piece is the stored conv-state CONTENT (the handoff write-back).
- LIVE candidate (HEAD `3cf98c21`, UNVALIDATED): `_fr10_node_state_source =
  cat(prior_window^T, node_x, zeros(state_len))` then `index_select(node_path.numel()+arange(
  state_len))` (`scripts/...:1308-1335`), plus a NEW `get_conv_copy_spec` override in
  `mamba_utils.py` (`_patch_mamba_state_utils_tree_conv_node_copy`, `:5415-5474`) that copies the
  full accepted node row `src_block_pos = cur_block_idx + num_accepted_tokens - 1` (`:5438`).
- MEASURED open gap (authoritative gateA, `FR13_L4_CONV_NOOP_DIAGNOSIS.md`): L4 conv1d_out
  first_nonzero **0.0125732421875**; tail-read made it 0.0252; uniform write-back was byte-
  identical (no-op). So the candidate write-back has NOT yet been shown to reach 0.0.
- ACTION for fr19: re-ladder L4 (filtered, PRE-kernel native capture) → require conv1d_out
  staircase `0.0556 → 0.0`. If residual remains, store native's EXACT `[history…, accepted…]`
  rolled-tail ordering (`FR13_L4_CONV_VERDICT.md` step 3/4).
- Couples to B.2 (the remap does not roll conv columns, so the column reconciliation MUST live in
  this write-back).

### B.2 (rank 2 — STRUCTURAL, coupled to B.1) State-handoff remap rolls ROWS not conv COLUMNS — A13/A3 [PROVEN]
- `launch_tree_state_linear_remap` / `_linear_remap_rows_kernel` (`fr10_gdn_tree_kernel.py:84-203`)
  copies WHOLE rows (`ROW_ELEMS = state.stride(0)`, `:118-120`) keyed accepted_paths→bank columns.
  Same instance is applied to both ssm_state and conv_state (`:186-203`) with identical whole-row
  semantics — CORRECT for ssm_state, INSUFFICIENT for conv_state's intra-row ring-buffer window.
- EXONERATED as the corruptor (no-op at accepted_len=0, `FR13_L4_CONV_VERDICT.md` line 12) but it
  is the structural reason the conv layout is never reconciled to native's rolled-tail.
- ACTION: do NOT add column-rolling here (graph-safety / ssm correctness). The fix belongs in B.1.

### B.3 (rank 3 — settled on SPINE, WATCH on BRANCHES) Caterpillar source-indices — A-class [PROVEN]
- Correct in SEQ: per-node ancestry window built at `:256-272` (metadata builder), consumed via
  `_fr10_source_flat = fr10_tree_conv_source_indices` (`:915`,`:1025-1027`). The FLAT-positional
  variant is diagnostics-only.
- WATCH: the caterpillar mislabel bites at depth≥3 — masked on the spine (spine output already 0.0
  there) but REAL for branches (`FR13_L4_CONV_VERDICT.md` line 6). This becomes a front when the
  branch oracle runs (§C).

### B.4 (SETTLED — do not re-discover) [PROVEN]
- **A1 scan 2-D reduction** — FIXED, live in HEAD (`:339/:341/:342/:547/:572`). Layers 0-3 bit-exact
  (`FR13_FR19_HANDOFF.md`). Do NOT regress to 1-D `tl.sum` / `BLOCK_V=1`; do NOT widen to BV=32.
  Sub-note: the in-kernel l2norm reductions `:334-335` (`tl.sum(b_q*b_q)`) are still bare 1-D but
  are scalar DIM_K normalizers gated by `USE_QK_L2NORM_IN_KERNEL` (OFF in the raw-gating path) —
  NOT the state contraction, NOT implicated in any measured divergence. Flag only if a future
  ladder points at a qk-norm channel.
- **A2 input-hidden** — FIXED (`# FR13_TREE_VERIFY_INPUT_IDS`, `:5106`). Input/pre_conv bit-exact
  0.0 (`FR13_SEQ_E2E_ROADMAP.md`).
- **A4/A6 bf16-tap conventions** — the SEQ scan is PURE fp32 (no bf16/FLA in
  `fr10_gdn_tree_kernel.py`; grep empty). The WY `FLA_BF16_BOUNDARIES` taps and the beta-bf16
  "fix" are the WRONG native for verify (oracle beta is fp32; matches SEQ `:332`). **DO NOT add
  bf16 to the SEQ scan** (`FR13_SEQ_SCANOUT_FIX.md`, `FR13_SEQ_LAYER2_REDTEAM.md`/`48c261b6`). The
  one load-bearing bf16-tap is in the CONV path (`_fr11_conv_tap_product`, `:938`), already aligned.
- **A10/A11 batch-invariance / fp8** — structurally satisfied: the SEQ kernel is a per-node
  `tl.static_range` walk (`:278-353`) resuming from the parent's `h_cache` checkpoint
  (`:280-286,:344`) — no cross-node accumulator whose order depends on N. The only N-dependent
  reduction (the 2-D state tile) is B.4-A1. fp8 GEMMs are outside this kernel, banked
  batch-invariant in FR12.

---

## C. Is the CURRENT stuck L4 conv prefill-seed bug a KNOWN replay chasedown? — YES [PROVEN]

**Yes — it is BANK entry A3 (conv prior-window / state-handoff convention), re-opened.** fr19 should
PORT, not re-derive:

1. **The proven READ fix** is `301f42fe` — gather the prior conv bank row from
   `spec_state_indices[:, clamp(accepted_len-1)]` (same linear-column contract as SSM), read the
   window with `_fr10_prior_col_base` HEAD cols. That fix is ALREADY LIVE in HEAD
   (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:918`, `:1015-1020`, `compact_head` `:1197`). **Do
   NOT re-touch the read** — the tail-read variant (`76eeb452`/`37a349f2`) was overturned as a
   misdiagnosis (`52b8323e`) and made L4 WORSE (0.0126→0.0252).

2. **The remaining piece is NEW within A3: the write-back CONTENT** (the handoff store), which the
   FR10/FR12 commits never had to reconcile because they predate the rolled-tail divergence at L4.
   The prior art that ports DIRECTLY is the methodology + capture caveat, not a single line:
   - From A13 (`d2e28e4e`, `FR12_CONV_STATE_DIAGNOSTIC.md`): **capture native's prior window
     PRE-`causal_conv1d_update`** (post-update reads roll the buffer and fake a divergence — the raw
     `6.05` was this artifact). Expect tree-col1 ≈ native-col0 (a 1-column roll), not a value seam.
   - From A4 (`ef32ab2d`): the conv tap arithmetic is already native-aligned (bf16 taps) — so this
     is purely a STORE-ordering/content fix, NOT a numeric/kernel one (the depth-3 clean cutoff
     proves it).
   - Target per `FR13_L4_CONV_VERDICT.md`: store native's exact `[history…, accepted…]` rolled-tail
     ordering so the NEXT step's HEAD-read matches; re-ladder to conv1d_out `0.0556 → 0.0`.

**Bottom line for fr19:** the read is solved-and-live (port = leave it alone); the write-back is
A3's never-before-built tail half — build it against the PRE-update native window capture
(reuse A13's diagnostic), measure on the authoritative gateA L4 ladder (0.0126), NOT the
geometry-confounded subop 0.0556. **CLASS = WIRING** (column convention / store ordering), not
kernel — do not build a kernel for it.

---

## D. NEXT FRONTS — predicted, so we fix ahead of the ladder

Ranked by when the ladder will hit them.

### D.1 Remaining GDN layers cascade (IMMEDIATE) — [PROVEN cause, NEEDS-CONFIRM the candidate clears it]
- Cause: B.1 conv write-back content + B.2 un-rolled conv columns. With A1 clearing L0-3, the L4
  conv content (0.0126) is THE next first-diverge; once it's 0.0 the next first-diverge moves
  deeper (expect the same conv-content seam re-surfacing at each GDN layer whose carried conv-state
  was last written by the tree-verify handoff — A12's compounding pattern). The HEAD write-back +
  `get_conv_copy_spec` override is UNVALIDATED against gateA.
- Fix-ahead: validate conv1d_out→0.0 at L4 FIRST; then re-ladder forward expecting the SAME fix to
  clear subsequent GDN layers (do not re-root each one — it is one seam, A3, repeated).

### D.2 Branch oracle (per-depth argmax) — [PROVEN watch-items]
- Cause: B.3 caterpillar source-index is correct on the spine but REAL at depth≥3 for branches;
  plus the SEQ ancestry-resume `tl.where` (`fr10_gdn_tree_kernel.py:280-286`) — cross-branch
  state-bleed shows as per-node argmax LAG (`FR13_SEQ_E2E_ROADMAP.md`). Gate per the bank's branch
  losslessness theorem: native-on-leaf-path oracle, **per-depth argmax 4/4**, NOT max_abs, NOT a
  joint distribution (validate per-node marginal).
- Fix-ahead: before the branch ladder, confirm the metadata builder feeds per-node ancestry windows
  to branches (not just the spine) and that the `strict_mask` ancestor gate at `:280-286` selects
  ONLY true ancestors (cross-branch bleed = a mask bug, per MEMORY's GDN-branch losslessness note).

### D.3 full_attention (16 layers, idx 3,7,…) — OTHER subsystem — [PROVEN it's not the GDN kernel]
- Cause: forked FA2 `.so` + `tree_attn.py` vs `flash_attn.py` (A6 prefill missing-extras ~7.2e-4,
  A7 decode ~0.00195 irreducible floor + depth-RoPE wiring A5). NOT in `fr10_gdn_tree_kernel.py`.
- Fix-ahead: A5 (depth-RoPE) and A7 (tree_bias fork, 2-ULP floor accepted) are done; A6 prefill is
  WRITTEN but flag-OFF (`patches/fr13_fa2_prefill_native.patch`, `FR13_FA2_PREFILL_NATIVE`). When
  the ladder reaches a full-attn layer with a prefill in the window, ENABLE A6's flag (apply the
  patch) rather than re-deriving. **Do NOT grind the GDN kernel for a full-attn divergence.**

### D.4 e2e verdict (after all per-layer 0.0) — [PROVEN methodology]
- Per-layer 0.0 is a DEV check only. The deliverable verdict is e2e vs **E5** (FLASH_ATTN native
  MTP-5, `output/fr10_native_mtp5_same8_*`): LOSSLESS = our distribution within E5 self-noise floor
  (NOT vs a TREE_ATTN baseline), SPEED = accept/event + TPS ≥ E5. Final gate = **B=4 +
  CUDA-graph-captured + SWE-Verified 4 tasks** (NOT eager/B1/toy — B=4 changes co-residency; the
  A7 2-ULP floor and any conv content must be re-confirmed there). Confirm the kernel graph-captures
  (hooks OFF) before measuring. This is a pass/fail/close gate — STOP and ask the user before
  declaring it.

---

## E. Honesty ledger
- **[PROVEN] against live code (I read the file:line):** A1, A2, A3/A3b, A4, A5, A13 code sites;
  B.0 (the exact uncommitted diff hunks); B.1 write-back `:1308-1335` + `get_conv_copy_spec`
  `:5415-5474`; B.4 SEQ kernel structure; the dirty working tree; all 19 cited commit hashes exist
  (`git cat-file -t`).
- **[PROVEN] against committed ladder docs:** the L4 0.0126/0.0252 numbers
  (`FR13_L4_CONV_NOOP_DIAGNOSIS.md`, `FR13_L4_CONV_VERDICT.md`), the staircase 0.0556→0.0, layers
  0-3 bit-exact (`FR13_FR19_HANDOFF.md`).
- **[NEEDS-CONFIRM] (gateA has not yet shown it):** that the HEAD write-back candidate + the dirty
  seed-row edits reach conv1d_out→0.0 at L4 (B.1); that the same fix clears deeper GDN layers (D.1);
  the branch per-depth argmax 4/4 (D.2). NO self-declare — the gateA ladder confirms each, AFTER
  the dirty edits are committed (B.0).
