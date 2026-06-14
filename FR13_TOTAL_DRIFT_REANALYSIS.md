# FR13 — TOTAL DRIFT REANALYSIS: fresh, skeptical, independent accounting of the 21 baked flips

Date 2026-06-14. READ-ONLY reanalysis (no GPU). Author: fresh subagent, NOT taking the prior binds on
faith. Inputs: the kernel patcher `scripts/fr10_phase4_patch_vllm_tree_gdn.py`, the serving kernel
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`, the locked launcher `scripts/fr13_launch_locked.sh`,
git log (97 FR13 commits + 27 remote branches), the banked flip records
(`output/fr13_shape_sweep/*_flips.json`, `output/fr13_verify_decisive/q3_*_classify.json`,
`output/fr13_node7_ladder/ladder_summary.json`), and online research (STree 2505.14969, lossless-specdec
literature). GB10 context: B=1 decode is **bandwidth-bound** (273 GB/s LPDDR5X), native at ~45% peak BW;
the in_proj_ba pad is speed-neutral precisely because the extra padded GEMM rows hide behind the bf16
weight DMA. That same bandwidth-bound regime is why M-keyed GEMM autotune picks shape-dependent kernels.

THE MOTIVATING FACT (re-confirmed from code + records): the in_proj_ba M-invariance pad is BAKED into
locked cat9 (`a666f9ec`, launcher L34-35 `LUMO_FB_KERNEL_ROWS=1` + `LUMO_FB_PROJ_PAD_ROWS=16`), VerifyGate
HOLD, but the baked build still shows **21 clear-margin flips [3,6,6,6]** vs native 3
(`FR13_BAKE_B1_HOLD_BIND`). So in_proj_ba was NOT the dominant driver — at best it removed ~3-5 same-boot
(26→18-21 spread is dominated by the ±3-4 cross-boot autotune fork floor, `feedback_no_cross_boot_byte_gate`).

---

## 1. FULL ACCOUNTING of the 21 baked flips (de-cascaded to independent events)

### 1a. The raw records and the cascade problem
The flip count is measured per-token-argmax vs each arm's OWN no-spec decode oracle (threshold dev≥1.0
nat), prompts_swe4. The binding instrument is the per-token argmax probe
(`reference_scalar_metric_per_token_blindspot`); the raw count is **necessary-never-sufficient** (bug class
#12, Measurement traps) because a single upstream argmax fork **cascades**: once a served token diverges,
the oracle is teacher-forced on the now-different served prefix and reads downstream positions as additional
"flips" though only ONE decision diverged. The de-cascade rule (FR13_PLUS2_DECASCADE, applied IDENTICALLY to
all arms incl. native): a clear flip is an INDEPENDENT event iff isolated from the prior clear flip by a
position gap and the stream re-converges to dev=0.000 immediately after; a CONTIGUOUS run that re-converges
is ONE event.

The **[3,6,6,6]=21 fingerprint** is the locked-bake boot (`FR13_BAKE_B1_HOLD_BIND` L18-20: served_lens
[104,116,128,128], stream sha1 [d32193ec,4df82e33,7c068e7e,b39b0580]). The re-run is BLOCKED so I worked
from the nearest banked baked-build record: **BA_PROJ_BI_ON** (in_proj_ba pad ON), raw 18 [4,4,4,6], and the
unbaked **q3_tree** (raw 22 [5,7,4,6]). The 18↔21 spread IS the cross-boot autotune floor — same lossless
build, different boot.

### 1b. De-cascade (my own independent pass, gap≥4 isolation heuristic)
| arm | topology | raw clear | INDEPENDENT events (my pass) | accept/event |
|---|---|---|---|---|
| **native E5** (FLASH MTP-5) | 5-spine | **3** | **3** (p1:1 p2:1 p3:1, all isolated) | 3.076 |
| **chain5** (our kernel, no branches) | 5-spine | 5 | **2** (decascade FR13_PLUS2: pos25-29 ONE fork + pos43 brace) | 2.664 |
| chain3 (our, shallow no-width) | 3-spine | 5 | **5** (dispersed, no cascade) | 2.295 |
| cat3w (our, shallow + width) | D3 + leaves | 25 | ~22 | 2.108 |
| **BA_PROJ_BI_ON (BAKED build)** | cat9 9-node | **18** | **~14** (p0:2 p1:4 p2:4 p3:4) | 3.017 |
| BA_PROJ_BI_OFF (pad OFF, same boot) | cat9 9-node | 26 | ~15 | 2.703 |
| cat9_bi (unbaked + BI) | cat9 9-node | 34 | ~20 | 3.109 |

### 1c. The QUANTIFIED breakdown of the baked ~21 (= native 3 + ~18 other, de-cascaded to ~11-13 independent)
Decomposing the baked build's ~14 independent events (from BA_PROJ_BI_ON, the cleanest baked record):

- **native floor = 3 independent** (irreducible; p1/94 prose-vs-codefence, p2/33 quote-style, p3/68
  prose-vs-codefence). These are genuine high-entropy format boundaries where ANY same-precision realization
  can cross. NOT removable (native itself crosses them).

- **+2 our-spine-vs-native intrinsic** (chain5 decascades to 2 ≤ native 3 = AT-OR-BELOW native on the DEEP
  spine; chain3 shows ~+2 real dispersed on a SHALLOW spine). This is the per-layer ~1-bf16-ULP realization
  gap of OUR GDN/attn kernels vs native's, accumulated over the spine. Diffuse, not a single seam. The deep
  spine masks it under cascade; the shallow tree exposes ~2 dispersed independent crossings. Bound: small,
  alignment-territory.

- **+~9-12 leaf/branch co-residency** (the dominant residual, the real target). The decisive evidence
  (`FR13_FA2_CARRIER_OVERTURNED_BIND`, commit 2fe2c567): **11/11 channel-2 flips land ON the spine, 0 on
  leaves** = SPINE_PERTURBATION. The branch (leaf) rows, by co-residing in the same batched verify forward,
  perturb the SPINE rows' computation by ~1 bf16-ULP/layer, which compounds and flips spine argmaxes. Note:
  chain5 (no branches) is at ~5 raw / 2 indep; cat9 (with 4 leaves) is at ~18-22 raw. The branches ADD the
  bulk. in_proj_ba (the ONE bf16 GEMM at L0 GDN) removed only ~3-5 of these → the co-residency carrier is
  NOT only in_proj_ba.

### 1d. Are these the SAME boundaries as native, or distinct? (decisive — answered fresh)
**DISTINCT.** I cross-checked positions:
- native 3 boundaries: {p1/94, p2/33, p3/68}.
- baked BA_PROJ_BI_ON 18 positions: overlap with native = **exactly 1** (p3/68 — the same prose-vs-codefence
  fork). The other 17 are at positions native does NOT flip (p0/34-37, p1/27/50/67/115, p2/21/74/77/95,
  p3/72/81/107/109/124).
- unbaked tree 22 positions: **zero overlap** with native's 3.

So the tree's extra flips are NOT "more of the same diffuse boundary set" — they are a near-disjoint SUPERSET
of boundaries that the tree-path crosses and native does not. This is consistent with **the served stream
itself diverging** (different served_lens, different downstream context) AND with the tree-verify being a
DIFFERENT numerical dispatch than no-spec decode (online research, below). It also means the raw count is
inflated by trajectory divergence (bug class #12): once the tree path forks early, ALL subsequent boundaries
are scored on a different prefix → not like-for-like with native.

### 1e. Honest bottom line on the count
The baked ~21 raw ≈ native 3 + ~2 spine-intrinsic + ~9-13 co-residency (decascaded to ~11-13 independent).
The deployable arbiter remains **accept/event = 3.0-3.15 ~ native 3.076** (the flips are near-ties /
trajectory forks that barely move the average — `reference_scalar_metric_per_token_blindspot`: a 4% argmax
flip rate hides in the accept band). HALF the flips (10/22) are dev≤2.5 nat near-ties; the other half
(12/22, up to 9.75 nat) are LARGE — but the large ones are overwhelmingly CASCADE consequences (oracle
confidently re-predicts on a diverged prefix), NOT 12 independent large-margin kernel defects.

---

## 2. RE-CHECK of every RULED-OUT channel against the actual kernel code (fresh read, not the bind)

| channel | prior ruling | my fresh-read verdict | evidence |
|---|---|---|---|
| **GDN scan** (BV/warps geometry) | bit-exact to native, D16=D32=0.0 | **HOLDS** | `fr10_gdn_tree_kernel.py:330-383` shared `_gdn_node_step` body: the reduction is `tl.sum(...,axis=1)` over **DIM_K** within each V-row; `BLOCK_V` only re-tiles WHICH V-rows a program owns, never the K-reduction order → geometry-invariant. FR13_BV_GEOMETRY measured RAW 0.0 ours-vs-REAL-native at D16/D32, N_PAD 1 and 16. The negative-control (independent fp32 torch scan = 0.0078 = 1 ULP) proves the harness reports true non-zeros. **HOLDS.** |
| **conv1d** (tree-fused) | row-M-invariant, per-row no GEMM | **HOLDS** | our fused tree conv is per-row (no batched GEMM); the prior-window bug (`project_fr13_conv_priorwindow_root`, 18.375 at num_accepted>1) was FIXED wiring. No M-keying. **HOLDS.** |
| **fp8 in_proj_qkvz + o_proj** | M-invariant, BLOCK_SIZE_M=64 constexpr | **HOLDS at cat9 geometry** | `_patch_fp8_utils_gb10_gemv_cfg` (L13602): GB10/sm_121 has NO tuned JSON → stock DEFAULT config `BLOCK_SIZE_M=64, BLOCK_SIZE_N=block_size[0], BLOCK_SIZE_K=block_size[1]=128, num_warps=4`. cat9 verify M = 6-10 (one tree) up to ~54-63 (packed) — all ≤ 64 → ONE M-tile → identical K-accumulation order regardless of active row count. The fp8 K-reduction `accumulator += tl.dot(a,b)*a_s*b_s` over `range(cdiv(K,128))` is M-independent. **HOLDS** — caveat: if a packed verify batch ever exceeds 64 rows the 2nd M-tile changes nothing on the reduction axis (still per-row). FR13_GB10_FP8_GEMV_CFG override is DEFAULT-OFF (locked build uses stock 64). |
| **gate (RMSNormGated)** | M-invariant, per-row rms | **HOLDS** | ROWS_PER_BLOCK=1 both M; rms is per-row. No cross-row reduction. **HOLDS.** |
| **FA2 fork (additive -inf tree bias)** | 14/16 calls whole-tree 0.0, 2 single-ULP floor | **HOLDS but NOT THE LIVE PATH** | `fr13_patch_fa2_tree_bias.py` adds `apply_tree_bias` after QK. BUT the locked decode backend is **TREE_ATTN, not FA2** (launcher `ATTENTION_BACKEND=TREE_ATTN`; FA2 fork is for PREFILL via `FR13_FA2_PREFILL_NATIVE=1`). So the FA2 2-ULP floor applies to prefill, not the decode tree-verify. **HOLDS for prefill; see §3 for the live TREE_ATTN seam.** |
| **oracle frame** (chunk-vs-recurrent) | the recurrent re-score found flips REAL (native 3/3, spine 5/5 byte-id, cat9 22/20 ours-only) | **HOLDS — but mis-states the reference** | The 22 flips and the L0 ladder are measured vs the **no-spec DECODE oracle** (a `fused_recurrent` sequential kernel = DIFFERENT dispatch than the tree-verify scan). "5/5 byte-id across chunked AND recurrent oracles" proves the count is not a chunk-vs-recurrent FRAME artifact, NOT that the tree-verify == native MTP. **FLAGGED**: the L0 first-nonzero 0.0078 vs decode oracle (§3) is partly the legitimate tree-verify-vs-sequential-decode dispatch difference, which is the academically-EXPECTED lossless gap (online research §4), not necessarily a co-residency carrier. |
| **reshape** (depth/width) | depth dead (chain3=chain5=5); width adds co-residency (cat3w 25 >> chain3 5) | **HOLDS** | chain3 (D3, no width) 5 ≈ chain5 (D5) 5 → depth dead. cat3w (D3 + leaves) 25 → WIDTH adds co-residency even at shallow depth and even with strict-mask-invisible root siblings. **HOLDS.** |
| **BI / warps scan codegen** | refuted (cat9+BI=34 > 22) | **HOLDS** | BI is COUNTERPRODUCTIVE (34 raw); the scan is already bit-exact so BI only perturbs autotune. **HOLDS.** |

**Net: every ruling HOLDS as a per-forward-kernel statement.** Two FLAGS, both about the REFERENCE and the
LIVE PATH, not the ruling itself: (a) the FA2 floor is prefill-only, the live decode is TREE_ATTN (§3);
(b) the L0/22-flip reference is the no-spec DECODE oracle, so some of the "drift" is the expected
tree-verify-vs-sequential-decode dispatch divergence, not a fixable kernel seam.

---

## 3. NEVER-EXAMINED SOURCES (the most valuable — what the prior accounting missed)

The prior decomposition declared "batch-invariance EXHAUSTED at in_proj_ba" (FR13_RESIDUAL13_RESOLVED) because
in_proj_ba is "the ONLY bf16 GEMM on the GDN data path." That statement is scoped to the GDN layers and
SKIPS the following live channels of the baked cat9 forward. Scanning end-to-end:

### 3a. **TREE_ATTN decode kernel — the LIVE full-attn path, NEVER M-invariance-checked post-bake** (LIVE)
The 16 full-attention layers route through `vllm/v1/attention/backends/tree_attn.py` with
`FR13_TREE_ATTN_EXP2_SOFTMAX=1` (patcher L11948-12040): the Triton unified-attention softmax is rewritten
`tl.exp → tl.exp2`, the KV-block iteration is REVERSED to FA2 order, and `num_warps=4/num_stages=3` are
pinned. This is an ATTEMPT to bit-match FLASH_ATTN — but it is a SEPARATE Triton kernel from native E5's CUDA
FLASH_ATTN. The memory note records the residual: **TREE_ATTN vs FLASH_ATTN = 0.00195** (per-MEMORY "do NOT
hand-wave as backend nature"). The full-attn layers' **qkv_proj** (the per-layer attention input GEMM) was
NEVER checked for M-keying the way in_proj_ba was — and the FA2-QPAD branch (`030a1c22`, commit `9ad6793f`)
DID measure the forked-FA2 query-tile as **M_DEPENDENT** (L31 3.9e-3 → 0.0 when query-padded). That QPAD fix
was OVERTURNED (`8b7684dd`) because e2e flips stayed 24 — BUT that overturn was BEFORE the in_proj_ba fix
landed and rests on the "first-nonzero is L0 GDN, upstream of L3 full-attn" ladder argument. **The full-attn
TREE_ATTN M-dependence is a LIVE, measured-nonzero, never-padded channel.** Whether padding the TREE_ATTN
query/KV tile (analogous to in_proj_ba) moves e2e flips POST-bake was never tested. LIVE SUSPECT.

### 3b. **The replay route `_tree_gdn_replay_kernel` cross-event durable-state handoff** (LIVE — prime suspect)
The locked build runs `FR13_REPLAY_ROUTE=1` (always on). At commit time it RE-EXECUTES the accepted chain
from h0 via `_tree_gdn_replay_kernel` (`fr10_gdn_tree_kernel.py:546`, patcher L7348 `_fr13_replay_launch`),
writing the durable next-event state into the bank. The byte A/B that "PASSED" (lineage table L27,
`FR13_REPLAY_GPU_GATES_BIND`) compared the **replay chain vs OUR OWN scan chain** — NOT vs native MTP's
durable state. Native MTP produces its durable state with `fused_sigmoid_gating_delta_rule_update`; our
replay uses a DIFFERENT rank-1 Triton kernel. The shared `_gdn_node_step` body comment itself warns:
"Codegen identity across the two compilations (FMA contraction/scheduling) is NOT spec-guaranteed" (bug
class #10). And the lineage table records the replay route **FAILED LIVE** (accept 2.02→1.58, within-boot
non-determinism) — claimed fixed by `02b1627a` (conv-remap page-stomp) but flagged "offline-bit-identical ≠
live multi-step, now proven TWICE" with the live state-logistics seam (publish-ordering vs next-event h0
read, native `get_temporal_copy_spec` neutrality, ring keying at request churn) UNLOCALIZED. **This is the
cross-event accumulation channel that exactly explains how ~14 e2e flips arise from per-forward-bit-exact
kernels: the durable state our replay writes differs from native's by ~1 ULP and accumulates across verify
events.** PRIME never-fully-examined LIVE suspect.

### 3c. **Sampler / committer (greedy path-LCP)** (likely clean)
`_lumo_tree_path_lcp_max_greedy_sample` (patcher L6344) is a pure-integer greedy max-LCP committer. Commits
were EXONERATED (`8bfd0854`: gold-margin gate 0/944 channel-1; `df1dfa07`: GDN h0 handoff byte-equal 160/160
but H1 ROWBUG — stock samples REJECTED-node hidden after partial accepts 84/164 events, FIXED by FIX-A
`FR13_TREE_SAMPLE_ROW`). OPT-1 GPU committer is DEFAULT-OFF. The committer is a byte-identical integer
decision → not a float drift source. **Likely clean** but the H1 ROWBUG class (wrong-row sampling after
partial accept, bug class #5) should be re-confirmed clean on the BAKED build (it was a real defect).

### 3d. **lm_head / final norm** (clean)
`compute_logits` (L13140) and the final `self.norm` are STOCK; the only patches are env-gated CAPTURE taps
(FR10_ROOT_LOGIT_CAPTURE). No behavioral change. The lm-head GEMV runs verbatim → the argmax flip is
inherited from the hidden state, not introduced at the head. **Clean** (but note the speed tax is the
lm-head GEMV over 9-10 verify rows vs 6 — `project_fr13_tree_reshape_unifying_lever`).

### 3e. **eager-pack + conv-fused replay** (FR13_EAGER_PACK=1, FR13_TREE_CONV_FUSED=1, always-on)
Both baked ON, both replay-coupled. EAGER_PACK is a DtoH-collapse (102→1 readback) — a SPEED logistics change,
byte-A/B-gated (`6c2f46d6`, `834bab16` fixed an AxisInfo codegen-identity break). TREE_CONV_FUSED emulates
conv in-graph (`ef4d7514`, byte A/B 283/283). These are logistics, not new math — but they are part of the
replay route (§3b) and share its "offline-bit-identical ≠ live" risk. Worth one same-boot A/B (pack/conv-fused
ON vs the legacy unpacked path) on the baked build to confirm they are byte-zero contributors.

### 3f. **RoPE / MRoPE position wiring for the tree** (per-MEMORY fixed, re-confirm)
Per-MEMORY (3680e6d2): flat MRoPE positions reached Qwen3Next full-attn RoPE instead of the scheduler's
tree-depth positions → q_after_rope diverged 4.52 → FIXED (position_ids=0, base off-by-one → num_computed_
tokens_cpu). The full-attn patch (L13450-13497) only adds capture taps around stock q_norm/k_norm/rotary_emb.
**Re-confirm the position fix is in HEAD** (`feedback_monitor_verify_work_committed`) — a regressed tree-depth
position would be a large M-independent carrier, but the L0-GDN-first-nonzero ladder (full-attn at L3, downstream)
argues the position wiring is currently correct.

---

## 4. THE BIGGEST REMAINING LEVER + irreducible-vs-missed verdict + online research

### 4a. Biggest lever
**The cross-event durable-state handoff of the replay route (§3b)**, then the **TREE_ATTN full-attn
query/KV M-invariance (§3a)**. Both are LIVE, both are measured-nonzero or never-A/B'd-vs-native, both
accumulate across the deep stack/events — which is the only mechanism that turns per-forward-bit-exact
kernels into ~14 e2e flips. The in_proj_ba pad was the right KIND of fix (authorized #42960 batch-invariance)
but applied to the WRONG dominant channel — it padded the one GDN bf16 GEMM and left the full-attn GEMM /
TREE_ATTN tile and the replay durable-state unaddressed. The decisive test the prior work itself named
(L0-GDN sub-op M10-vs-M5 A/B) is BLOCKED on infra (device-assert in FLA fused_post_conv_prep:215) — but the
MORE valuable A/B is **replay-durable-state vs native-MTP-durable-state** (cross-event), which has NEVER been
run (the byte A/B was replay-vs-our-scan).

### 4b. Is the residual irreducible diffuse/cascade floor, or a missed paddable/alignable channel?
**Mixed, honestly.** Three layers:
1. The native-3 floor + ~2 spine-intrinsic = genuine diffuse/cascade floor (native crosses the same kind of
   boundary; accept/event ~native; sub-deployment-impact). Irreducible at this precision.
2. The ~9-13 co-residency residual is **NOT proven irreducible** — it is "diffuse" only in the sense of
   `reference_diffuse_gdn_accumulation_explained` (native = existence proof that a 3-flip realization exists
   at the same fp8/64-layer precision). The SPINE_PERTURBATION evidence (11/11 ch2 on spine) localizes the
   MECHANISM (co-residency) without yet localizing the OP. in_proj_ba was ONE op; the FA2-QPAD measured a
   SECOND M-dependent op (full-attn query tile). The replay durable-state is a THIRD un-A/B'd channel. So
   "diffuse" here = "≥2 un-aligned seams nobody drove to zero," not a thousand independent ones
   (`feedback_math_correct_vs_bitexact`). **There IS at least one missed alignable channel** (TREE_ATTN
   query M-pad + replay-vs-native durable state), and the unifying lever (`project_fr13_tree_reshape_unifying
   _lever`) — a shallower/root-sibling tree that reduces co-residency depth-accumulation — was never tried
   POST-bake.
3. The cascade/trajectory inflation (raw 22 → ~14 independent; near-disjoint from native's boundaries) means
   the HEADLINE 21 OVERSTATES the per-forward defect by ~1.5x. The honest per-forward independent-defect
   count is ~11-13, of which 3 are native floor.

### 4c. Online research grounding
- STree (arXiv 2505.14969, the GDN-hybrid tree-decode reference): tree decoding on SSMs requires
  ACCUMULATING state-transition matrices per the tree structure; the recurrent state replay is the
  bf16-sensitive path — directly corroborates §3b (the durable-state handoff is the hard part for
  GDN/Mamba hybrids, not the per-forward scan).
- Lossless-specdec literature (emergentmind): "Lossless speculative decoding ensures every emitted token is
  the target model's greedy argmax at verification time, though output can still differ from pure
  autoregressive because of **numerical dispatch divergence**." This is EXACTLY the FR13 situation: the
  21 flips are tree-verify (a different numerical dispatch) vs no-spec decode (the autoregressive oracle).
  The academically-correct lossless gate is "served == verify-time argmax," NOT "served == no-spec-decode
  argmax." Part of the 21 is the EXPECTED dispatch gap, not a bug — which means the binding gate should be
  cat9-tree-verify vs native-MTP-tree-verify (same dispatch class), not vs the no-spec decode oracle. This
  reframes the count (and was flagged but never re-measured: see FR13_BV_GEOMETRY §"NEXT localization").
- Component-aware self-spec in hybrids (arXiv 2605.01106) and SpecMamba (2509.19873): bf16 recurrent-state
  replay and small projections are the numerically fragile paths that must be "stabilized across speculative
  cycles" — corroborates that the projection (in_proj_ba, qkv_proj) AND the recurrent replay are the right
  targets, and that more than one needs stabilizing.

Sources:
- STree: https://arxiv.org/abs/2505.14969 ; https://openreview.net/forum?id=a95Vd41o1u
- Lossless specdec (numerical dispatch divergence): https://www.emergentmind.com/topics/lossless-speculative-decoding
- Component-aware self-spec hybrids: https://arxiv.org/html/2605.01106
- SpecMamba: https://arxiv.org/pdf/2509.19873
- Gated Delta Networks (ICLR 2025): https://openreview.net/pdf?id=r8H7xhYPwz

---

## VERDICT (skeptical, fresh)
- The 21 baked = native 3 + ~2 spine-intrinsic + ~9-13 co-residency, de-cascaded to ~11-13 INDEPENDENT
  events (the raw 21 overstates by ~1.5x via cascade/trajectory inflation; bug class #12). The tree flips
  are at boundaries NEAR-DISJOINT from native's 3 (1/18 overlap) = a superset of crossings, not "more of
  the same set."
- Every per-forward kernel ruling HOLDS (scan/conv/fp8/gate/FA2-fork/reshape/BI). Two FLAGS: the FA2 floor is
  PREFILL-only (live decode is TREE_ATTN); the 22-flip reference is the no-spec DECODE oracle, so part of the
  count is the expected tree-verify-vs-sequential-decode dispatch gap, not a kernel bug.
- The biggest UN-examined live channels: (1) replay-route cross-event durable-state vs native MTP (NEVER
  A/B'd vs native — only vs our own scan; live-fail history), (2) TREE_ATTN full-attn query/KV M-invariance
  (measured M_DEPENDENT on the FA2 fork, overturned PRE-bake, never re-tested POST-bake), (3) the
  reframe-the-gate-to-tree-verify-vs-native-MTP that the literature says is the correct lossless reference.
- NOT a proven irreducible floor: native is the existence proof. ~3 of the 21 is irreducible; the rest is
  ≥2 un-aligned/un-padded seams + cascade inflation. The arbiter accept/event 3.0-3.15 ~ native 3.076 means
  the residual is currently SUB-DEPLOYMENT-IMPACT — but the headline "21 flips" is not the irreducible floor
  it was presented as.
