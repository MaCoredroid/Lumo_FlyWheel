# FR13 — TOTAL DRIFT REANALYSIS: fresh, skeptical, independent accounting of the 21 baked flips

Date 2026-06-14. READ-ONLY reanalysis (no GPU; re-run is BLOCKED — the SUBOP_MAB capture arm
device-asserts in FLA `fused_post_conv_prep:215`). Author: fresh subagent, re-derived from code + git
history + the on-disk flip records + online research — NOT taking prior binds on faith. Several prior
conclusions were OVERTURNED this session (FA2-tile carrier, depth model, BV seam, oracle frame), so EVERY
ruling below was re-checked against the actual kernel code, not the bind that asserted it.

**Inputs (all read fresh):** kernel patcher `scripts/fr10_phase4_patch_vllm_tree_gdn.py`; serving kernel
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`; locked launcher `scripts/fr13_launch_locked.sh` +
forked `scripts/fr13_launch_forked_fa2_tree_server.sh`; the CANONICAL baked flip record
`output/fr13_bake_verify_20260614T193848Z/tree_flips.json` (= the [3,6,6,6]=21 of
`FR13_BAKE_B1_HOLD_BIND`), `output/fr13_shape_sweep/{BA_PROJ_BI_ON,BA_PROJ_BI_OFF,chain3,chain5}_flips.json`,
`output/fr13_verify_decisive/q3_{native,tree}_classify.json`, `output/fr13_node7_ladder/ladder_summary.json`;
git log (97 FR13 commits, 28 remote branches); online research (sglang #25587, SpecMamba 2509.19873,
dflash-mlx, emergentmind lossless-specdec).

**GB10 hardware context (named, load-bearing):** B=1 decode is **bandwidth-bound** (273 GB/s LPDDR5X);
the bf16 weight DMA dominates per-forward time, so (a) the in_proj_ba pad is speed-neutral — the extra
padded GEMM rows hide behind the weight read (measured s/fwd 0.2248 vs 0.2249 OFF); (b) M-keyed GEMM
autotune picks **shape-dependent** kernels, which is exactly why co-residency (tree M ≠ spine M) can perturb
spine rows at the ULP level; (c) cross-boot autotune forks the served stream — the **18↔21↔22↔26 spread is
the ±3-4 same-build autotune floor**, not a behavior change (`feedback_no_cross_boot_byte_gate`).

---

## 0. The canonical number, re-derived from the bytes

The "21 [3,6,6,6]" is NOT the on-disk `BA_PROJ_BI_ON_flips.json` (that boot = **18 [4,4,4,6]**). The canonical
[3,6,6,6]=21 is the bake-VerifyGate boot `output/fr13_bake_verify_20260614T193848Z/tree_flips.json` +
`verify_summary.json` gate3 (`research/fr13_workflows/bake_b1_wc9kiwfi7.raw.json`). All 33 flips
within_boot_det=True (zero nondeterminism), `spec_delta_during_oracle` all 0.0 (clean teacher-force decode
oracle, thr 1.0 nat, prompts_swe4). Confirmed by independent re-count of the per-prompt clear-margin list:

| prompt | served_len | clear positions | n_clear |
|---|---|---|---|
| 0 | 104 | 17, 34, 103 | 3 |
| 1 | 116 | 50, 58, 94, 100, 112, 115 | 6 |
| 2 | 128 | 21, 60, 73, 100, 104, 125 | 6 |
| 3 | 128 | 61, 80, 99, 123, 125, 127 | 6 |

Three records of the SAME lossless build at different boots: 18 [4,4,4,6] / 21 [3,6,6,6] / 22 [5,7,4,6]
(unbaked) / 26 [2,10,7,7] (same-boot pad OFF). The bake took 26→21 same-boot and held ≤22 banked = it DID
NOT raise flips = lossless-preserving. **The 21 is one draw from a ±3-4 distribution, not a fixed defect
count** — any "accounting to the unit" is over-precise; the honest target is the SHAPE of the residual.

---

## 1. FULL ACCOUNTING of the 21 (de-cascaded, same rule applied to native)

### 1a. De-cascade (bug class #12 — measurement traps): raw 21 → **18 independent**
De-cascade rule (identical to `FR13_PLUS2_DECASCADE`, applied symmetrically to native): a clear flip is an
INDEPENDENT event iff it is gap ≥ 4 from the previous clear flip in the same stream (a contiguous run
gap ≤ 3 on an already-diverged served prefix = ONE event, cascade tails excluded). Computed fresh on the
canonical record:

| prompt | raw clear | independent | collapsed |
|---|---|---|---|
| 0 | 3 | 3 | — |
| 1 | 6 | 5 | 112↔115 (gap 3) |
| 2 | 6 | 6 | — |
| 3 | 6 | 4 | 123↔125↔127 (gaps 2,2) |
| **total** | **21** | **18** | 3 |

**FLAG on the prior adversarial block:** it claimed the de-cascade collapses 21 → "~11-13 independent"
(a ~1.5× overstatement). The bytes do NOT support that. With the same gap≥4 rule that gave chain5 5→2 and
chain3 5→5 in `FR13_PLUS2_DECASCADE`, the canonical [3,6,6,6] collapses only 21 → **18** (~1.17×). The
cascade inflation on the deep cat9 spine is REAL but MILD — most of the 21 are isolated, distinct-boundary
crossings, not one big fork. The headline overstates the per-forward defect by ~17%, not ~50%. (A looser
gap≤2 rule collapses one more → 17; the order of magnitude is ~17-18 independent, not ~12.)

### 1b. Native floor = 3 independent (re-derived, like-for-like)
`q3_native_classify`: [0,1,1,1] at p1/94 (`Let`/codefence), p2/33 (quote-style), p3/68 (`Let`/codefence) —
genuine high-entropy FORMAT boundaries, isolated, distinct. Native itself crosses them; **irreducible at
this precision** (existence proof that 3-flip realization exists for this model+fp8+64 layers).

### 1c. Distinctness — DECISIVE, checked fresh: the tree set is NEAR-DISJOINT from native's
Native clear positions: `{(p1,94),(p2,33),(p3,68)}`. Canonical baked clear positions (21): intersection
with native = **exactly 1** — `(p1,94)`. So the tree flips are **NOT "more of the same diffuse set"** — they
are a SUPERSET of crossings the **tree path makes and native does not** (the served streams fork: different
served_lens [104,116,128,128] vs native, different downstream context). 20 of 21 are at boundaries native
never crosses. This is the single most important structural fact: the residual is tree-path-specific drift,
not amplified native drift. (Unbaked `q3_tree` shares ZERO with native; BA_PROJ_BI_ON shares 1 = p3/68. The
"~1 shared" is stable across boots.)

### 1d. Back-loading — the cross-event accumulation signature (NEW, fresh measurement)
Normalized flip position (pos/served_len) **mean = 0.696** — flips are heavily BACK-LOADED. p1/p2/p3 each
put 4-5 of 6 clear flips in the back HALF of the stream:

| prompt | early (<½) | late (≥½) |
|---|---|---|
| 0 | 2 | 1 |
| 1 | 1 | 5 |
| 2 | 2 | 4 |
| 3 | 1 | 5 |

Native's 3 are NOT back-loaded (scattered single boundaries 94/33/68). Back-loading is the fingerprint of a
divergence that **GROWS with the number of chained verify events** — i.e. cross-event durable-state
accumulation, not a per-forward op error that would hit uniformly. This matches sglang #25587's external
observation ("output diverges from non-spec after ~100 tokens, accumulates with generation length"; §6).

### 1e. Deviation distribution
12/21 near-tie (dev ≤ 2.5 nat), 9/21 large (up to 14.94 nat). The 9 large ones cluster LATE (back-loaded) =
overwhelmingly cascade consequences (the oracle re-predicts confidently on a diverged prefix), not 9
independent large-margin defects. The arbiter accept/event = **3.1513 ~ native 3.1613** (spine floor 2.66)
= **sub-deployment-impact**: the branch/leaf accept edge is intact, the flips do not collapse acceptance.

### 1f. Accounting summary (honest, with error bars)
```
21 raw  →  18 independent (de-cascade, ±1 by rule choice)
  3  : native floor (irreducible diffuse format boundaries; native crosses these too)
  ~2 : our-spine-vs-native intrinsic (chain3 5→5 = +2 over native; chain5 5→2 cascade-masked;
       = per-layer ~1-bf16-ULP GDN/attn realization gap over the spine, alignment-territory)
 ~13 : tree-path co-residency residual (SPINE_PERTURBATION + cross-event accumulation),
       20/21 at boundaries NATIVE NEVER CROSSES, back-loaded → cross-event durable-state.
       in_proj_ba pad addressed ~4-8 of the original +17; this is the un-padded remainder.
```
The residual is a SUPERSET of native crossings, dominated by tree-path co-residency that is back-loaded
(accumulation-shaped), NOT amplified native drift.

---

## 2. RE-CHECK of every RULED-OUT channel against the actual kernel code

| channel | ruling | re-check verdict | code evidence |
|---|---|---|---|
| **GDN scan** | M/geometry-invariant, RAW 0.0 to native | **HOLDS** | `fr10_gdn_tree_kernel.py:330-383` `_gdn_node_step`: the three reductions (`tl.sum(b_q*b_q)`, `tl.sum(state_i*b_k[None,:],axis=1)`, `tl.sum(state_i*b_q[None,:],axis=1)`) are over **DIM_K** within each V-row. `BLOCK_V`/`pid_v` (L429) re-tile WHICH V-rows a program owns; they never touch the K-reduction order. So geometry-invariant by construction. `FR13_BV_GEOMETRY` measured RAW 0.0 vs the **REAL** native `fused_sigmoid_gating_delta_rule_update` at D16=D32, N_PAD 1+16; neg-control (independent fp32 torch scan = 0.0078 = 1 ULP) proves the harness reports true zeros, not clamped. |
| **conv1d (tree)** | row-M-invariant, prior-window fixed | **HOLDS** structurally | fused per-row tree conv, no GEMM, no M-keying. BUT see §3(2): the prior-window FIX is wiring that lives in the live conv-fused REPLAY path — re-verify it is engaged post-bake. |
| **fp8 in_proj_qkvz + o_proj** | M-invariant (BLOCK_SIZE_M=64 constexpr) | **HOLDS at cat9 geometry** | `_patch_fp8_utils_gb10_gemv_cfg` (L13602) is **DEFAULT-OFF** → the **stock** default config runs: BLOCK_SIZE_M=64, BLOCK_SIZE_N/K pinned to block_size (K=128). The fp32 K-accum (`accumulator += tl.dot(a,b)*a_s*b_s` over `range(cdiv(K,128))`) is per-row, M-independent. cat9 verify packs ~33 real rows; padded in_proj_ba = 16×row_len. **CAVEAT (checked):** if a single packed GEMM exceeds 64 rows the M-tile splits, but that does not change per-row K-accum order (each output row = W@src[row]); M-tiling only changes WHICH rows a CTA owns. So bit-identical regardless. `FR13_GB10_FP8_GEMV_CFG` override is DEFAULT-OFF (not in the locked env). |
| **gate (RMSNormGated)** | M-invariant | **HOLDS** | ROWS_PER_BLOCK=1, per-row rms; no cross-row reduction. |
| **BV/warps scan codegen** | refuted as seam | **HOLDS** | `FR13_BV_GEOMETRY` silicon 0.0 at BV16 AND BV32; the static "BV reshapes the reduction tree" reading conflated V-tiling with K-reduction. |
| **chunk-vs-recurrent oracle FRAME** | NOT a frame artifact (5/5 byte-id across oracles) | **HOLDS — but FLAGGED** | The count is not a chunk-vs-recurrent artifact (cascade reproduces under any honest oracle). **FLAG:** the 21-flip + the L0 ladder reference is the **no-spec DECODE oracle** (`fused_recurrent`, a DIFFERENT dispatch than the tree-verify scan). So part of the L0 0.0078 first-nonzero is the LEGITIMATE/EXPECTED tree-verify-vs-sequential-decode dispatch divergence — emergentmind: "every emitted token is the target's greedy argmax at verify, though output can still differ from pure autoregressive because of NUMERICAL DISPATCH DIVERGENCE." Some fraction of the 21 is that expected gap, not a bug. The binding-gate-vs-native-MTP-tree-verify (same dispatch class) was flagged in `FR13_BV_GEOMETRY` NEXT and **never re-measured**. |
| **reshape (depth)** | depth dead (chain3=chain5=5), width adds co-residency | **HOLDS** | cat3w(25) ≫ chain3(5) = width adds co-residency even for strict-mask-invisible root siblings. |

**FA2 fork floor (14/16 whole-tree 0.0, 2 single-ULP):** HOLDS as a fact, but **NOT the live decode path** —
the locked launcher sets `ATTENTION_BACKEND=TREE_ATTN` (`fr13_launch_forked:11`), NOT the FA2 fork. The FA2
fork is the PREFILL path. So the FA2 2-ULP floor does not directly carry the decode tree-verify flips. (See
§3(1) — TREE_ATTN is a SEPARATE kernel that was never M-keyed the way the FA2 fork was.)

**Net:** all per-forward-kernel rulings HOLD on a fresh read. Two FLAGS are about the REFERENCE/LIVE-PATH,
not the kernels: (i) the gate uses the decode oracle (a different dispatch from tree-verify); (ii) the live
decode backend is TREE_ATTN (the FA2-fork floor and the FA2-QPAD finding are on a different kernel).

---

## 3. NEVER-EXAMINED SOURCES (the most valuable output — what the prior decomp SKIPPED)

The prior decomp declared "batch-invariance EXHAUSTED at in_proj_ba (the only GDN-data-path bf16 GEMM)".
That scoping SKIPS these LIVE channels of the baked cat9 forward (all baked ON per `fr13_launch_locked.sh`:
`FR13_REPLAY_ROUTE=1`, `FR13_EAGER_PACK=1`, `FR13_TREE_CONV_FUSED=1`, `ATTENTION_BACKEND=TREE_ATTN`,
`FR13_TREE_ATTN_EXP2_SOFTMAX=1`, `LUMO_FB_KERNEL_ROWS=1`):

### (1) TREE_ATTN decode kernel — the 16 full-attn layers [LIVE; prime un-padded]
`FR13_TREE_ATTN_EXP2_SOFTMAX=1` patches the unified Triton attn into `tl.exp2` log2-space (patcher
L12002-12040) — an ATTEMPT to bit-match FLASH_ATTN, but it is a **SEPARATE Triton kernel** from native CUDA
FLASH_ATTN (memory records TREE_ATTN-vs-FLASH_ATTN = 0.00195). The full-attn **qkv_proj GEMM and the
TREE_ATTN query-tile were NEVER M-keyed-checked** the way in_proj_ba was. The FA2-QPAD A/B (`9ad6793f`,
`FR13_FA2_MDEPENDENT_BIND`) MEASURED the forked-FA2 query-tile **M_DEPENDENT** (L31 3.9e-3→0.0 when
query-padded; 26/224 sweep cells nonzero, every value an exact bf16 power-of-2). That QPAD fix was
OVERTURNED (`FR13_FA2_CARRIER_OVERTURNED`/`8b7684dd`) — BUT on TWO grounds that I re-checked and that have a
gap: (a) it didn't move e2e flips (24, but the stream FORKED so class-12 confounded — a complete
non-response can't be a confound but the magnitude can); (b) the node7 ladder first-nonzero is **L0 GDN
(0.0078) upstream of L3 full-attn (0.00409)**, so a fix to L3+ cannot remove an L0-born divergence. Ground
(b) is sound — BUT (i) it was measured vs the DECODE oracle (the §2 frame flag — part of L0 0.0078 is
dispatch-expected), and (ii) the **TREE_ATTN** query-tile M-invariance (as opposed to the FA2-fork that QPAD
patched) was **never tested**, pre- OR post-bake. TREE_ATTN is the live decode kernel; the FA2 fork QPAD was
on the prefill kernel. So "FA2-QPAD is moot" does NOT discharge "TREE_ATTN full-attn tile M-invariance".

### (2) REPLAY-ROUTE cross-event DURABLE-STATE handoff [LIVE; PRIME SUSPECT — code-confirmed gap]
`FR13_REPLAY_ROUTE=1` always-on re-executes the accepted chain via `_tree_gdn_replay_kernel`
(`fr10_gdn_tree_kernel.py:546`) and publishes the durable next-event state to the bank's LINEAR columns.
**The byte A/B that PASSED compared replay-vs-OUR-OWN-SCAN, NEVER vs native MTP's durable state.** Confirmed
verbatim in two places:
- `FR13_GDN_KERNEL_LINEAGE.md:30`: "the only new claim — **'the replayed chain equals the scan's chain
  bit-for-bit'** — holds by IEEE determinism iff the compiler emits identical code". That is class #10
  (shared-source ≠ shared-SASS), OUR-kernel-vs-OUR-kernel.
- The shared-body comment itself (`fr10_gdn_tree_kernel.py:360-363`): "Codegen identity across the two
  compilations … is **NOT spec-guaranteed**: it is gated by the one-time byte A/B on captured payloads."
- Native's reference is `fused_sigmoid_gating` (lineage L20: "the gold reference — everything is judged
  against its op order"). The replay was judged against the SCAN, not against `fused_sigmoid_gating`'s
  durable handoff. A different kernel, different op order.

Replay also FAILED LIVE (gate-4 class: accept 2.02→1.58, within-boot nondeterminism, native forks at pos
11-17). Root-caused to a conv-remap page-stomp (`stride(0)` row-extent on a shared-page as_strided view) and
fixed PURE-WIRING `02b1627a` (confirmed ANCESTOR of HEAD; `fr13_replay_conv_remap.py` present). BUT the fix
was the CONV half; the SSM durable publish is still the replay kernel's, A/B'd only vs our scan. `class #8`
(offline single-forward ≠ live multi-step) was "proven TWICE" here. **This is the cross-event accumulation
channel that explains §1d back-loading: per-forward-bit-exact kernels can still drift across EVENTS through
the durable handoff.** External corroboration is strong (§4).

### (3) committer / sampler [class #5] — EXONERATED, but re-confirm on baked build
0/944 ch1 clear-margin violations (`FR13_BRANCH_FLIP_LOCALIZED`: 11/11 ch2 flips on the SPINE, 0 on leaves);
H1 ROWBUG fixed by FIX-A. The committer does not fail its own verification — branches perturb the co-resident
spine. Re-confirm clean on the baked build (cheap, 0/944 should reproduce).

### (4) lm_head / final norm — CLEAN. Stock modules; only env-gated capture taps (patcher L13140). Not a
drift source.

### (5) eager-pack (`FR13_EAGER_PACK=1`) + conv-fused replay (`FR13_TREE_CONV_FUSED=1`) [LIVE]. Patcher
L332-591. Logistics, not new math; byte-A/B-gated. BUT they SHARE the replay live-risk (§3(2)) — both are
replay-coupled (FIX-2 "replay-coupled", FIX-3 "requires REPLAY_ROUTE=1"). Worth one same-boot A/B vs OFF.

### (6) RoPE / MRoPE tree positions — per-memory FIXED (`3680e6d2`); re-confirm in HEAD (the full-attn
depth-RoPE wiring fix). Not re-verified this pass; flagged for the cheap re-check.

**The empirically-real-but-unlocalized carrier (`FR13_BRANCH_FLIP_LOCALIZED`):** SPINE_PERTURBATION is real
(topology + chain5), but the EXACT perturbing op is UNCERTAIN — fp8 GEMM ruled out (config=None=M-invariant
on GB10), TREE_ATTN ruled out (strict -inf mask), scan/conv/gate/o_proj ruled out. The in_proj_ba bf16 GEMM
was the one identified bf16-data-path GEMM (padded → ~4-8). The remainder's op is NOT yet pinned. The two
LIVE un-A/B'd channels above — TREE_ATTN full-attn tile + replay durable-vs-native — are the unpinned
candidates.

---

## 4. THE BIGGEST LEVER + irreducible-or-missed verdict

### Biggest lever (diagnostic, NOT a reward-hack): the replay DURABLE-STATE A/B vs NATIVE MTP
The one channel that ACCUMULATES across verify EVENTS (which is exactly how ~13 back-loaded e2e flips arise
from per-forward-bit-exact kernels) and was NEVER A/B'd against native's `fused_sigmoid_gating` durable
state. Concretely: capture event-N's published durable bank row, byte-diff vs native MTP's durable state on
the SAME accepted path. This is the boundary-instrument of class #8 (capture consumer-input-as-read vs
producer-output-as-written), aimed at native instead of our scan. NOT a reroute — it is a measurement.

Second lever: **TREE_ATTN full-attn query/KV M-invariance pad** — the same authorized #42960
batch-invariance that worked for in_proj_ba, applied to the live TREE_ATTN tile (which the FA2-fork A/B
measured M_DEPENDENT and which, as the SEPARATE live decode kernel, was never tested). in_proj_ba was the
RIGHT KIND of fix (M-pad a bf16 op) on a PARTIAL channel; the full-attn tile and the replay durable state
are unaddressed.

Third (already-banked) lever: tree RESHAPE post-bake (shallower / root-sibling) to cut co-residency
depth-accumulation — never tried post-bake (`project_fr13_tree_reshape_unifying_lever`).

### Is the residual irreducible? — MIXED, NOT the proven irreducible floor it was framed as
- **~3 native + ~2 spine-intrinsic = genuine diffuse floor.** Native crosses the same format-boundary
  class; accept/event ~ native; sub-deployment-impact. Irreducible at this precision (the ~1-bf16-ULP
  per-layer GDN/attn realization gap over 64 layers, `reference_diffuse_gdn_accumulation_explained`).
- **The ~13 co-residency residual is NOT proven irreducible.** Native E5 (SAME model, SAME fp8, SAME 64
  layers, 3 flips) is the EXISTENCE PROOF that a 3-flip realization exists at this precision. "Diffuse" here
  means "≥ 2 un-aligned/un-padded SEAMS nobody drove to zero" — NOT a thousand independent ones. in_proj_ba
  was ONE; the FA2-QPAD A/B measured a SECOND M-dependent op (on the prefill kernel; the TREE_ATTN analog
  untested); replay-durable-vs-native is a THIRD un-A/B'd channel. There IS at least one missed
  alignable/paddable channel.
- **Cascade/distinctness:** raw 21 → 18 independent (NOT ~12 — §1a flag), 20/21 at boundaries NATIVE NEVER
  CROSSES (§1c), back-loaded (§1d). The headline overstates the per-forward defect by ~17%, and the residual
  is tree-path-specific (cross-event-shaped), not amplified native drift.

**Honest verdict:** ~3-5 irreducible; the rest is missed seams (TREE_ATTN tile + replay-vs-native durable
state, both un-A/B'd, both LIVE) + mild cascade inflation. It is NOT "the irreducible diffuse floor." The
deployable arbiter (accept/event 3.1513 ~ native) means it is **sub-deployment-impact TODAY**, so the call
is: either accept the diffuse floor and proceed to speed/B=4 (defensible — accept/event already at native),
OR cheap-test the ONE highest-value never-examined channel (replay durable-vs-native A/B) before accepting.

---

## 5. Git-history findings (what was tried / overturned / never finished)

- **FA2 query-tile carrier:** `9ad6793f` MEASURED forked-FA2 query M_DEPENDENT → built FR13_FA2_QPAD
  (`030a1c22`, branch `fr13-fa2-qpad`, UNMERGED) → OVERTURNED `8b7684dd`/`FR13_FA2_CARRIER_OVERTURNED`
  (didn't move flips; L0 GDN upstream of L3). Overturn was on the FA2-FORK/PREFILL kernel; the TREE_ATTN
  DECODE tile was never tested (§3(1)).
- **BV/warps scan geometry:** proposed `850e191a` → REFUTED twice (`7c234c37`, `8d01ac6d`; silicon 0.0).
- **conv prior-window:** 18.375 root found then FIXED+STALE (`b6c30b4b`; `project_fr13_conv_priorwindow_root`).
- **+2 spine WY-wall:** overturned to oracle-frame then de-cascaded to cascade artifact (`ec342d86`,
  `38a78473`).
- **Replay route:** `d2a0ff51` default-ON, `02b1627a` page-stomp fix (ANCESTOR of HEAD). Live-fail class
  fixed for the CONV half; SSM durable A/B-vs-native never run (`FR13_REPLAY_GPU_GATES_BIND`).
- **NEVER-FINISHED:** the L0-GDN sub-op M10-vs-M5 A/B (the named decisive co-residency discriminator) FAILED
  3× on infra (CUDA device-assert in FLA `fused_post_conv_prep:215`, env-not-in-EngineCore-worker) — STILL
  BLOCKED (`fe0af022` "one-more-fix"). residual "RESOLVED depth-intrinsic" (`e89c4003`) was decided on KERNEL
  EVIDENCE because the empirical A/B never ran — i.e. by elimination, not measurement.

---

## 6. Online research

- **sglang #25587 "Hybrid-GDN MTP speculative decoding is NOT lossless" — the closest external
  corroboration.** Root cause = `conv_state` corruption after PARTIAL acceptance during verify:
  "the verify forward runs all N draft tokens, updating conv_states assuming complete acceptance; when only
  k≤N are accepted, `conv_state_rollback` right-shifts existing (already-corrupted) values — it does not
  restore what the window would have contained had only k tokens been processed." The fix (which NVIDIA has,
  NPU lacks): "during verify forward `causal_conv1d_update` saves the K-1 window state **after consuming
  each draft token**." Quantified: "output tokens begin to diverge from non-spec after **~100 tokens**,
  accumulates with generation length, 15-20% acceptance loss." This is EXACTLY (a) the conv prior-window /
  per-step-snapshot class FR13 chased, (b) the cross-event accumulation that produces §1d back-loading
  ("diverge after ~100 tokens"). Independent confirmation that conv-state-handoff / per-step snapshots are
  the lossless-killer for hybrid-GDN spec decode.
- **SpecMamba (2509.19873) + dflash-mlx + emergentmind:** numerical coherence for hybrid-GDN spec decode is
  maintained through "bf16-sensitive paths, including **recurrent state replay** AND **small projections**,
  stabilized across speculative cycles" — confirms BOTH the replay durable-state AND the projection
  (in_proj_ba/qkv_proj) are the right targets, and that **MORE THAN ONE** needs stabilizing (in_proj_ba was
  one; ≥ one more remains). emergentmind: lossless = "every emitted token is the target's greedy argmax at
  verify, though output can differ from pure autoregressive due to **numerical dispatch divergence**" — the
  §2 frame flag (the gate should compare same-dispatch-class, cat9-tree-verify vs native-MTP-tree-verify).
- Sources: github.com/sgl-project/sglang/issues/25587, arxiv.org/pdf/2509.19873, github.com/bstnxbt/dflash-mlx,
  emergentmind.com/topics/lossless-speculative-decoding.

---

## 7. Playbook rows + disposition

Bug classes quoted (`FR13_BUG_CLASS_PLAYBOOK.md`): **#12 Measurement traps** (de-cascade / cross-boot
autotune spread / per-pos counter inflation — central to §0/§1a); **#10 shared-source ≠ shared-SASS** (the
replay vs scan codegen-identity caveat, §3(2)); **#8 offline single-forward ≠ live multi-step** (the replay
live-fail, "proven twice", §3(2)); **#5 wrong-row sampling** (H1 ROWBUG, exonerated §3(3)); **#11 BI-flag
sensitivity** (cat9+BI = 34 counterproductive). First gate of any live re-test = B=1 same-seed byte-identical
repeat (class 8).

**Reward-hacks BANNED — honored:** no copy/dense/splice/reroute proposed. The two recommended levers
(TREE_ATTN query M-pad via authorized #42960 batch-invariance; replay-durable-vs-native byte A/B) are
diagnostic / authorized-batch-invariance, not reroutes; our kernel still computes.

**Disposition:** the 21 is ~18 independent (mild cascade), ~3-5 irreducible (native + spine-intrinsic), the
remaining ~13 dominated by tree-path co-residency that is back-loaded (cross-event-shaped) and sits at
boundaries native never crosses — NOT amplified native drift, and NOT proven irreducible. Two LIVE
never-A/B'd channels (TREE_ATTN full-attn tile; replay durable-state vs native MTP) are the highest-value
cheap tests. accept/event 3.1513 ~ native = sub-deployment-impact, so accepting the diffuse floor and
proceeding to speed/B=4 is defensible; if one cheap test is run first, make it the replay-durable-vs-native
A/B (strongest external corroboration via sglang #25587). No close/pass-fail asserted (user decision class).
