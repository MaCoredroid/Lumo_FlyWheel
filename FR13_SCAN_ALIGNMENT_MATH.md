# FR13 — Scan ↔ native-packed-decode alignment MATH (read-only, code+math; FIX ready for the A/B)

Date 2026-06-14. CPU-only, READ-ONLY companion to the decisive scan-vs-native-packed A/B (we834923g). I did NOT
edit code. Sources read live: native
`/tmp/vllm_live_019/vllm/model_executor/layers/fla/ops/fused_recurrent.py` and `.../fla/ops/op.py`; ours
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`. Bar = bit-exact to the INCUMBENT SASS (native packed-decode),
NOT ℝ-correct vs a serial ref (feedback_math_correct_vs_bitexact; bug-class #10). Build OUR kernel bit-exact; NO
native call in the served path (feedback_no_reroute_reward_hacking). Be skeptical — this session overturned BV/warps,
FA2-tile, the BV=4 lead, and the chunked-prefill reference as overstated/wrong; I overturn one more below (BV32/w4).

---

## 1. THE PIVOTAL QUESTION — is native DECODE recurrent or chunked? → **RECURRENT rank-1, identical math shape to ours.**

The kernel that LIVE tree-verify / no-spec decode actually dispatches is
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (fused_recurrent.py:256-335), launched by
`fused_recurrent_gated_delta_rule_packed_decode` (:338-477). It is a **single-token, sequential rank-1 recurrent
state update** — there is NO `for i_t in range(0, T)` loop, NO chunking, NO `tl.dot`, NO WY/UT transform, NO
matmul over a chunk. It processes exactly ONE token per program (`i_v, i_nh = program_id`), loads the prior state
`b_h` from the bank, applies the SAME five rank-1 ops our `_gdn_node_step` applies, stores out + final state. Core
body (fused_recurrent.py:326-331):

```
b_h *= exp(g_val)                       # state decay
b_v -= tl.sum(b_h * b_k[None, :], 1)    # delta correction, K-reduce axis=1
b_v *= beta_val                         # beta scale
b_h += b_v[:, None] * b_k[None, :]      # rank-1 state write
b_o = tl.sum(b_h * b_q[None, :], 1)     # readout, K-reduce axis=1
```

Our `_gdn_node_step` (fr10_gdn_tree_kernel.py:378-382):

```
state_i *= tl.exp(b_g)
b_v -= tl.sum(state_i * b_k[None, :], axis=1)
b_v *= b_beta
state_i += b_v[:, None] * b_k[None, :]
out_i = tl.sum(state_i * b_q[None, :], axis=1)
```

**These are the same five operations, in the same order, on the same fp32 accumulators, with the same K-reduction
axis.** native's `exp` resolves to `tl.exp` (op.py:21, default `FLA_USE_FAST_OPS=0`) — identical to our `tl.exp`.

### Reframing (this is the load-bearing correction)
The conv-doc named the carrier as a "chunk-vs-recurrent ~1-ULP gap born at L0." That gap, where it was ever
measured, was measured against the CHUNKED-PREFILL realization (`chunk.py` / `chunk_delta_h.py` — those DO use
`tl.dot` and a chunk loop). **For the DECODE path that is the live no-spec oracle, the incumbent is NOT chunked.**
native decode is recurrent rank-1 — the SAME family as our scan. So:
- There is **no structural recurrent-vs-chunked difference** between our scan and the kernel we must match. The two
  are recurrent-vs-recurrent.
- The "diffuse irreducible chunk-vs-recurrent" pessimism rests on the WRONG reference for the decode/verify oracle
  (bug-class #10: bit-exact-to-the-wrong-incumbent). Any residual bit-difference between our scan and native-decode
  is therefore **CODEGEN-class (alignable), not algorithm-class**, modulo the few seams in §3.
- This also re-validates EXIT-1 of FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND: the deployed BV16/w8 scan was only
  ever int-checked vs `native_update_serial_per_path` (a serial TORCH ref, bug-class #10) — never vs THIS kernel.
  The A/B (we834923g) is the first time the right reference is used.

Caveat held: "recurrent-identical math" ≠ "bit-identical SASS." Codegen (FMA contraction, K-reduce tree, tile
shape) can still differ (§3). But the question "is the carrier a deep algorithmic chunk/recurrent split?" is
answered NO — it is at most a codegen seam, and codegen seams are the alignable kind we've closed before (conv
bf16-tap, scan static_range).

---

## 2. OP-BY-OP MAP — our `_gdn_node_step` scan vs native packed-decode

| step | OURS (fr10_gdn_tree_kernel.py) | NATIVE packed-decode (fused_recurrent.py) | match? |
|---|---|---|---|
| state load (h0) | `tl.load(...).to(tl.float32)` :449-453 | `tl.load(p_h0...).to(tl.float32)` :302 | SAME (fp32 accumulator) |
| q/k/v load + dtype | `tl.load(...).to(tl.float32)` :469-483 (from per-node q/k/v tensors) | `tl.load(p_mixed+off...).to(tl.float32)` :308-310 (from PACKED `mixed_qkv`) | SAME cast; layout differs (packed vs split) — value-neutral if upstream proj matches |
| beta (`b`) | RAW_GATING: `b_beta = tl.sigmoid(b_raw_b.to(tl.float32))` :373 | `beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` :324 | **SEAM (e)** — native round-trips sigmoid through `b.dtype` (bf16) then back to fp32; ours stays fp32. See §3(e). |
| g (raw gating) | `x=b_raw_a+b_dt_bias; softplus(thr 20.0); b_g=-exp(b_a_log)*softplus` :366-372 | `x=a_val+dt_bias_val; softplus(SOFTPLUS_THRESHOLD); g_val=-exp(A_log_val)*softplus` :321-323 | SAME; threshold 20.0 both (ours literal :367, native `SOFTPLUS_THRESHOLD=20.0` :472). MATCH. |
| q/k l2norm | `b_q*=tl.rsqrt(tl.sum(b_q*b_q)+1e-6)`; same k :374-376 | `b_q/=tl.sqrt(tl.sum(b_q*b_q)+1e-6)`; same k :313-314 | **SEAM (d)** — ours `rsqrt`, native `1.0/sqrt` (`/`). Different opcode → possibly different ULP. See §3(d). eps 1e-6 MATCH. |
| q scale | `b_q = b_q * OUTPUT_SCALE` :377 | `b_q = b_q * scale` :315 | SAME op-position (after l2norm, before recur). Value must match: OUTPUT_SCALE vs native `scale=K**-0.5`. |
| decay | `state_i *= tl.exp(b_g)` :378 | `b_h *= exp(g_val)` :326 (exp=tl.exp) | SAME |
| delta K-reduce | `b_v -= tl.sum(state_i * b_k[None,:], axis=1)` :379 | `b_v -= tl.sum(b_h * b_k[None,:], 1)` :327 | **SEAM (a)** — same math; FMA/reduce-tree order is a codegen artifact of tile shape. |
| beta scale | `b_v *= b_beta` :380 | `b_v *= beta_val` :328 | SAME |
| rank-1 write | `state_i += b_v[:,None] * b_k[None,:]` :381 | `b_h += b_v[:,None] * b_k[None,:]` :329 | SAME |
| readout K-reduce | `out_i = tl.sum(state_i * b_q[None,:], axis=1)` :382 | `b_o = tl.sum(b_h * b_q[None,:], 1)` :330 | **SEAM (a)** — same math; reduce-tree codegen. |
| out store dtype | `tl.store(out..., out_i, ...)` :528 (out tensor dtype) | `tl.store(p_o, b_o.to(p_o.dtype.element_ty), ...)` :331 | check out dtype parity (both downcast to out tensor dtype) |
| state store | `tl.store(state...)` (STORE_NODE_STATES export) | `tl.store(p_ht, b_h.to(...))` :335 | export-only in ours; native writes back to bank |
| **launch geom** | grid `(num_vh, cdiv(dim_v, BV=16))`; `num_warps=8`; stages unset :1554-1591 | grid `(NV=cdiv(V,32), B*HV)`; `BV=min(npow2(V),32)=32`; **`num_warps=1`**; `num_stages=3` :436-475 | **SEAM (b)** — geometry mismatch. See §3(b) + the BV32/w4 correction below. |

**Spine = branch for losslessness.** The per-node body is the SAME for spine and branch nodes; the only branch
difference is which prior state `state_i` seeds the step (h_cache ancestry select :460-467). native decode has no
branch counterpart, so the branch oracle = native-on-the-branch-path (reference_gdn_tree_branch_oracle_losslessness);
the per-NODE step we must bit-match is exactly this one body. So aligning `_gdn_node_step` to native packed-decode
aligns BOTH arms.

---

## 3. CANDIDATE BIT-EXACT SEAMS — located, scored, mapped to the A/B branch

Legend: **ALIGNABLE** = a value-neutral edit drives it →0.0 (like conv bf16-tap / scan static_range);
**STRUCTURAL** = needs a kernel/shape rewrite. A/B branch: **geometry-seam** (BV/warps/stages codegen) vs
**kernel-math** (an op/cast/order edit inside `_gdn_node_step`).

### (a) fp32 op-order / FMA in the rank-1 update + the two `tl.sum(axis=1)` over K (K=128)
- The five ops are mathematically identical (§2). Any bit difference is from how the compiler contracts
  `state*k` into the K-reduce (FMA vs mul+add), the reduction TREE order across the 128-wide K axis, and tile
  shape (a `[BV,128]` tile reduces differently than a `[32,128]` tile if BV≠32).
- **Score: ALIGNABLE, but its realization is GATED BY THE TILE SHAPE (b).** With identical tile shape + warps +
  stages, the unrolled fp32 instruction stream is the same → 0.0. With BV16 vs BV32 the per-program V-extent
  differs and the K-reduce can schedule differently.
- **A/B branch: geometry-seam.** This is NOT a separate math edit; it rides on (b). bug-class #10.

### (b) BV / num_warps / num_stages codegen  — **and the BV32/w4 correction**
- Deployed scan: **BV=16, num_warps=8, num_stages=unset** (fr10:18, :1547, :1554-1591).
- Native packed-decode: **BV=32, num_warps=1, num_stages=3** (fused_recurrent.py:436-438, computed: V=128 →
  `min(npow2(128),32)=32`).
- **CORRECTION (overturn):** FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND and the lineage describe native geom as
  **"BV32/w4"**. The packed-decode SOURCE is **w1 (num_warps=1)**, num_stages=3 — NOT w4. (The w4/stages-3 figure
  belongs to a different FLA entry, not the decode kernel live-verify dispatches.) Any "recompute-from-spine @
  BV32/w4" plan must be re-pinned to **BV32 / w1 / stages=3** to be native-bit-exact by construction. This is the
  single most important factual correction in this doc — STOP+REPORT candidate for the lineage table (a byte-A/B
  geometry row changes).
- The K-reduction TREE differs across (BV, warps): with BV=32 the V-tile is one program covering 32 value rows;
  with BV=16 it is two programs of 16. The K-axis (128) reduce is per-(value-row) and nominally independent of BV,
  BUT warps=8 vs warps=1 changes how the 128-lane reduce is partitioned across warps and recombined → different
  partial-sum grouping → sub-ULP fp32 differences (bug-class #10, exactly the matrix-R4 signature).
- **Score: ALIGNABLE via geometry match (the cheap path) OR STRUCTURAL-ish (recompute-from-spine) if BV32 spills.**
- **A/B branch: geometry-seam.** If the A/B says BV16/w8 ≠ native-packed but BV32/w1 == native-packed, the fix is
  geometry, designed in §4(geometry).

### (c) bf16 ↔ fp32 cast boundaries
- INPUT casts match: both `.to(tl.float32)` on h0, q, k, v at load (§2 rows 1-2).
- OUTPUT cast: native `b_o.to(p_o.dtype.element_ty)` (:331); ours stores `out_i` to the out tensor (:528) — confirm
  the out tensor is the same dtype both arms (the A/B harness must pin out dtype). If our out tensor is fp32 while
  native's is bf16, the int-view will mismatch by the downcast — that is a HARNESS-dtype issue, not a kernel-math
  bug; pin it.
- STATE write-back cast: native `b_h.to(p_ht.dtype.element_ty)` (bank dtype, bf16) :335; ours exports `state_i`
  (STORE_NODE_STATES) at the export tensor dtype. For the CHILDREN's seed, ours resumes from the fp32 `h_cache`
  REGISTER (no bf16 round-trip), whereas native decode resumes the NEXT token from the bf16 BANK. **This is a real
  difference in the multi-token case**: ours keeps fp32 between sibling/child nodes; native round-trips through
  bf16 each token. For a SINGLE decode token (N_PAD=1) there is no intermediate round-trip → should match. For
  N_PAD>1 (the tree), our fp32-carried state is MORE precise than native's bf16-banked chain — so our tree is
  closer to the ℝ ideal but NOT bit-identical to a native multi-token chain. Native multi-token decode, however,
  does not exist as the single oracle (the verify dispatches ONE packed token per program); the branch oracle is
  native-on-path which itself re-banks bf16 per token. **This is the one genuinely STRUCTURAL seam.** See §4.
- **Score: (single-token) ALIGNABLE/already-equal; (multi-token fp32-carry vs bf16-rebank) STRUCTURAL but
  LOSSLESS-FAVORABLE (ours is more precise) — gate is per-depth argmax vs native-on-path, not abs-0.0 (user
  2026-06-09).** bug-class #12 (don't demand abs-0.0 where the floor is bf16-rebank).
- **A/B branch: kernel-math (cast), but expected at N_PAD=1 to be 0.0; the divergence appears only at N_PAD≥2.**

### (d) l2norm: `rsqrt` vs `1.0/sqrt`; eps 1e-6
- Ours: `b_q * tl.rsqrt(tl.sum(b_q*b_q)+1e-6)` (:375-376). Native: `b_q / tl.sqrt(tl.sum(b_q*b_q)+1e-6)` (:313-314).
- `tl.rsqrt(x)` and `1.0/tl.sqrt(x)` are DIFFERENT opcodes (rsqrt approximation vs IEEE div-of-sqrt) and can differ
  by 1 ULP. eps (1e-6) and the `tl.sum(·*·)` MATCH.
- **Score: ALIGNABLE — cheap one-line edit.** Change ours to `b_q / tl.sqrt(...)` / `b_k / tl.sqrt(...)` to match
  native opcode-for-opcode. This is the same flavor as the conv bf16-tap fix: a value-neutral opcode swap that
  removes a 1-ULP seam. (USE_QK_L2NORM_IN_KERNEL is the path the live model uses — confirm with the model config;
  if l2norm is done PRE-kernel instead, this seam is moot and both arms read pre-normed q/k.)
- **A/B branch: kernel-math (op-choice).** Highest-value cheap candidate if l2norm-in-kernel is on.

### (e) beta cast round-trip
- Native: `beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` (:324) — sigmoid in fp32, downcast
  to bf16, re-upcast to fp32. Ours: `b_beta = tl.sigmoid(b_raw_b.to(tl.float32))` (:373) — stays fp32, no bf16
  round-trip.
- The `.to(bf16).to(fp32)` is a deliberate native quantization of beta to bf16 precision. Ours skips it → ours is
  more precise but NOT bit-identical to native's beta.
- **Score: ALIGNABLE — cheap.** Insert the bf16 round-trip: `tl.sigmoid(b_raw_b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)`
  to match native opcode-for-opcode. Value-neutral to ℝ within bf16; bit-aligns to native.
- **A/B branch: kernel-math (cast).** Second cheap candidate.

### (f) g / softplus / gate-application order
- softplus threshold 20.0 MATCHES (ours literal :367, native constexpr :472). `-exp(A_log)*softplus` order
  MATCHES (:372 vs :323). `exp`=`tl.exp` both. **No seam.** Gate-application order (decay before delta before
  rank-1 write before readout) MATCHES exactly (§2). **No seam.**
- **Score: already 0.0.** No edit.

### Seam summary (scored)
| seam | alignable? | A/B branch | cost | expected at N_PAD=1 |
|---|---|---|---|---|
| (a) fp32 op-order/FMA + 2× K-reduce | ALIGNABLE (rides on geom) | geometry | — | 0.0 iff geom matches |
| (b) BV/warps/stages (BV16w8 vs BV32**w1**s3) | ALIGNABLE (match geom) or recompute-from-spine | geometry | cheap→structural | the suspected source |
| (c) bf16↔fp32 casts (state carry) | single-tok ALIGNABLE; multi-tok STRUCTURAL (ours more precise) | kernel-math | structural | 0.0 at N_PAD=1, diverges N_PAD≥2 |
| (d) l2norm rsqrt vs 1/sqrt | ALIGNABLE | kernel-math | 1-line | 1-ULP if l2norm-in-kernel on |
| (e) beta bf16 round-trip | ALIGNABLE | kernel-math | 1-line | 1-ULP |
| (f) g/softplus/gate order | already 0.0 | — | — | 0.0 |

---

## 4. PRE-DESIGNED ALIGNMENT, ready-to-apply per A/B branch

The A/B (we834923g) int-views deployed BV16/w8 scan out_i vs native-packed out at N_PAD=1 AND 16, spine + branch
winner. Two outcome branches:

### Branch GEOMETRY-SEAM (A/B says: BV16/w8 ≠ native-packed, but the math edits (d)(e) don't close it; geometry does)
Root = (a)+(b): the K-reduce tree / FMA scheduling differs because our tile is BV16/w8 vs native BV32/w1/s3.
- **PRIMARY = recompute-from-spine @ native BV32 / w1 / stages=3** (NOT w4 — see the §3(b) correction;
  re-pin FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND EXIT-2 to w1). Drop the `h_cache` N_PAD tile; hold ONE
  `[BV=32,128]` fp32 register tile; replay ancestry via the existing `tl.where(strict_mask)` on the shared
  `_gdn_node_step`. Bit-exact BY CONSTRUCTION (compiles native's exact `[32,128]` tile + the two `tl.sum(axis=1)`
  in native's order at native's warp count), spill-free at any tree size (O(1) in tree size, ~64-90 regs), and
  lifts the N_PAD≤16 cap. This is build-OUR-kernel (no splice). Cross-reference: this is EXIT-2 of
  FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND — DO NOT redo the SRAM/spill arithmetic here; the only change is the
  warp count w4→**w1** and stages→3 to match the packed-decode source.
- **CHEAPER pre-test before the rewrite:** the TEST-ONLY `FR13_TREE_GDN_GEOM_OVERRIDE="BV=32,num_warps=1,num_stages=3"`
  (the additive override at fr10:1545-1553, already in place — value-neutral) lets the A/B's NEXT arm measure
  whether the DEPLOYED scan kernel, merely re-launched at native geom, already int-matches. If BV32/w1/s3 on the
  unchanged `_tree_gdn_kernel` == native-packed at N_PAD=1, then geometry alone is the seam and the recompute-
  rewrite is only needed because BV32 SPILLS at N_PAD=16 (the SRAM tension) — i.e. recompute-from-spine is the
  spill-free way to DEPLOY the geom that the override proves bit-exact. (This is exactly the SRAM×N_PAD tension's
  resolution path.)
- CAVEAT (held from the bind): the recompute/replay path runs has a w8 existence-proof + a broken-live gate-4
  history → losslessness is a GPU obligation, must be re-gated live, spine AND branches.

### Branch KERNEL-MATH (A/B says: even at native geom / N_PAD=1, out_i ≠ native-packed → an op/cast inside the body)
Apply these to `_gdn_node_step` IN DEPENDENCY ORDER (upstream proj/norm → recurrent update → gate). All are
value-neutral-to-ℝ opcode/cast alignments (the conv-bf16-tap / scan-static_range flavor), NONE are structural:
1. **(d) l2norm opcode** (upstream-most): `b_q = b_q * tl.rsqrt(...)` → `b_q = b_q / tl.sqrt(...)`; same for `b_k`.
   Match native's `/`+`sqrt` (:313-314). Do FIRST — it feeds q (and thus the readout) and k (and thus the delta).
2. **(e) beta bf16 round-trip**: `b_beta = tl.sigmoid(b_raw_b.to(tl.float32))` →
   `tl.sigmoid(b_raw_b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)`. Match native (:324). Feeds the beta-scale.
3. **(a) reduce order**: only if (1)(2)+geom still leave a diff — there is no body edit that changes (a) without
   changing tile shape; (a) is resolved by the GEOMETRY branch, not here. So if a residual survives (1)(2)+native-
   geom, it IS the geometry branch (route to that fix).
4. **(c) state-carry cast**: do NOT "fix" by forcing a bf16 round-trip between tree nodes — that would make our tree
   LESS precise to match native's bf16 chain, and the lossless bar is per-depth argmax vs native-on-path (within
   floor), NOT abs-0.0 to a bf16-rebanked chain (user 2026-06-09; bug-class #12). At N_PAD=1 (single token) there is
   no intermediate round-trip, so (c) does not appear; if the A/B shows N_PAD=1 clean but N_PAD=16 diverges with
   geom+`(1)(2)` applied, the residual is the fp32-carry-vs-bf16-rebank STRUCTURAL difference — that is EXPECTED and
   FAVORABLE (ours more precise), gate it as within-floor per-depth-argmax, do NOT chase abs-0.0.

Cheap (cast/op-order) edits: (d), (e). Structural: (b)-recompute (only if BV32 spills) and (c)-carry (do not
"fix"). Order to TRY: override→BV32/w1/s3 (free) → (d) → (e) → recompute-from-spine if spill. All BUILD-OUR-KERNEL;
no native call in the served path (native packed-decode = A/B oracle only).

---

## buildOurKernel / no-reroute confirmation
The served live path computes via OUR `_tree_gdn_kernel` (launch :1554-1592) / `_gdn_node_step`. native
`fused_recurrent_gated_delta_rule_packed_decode` is referenced ONLY as the A/B int-view ORACLE (and as the no-spec
decode oracle). NONE of the §4 edits route the spine/branch through a native call — they align OUR kernel's
opcodes/geometry to be bit-exact to the incumbent SASS (feedback_no_reroute_reward_hacking). The
`FR13_TREE_GDN_GEOM_OVERRIDE` is an additive, value-neutral, default-OFF diagnostics lever (bug-class #10 A/B), not
a served-path value change.

## Bug-class anchors
- **#10 (shared-source ≠ shared-SASS / codegen identity):** the entire scan-vs-native-packed question is #10. The
  body is the same source-shape; the open question is SASS identity, gated by the byte A/B (int-view, NEVER atol).
  The deployed scan was only checked vs a serial TORCH ref — bit-exact-to-serial-ref ≠ bit-exact-to-incumbent-SASS.
- **#12 (depth / co-residency / measurement traps):** the N_PAD=1 vs 16 split IS the discriminator — at N_PAD=1 the
  recurrent body has no intermediate state round-trip so a clean N_PAD=1 + dirty N_PAD=16 localizes to the
  multi-token state-carry seam (c), which is fp32-carry-vs-bf16-rebank = favorable, gate within-floor not abs-0.0.

## One-line verdict
native DECODE is RECURRENT rank-1, the SAME math family as our scan — there is NO structural chunk-vs-recurrent
gap to the live oracle; the "diffuse irreducible" carrier rested on the WRONG (chunked-prefill) reference. Residual
bit-difference, if the A/B finds any, is CODEGEN-class: geometry (BV16/w8 vs native BV32/**w1**/s3 — correcting the
prior BV32/w4) and two cheap opcode/cast seams (l2norm rsqrt→1/sqrt, beta bf16 round-trip), all ALIGNABLE by
building OUR kernel bit-exact; the only genuinely structural seam (fp32 state-carry vs bf16 rebank, N_PAD≥2) makes
ours MORE precise and is gated within-floor, not chased to abs-0.0.
