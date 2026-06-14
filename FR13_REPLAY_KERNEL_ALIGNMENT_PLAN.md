# FR13 Replay-Kernel Alignment Plan — OUR `_tree_gdn_replay_kernel` → bit-exact to NATIVE sequential

**Status:** CONDITIONAL pre-design (read-only). The live A/B `w2vaqcsmx` measures
`max_abs(H_ours − H_native_seq)` per-layer per-event. THIS doc is the ready-to-apply fix the
moment that verdict lands. **Do not presume the outcome** — §6 caveats both branches.

**Frame (binding):** The accepted tokens are committed/correct, so the durable next-event `h0`
MUST equal the SEQUENTIAL recurrent state after those accepted tokens (a no-spec decode). The
reference is therefore native `fused_sigmoid_gating_delta_rule_update` (sequential rank-1).
Aligning OUR replay kernel to it is **BUILD-OUR-KERNEL-BIT-EXACT-TO-THE-INCUMBENT**
([[feedback_no_reroute_reward_hacking]], [[feedback_math_correct_vs_bitexact]]: the bar is
BIT-EXACT, not ℝ-correct). The fix is a numerics alignment of OUR kernel — **never** "call native
fused_sigmoid for the durable state" (that is the banned splice/reroute; native stays an ORACLE in
the A/B only). No native call enters the live path.

**Playbook rows in force:**
- **#10 Shared-source ≠ shared-SASS (codegen identity):** "two kernels inline the same body but
  compile differently (constexpr/pressure) … byte A/B on captured payloads, int-view equality
  (NEVER atol), SASS hash pin … any 'bit-exact by re-execution' claim." The replay's whole
  losslessness claim ("bit-exactness by re-execution of the identical fp32 instruction sequence",
  kernel comment L356-363) is EXACTLY this class — and it is gated, not proven. Two compilations
  with DIFFERENT `num_warps`/`BV`/`num_stages` are NOT the same SASS.
- **#12 Measurement traps:** the A/B reference is varlen `cu_seqlens=[0,M]`, `IS_SPEC_DECODING=
  False`. Its inner loop (L136-192) is byte-identical to the live incumbent's inner loop; the
  IS_VARLEN vs IS_SPEC_DECODING branches only change h0/store ADDRESS arithmetic, not the per-token
  op sequence. So the A/B reference is a like-for-like proxy for the real incumbent's durable
  state. (Label every estimate; this is an equality-by-source-inspection claim, re-checked below.)

---

## 1. Op-by-op kernel map

Both kernels are **pure sequential rank-1 delta-rule** (NO chunk/WY in either) over a token
sequence, fp32 state accumulator. Same mathematical recurrence. The map below is per-token.

Model dims (Qwen3-Next, `/models/qwen3.6-27b-fp8/config.json`): `head_k_dim(K)=128`,
`head_v_dim(V)=128`, `num_k_heads=16`, `num_v_heads=48`. Reduction axis K=128 is the FULL
contraction in BOTH (native asserts `NK==1`; ours uses full `DIM_K`), so the contraction LENGTH is
identical; what differs is tiling of the V/output dim and warp partition of the K-reduction.

| Op (per token t) | NATIVE `fused_sigmoid_gating_delta_rule_update_kernel` (`/tmp/vllm_live_019/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`) | OURS `_gdn_node_step` (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:337-383`, called by `_tree_gdn_replay_kernel:674`) | Match? |
|---|---|---|---|
| load q,k,v,b,a | `:137-140,143` `.to(tl.float32)` from consumed dtype | replay loads from RING (`k_ring,v_ring,a_ring,b_ring`) `.to(tl.float32)`; ring stores `key_spec` (pre-l2norm), `value_tree`, `a`, `b` byte-copies at consumed precision (patcher L4824-4835) | input VALUES bit-equal IFF ring byte-copy == native's source tensor (k pre-l2norm). **Verify** ring k == the same `key_spec` native consumed (it is the same activation; copy is `.copy_`, no recast). |
| softplus / g | `:143-147` `x = a.f32 + dt_bias.f32; sp = where(beta*x<=thr, (1/beta)*log(1+exp(beta*x)), x); b_g = -exp(A_log.f32)*sp` with **beta=1.0, thr=20.0** | `:366-372` `x = raw_a + dt_bias; sp = where(x<=20.0, log(1+exp(x)), x); b_g = -exp(A_log)*sp` | algebraically identical at beta=1.0 BUT native emits `beta*x` and `(1/beta)*log(...)` MULTIPLIES — see seam (d). |
| beta (gate) | `:150` `b_beta = sigmoid(b_b.to(f32))` | `:373` `b_beta = sigmoid(b_raw_b.to(f32))` | identical |
| l2norm q,k | `:152-154` `q*=rsqrt(sum(q*q)+1e-6); k*=rsqrt(sum(k*k)+1e-6)` (in-kernel) | `:374-376` identical | identical (BOTH in-kernel; ring stores PRE-l2norm k, so replay re-normalizes exactly like native) |
| q scale | `:155` `b_q *= scale` (scale=K**-0.5) | `:377` `b_q *= OUTPUT_SCALE` (=head_k_dim**-0.5, patcher L4860) | identical value; q never touches state — IRRELEVANT to durable state |
| state decay | `:158` `b_h *= exp(b_g)` | `:378` `state_i *= exp(b_g)` | identical op |
| delta read | `:162` `b_v -= tl.sum(b_h * b_k[None,:], 1)` reduce over K=128 (axis1, BK=128) | `:379` `b_v -= tl.sum(state_i * b_k[None,:], axis=1)` reduce over K=128 | SAME op, SAME length — but warp partition of the 128-reduction differs (seam a) |
| delta scale | `:163` `b_v *= b_beta` | `:380` `b_v *= b_beta` | identical |
| rank-1 update | `:165` `b_h += b_v[:,None] * b_k[None,:]` | `:381` `state_i += b_v[:,None] * b_k[None,:]` | identical op |
| output | `:167` `b_o = sum(b_h*b_q,1)` then store | `:382` `out_i = sum(state_i*b_q,axis=1)` (replay passes q=0, out discarded) | IRRELEVANT to durable state |
| state store | `:171-184` `b_h.to(p_ht.dtype)` → ht (fp32 bank) | replay `:699-706` store `state` (fp32) → bank LINEAR column | store DTYPE fp32 both — match (seam f checks the rounding boundary) |
| h0 load | `:107-134` from initial_state row (`.to(f32)`) | replay `:609-616` from bank row (`.to(f32)`) | match (both fp32 bank) |

**Launch-config divergence (the codegen seam, NOT visible in the body):**

| Knob | NATIVE (`fused_sigmoid_gating.py:223,226-227`) | OURS (`fr10_gdn_tree_kernel.py:812-844`) | |
|---|---|---|---|
| `BV` (V/output tile) | `min(next_pow2(128), 32) = 32` → NV=cdiv(128,32)=**4 programs/head** | `BV=16` (module const L18) → cdiv(128,16)=**8 programs/head** | **DIVERGES** |
| `BK` | `next_pow2(128)=128` (full) | full `DIM_K=128` | match |
| `num_warps` | **4** | **8** (L843) | **DIVERGES** |
| `num_stages` | **3** | unset → Triton default (≈2) | **DIVERGES** |

---

## 2. Candidate bit-exact divergence seams (WHERE + alignable vs structural)

Ranked by likelihood of being live, with the discriminator the A/B already provides.

### (a) `num_warps` / `BV` codegen of the K-reduction — **PRIME, alignable**
- **WHERE:** launch `_tree_gdn_replay_kernel[grid](..., BLOCK_V=BV=16, num_warps=8)` (L829,843)
  vs native `BV=32, num_warps=4, num_stages=3`. The two `tl.sum(..., axis=1)` over K=128
  (delta read `:379`, output `:382`) are partitioned across a DIFFERENT number of warps and a
  DIFFERENT V-tile, so the partial-sum tree of the 128-wide fp32 reduction is reordered. fp add is
  non-associative ⇒ ~1-bf16-ULP per-token realization difference that COMPOUNDS over the accepted
  chain and (cross-event) over depth.
- **Corroboration:** `FR13_DIFFUSE_GDN_EXPLAINED.md:42-44` names this EXACT seam for the sibling
  SCAN kernel — "the GDN scan `num_warps=8/BV=16` codegen (vs native 4/BV=32), only ever
  atol=1e-3-gated." The replay SHARES `BV=16`+`num_warps=8`+no-`num_stages`. Same axis.
- **ALIGNABLE → 0.0.** Pure launch-config change in OUR kernel; no math/formula change. This is the
  cheapest, highest-prior fix. It does NOT route through native — it makes our kernel COMPILE the
  same partition.
- **Class #10 note:** matching `num_warps`/`BV`/`num_stages` is necessary but NOT automatically
  sufficient for byte-identity (constexpr/unroll/register pressure can still diverge). The A/B
  (int-view, never atol) is the gate; if it lands 0.0 after matching the three knobs, seam closed.

### (b) `num_stages` pipelining — **alignable, couple with (a)**
- **WHERE:** ours omits `num_stages` (Triton default ≈2) vs native `num_stages=3`. Software
  pipelining can re-order independent FMAs across loop iterations and change accumulation grouping.
- **ALIGNABLE → 0.0.** Add `num_stages=3` to the replay (and all-layers) launch. Free; bundle with (a).

### (c) Ring byte-copy vs native source dtype (cast boundary) — **alignable, verify-first**
- **WHERE:** patcher L4824-4835 copies `key_spec[..]`/`value_tree[..]`/`a`/`b` into the rings via
  `.copy_`. The replay re-loads from the ring `.to(tl.float32)` (L654,663,668,673). The scan loads
  the SAME `key_spec`/`value_tree`/`a`/`b` `.contiguous()` `.to(tl.float32)` (L4838-4843). Native
  durable state (real incumbent) consumes the SAME activations.
- **Risk:** if the ring dtype differs from the native source dtype, the `.copy_` re-rounds and the
  replay sees a DIFFERENT bf16/fp value than native. The patcher's shape/dtype guard (L4773-4786)
  asserts `ring.dtype == key_spec.dtype` etc, so the copy is a same-dtype memcpy — **no recast**.
  This seam should be 0 BY CONSTRUCTION; the A/B's use of the SAME rings as input to BOTH arms
  (harness L6535-6538 index_selects the rings for native too) means a ring/source mismatch would
  NOT show in this A/B (both arms read the ring). **So (c) cannot be the A/B carrier**, but it
  could be a live carrier vs the TRUE incumbent. **Verify separately** (ring k == the `key_spec`
  the live native path would consume) — out of scope for this A/B, note for the live gate.
- **ALIGNABLE** if ever live: store the ring at native's consumed dtype (already enforced).

### (d) softplus `beta*x` vs `x` instruction shape — **alignable, low prior**
- **WHERE:** native `:145` literally emits `beta * x` and `(1/beta) * log1pexp(beta*x)` with the
  runtime scalar `beta=1.0`; ours `:367-369` emits bare `x` / `log1pexp(x)`. At beta=1.0 these are
  the same VALUE, but native inserts two extra fp mults (`1.0*x`, `1.0*result`). `1.0*x == x`
  exactly in IEEE (no rounding), so this is bit-identical — **but** the threshold COMPARE differs:
  native `beta*x <= threshold` (= `x <= 20.0` at beta=1.0) vs ours `x <= 20.0`. Same branch.
- **ALIGNABLE / likely already 0.0.** Only act if the A/B isolates the gate. If acting: mirror
  native's `beta`/`threshold` scalar form (carry `beta=1.0`, `threshold=20.0` as kernel args and
  emit `beta*x`) so the SASS matches. Lowest prior — `1.0*` is exact.

### (e) Reduction ORDER over the chain — **N/A (both sequential rank-1), NOT structural**
- Native iterates `for i_t in range(0,T)` strictly sequentially; our replay iterates
  `for t in tl.static_range(0, PATH_COLS+1)` strictly sequentially along the accepted LINEAR chain
  (root node 0 + accepted path, L624-689). **Same sequential order, same recurrence.** The replay
  is NOT a chunked/parallel-scan formulation — it is the identical sequential delta-rule.
  ⇒ **NO structural rewrite needed.** This is the key skeptical finding: the replay is the SAME
  formulation as native sequential, so bit-exactness is reachable by op-order/codegen alignment,
  not a rewrite. (Contrast the PARKED batched-WY kernel, which WAS a different formulation —
  [[reference_gdn_verify_sequential_dispatch]]: chunked-WY is prefill-only, can't be bit-exact to
  sequential. The replay deliberately AVOIDS that by being sequential rank-1.)

### (f) state store/h0 rounding boundary — **alignable, should be 0.0**
- **WHERE:** native stores `b_h.to(p_ht.dtype)` (`:180,184`); our replay stores `state` (L704).
  Bank is fp32 (launch asserts L742-745), native A/B uses `inplace_final_state=False` → fresh
  `initial_state.dtype` tensor = fp32 (`_h0` cloned `.to(torch.float32)` L6544). Both fp32 → no
  store rounding. h0 load both fp32. **0 by construction.** Only a seam if the live bank were bf16
  (it is not).

### (g) Parent-handoff `+ 0.0` (-0.0→+0.0 flip) — **alignable, replay-specific, already handled**
- **WHERE:** replay L645 `state = state + 0.0` on each non-root edge, to reproduce the scan's
  `tl.sum(tl.where(...), axis=0)` -0.0→+0.0 flip. Native sequential has NO such gather (it streams
  one contiguous sequence). The A/B native reference ALSO streams contiguously (varlen [0,M]) so it
  has no -0.0 flip either. **If** the A/B isolates a sign-of-zero-only diff at an edge, the `+ 0.0`
  is matching the SCAN, not native sequential — and should be REMOVED for the durable-state path
  (the durable reference is native sequential, which never flips). Low prior (only -0.0 channels),
  but note: the comment L641-644 ties this to the scan's behavior, which is the WRONG reference for
  the durable state. **Flag:** align to native-sequential (no `+0.0`), not to the scan.

---

## 3. Alignment design — concrete edits to OUR replay kernel, dependency order

Apply upstream→downstream. Each is BUILD-OUR-KERNEL (launch-config / op-shape in our file); no
native call enters the live path.

**Step 0 (free, do first regardless):** in `launch_tree_gdn_replay` (L812-844) AND
`launch_tree_gdn_replay_all_layers` (L1245-1284) set the launch to MATCH native:
```
BLOCK_V = 32          # was BV=16  → match native min(next_pow2(V),32)
num_warps = 4         # was 8       → match native
num_stages = 3        # was unset   → match native
```
This is seams (a)+(b) together — the PRIME hypothesis. `BV=32` requires the module const `BV` (L18)
NOT be reused blindly for the scan path (the scan's `h_cache` register budget may differ); scope
the change to the REPLAY launches only, or introduce `REPLAY_BV=32` so the scan is untouched.
(Skeptic note: the scan ALSO has the 8/16 seam per `FR13_DIFFUSE_GDN_EXPLAINED`, but the scan
output is the OUT/accept path; THIS doc's reference is the durable STATE = replay. Fix the replay
first; the scan's seam is a separate front.)

**Step 1 (verify, no edit unless A/B says):** confirm ring dtype == native consumed dtype (seam c)
— the guard at patcher L4773-4786 already enforces this; assert it holds at the live gate.

**Step 2 (conditional, only if A/B isolates a sign/edge diff):** drop the replay's `state = state
+ 0.0` (L645) for the durable-state path — its reference (scan) is wrong for the durable state;
native sequential has no edge flip. (seam g)

**Step 3 (conditional, only if A/B isolates the gate channel after Step 0):** mirror native's
softplus `beta*x` / `(1/beta)*log1pexp(beta*x)` scalar form with `beta=1.0, threshold=20.0` passed
as kernel args (seam d). Lowest prior; `1.0*` is exact so this likely does nothing.

**No structural rewrite is anticipated** (seam e): the replay is already native's sequential
rank-1 formulation. If — and only if — after Steps 0-3 the A/B still shows a growing diff that
localizes to the recurrent UPDATE op (not the gate, not the reduction partition), re-open the
possibility that some OTHER constexpr (DIM_K unroll, register pressure under BV=32) re-orders the
FMA; that is STILL alignable (class #10: pin SASS, adjust the offending constexpr), not a rewrite.

**Cost:** Step 0 = trivial (3 launch kwargs). Steps 1-3 = trivial/conditional. NONE is a
formulation rewrite. This is a CHEAP grind, consistent with the scan-grind history (conv bf16-tap,
scan static_range, intra-chunk A cast — each a one-liner once located).

---

## 4. Alignable vs structural — per-seam verdict

| Seam | Alignable (→0.0) | Structural (rewrite to native sequence) |
|---|---|---|
| (a) num_warps/BV codegen | **YES** — launch kwargs | no |
| (b) num_stages | **YES** — launch kwarg | no |
| (c) ring cast boundary | **YES** — dtype already enforced; verify | no |
| (d) softplus beta-shape | **YES** — emit native scalar form | no |
| (e) chain reduction order | **N/A** — already identical sequential rank-1 | **NO rewrite needed** |
| (f) store/h0 rounding | **YES** — already fp32, 0 by construction | no |
| (g) parent-handoff +0.0 | **YES** — remove for durable path | no |

**Overall: every located seam is ALIGNABLE. No structural rewrite is required** because the replay
is ALREADY native's sequential rank-1 formulation (the deliberate design choice that distinguishes
it from the parked batched-WY kernel). The ONLY thing standing between our replay and bit-exact is
the launch-config codegen (a/b) plus two conditional one-liners (g/d).

---

## 5. Build-our-kernel, not reroute (confirmation)

Every edit in §3 is to OUR `_tree_gdn_replay_kernel` / its launch in our repo file. The native
`fused_sigmoid_gating` appears ONLY as the A/B ORACLE (harness `_fr13_replay_durable_ab`, default
OFF, `inplace_final_state=False` into a throwaway tensor) — it never writes the served bank, never
enters the live durable-state path. The live path computes durable state with OUR replay kernel,
made to match native's realization. This is build-our-kernel-bit-exact-to-the-incumbent
([[feedback_no_reroute_reward_hacking]]), NOT splice/reroute. The gate is verified replay-ON, our
kernel computing. We do NOT propose "call native fused_sigmoid for the durable state."

---

## 6. Conditionality (do NOT presume the A/B outcome)

- **IF `w2vaqcsmx` shows nonzero + growing `max_abs(H_ours − H_native_seq)`** (the PRIME 21-flip
  carrier hypothesis confirmed): apply §3 Step 0 first (the named codegen seam), re-run the A/B
  int-view. If 0.0 → seam (a/b) was the carrier (cheap, the user's read). If still nonzero, Steps
  2/3 then the §3 last-paragraph constexpr chase — all alignable.
- **IF the A/B lands ~0** (per-layer per-event, all chains): the replay is ALREADY faithful to
  native sequential, and §1's op-by-op map is the ANALYTICAL CONFIRMATION of why (same sequential
  rank-1, same in-kernel l2norm/raw-g, fp32 bank, same recurrence). Then the 21 flips are NOT in
  the durable handoff and the carrier is elsewhere (back to the scan-codegen front on the OUTPUT/
  accept path, or branch co-residency — `FR13_DIFFUSE_GDN_EXPLAINED:47-50`). In that case this doc
  is the durable-handoff EXONERATION, and §2(a) redirects to the scan kernel as the next front.
- **Skeptic guard (this session overturned several single-seam overstates — FA2-tile carrier
  OVERTURNED, conv "carrier" refuted, multispine):** seam (a) is the highest-prior but is NOT
  asserted as THE carrier until the int-view A/B (NEVER atol) confirms 0.0 after Step 0. The diff
  could be DISTRIBUTED across (a)+(b)+(g); match all three before declaring. Class #10: source
  identity ≠ SASS identity — even after matching three knobs, the A/B is the only proof.

---

## 7. Files / line anchors
- OUR replay: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` — `_gdn_node_step` 337-383;
  `_tree_gdn_replay_kernel` 545-706; `launch_tree_gdn_replay` 709-844 (knobs L829 `BLOCK_V=BV`,
  L843 `num_warps=8`); all-layers launch 1111-1284 (L1264 `BLOCK_V=BV`, L1284 `num_warps=8`);
  `BV=16` module const L18.
- NATIVE incumbent + A/B oracle: `/tmp/vllm_live_019/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`
  — inner loop 136-192; gate 143-150; l2norm 152-154; knobs 223 (`BV=min(np2(V),32)=32`), 226-227
  (`num_stages=3, num_warps=4`).
- Ring fill (cast-boundary seam c): `scripts/fr10_phase4_patch_vllm_tree_gdn.py` L4747-4835;
  dtype guard L4773-4786; replay launch sites L7621-7637 / L8194-8210; A/B harness
  `_fr13_replay_durable_ab` L6475-6580.
- Corroboration: `FR13_DIFFUSE_GDN_EXPLAINED.md:42-44` (num_warps=8/BV=16 named seam).
