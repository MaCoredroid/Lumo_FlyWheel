# FR13 +2 Spine Alignment — Locating the Last Mile to Native (READ-ONLY research)

Date 2026-06-14. GB10 DGX Spark (sm_121, unified LPDDR5X, mem-bandwidth-bound; GDN routes to
`fla_chunk_gated_delta_rule` raw-g / l2norm-in-kernel — flashinfer GDN is Hopper sm_90 only, see
`reference_gb10_gdn_backend_fla`). READ-ONLY: this doc only; a concurrent GPU workflow owns
`gdn_linear_attn.py` + the forked launcher (the ba-proj +17 fix). No kernel edits here.

## TL;DR
- cat9 22 = native-floor **3** + REAL **+2 spine** + REAL **+17 leaf co-residency** (the +17 is the
  bf16 `in_proj_ba` GEMM M-keyed batch-variance, being fixed in parallel; not this doc).
- The **+2 is the our-tree-scan-vs-native-recurrence REALIZATION seam** (class-10 "shared-source ≠
  shared-SASS", `FR13_BUG_CLASS_PLAYBOOK.md` row 10). conv is bit-exact (`FR13_CONV_NOT_CARRIER`),
  fp8 GEMMs M-invariant, softplus/op-order line-for-line identical (verified below). The ONLY
  un-aligned seam left in the recurrence is the **Triton codegen layout: our scan launches
  `num_warps=8 / BV=16`, native launches `num_warps=4 / BV=32`** over a V=128 head.
- **(1) Localization:** the +2 are NOT diffuse over 5 positions. chain5's 5 clear-margin flips are
  ALL in prompt 2, and 4 of them (pos 25–28) are a **single branch-point cascade** (one format-token
  flip at `\`\`\`tool_call` vs native `\`\`\`bash`, then 3 positions re-scored against a now-diverged
  prefix). Native's 3 flips are one-per-prompt at DIFFERENT high-entropy boundaries. So the e2e +2 =
  **a small number (≈1–2) of extra argmax-boundary crossings carried by the diffuse per-layer ULP
  drift**, concentrated where the spine hits a near-tie format choice. Diffuse CAUSE, localized EFFECT.
- **(2) Alignment knobs (ranked):** match native's launch config in our tree scan — `BV=32`
  (top), then `num_warps=4`, then the K-reduction emission. All are non-WY, non-reroute.
- **(3) Achievability:** 5→3 is reachable IN PRINCIPLE — native runs the SAME model/fp8/64 layers and
  sits at 3 (existence proof, `FR13_DIFFUSE_GDN_EXPLAINED`). The honest open question is **few vs many
  seams**: the V-tiling knob is the one un-closed seam (only ever atol-gated). Likely outcome: the
  cascade-anchor flip (pos 25) is a near-tie that ULP-realignment can move back; whether ALL +2 clear
  is what the cheap test resolves. There IS a true ~2-ULP MMA-grouping floor far below an argmax flip
  (do-not-grind), so a residual +0..+1 may persist if the spine sits exactly on a knife-edge tie.

---

## (1) +2 LOCALIZATION — which of chain5's 5 are the excess vs native's 3

Evidence: `output/fr13_shape_sweep/chain5_flips.json` (our pure-spine, chain depth-5, FLASH spine),
`output/fr13_verify_decisive/q3_native_classify.json` (native E5), and the q1 deep-row finding
`output/fr13_verify_decisive/q1_recur_vs_chunked.json`.

**chain5 — 5 clear-margin flips, ALL in prompt 2, 4 of them one contiguous cascade:**

| pos | served | oracle argmax | dev_nat | clear | note |
|----|--------|---------------|---------|-------|------|
| 25 | `tool` | `bash` | 2.75 | ✓ | **CASCADE ANCHOR** — format fork `\`\`\`tool_call` vs `\`\`\`bash` |
| 26 | `_call`| `\n` | 2.75 | ✓ | re-scored against the pos-25-diverged prefix |
| 27 | `\n` | `_name` | 4.25 | ✓ | same cascade |
| 28 | `<` | `\`\`\`` | 4.0 | ✓ | same cascade |
| 43 | `"{` | `{"` | 2.94 | ✓ | separate JSON-brace-order near-tie |

The cascade context: served = `...locate the relevant files.\n\n\`\`\`tool_call\n<function=...`,
oracle wanted `\`\`\`bash\n...`. pos 25 is a genuine high-entropy fork (two valid tool-call surface
forms). Once pos 25 commits, pos 26–28 are teacher-forced against an oracle conditioned on the
*served* prefix, so they read as flips even though only ONE independent decision diverged. **So the 5
clear-margin flips are really ≈2 independent divergence events (pos 25 cascade + pos 43 brace).**

**native E5 — 3 clear-margin flips, one per prompt, at different boundaries:**

| prompt | pos | served | clean argmax | dev_nat | context |
|--------|-----|--------|--------------|---------|---------|
| 1 | 94 | `Let` | `\`\`\`` | 1.88 | ` head -5\n\`\`\`\n\n` |
| 2 | 33 | ` "` | ` '` | 6.38 | `Name: find\nArguments:` (quote choice) |
| 3 | 68 | `Let` | `\`\`\`` | 8.38 | ` -20\n\`\`\`\n\n` |

Native's prompt-2 flip is at pos **33** (quote `"` vs `'`) — a DIFFERENT boundary than our pos-25
cascade. Native flips at one near-tie per prompt; ours additionally trips the pos-25 format fork and
the pos-43 brace order.

**VERDICT (localization):** the +2 excess is **localized in effect to prompt 2's format region (the
pos-25 cascade anchor + pos-43 brace)** but **diffuse in cause** — it is NOT a single bad op/position
in the kernel. It is the diffuse L0→L58 GDN ULP accumulation (q1: L0 = 0.000854 vs the TRUE recurrent
oracle ≈ 1 bf16 ULP; amplified by gate `1/rms` and deep full-attn L59→L63 = 33.75) pushing the spine
across **one extra near-tie argmax boundary** (pos 25) than native crosses. This matches q1's
verdict_component: "L0 is NOT a fixable single op (it is at the bf16-ULP floor vs the true oracle)…
carried by DIFFUSE fp-accumulation L0–L58 amplified by deep full-attn." (`q1_summary.json`).

Caveat (class-12 measurement, `FR13_BUG_CLASS_PLAYBOOK.md` row 12): the cascade inflates the raw
clear-flip COUNT (5) above the independent-event count (≈2). The honest comparison to native's 3 is
"number of independent boundary crossings," not raw flip count. The e2e accept/event gate (cat9) is
the real arbiter; this localization is the mechanism, not a second verdict.

---

## (2) BIT-EXACT ALIGNMENT DRAFT — our fr10 tree-scan → native fused_sigmoid_gating recurrence

Both kernels read side-by-side. Native is RECURRENT (`for i_t in range(0,T)`), our verify path is also
RECURRENT (`reference_gdn_verify_sequential_dispatch` — vLLM Qwen3-Next verify uses
`fused_sigmoid_gating` sequential, NOT chunked; WY/chunked is the WRONG tool and is PARKED). So this is
recurrent-vs-recurrent codegen alignment — the SAME family that cracked the conv bf16-tap and the prior
scan `static_range`→`tl.range`.

Files cited:
- OURS: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` — `_gdn_node_step` (L337–383, shared body),
  `_tree_gdn_kernel` (L386–543, tree scan), launch at L1500–1536 (`BV=16` const L18, `num_warps=8`
  L1536).
- NATIVE: `/tmp/vllm_live_019/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` —
  `fused_sigmoid_gating_delta_rule_update_kernel` (L24–193, recurrence body L136–192), launch
  `BK=next_power_of_2(K)=128`, `BV=min(next_power_of_2(V),32)=32`, `num_warps=4`, `num_stages=3`
  (L223–227, L291–292).
- Model dims (`/models/qwen3.6-27b-fp8/config.json` text_config): `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=48`. So
  K=128, V=128 per head.

### Op-order audit — what is ALREADY identical (do NOT touch; not carriers)
The per-node arithmetic is line-for-line identical between `_gdn_node_step` and native L143–167:
1. softplus: ours `tl.where(x<=20.0, tl.log(1+tl.exp(x)), x)` (L367–371) vs native
   `tl.where(beta*x<=threshold, (1/beta)*tl.log(1+tl.exp(beta*x)), x)` with beta=1.0, threshold=20.0
   (L144–146). **beta=1.0 makes `beta*x`/`(1/beta)` identity multiplies by 1.0 = bit-exact no-ops in
   fp32.** NOT a carrier. (Confirm beta/threshold default 1.0/20.0 at the call site — if the deployed
   image passes beta≠1, that WOULD be a seam; live shows the default.)
2. `b_g = -tl.exp(A_log) * softplus_x` — identical (L372 vs L147).
3. `b_beta = tl.sigmoid(raw_b)` — identical (L373 vs L150).
4. l2norm `q,k *= rsqrt(sum(x*x)+1e-6)` — identical (L375–376 vs L153–154), `USE_QK_L2NORM_IN_KERNEL`.
5. `b_q *= scale` AFTER l2norm — identical order (L377 vs L155). OUR `OUTPUT_SCALE` = native `scale` =
   `K**-0.5 = 128**-0.5`. **VERIFY this value is passed identically** (cheap, see knob #4).
6. state decay `state *= exp(b_g)` (L378 vs L158, non-KDA branch) — identical.
7. delta `b_v -= sum(state*k[None,:],axis=1); b_v *= b_beta; state += b_v[:,None]*k[None,:]` — identical
   (L379–381 vs L162–165).
8. output `out = sum(state*q[None,:],axis=1)` — identical (L382 vs L167).

**⟹ the recurrence MATH and op-order are already aligned. The only divergence is CODEGEN LAYOUT.**

### The class-10 codegen seam (cited to both files)
| knob | OURS (`fr10_gdn_tree_kernel.py`) | NATIVE (`fused_sigmoid_gating.py`) | effect |
|------|-----|--------|--------|
| **BV (V-tile)** | `BV=16` (L18); tile `(N_PAD, BLOCK_V=16, DIM_K)` | `BV=min(npo2(128),32)=32` (L223) | DIFFERENT — drives the whole layout |
| **num_warps** | `8` (L1536/L843/L1284) | `4` (L227) | DIFFERENT — changes warp partition of the K-reduction |
| NV (programs over V) | `cdiv(128,16)=8` | `cdiv(128,32)=4` | DIFFERENT (consequence of BV) |
| BK (K-tile) | DIM_K=128 whole | `npo2(128)=128` whole | same |
| K-reduction `tl.sum(...,axis=1)` | over DIM_K=128 with BV=16 rows, 8 warps | over K=128 with BV=32 rows, 4 warps | **DIFFERENT reduction tree / FMA schedule** |
| `static_range` unroll | tree scan uses `tl.static_range(0,N_PAD)` + nested `tl.static_range(0,i)` ancestor gather (L459–467) | `range(0,T)` (L136, runtime loop) | ours fully unrolls + a per-i `tl.where`-gather of `h_cache`; native is a flat loop — extra register pressure |
| raw-g / l2norm | RAW_GATING=True, l2norm-in-kernel True | same | same (not a seam) |
| cast boundaries | all loads `.to(tl.float32)`, fp32 accum, no intermediate bf16 | same `.to(tl.float32)` | same |

**Root carrier (ranked #1): `BV` (and the `num_warps` that follows it).** With V=128:
- native: BV=32 → each program owns a 32-row V-tile, 4 warps. The `tl.sum(b_h * b_k[None,:], 1)`
  K-reduction (contract over K=128) and `tl.sum(b_h * b_q[None,:],1)` output reduction are emitted as a
  specific warp-level reduction tree for the (32,128) tile under 4 warps.
- ours: BV=16 → 16-row V-tile, 8 warps. Triton lays out the SAME `tl.sum` over a (16,128) tile under 8
  warps as a DIFFERENT reduction tree (more warps, narrower tile) → different partial-sum grouping →
  **fp non-associativity yields a ~1-bf16-ULP-different result per node-step**, compounding over 48 GDN
  layers (online research: Triton `tl.sum` order is num_warps-dependent; FLA is documented
  non-deterministic, Megatron ships a separate torch chunked impl for deterministic mode — see Sources).

**The fix (real numerics matching, NOT reroute):** change OUR tree-scan launch to native's blocking —
`BV=32`, `num_warps=4` — so the K/output reductions over the V-tile compile to the same reduction tree
and FMA schedule as native. This is a launch-constexpr change to OUR kernel (the `BV` module const L18 +
the `num_warps=` at the four launch sites); it does NOT route the spine through native's kernel (reward-
hack banned, `feedback_no_reroute_reward_hacking`). The h_cache register tile becomes `(N_PAD,32,128)`
fp32 — at N_PAD=16 this is 16·32·128·4 = 256 KiB/program, which **may spill** (the same h_cache-scaling
pressure that motivated `num_warps=8` originally, `FR13_CACHE_SCALING_FUTURE`). Mitigations, in order:
1. **For the SPINE (N_PAD≤6 chain) the spill is far smaller** — chain5 uses N_PAD=8 padded; at BV=32
   that's 8·32·128·4 = 128 KiB. The +2 is a SPINE phenomenon, so test BV=32 on the spine shape first.
2. If N_PAD=16 deployed tree spills at BV=32/warps=4, keep BV=32 but raise num_warps back to 8 ONLY for
   the wide tree and pin BV/warps per-shape — but note **warps=8 re-introduces the reduction-tree seam**,
   so the spine arm (which carries the +2) must use the native-matched (BV=32, warps=4) config.
3. Alternative if spill is hard: emit the K-reduction with an explicit accumulation order that matches
   native's warp layout independent of num_warps (a manual tree-reduce), so warps can stay 8 for
   occupancy while the partial-sum grouping matches native. Higher effort; only if knob #1 leaves residual.

### Ranked alignment knobs (all non-WY, non-reroute, OUR kernel computes)
1. **BV 16→32** (match native `min(npo2(V),32)`). Single const (L18) + re-tile h_cache. Highest leverage:
   it changes the V-tile shape that the reduction tree is built over. **Top knob.**
2. **num_warps 8→4** (match native L227). One arg at each launch site (L843/L1284/L1536). Changes warp
   partition of the reduction; pairs with #1 (native uses 32/4 together).
3. **num_stages**: native `num_stages=3` (L226); ours unspecified (Triton default). Pin to 3 to match
   pipelining (minor; affects scheduling not numerics much, but pin for codegen identity per row-10).
4. **scale value parity**: assert OUR `OUTPUT_SCALE == 128**-0.5` byte-equals native `scale` (k.shape[-1]
   **-0.5). Trivial, removes a confound (`FR13_BUG_CLASS_PLAYBOOK` row 9 engagement assert).
5. **Unroll form**: native is a runtime `range` loop; ours `static_range`. For the SPINE (no branches)
   the ancestor-gather collapses to "j is the immediate predecessor," so a spine-only path could emit a
   flat `range` loop matching native's instruction stream. Lower priority — only chase if 1–3 leave residual.

### Re-arm the A/B gate (row 10 discipline)
After ANY knob change, re-run the byte A/B on captured payloads with int-view equality (NEVER atol) per
`FR13_BUG_CLASS_PLAYBOOK.md` row 10 ("byte A/B on captured payloads, int-view equality, SASS hash pin …
the A/B gate re-armed per toolchain"). The existing harness `scripts/fr13_gdn_scan_warp_gate.py` does
exactly this (our tree kernel vs `native_update_serial_per_path` at N_PAD=1 and N_PAD=16) — extend it to
A/B the (BV=32,warps=4) config and report raw `max_abs` + int-view-equal, NOT the loose atol.

---

## (3) ACHIEVABILITY — is 5→3 reachable, or is ≈5 an irreducible floor?

**Reachable in principle — existence proof stands.** Native runs the SAME model, SAME fp8, SAME 64
layers, and sits at 3 clear flips (`q3_native_classify.json`; `FR13_DIFFUSE_GDN_EXPLAINED`: "native …
drifts only 3 … if we matched native bit-for-bit at each op (op order, cast boundaries, reduction order,
the Triton num_warps/BV codegen), we'd … be at native's 3"). A recurrent chain CAN sit at 3; our kernel
is the SAME recurrence with a DIFFERENT codegen layout.

**The real question is FEW vs MANY seams (per `FR13_DIFFUSE_GDN_EXPLAINED` §"few vs many").** The
drift-localize already ruled out fp8 / conv-tap / conv-window and left exactly ONE un-closed recurrence
seam: **the GDN scan `num_warps=8/BV=16` codegen (vs native 4/BV=32), only ever atol=1e-3-gated.** This
doc confirms (op-order audit above) that the MATH is identical — so if a few seams remain they are all
codegen-layout, not algebra.

**Honest bounds:**
- BEST case: BV/warps alignment makes the per-node result bit-exact (or within the ~2-ULP MMA-grouping
  floor) → the diffuse accumulation tracks native → the pos-25 near-tie no longer crosses → 5→3 (or →3±1).
- WORST honest case: a true **~2-ULP MMA-grouping floor (~1e-12)** exists and is irreducible (do-not-
  grind, far below an argmax flip per `FR13_DIFFUSE_GDN_EXPLAINED`). If the spine sits EXACTLY on a
  knife-edge tie at pos 25 (dev_nat 2.75 is NOT a knife-edge — it's a real 2.75-nat margin in the
  ORACLE, but the SERVED top-2 are near-tied: chain5 tree_served_top5 shows ties to 1e-4), a residual
  +0..+1 could persist. The pos-43 brace (dev 2.94) is similar. So a plausible landing is **3→4, not
  always exactly 3** — still a strict win over 5 and within the diffuse-floor story.
- The +17 leaf co-residency fix (parallel) is ORTHOGONAL — it removes the M-keyed in_proj_ba variance,
  not the scan codegen. Both must land for cat9 → ≈3–4.

**Not a dead-end (research-before-deadend, `feedback_research_before_deadend`):** there is an un-tested
alignment knob (BV/warps) with a cheap GPU test and an existence proof. Do NOT call ≈5 irreducible until
the BV=32/warps=4 A/B is run.

---

## CHEAPEST GPU TEST FOR THE TOP KNOB (BV 16→32, num_warps 8→4)

**In-process scan A/B, no model boot** — reuse `scripts/fr13_gdn_scan_warp_gate.py` (already loads a
captured tree-GDN payload and compares our `launch_tree_gdn_prepared` against `native_update_serial_
per_path` at N_PAD=1 and N_PAD=16). Steps:
1. Run it AS-IS to record the current (BV=16, warps=8) raw `max_abs` vs native per-path (the baseline
   seam magnitude). This is the class-10 byte A/B, int-view (NOT atol).
2. Add a (BV=32, warps=4) variant launch (or env-flag the const) and re-run on the IDENTICAL captured
   payload. Compare raw `max_abs` and int-view-equal count vs native.
3. **Decision:** if (BV=32,warps=4) drives our scan output bit-exact (or to the ~2-ULP floor) vs the
   per-path native recurrence where (BV=16,warps=8) did not → the codegen seam is THE carrier → expect
   5→3 e2e (then confirm e2e: spine chain5 vs E5 same-prompts, accept/event ≥ native, per-token argmax
   probe `fr13_gold_margin_probe.py` per `reference_scalar_metric_per_token_blindspot`). If it does NOT
   close the gap on the IDENTICAL payload → more seams (short grind: pin num_stages, flat-range spine
   unroll) or the residual is the MMA-grouping floor.

This is the cheapest possible test: a single in-process CUDA replay on a captured spine payload, our-vs-
native-recurrence, identical input, one constexpr flip — no boot, no serving, no graph capture. It
directly isolates the top knob from the +17 co-residency and from full-attn amplification. Watch the
h_cache spill at N_PAD=16 (256 KiB) — if it spills, run the SPINE shape (N_PAD≤8) where the +2 lives.

### Playbook rows quoted (`FR13_BUG_CLASS_PLAYBOOK.md`)
- **Row 10 (Shared-source ≠ shared-SASS / codegen identity):** "two kernels inline the same body but
  compile differently (constexpr/pressure)" → discriminator "byte A/B on captured payloads, int-view
  equality (NEVER atol), SASS hash pin" → fix "one shared body + identical constexprs/num_warps + the
  A/B gate re-armed per toolchain." THIS IS THE +2.
- **Row 11 (Batch-composition / BI-flag sensitivity):** pin BI on BOTH arms; near-ties flip on sub-ULP
  shifts — the pos-25/43 near-ties are exactly this class of crossing.
- **Row 12 (Measurement traps):** the cascade inflates raw flip count (5) above independent-event count
  (≈2); compare like-for-like (independent crossings, e2e accept/event), not raw counts; capture-once
  pinned spine payload for the A/B.
- **Row 9 (Silent fallback / vacuous instrument):** assert the scan engages + flag-state header in the
  A/B artifact before trusting any number; assert `OUTPUT_SCALE==native scale`.

## Sources (online)
- Triton `num_warps` ↔ reduction layout / `tl.sum`: triton-lang/triton#5327 (num_warps vs BLOCK_SIZE),
  triton-lang/triton#4602 (num_warps necessity), PyTorch "Triton Kernel Compilation Stages".
  https://github.com/triton-lang/triton/issues/5327 , https://github.com/triton-lang/triton/issues/4602 ,
  https://pytorch.org/blog/triton-kernel-compilation-stages/
- FLA non-determinism + chunked-vs-recurrent numerics: fla-org/flash-linear-attention#389, #104;
  Megatron Core `gated_delta_net` (torch chunked impl for deterministic mode, "FLA is not
  deterministic"). https://github.com/fla-org/flash-linear-attention/issues/389 ,
  https://github.com/fla-org/flash-linear-attention/issues/104 ,
  https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/apidocs/core/core.ssm.gated_delta_net.html
