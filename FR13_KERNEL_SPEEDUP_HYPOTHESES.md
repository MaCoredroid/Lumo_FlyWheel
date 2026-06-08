# FR13 Kernel Speedup Hypotheses — Ranked Deliverable

**Scope:** the two FR13 verify kernels on DGX Spark GB10 (sm_120/sm_121, Grace-Blackwell, unified LPDDR5X, ~273 GB/s).
**Workload:** speculative-decode VERIFY forward, B=4, 9-node token tree (spine 5 + 4 branches), temp 0.6, Qwen3.6-27B-fp8 (48 GDN linear_attn + 16 full_attn layers).

**The binding constraint (re-stated and used everywhere below):** the forward is WEIGHT-BANDWIDTH-BOUND. The fp8 weight stream is ~27 GB / 273 GB/s = **98.9 ms/forward** (reproduced `/tmp/bw_floor.py`). A kernel change helps wall-time ONLY if it (a) cuts the kernel's own HBM traffic, (b) is CUDA-graph FULL-capturable (removes eager-launch tax), or (c) removes wasted work/launches. Pure FLOP/ALU/SMEM/tensor-core throughput is HIDDEN behind the weight stream and does not move wall-time.

**Evidence status:** every claim below is tied to a reproduced CPU number, a file:line, or a citation. All five CPU evidence scripts in `/tmp` and the four validation/cost scripts were re-run for this synthesis and reproduce the cited values. GPU measurement remains the final arbiter for every `needs-gpu` item (per MEMORY: GDN/TreeAttention could not full-capture on vLLM 0.19 — must re-confirm on cu130-nightly).

---

## 1. Ranked table (obvious-now first within the helps/prototype tiers, then drops)

| Rank | Kernel | Speedup | HBM / capture impact | Expected gain | Implementability | Lossless? | Recommendation |
|---|---|---|---|---|---|---|---|
| **1** | GDN_tree | **WY one-pass + accept-only state commit** (replace ancestor-replay) | **Cuts the binding metric:** GDN-state r+w 9.0x→1.0x native. Removes +2.416 GB/req (8.95% of 27 GB) up to +9.664 GB at B=4 (35.79%). Capture-neutral (static loop). | **#1, largest, highest-confidence.** Single-digit → low-double-digit % of forward wall-time depending on B-accounting; much larger on GDN-layer time alone. | hard | yes (fp32 4.19e-9 / 0-flip, bf16 6.10e-5 < native floor 9.55e-5) | **do-now** |
| **2** | GDN_tree | **Preallocate `tree_state_all` / `core_attn_out_spec` / masks** (persistent buffers) | No HBM byte change, but **capture-ENABLING**: removes per-forward `torch.empty` (201.3 MB alloc/fwd) that freezes pointers / forces fresh pool inside a captured region. | Small direct; prerequisite-class for the B=4 CUDA-captured gate. Pairs with #1 / #3. | obvious | yes (same buffers, same math) | **do-now** |
| **3** | GDN_tree | **Fully-fused WY** (W/U/T/S on-chip, one Triton kernel) | Intermediate round-trip saving tiny (124.7 KB/layer). Win = launch collapse + being the single capturable kernel carrying #1's accept-only commit. | Moderate; mostly launch reduction. Bundle with #1 (this is #1's implementation FORM). | hard | yes (inherits #1, requires native fp32 op-order on within-path solve) | **do-now** |
| **4** | GDN_tree | **Shared-prefix state reuse across branches** (no-copy spine factor, append-only suffix) | Carries O(sum path_len)≤16 reflector columns, NOT O(branches) full states. **This is the property that makes #1's HBM=1.0x hold.** | Indirect — score as part of #1. Standalone FLOP saving (spine recomputed 4x in replay) is hidden behind weights. | hard | yes (coresidency byte-invariance 0.0; explicitly NOT banned per-branch copy) | **do-now** |
| **5** | FA2_tree_attn | **fp8 q/k/v descale guard** (verify tree-bias op adds no dequant HBM pass) | Defensive: confirms NO double-KV-read regression today (runtime KV is bf16, not fp8 — `kv_cache_dtype` rewritten fp8_e5m2→auto). | No speed gain; flags a latent correctness break if true fp8 KV is ever enabled. | moderate | yes | **do-now** |
| 6 | both | **CUDA-graph FULL capture of the verify forward** | The capture LEVER (rule b) + wasted-launch removal (rule c). No HBM byte change. | Potentially significant at B=4/9-node IF any op forces an eager break; ~0 if already captured. **GPU-gated.** | moderate | yes (replays identical kernels) | **prototype** |
| 7 | GDN_tree | **Fuse B-batch Python launch loop into one grid** (batch as program axis) | Copy traffic 72.1 MB = 0.267% of 27 GB (HIDDEN). Real lever = launch tax 192→48; **also gives fixed grid = cleaner capture.** | Moderate-conditional: 0.72 ms (eager 5us) → 28.8 ms (200us) — **size is GPU-gated.** | moderate | yes | **prototype** |
| 8 | FA2_tree_attn | **Confirm + lock FULL CUDA-graph capture of the tree-bias varlen op** | The only FA2-side real lever (rule b). Splitkv capture-blocker does NOT exist on this path (swap OFF, no runtime accum alloc). vLLM PR #20059 makes max_query_len>1 decode graph capturable. | Rank-1 of FA2 items but **BOUNDED**: launch-tax ceiling ~2.7% of forward; attention's own slice ~0.1%. Near-zero if already cgfull. | moderate | yes | **prototype** |
| 9 | GDN_tree | **Keep FR12 diagnostic clones/branches out of captured hot path** | Flags OFF by default = no extra HBM, no clone. If any flag live during capture, `.clone()/.cpu()` HARD-BREAKS capture (footgun). | Negligible direct (flags OFF). Value = removing capture-break footgun. | obvious | yes (observational) | **do-after-lossless** |
| 10 | GDN_tree | Eliminate masked-out replay work (bound outer loop to N_ACTUAL) | Pure FLOP (136→45 inner updates, 35.6% residual waste), unchanged state writes. Hidden behind weights. | Negligible; strictly dominated by #1 which deletes the replay. | obvious | yes | drop |
| 11 | FA2_tree_attn | Restrict apply_tree_bias to single query-suffix n_block | Pure instruction-count; bias matrix 324 B is L1-resident, no HBM/capture change. | Negligible; only bundle IF a capture change already touches this kernel. | obvious | yes | drop |
| 12 | FA2_tree_attn | Force num_splits=1 to kill splitkv combine launch | **Premise FALSE for the tree op:** swap OFF (max_seqlen_q=9), splitkv never set, combine never launches, no accum HBM allocated. Nothing to remove. | ~0. Hypothetical 16×7us = 0.11% of floor. | obvious | yes | drop |
| 13 | FA2_tree_attn | Replace dense fp32 [9,9] bias with compact bitmask (SpecInfer BCM) | 324 B → 72 B, both single-cache-line, immaterial vs 422 MB/layer weight stream. Quadratic — only matters for trees >16 nodes. | Negligible at 9 nodes; pure cleanup. | moderate | yes | drop |
| 14 | both | TMA async bulk copy / cp.async.bulk for q/k/v/state staging | **Moves the SAME bytes** = zero HBM reduction. sm_121 single-block/persistent restrictions; no cluster multicast on GB10 (1x1x1). | Small/uncertain; #1 removes the bytes outright. | moderate | yes | drop |
| 15 | GDN_tree | Blackwell TMA for GDN state load/store | Same bytes (zero HBM cut); unsupported on sm_121 (dgx-spark-playbooks #22; ptxas refuses for sm_121a). | Zero on binding metric + blocked. #1 meets the goal algorithmically. | hard | yes (effort-risk only) | drop |
| 16 | both | Persistent kernel / megakernel to cut launch overhead | Same target as capture, no HBM cut. Partly blocked on sm_121 persistent/cluster path. Large rewrite = bit-exact reduction-order risk. | Marginal over capture, high cost. | hard | yes | drop |
| 17 | FA2 / both | FA3/FA4 warp-specialized / tcgen05 + TMEM / 5th-gen tensor cores | **Structurally unbuildable on GB10:** FA4 needs tcgen05+TMEM (SM100), FA3 needs WGMMA (SM90); sm_12x has neither (CUTLASS #2947, vLLM #22279). Compute/SMEM win anyway = hidden behind weights. | Zero (unbuildable AND off-metric). Documents FA2 fork as the correct ceiling on GB10. | hard | yes | drop |
| 18 | both | Thread-block clusters + DSMEM / multicast | GB10 limited to 1x1x1 cluster; `.shared::cluster` unsupported on sm_121a (#22). The HBM-cutting feature (multicast) is exactly what GB10 lacks. | Zero on GB10. | hard | yes | drop |
| 19 | both | FP8/FP6/FP4 microscaling (NVFP4/MXFP4) for in-kernel GEMMs | **Neither verify kernel streams model weights** — GDN args are all activations/state, FA2 reads q/k/v + KV. Nothing to requantize. | Zero for these kernels. Model-wide fp4 is out-of-scope, lossy, near banned "weight re-streaming". | hard | **NO** | drop (see §5) |

---

## 2. Do-now shortlist (cheap, safe, bandwidth-relevant) per kernel

**GDN_tree (where the real bandwidth win lives):**
- **[#1] WY one-pass + accept-only state commit.** The ONLY kernel-side change that touches the binding HBM metric. Removes the GDN-state amplification stream (9.0x→1.0x native r+w): +2.416 GB/req (8.95% of 27 GB) → +9.664 GB at B=4 (35.79%). Reproduced `/tmp/gdn_cost_verify.py`. Lossless: fp32 max_out 4.190951585769653e-09, max_state 1.043e-07, 0/9 argmax flips all 5 seeds; bf16 6.10e-5 < native bf16 self-noise floor 9.55e-5 (`scripts/fr13_lossless_fast_derivation_validate.py`, re-run, reproduces). Implement as #3 (fused) carrying #4's shared-prefix no-copy property.
- **[#2] Preallocate buffers.** `core_attn_out_spec` (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:1930`) and `tree_state_all` (`:1935`) are `torch.empty` INSIDE the forward (201.3 MB alloc/fwd, reproduced). `launch_tree_gdn_prepared` already accepts `out=`/`state=` (kernel docstring) — wire persistent buffers. Zero numeric change; unblocks capture.
- These bundle: #1+#3+#4 are one deliverable; #2 is its capture prerequisite.

**FA2_tree_attn (defensive only — no positive bandwidth win available here):**
- **[#5] fp8 descale guard.** Confirm the tree-bias decode call (`scripts/fr13_patch_fa2_tree_bias.py` ~line 570) stays correct: runtime KV is bf16 (`kv_cache_dtype` rewritten fp8_e5m2→auto for fp8 ckpts, MEMORY `project_l0c_p5b_fp8_kv_passthrough`), so the omitted descale is SAFE and adds no fp8→bf16 dequant HBM pass. If KV were fp8-coded and read without descale, attention diverges by max 35.9 (`/tmp/fa2_fp8_descale_and_suffix.py`) — re-arms as a hard correctness break only if vLLM ever enables true fp8 KV. No speed gain; keep as a guard/assert.

---

## 3. Blackwell-specific items worth a prototype (GPU is the arbiter)

These are the only items whose gain is real-but-unknowable on CPU. All are launch/capture levers (rule b/c), none cut HBM — so the upper bound is the launch-tax ceiling, not the 98.9 ms weight floor.

- **[#6] CUDA-graph FULL capture of the verify forward (both kernels).** GDN kernel is capturable in principle: tree-size/h0 reads are device-side `tl.load` (`fr10_gdn_tree_kernel.py:262-267`), N_PAD is a constexpr (static shape), batch loop uses static tree offsets (no GPU→CPU sync). Capture-breakers exist TODAY: per-forward `torch.empty` (fix via #2) and live diagnostic flags (fix via #9). Re-confirm capture state on cu130-nightly (MEMORY: failed on 0.19). Ceiling: at 384 launches × 7us = 2.69 ms = 2.72% of floor if currently eager; ~0 if already cgfull.
- **[#7] Fuse the B-batch launch loop into one grid (GDN).** 192→48 launches. Copy/HBM benefit hidden (0.267%); launch+capture benefit real but size is GPU-gated: 0.72 ms (eager) → 28.8 ms (200us pathological) per forward. Also yields a fixed grid = cleaner capture. Loop at `fr10_phase4_patch_vllm_tree_gdn.py:1946`, 7 `.contiguous()` slices at 2350-2356.
- **[#8] Confirm + lock FA2 tree-bias varlen full capture.** The splitkv capture-blocker does NOT exist on this path (swap OFF at `flash_api.cpp:632`, max_seqlen_q=9 → no runtime accum alloc, no data-dependent num_splits). vLLM PR #20059 (merged Aug 2025) makes the max_query_len>1 decode graph capturable. BOUNDED: attention's own launch slice is ~0.1% of forward; most of any win is non-attention launches. Measure whether the bundle is already `cgfull` (bundle names show `-cgfull-cgpacked` vs `-cgpiecewise-cgpacked`).

**Note on sm_121 reality:** every datacenter-Blackwell accelerant (tcgen05, TMEM, TMA tile control, DSMEM/multicast, thread-block clusters >1×1×1, WGMMA, FA3/FA4) is **structurally absent or restricted on GB10/sm_121** (CUTLASS #2947, NVIDIA dgx-spark-playbooks #22, vLLM #22279). They are all in §1 ranks 14–18 as drops. The FA2 fork is the correct attention ceiling on this hardware.

---

## 4. Explicitly irrelevant on a bandwidth-bound forward (with why)

| Item | Why irrelevant on this workload |
|---|---|
| Force num_splits=1 (FA2) | Premise FALSE: combine kernel never launches on the 9-query tree path (swap OFF). Nothing to remove. |
| Restrict apply_tree_bias to suffix block | Pure ALU; 324 B bias is L1-resident. FLOP hidden behind 98.9 ms weight stream. |
| Dense→BCM bitmask bias | 324 B → 72 B, both single cache line. Immaterial at 9 nodes; only matters for trees >16. |
| Bound replay outer loop to N_ACTUAL | Pure FLOP (136→45), unchanged state writes. Dominated by #1 (deletes replay). |
| TMA / cp.async.bulk staging | Moves the SAME bytes → zero HBM cut. Plus sm_121 restrictions, no GB10 multicast. |
| Blackwell TMA for GDN state | Same bytes; unsupported on sm_121. #1 removes the bytes algorithmically. |
| Persistent/megakernel | No HBM cut; capture gets most of the launch win with less risk + no sm_121 landmine. |
| FA3/FA4, tcgen05+TMEM, 5th-gen tensor cores, DSMEM/clusters/multicast | Unbuildable/absent on sm_121 AND compute/SMEM-only = hidden behind weights even where present. |

**General rule applied:** any item whose only effect is on FLOPs, ALU occupancy, shared-memory traffic, or tensor-core throughput is irrelevant here because the 27 GB weight stream (98.9 ms) dominates wall-time. Only HBM-byte reduction (rule a), full-capture (rule b), or wasted-launch removal (rule c) can move the number.

---

## 5. Items that RISK the lossless gate (flagged — NOT recommended)

- **FP8/FP6/FP4 microscaling (NVFP4/MXFP4) model-wide (rank 19).** `preserves_lossless: false`. The only fp-narrowing that would cut the binding 27 GB weight stream is requantizing model weights fp8→fp4, which is LOSSY (NVFP4 ~1% degradation, breaks the bit-exact lossless gate) and brushes the **BANNED "weight re-streaming"** rule. Neither verify kernel even has a weight operand (GDN args are all activations/state per `fr10_gdn_tree_kernel.py:206-241`; FA2 reads q/k/v + KV), so this is out-of-scope as a kernel change regardless. **Do not propose.**
- **fp8 KV descale omission (rank 5) — latent, not active.** Recommended as a do-now *guard*, but flagged here: it is lossless ONLY because runtime KV is currently bf16. If vLLM ever enables true fp8 KV for fp8 checkpoints, the omitted descale silently diverges attention by up to 35.9 (`/tmp/fa2_fp8_descale_and_suffix.py`). Ship with an assert that KV dtype is bf16 (or wire descales) before any fp8-KV config is enabled.

**Banned shortcuts confirmed avoided** by the recommended bundle (#1/#3/#4): per-branch state COPY, copy-recurrent multi-spine, reroute/splice, dense order-seq² materialization, weight re-streaming. #4 explicitly carries O(sum path_len)≤16 reflector columns, NOT O(branches) full states — verified by coresidency byte-invariance 0.0.

---

## Bottom line

One change moves the binding metric: **#1 WY one-pass + accept-only commit** (built as fused #3, enabled by prealloc #2, exploiting shared-prefix #4), removing the 8.9%→35.8% GDN-state amplification HBM stream losslessly (fp32 4.19e-9, 0 flips). Everything else is either a GPU-gated launch/capture prototype (#6/#7/#8, ceiling ~2.7% of the 98.9 ms floor), a cheap defensive guard (#5), or a documented drop. The FA2 fork is the correct, lossless attention ceiling on GB10 — its only real lever is full capture, and that win is bounded to ~0.1% on the attention slice itself.

**GPU measurement is the final arbiter for every `needs-gpu` item.** CPU evidence establishes losslessness and HBM-byte deltas (firm); it cannot establish the launch/capture wall-time fraction, which depends on the current capture state on cu130-nightly.
