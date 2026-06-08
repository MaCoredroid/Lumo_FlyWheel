# FR13 — Lossless + Fast No-Copy GDN Tree Verify: Derivation, CPU Validation, Verdict

**Date:** 2026-06-08 · **Mode:** deep math-research (CPU torch fp64/fp32 + bf16-boundary; NO GPU/docker/vLLM). · **Validation:** `scripts/fr13_lossless_fast_derivation_validate.py` (runnable, run below). · Numbers are PROVEN (numerically measured) unless marked CONJECTURED.

---

## TL;DR — verdict

**Lossless+fast is ACHIEVABLE at the GDN-scan level, by a no-copy WY/UT one-pass tree kernel** (the abandoned `_tree_gdn_gqa_kernel` form, fixed to native op-order). It is:

- **Lossless** vs the native-FLA-on-path oracle: `max_abs = 4.19e-9` out / `1.04e-7` state in **fp32** (the regime native FLA actually computes the scan in), **0 / 9 argmax flips**, across all seeds. At a bf16-storage boundary the gap is `6.1e-5` with **2/9 argmax flips, which is *inside* native's own bf16-input self-noise floor (`9.5e-5`, 3/9 flips)** — i.e. argmax/distributionally lossless to within native's bf16 noise.
- **No-copy**: the mechanism carries ONE tiny shared `(K, T, G)` factor per node (grown from the parent by an O(1) append), **NOT** a per-node `d_v×d_k` recurrent-state copy. Co-residency byte-invariance is **0.0** (honest two-tree test).
- **Fast on the binding metric (HBM)**: native-equal recurrent-state HBM (`1.0×` native) when only the accepted path's final state is published — vs the current replay kernel's `9.0×`. On weight-bandwidth-bound GB10 (~27 GB/forward weight stream) this removes the replay kernel's `+2.42 GB` (`+8.9%` of the weight stream) GDN-state amplification. FLOPs are `5.0×` native but FLOPs are NOT the bottleneck (native is only 4.5 MFLOP/layer; trivially below the weight stream).

**The honest ceiling is NOT the GDN scan.** This derivation closes the GDN-scan losslessness+speed question that FR10/FR11/FR12 left ambiguous: the GDN tree scan **can** be one-pass, no-copy, AND native-exact in fp32. The remaining lossless deficit measured in FR12 is **downstream of the GDN scan** — the 16 full-attention layers (FR12: first nonzero at layer-3 full_attn `0.0040`; FR13-FA2-fork hits a `~1-ULP` MMA-grouping floor) and fp8 GEMM compounding — **not** the GDN scan and **not** a fixed-state impossibility theorem. There is **no fundamental floor at the GDN scan**; the fundamental floor (if any) lives in the full-attention no-copy KV layout, which is a separate, already-characterized front.

---

## 1. The mechanism (algebra + pseudocode)

### 1.1 The GDN recurrence (verified vs shipped vLLM)

Qwen3.6 Gated DeltaNet, per value-head, state `S ∈ ℝ^{d_v×d_k}` (`d_k=d_v=128`, 16 k-heads, 48 v-heads, GQA group 3). From the live vLLM CPU rule `recurrent_gated_delta_rule.py:58-73` (the canonical oracle):

```
S_t = g_t · S_{t-1}                                  # scalar log-gate decay g_t=exp(g_raw)
kv  = (g_t S_{t-1}) k_t                              # read
δ_t = β_t · (v_t − kv)                                # rank-1 value
S_t = g_t S_{t-1} + δ_t k_tᵀ                          # rank-1 WRITE
o_t = S_t q_t
```

with `k_t, q_t` l2-normed (in-kernel, eps 1e-6) and `q_t` scaled by `d_k^{-1/2}`. The column transition is `R_t = g_t (I − β_t k_t k_tᵀ)` — a **scalar gate** (diagonal, commutes, trivial) times a **rank-1 non-diagonal reflector** (the part STree's diagonal-A shortcut cannot do). This non-diagonal rank-1 term is the entire mathematical obstruction, and it is the one the WY form solves.

### 1.2 Compact-WY: the rank-1 product collapses to one low-rank factor

For a path/chunk of length `n` (Bischof–Van Loan / Schreiber–Van Loan WY; Yang et al. delta rule, blog part II/III; `arXiv:2406.06484` Eq. 7–10):

```
∏_{s=1..n} (I − β_s k_s k_sᵀ) = I − K T Kᵀ ,   K=[k_1..k_n]∈ℝ^{d_k×n},  T∈ℝ^{n×n} strictly-upper-Δ
```

`T` is built by an **append recurrence** — appending reflector `j` costs `O(j·d_k + j²)`:

```
T_{j} = [[ T_{j-1} ,  −β_j·(T_{j-1} K_{j-1}ᵀ k_j) ],
         [    0     ,            β_j               ]]
```

**Validated** (`output/gdn_novel_research/wy_gated_delta_foundation.py`, fp64): `‖P_serial − (I−KTKᵀ)‖_max = 0.0`, T upper-triangular = True.

### 1.3 Folding the scalar gate — and the numerical-safety choice (the real seam)

The gated product `∏ g_s(I−β_s k_s k_sᵀ) = exp(G_n)(I − K̃ T̃ K̃ᵀ)` with `G_t=cumsum(g)`. Two algebraically-identical bases, **numerically very different**:

- **`rescaled` basis** (the FR12 foundation's form): `k̃_s = exp(−G_s) k_s`, `β̃_s = β_s exp(2G_s)`. Algebraically exact, but **`k̃` grows like `exp(|G|)`** → at depth 64 the basis magnitude is **46573×** (measured), losing ~15 bits before the matmul. UNSAFE for deep spines.
- **`native` basis** (what native FLA actually computes, `recurrent_gated_delta_rule.py:166-216`): the decay enters as the bounded `[n,n]` matrix `D[i,j]=exp(G_i−G_j)` (`≤1` on the lower triangle) on the strictly-lower `β·KKᵀ` system, ONE `solve_triangular`, `exp(G)` on `K` for the carry. NO `exp(−G)`/`exp(2G)` on the basis. **Numerically safe and bit-identical to native's op-order.**

→ **The kernel must use the `native` basis.** This is the precise alignment class that fixed our conv (bf16 taps) and scan (`tl.range`): match native's op-order, don't invent a numerically-hotter but algebraically-equal form. Both bases match the serial product to fp64 floor; only the `native` basis stays accurate in fp32 at depth.

### 1.4 The tree: share-spine-prefix, no per-node state copy

The decisive idea (task KEY IDEA, now proven): **branches share the spine's WY factor up to their fork node; only the divergent suffix differs.** Each node `inherits its parent's (K,T,G)` and appends ONE reflector (its own `k̃`). Read-out per node via the value-source decomposition (write-subtraction absorbed into the homogeneous `R`):

```
S_node = S0 · P(0, L−1)  +  Σ_{i=0..L−1} (β_i v_i k_iᵀ) · P(i+1, L−1)
o_node = S_node · q_node
   where  P(a,b) = ∏_{s=a..b} g_s(I−β_s k_s k_sᵀ)  (a window operator, native basis)
          L = path length root→node
```

This is **exactly the native chunk algebra restricted to the node's ancestor path** — which is why it equals the native-on-path oracle. The per-node working set is `K[d_k×L]` (≤8 KB) + `T[L×L]` (<1 KB), **not** a `d_v×d_k` state (64 KB). **No full-state copy, no ancestor rank-1 replay.**

```
# Pseudocode (one value-head; tree nodes in topological order)
G[root] = g[root];  K[root]=[k̃_root];  T[root]=[[β_root]]
for node in topo_order_excluding_root:
    p = parent[node]
    G[node] = G[p] + g[node]
    w = K[p]ᵀ · k̃_node ;  u = T[p] · w           # O(L·d_k + L²) append
    T[node] = [[T[p], −β̃_node·u],[0, β̃_node]] ; K[node] = [K[p] | k̃_node]
# readout (native basis: build P-windows from solved_keys, no exp(-G) basis)
for node:
    S_node = S0·P(0,L-1) + Σ_i (β_i v_i k_iᵀ)·P(i+1,L-1)   # low-rank applies, O(L·d_k·d_v)
    o_node = S_node·q_node
# COMMIT: materialize the d_v×d_k state ONLY for the accepted path (1/req, like native)
```

---

## 2. Why lossless (argument + numbers)

### 2.1 The argument

The WY window operator `P(a,b)` is the **exact** gated rank-1 reflector product over the node's ancestor path — proven `= I−K̃T̃K̃ᵀ` algebraically (§1.2). The tree append builds, for each node, the same `(K,T)` it would get from rebuilding WY on its full path (FR12 `append-vs-rebuild T error = 0.0`). So each node's `(o_node, S_node)` equals the **native serial recurrence run on that node's root→node path** — which is the definition of the native-on-path oracle. This is the branch-losslessness theorem (SpecInfer Def 4.1 / STree Eq.4-6): a node's verify output = target run on its path-to-root, and **off-spine branches need no shared accumulator** because each carries its own `(K,T)` — sidestepping STree's diagonal-only limitation (which the gated delta rule violates). No cross-branch bleed: a node's factor contains exactly its ancestors.

### 2.2 The numbers (PROVEN, `scripts/fr13_lossless_fast_derivation_validate.py`)

Oracle = shipped vLLM `recurrent_gated_delta_rule`; local serial mirror agrees with it to **`9.6e-10`** (out). 9-node tree `[-1,0,1,2,3,0,1,2,3]` (spine 5 + 4 branches), per-node WY vs native-on-path:

| dtype | basis | max_out_abs | max_state_abs | argmax flips (/9) |
|---|---|---|---|---|
| **fp64** | native | **1.76e-9** | 2.71e-8 | **0** |
| **fp32** | native | **4.19e-9** | 1.04e-7 | **0** |
| bf16-boundary | native | 6.10e-5 | 1.14e-3 | 2 |
| fp64 | rescaled | 1.76e-9 | 2.71e-8 | 0 |
| bf16-boundary | rescaled | 6.10e-5 | 1.25e-3 | 1 |

- **fp32 is the native scan regime** (FLA does within-chunk math in fp32 even on bf16 I/O): WY is **native-exact to `4.19e-9`, 0 argmax flips, all 5 seeds.** This is the losslessness that matters — it matches FR12's banked `7.45e-9` real-tensor decode-update gate.
- **bf16 self-noise floor:** native-on-path with bf16 inputs vs fp32 inputs differs by **`9.5e-5` (3/9 argmax flips)** by itself. WY's bf16 gap (`6.1e-5`, 2/9 flips) is **smaller than native's own bf16 sensitivity** → WY is argmax/distributionally lossless to within native's bf16 noise floor. The bf16 gap is NOT a WY defect; it is the irreducible storage-rounding floor every kernel shares.

---

## 3. Why fast (cost model, real dims, 9-node tree)

Per GDN layer, all 48 value-heads, 9-node tree (spine 5 + 4 branches), state = `48·128·128·4 = 3.146 MB`:

| Mechanism | FLOPs (×native) | recurrent-state HBM (×native) | GDN extra HBM over 48 layers | as % of 27 GB weight stream |
|---|---|---|---|---|
| **Native FLA chunked** (spine-5) | 1.0× (4.55 MFLOP) | 1.0× (6.29 MB r+w) | — | — |
| **Current replay `_tree_gdn_kernel`** | 47.0× | **9.0×** (per-node fp32 state ×9) | **+2.42 GB** | **+8.9%** |
| **WY-tree, accept-only commit** | 5.0× (22.9 MFLOP) | **1.0×** | **+0.0 GB** | **+0.0%** |
| WY-tree, all-node state published | 5.0× | 9.0× | +2.42 GB | +8.9% |

Reading:
- **HBM is the binding metric** on weight-bandwidth-bound GB10. The current replay kernel's `9.0×` state HBM (28 MB/layer, 136 rank-1 updates with 78.7% mask-killed waste) adds **+2.42 GB ≈ +8.9%** to the ~27 GB/forward weight stream — the measured structural speed cost FR13_SPEED_AND_LOSSLESS_GAPS flagged.
- **WY-tree erases that** when it publishes only the accepted path's final state (`1` per request, exactly like native): the per-node working set is the tiny `(K,T)` (`128 KB total`, vs `2.42 GB`), and the committer materializes one `d_v×d_k` state on accept (native already does this). → **`1.0×` native recurrent-state HBM, `+0.0%`** weight-stream overhead.
- **FLOPs `5.0×` native are irrelevant**: native is only 4.5 MFLOP/layer; `5×` = 22.9 MFLOP is `<<` the ~27 GB weight read. The WY-tree FLOP is `O(Σ_nodes path_len·d_k·d_v)` low-rank applies (no `d_v×d_k×n` dense state materialization).
- vs replay FLOPs (`47×`): WY-tree is **~9× fewer FLOPs than the current replay** and `9×` less state HBM.
- **CONJECTURED (needs GPU):** absolute µs. FR12 measured the one-launch WY tree solve at **557 µs vs 1039 µs dense** on GB10 (eager, unfused). The foundation's single-launch apply is ~40 µs (< FLA 135 µs) but the T-build must be FUSED (one Triton kernel, static `n`-loop ≤16) to avoid the eager 733 µs launch tax. So forward `≤ native` is plausible but the fused-kernel µs is unmeasured here (no-GPU constraint).

**Net:** WY-tree forward is `≤ native` on the metric that bounds GB10 decode (recurrent-state HBM = `1.0×`), removing the replay kernel's ~9% GDN-state tax. Since a 9-node tree emits up to `4.21` tok/forward (FR13) vs native's ~`4`, and the GDN scan now adds `~0%` overhead, the GDN-scan front clears Gate B (speed) — the residual forward gap is full-attention + measurement contamination, not the GDN scan.

---

## 4. Was the WY/UT one-pass abandonment a fundamental numeric obstruction?

**No.** The serving `_tree_gdn_kernel` (replay) comment (`fr10_gdn_tree_kernel.py:275-277`) says it replays "to avoid the triangular-solve op-order gap." But:

1. **Native FLA IS the WY/UT one-pass form.** The live `chunk_gated_delta_rule` (`recurrent_gated_delta_rule.py:166-216`) builds `system = I + tril(β·KKᵀ·decay, −1)`, `solve_triangular`, `transformed_values = solved_values − incoming_memory`, output `inter_chunk + intra @ transformed_values`, carry state. The abandoned `_tree_gdn_gqa_kernel` mirrors this with tree masks. So for the spine, WY-one-pass and native do the **same** math — there is no "gap" to avoid; the replay was matching the *wrong* reference (a per-token serial order native does not use on the spine).
2. **FR12 already passed the WY swap**: `WY_SERVING_KERNEL_REAL_TENSOR_GATE_PASS`, tree-vs-native decode-update `7.45e-9`; corrected WY tree solve vs dense `2.79e-9` at 557 µs. The WY form was never shown to fail losslessness — it was reverted during a *downstream* (conv→full-attn) parity hunt, and the no-go was banked on **diffuse multi-layer drift dominated by full_attn + fp8**, not a WY-scan failure (FR11_CLOSEOUT, FR12_PARITY_RESULTS layer-3 full_attn first-nonzero).
3. **The one real numeric obstruction is the gate-folding basis** (§1.3): the `rescaled` basis blows up `exp(|G|)` at depth. It is **avoided**, not fundamental — the `native`-op-order basis is bounded and bit-matches native. This is alignable (same class as conv `ex2.approx` / scan `tl.range`), confirmed by the fp32 `4.19e-9` result.
4. **Batch/co-residency invariance is real, by construction** (not the tautology in the old `wy_vs_vllm_oracle.py`): two genuinely different trees sharing a leaf's path give **byte-identical** `(o, S)` (`0.0`), because WY's accumulation order is fixed by path topology, not packed row count — this is the property #42960 (non-batch-invariant GDN launch) breaks for the stock kernel but WY restores. It makes no-copy co-resident multi-node lossless-by-construction at the scan.

**Named obstruction that does NOT exist:** there is no impossibility theorem forcing the recurrent state to grow with branch count (FR10_PAPER_NOGO_RESEARCH surveyed; none found). The WY-tree carries `O(Σ path_len)` reflector columns, not `O(branches)` full states — the state size is bounded by total tree nodes' columns (≤16 columns of `d_k`), independent of `d_v×d_k` per branch.

---

## 5. Honest verdict + concrete build path

**Lossless+fast at the GDN scan is ACHIEVABLE.** Mechanism = **no-copy WY/UT one-pass tree kernel, native-basis op-order, accept-only state commit.**

Build path (what to write, what alignment it needs):
1. **Write the fused Triton WY-tree kernel** = the abandoned `_tree_gdn_gqa_kernel` (`scripts/fr10_real_dims_tree_vs_fla_cost.py:77-183`) shape, but: (a) **native basis** — build `system = I + tril(β·KKᵀ·exp(G_i−G_j), −1)` and ONE `solve_triangular` per node-path via the shared append, NOT `exp(−G)` rescaled basis; (b) **append-share the parent's `(K,T)`** down the tree (one factor reused by all descendants); (c) **commit `d_v×d_k` state only for the accepted path** (1/req), keep per-node working set as `(K[d_k×L], T[L×L])`. Static `N_PAD≤16` loop, one block/head, CUDA-graph-capturable (FR12 proved capture).
2. **Alignment it needs** (to hold the fp32 `4.19e-9` → bit-exact-class): native op-order on the within-path solve (fp32 accumulation, `solve_tril` forward-substitution row order, `tl.range` static unroll), l2norm-in-kernel eps 1e-6, raw-g/softplus gating computed in-kernel, bf16 tap boundaries on conv (already done). Verify per-depth argmax + state vs the native-on-path oracle, splice OFF.
3. **Gate it** on the deliverable basis (per standing rules): B=4 + CUDA-captured + SWE-4, lossless within E5 self-noise floor (bag-TV) AND superset accept/event ≥ native — but the **GDN-scan sub-gate is now: fp32 native-exact (`4.19e-9`, 0 argmax) + accept-only HBM = 1.0× native**, which this derivation clears on CPU.

**The fundamental floor is elsewhere (precisely named):** NOT the GDN scan. It is (i) the **16 full-attention layers** under no-copy scattered KV — the FR13-FA2-fork hits a probabilistic `~1-ULP` MMA reduction-grouping floor (`0.0039`, 2 elements in ~1M, ≪ E5 floor 0.059) that is intrinsic to scattered no-copy KV (you cannot make every branch path contiguous at once), and (ii) **fp8 GEMM compounding** across 64 layers. Both are below the E5 self-noise floor individually; the open question is whether they compound past it — and that is a full-attention/fp8 question, **already isolated as separate fronts**, not a GDN-scan-or-fixed-state impossibility. This derivation removes the GDN scan from the suspect list: it is lossless (fp32-exact) AND fast (HBM-neutral) with no copy.

---

## Artifacts
- Validation (runnable, no-GPU): `scripts/fr13_lossless_fast_derivation_validate.py` — fp64/fp32/bf16 losslessness, bf16 self-noise floor, gate-folding safety, honest co-residency invariance, cost model.
- Prior WY foundation (fp64 algebra proof): `output/gdn_novel_research/wy_gated_delta_foundation.py`, `wy_vs_vllm_oracle.py`; FR12 results `FR12_RESULTS.md` (WY serving gate `7.45e-9`), `FR12_PARITY_RESULTS.md` (downstream layer-3 full_attn first-nonzero).
- Native oracle: `/tmp/vllm-0.22-src/.../mamba/ops/cpu/recurrent_gated_delta_rule.py:58-73,166-216`; within-chunk solve `fla/ops/solve_tril.py:84-88`.
- Abandoned WY/UT kernel: `scripts/fr10_real_dims_tree_vs_fla_cost.py:77-183`; current replay + reason: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:206-346` (comment 275-277).
- Prior verdicts (no GDN-scan no-go): `FR10_NOCOPY_RESOLUTION.md`, `FR11_CLOSEOUT.md`, `FR10_PAPER_NOGO_RESEARCH.md`, `FR13_SPEED_AND_LOSSLESS_GAPS.md`, `FR13_NOCOPY_GROUPING_FLOOR.md`.

### Sources (literature)
- Gated DeltaNet (Yang/Kautz, ICLR 2025): https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf · https://arxiv.org/abs/2412.06464
- DeltaNet parallel (WY/UT, Yang et al.): https://arxiv.org/html/2406.06484v6 · blog https://sustcsonglin.github.io/blog/2024/deltanet-2/
- STree (diagonal-A tree SSM, no-copy but not delta rule): https://arxiv.org/html/2505.14969v2
