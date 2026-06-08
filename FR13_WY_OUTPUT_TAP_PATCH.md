# FR13 WY GDN #6 OUTPUT-TAP — ready-to-apply package (workflow w14ozz0qh, 2026-06-08)

CPU-PROVEN (tap closes spine 1.22e-4 -> 0.0; 48-layer coherent accumulation 5.86e-3 made MOOT for spine). GPU-RE-LADDER PENDING (codex applies after the live ladder frees the GPU). Flag FLA_BF16_OUTPUT_SPLIT default OFF -> fp32-oracle path byte-identical. NO self-declare; GPU re-ladder confirms.

---

The source matches the patch proposal's structure exactly. Now I have everything I need to synthesize the deliverable. The line numbers in the proposal are close but I'll use the actual verified line numbers from my reads (433, 615-617, 673-677, 683, 928, 712/744, 776).

Let me verify one detail: the patch author's Edit 3 references `OUTPUT_SCALE` already folded into `q_i` at L614, and `decay_ij`/`trans_j`/`k_j` are at L624/L621/L622 — all confirmed in scope inside the j-loop. The proposal's `out_intra_i` accumulation approach (CPU-validation script) uses `tl.sum(state_update_ij * q_i)` while the patch-author edit uses `tl.sum(q_i * k_j) * decay_ij * trans_j`. These are algebraically identical per the proposal's own derivation. I'll use the patch-author's exact edits as the canonical block since they were "validated for structure and indentation," and note the equivalence.

Here is the synthesized deliverable.

---

# FR13 WY GDN #6 OUTPUT-TAP — READY-TO-APPLY PACKAGE

**Status: CPU-PROVEN, GPU-RE-LADDER PENDING. Do NOT self-declare PASS — the live ladder confirms.**

Target file (verified against live source): `/home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`
Target kernel: `_tree_gdn_wy_kernel` (the WY path). The non-WY `_tree_gdn_kernel` has the identical fused readout `out_i = tl.sum(state_i * q_i[None, :], axis=1)` but is NOT touched here (WY is the deliverable; an identical edit can be ported if both paths are later needed).
New flag: `FLA_BF16_OUTPUT_SPLIT` constexpr / `fla_bf16_output_split` host bool — **default OFF**, fp32-oracle path stays byte-identical.

All line numbers below were re-verified against the live file (not the proposal's approximations). All old_string blocks are copy-paste exact.

---

## 1. THE EXACT PATCH (6 edits, directly appliable)

### Edit 1 — add the constexpr to the WY kernel signature (around L433)

old_string:
```
    FLA_BF16_BOUNDARIES: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    head_group = NUM_VH // NUM_KH
```
new_string:
```
    FLA_BF16_BOUNDARIES: tl.constexpr,
    FLA_BF16_OUTPUT_SPLIT: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    head_group = NUM_VH // NUM_KH
```
The anchor line `head_group = NUM_VH // NUM_KH` (L439, WY-only) disambiguates from the non-WY `_tree_gdn_kernel`, which also has a `FLA_BF16_BOUNDARIES,` constexpr line.

### Edit 2 — capture inter-only state + intra accumulator at the readout loop head (L615-617)

old_string:
```
        state_i = b_h0 * tl.exp(cumg_i)
        state_store_i = tl.zeros((BLOCK_V, DIM_K), dtype=tl.float32)
        state_store_i += b_h0
```
new_string:
```
        state_i = b_h0 * tl.exp(cumg_i)
        state_inter_i = state_i
        out_intra_i = tl.zeros((BLOCK_V,), dtype=tl.float32)
        state_store_i = tl.zeros((BLOCK_V, DIM_K), dtype=tl.float32)
        state_store_i += b_h0
```
`state_inter_i` snapshots the carried-in (inter) state at L615 BEFORE the j-loop folds intra updates into `state_i` in-place at L673. `out_intra_i` accumulates the separately-bf16-rounded intra term. This addresses the patch author's own refinement: `state_i` at the L683 readout is the FULL state (inter+intra), so the inter readout must use the L615 snapshot, not `state_i`.

### Edit 3 — build + bf16-round the per-node intra score inside the j-loop (L673-677)

old_string:
```
            state_i += tl.where(
                vis & (i < N_ACTUAL) & (j < N_ACTUAL),
                state_update_ij,
                0.0,
            )
```
new_string:
```
            if FLA_BF16_OUTPUT_SPLIT:
                a_ij = tl.sum(q_i * k_j) * decay_ij
                a_ij = tl.where(
                    vis & (i < N_ACTUAL) & (j < N_ACTUAL), a_ij, 0.0
                )
                a_ij = a_ij.to(tl.bfloat16).to(tl.float32)
                out_intra_i += a_ij * trans_j
            state_i += tl.where(
                vis & (i < N_ACTUAL) & (j < N_ACTUAL),
                state_update_ij,
                0.0,
            )
```
In-scope inside the j-loop (all verified): `q_i` = `b_q[i] * OUTPUT_SCALE` (L614), `k_j` (L622), `decay_ij = exp(cumg_i - cumg_j)` (L624), `trans_j` (L621). The mask is applied to `a_ij` BEFORE the bf16 cast (mirrors native `chunk_o.py` L125 mask → L137 cast). The `state_i +=` and `state_store_i =` updates below are untouched, so the stored recurrent state (L689) and the OFF path are bit-identical to current.

### Edit 4 — split the readout (L683)

old_string:
```
        out_i = tl.sum(state_i * q_i[None, :], axis=1)
```
new_string:
```
        if FLA_BF16_OUTPUT_SPLIT:
            out_inter_i = tl.sum(state_inter_i * q_i[None, :], axis=1)
            out_i = out_inter_i + out_intra_i
        else:
            out_i = tl.sum(state_i * q_i[None, :], axis=1)
```

### Edit 5 — thread the flag at the WY launch site (L928)

old_string:
```
            FLA_BF16_BOUNDARIES=bool(fla_bf16_boundaries),
            RAW_GATING=raw_gating,
            COUNT_INVOCATION=count_invocation,
        )
        return out, state
```
new_string:
```
            FLA_BF16_BOUNDARIES=bool(fla_bf16_boundaries),
            FLA_BF16_OUTPUT_SPLIT=bool(fla_bf16_output_split),
            RAW_GATING=raw_gating,
            COUNT_INVOCATION=count_invocation,
        )
        return out, state
```
(`return out, state` here is the WY-branch return at L932, disambiguating from the non-WY launch below it.)

### Edit 6a — host param on `launch_tree_gdn_prepared` (L776)

old_string:
```
    invocation_counter: torch.Tensor | None = None,
    fla_bf16_boundaries: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch with precomputed graph-safe tree descriptors."""
```
new_string:
```
    invocation_counter: torch.Tensor | None = None,
    fla_bf16_boundaries: bool = False,
    fla_bf16_output_split: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch with precomputed graph-safe tree descriptors."""
```
(The trailing docstring anchor disambiguates this signature's `fla_bf16_boundaries,` block from the one in `launch_tree_gdn`.)

### Edit 6b — host param on `launch_tree_gdn` (L712)

old_string:
```
    invocation_counter: torch.Tensor | None = None,
    fla_bf16_boundaries: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the FR10 dense tree verifier.
```
new_string:
```
    invocation_counter: torch.Tensor | None = None,
    fla_bf16_boundaries: bool = False,
    fla_bf16_output_split: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the FR10 dense tree verifier.
```

### Edit 6c — forward the flag in the `launch_tree_gdn` → `launch_tree_gdn_prepared` call (L744)

old_string:
```
        invocation_counter=invocation_counter,
        fla_bf16_boundaries=fla_bf16_boundaries,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=A_log,
        dt_bias=dt_bias,
        use_wy=use_wy,
    )
```
new_string:
```
        invocation_counter=invocation_counter,
        fla_bf16_boundaries=fla_bf16_boundaries,
        fla_bf16_output_split=fla_bf16_output_split,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=A_log,
        dt_bias=dt_bias,
        use_wy=use_wy,
    )
```

**Flag flow:** constexpr declared at kernel (Edit 1) → consumed at Edit 3/Edit 4 → host bool at both wrapper signatures (Edit 6a/6b) → forwarded `launch_tree_gdn`→`prepared` (Edit 6c) → passed to kernel at the WY launch (Edit 5). Default `False` at every site → OFF path unchanged.

---

### chunk_o.py line-correspondence (native FLA `chunk_scaled_dot_kkt_fwd` / `chunk_o`)

| Native `chunk_o.py` | Patched WY equivalent | Rounding / scale |
|---|---|---|
| L90 `b_o = zeros(fp32)`; L111 `b_o += dot(b_q, trans(b_h))` | `state_inter_i = b_h0*exp(cumg_i)` (Edit 2) → `out_inter_i = sum(state_inter_i * q_i)` (Edit 4) | inter, fp32, carried-in state |
| L119 `b_o = b_o*exp(b_g)` cumulative decay | `exp(cumg_i)` already folded into `state_inter_i` | fp32 |
| L91 `b_A = zeros(fp32)`; L113 `b_A += dot(b_q,b_k)` | `a_ij = tl.sum(q_i * k_j)` (Edit 3) | intra score, fp32 |
| L120 `b_A *= exp(b_g[:,None]-b_g[None,:])` pairwise decay | `* decay_ij` (`decay_ij = exp(cumg_i-cumg_j)`, L624) | fp32 |
| L122-125 causal/range mask `b_A = where(m_A, b_A, 0)` | `a_ij = where(vis & i<N_ACTUAL & j<N_ACTUAL, a_ij, 0.0)` (Edit 3) — mask BEFORE cast | — |
| **L137 `b_A.to(b_v.dtype)`** (bf16 cast of scores) | **`a_ij = a_ij.to(tl.bfloat16).to(tl.float32)`** (Edit 3) | **the single new rounding — 1 bf16 ULP** |
| L137 `tl.dot(b_A.to(bf16), b_v)` | `out_intra_i += a_ij * trans_j` (Edit 3), `trans_j` = WY-transformed value (L621) | accumulated fp32 |
| L137 `b_o*scale + (...)*scale` per-term scale | `OUTPUT_SCALE` folded once into `q_i` (L614), distributes over `out_inter_i + out_intra_i` (Edit 4) | one multiply, exact-distributive over the sum |

**WY-vs-dense algebraic identity (why the tap is not a reroute):** native's `b_A` is a dense `[BT,BT]` chunk score; our intra term is the WY/UT-reformulated `state_update_ij ⊗ q`. They are algebraically identical:
`sum_k state_update_ij[v,k]*q_i[k] = trans_j[v]*decay_ij*sum_k(k_j[k]*q_i[k]) = a_ij * trans_j[v]`, where `a_ij = (q_i·k_j)*decay_ij`.
The tap reconstructs the per-node score `a_ij` explicitly inside the j-loop and casts each scalar to bf16 (native's `b_A.to(b_v.dtype)`), then accumulates `out_intra_i = sum_j bf16(a_ij)*trans_j` — exactly native's `dot(bf16(b_A), b_v)` expressed per-ancestry-edge. Two tree-forced subtleties: (1) per-node scalar cast vs native's whole-tile cast — same rounding of each score element, spine (single ancestry chain) reproduces native's lower-triangular `b_A` row exactly; (2) mask-then-cast ordering — off-ancestry edges contribute exactly 0 and add no rounding. This builds native's arithmetic inside our kernel; no call/splice into native code.

---

## 2. CPU VALIDATION EVIDENCE

CPU/torch fp32 re-derivations from captured L1 scan inputs (`fr10_tree_gdn_scan_l1.pt`), mirroring kernel arithmetic exactly. Read-only, no GPU/docker/Triton touched. Scratch: `/tmp/fr13_cpu_tap_validate.py`, `/tmp/fr13_output_split_analysis.md`, `/tmp/fr13_patch_proposal.py`.

Per-row readout max_abs over all 48 value-heads × 128 dims, vs reference (1) = native two-term split `b_o*scale + (b_A.to(bf16) @ v_new)*scale` (mirror of `chunk_o.py:137`):

| | SPINE rows [0,1,2,4,6,8] | BRANCH rows [3,5,7,9] |
|---|---|---|
| **(2) WY one-pass** (fused `out_i=sum(state_i*q_i)`, current kernel:683) vs (1) | **1.22e-4** (block max) | **6.10e-5** (block max) |
| **(3) WY + #6 tap** (inter fp32 / intra bf16(A) split, per-term scale) vs (1) | **0.0** | **0.0** (CPU serial; see floor caveat) |

Per-spine-node (2)-vs-native: node0 1.53e-5, node1 1.22e-4, node2 6.10e-5, node4 3.05e-5, node6 6.10e-5, node8 3.05e-5 — matches live `wy_l1_spine_scan_live_fla_bf16.json` `by_depth` out_max_abs to the ULP. The 1.22e-4 worst case = exactly 1 bf16 ULP at GDN attn-output magnitude (~0.026-0.031, exponent −6 → ULP = 2^(−6−7) = 1.221e-4); the 2.44e-4 seen in the 6-tap probe = 2 ULP at ~0.037-magnitude rows.

**Mechanism isolation (control):** rebuilding the tap with the intra A matrix kept fp32 (no cast) re-introduces the 1.22e-4 gap; with `b_A.to(bf16)` it goes to 0.0. The entire gap is the bf16 rounding of the intra (A·v_new) term — (2) fuses and rounds once in fp32; native rounds the intra matrix to bf16 before the dot. That single cast + the inter/intra split is the whole tap.

**Branch floor caveat (honest):** the CPU serial reference has no MMA tiling, so (3)-vs-(1) is a clean 0.0 on branches in CPU. The live branch single-ULP floor is a GPU MMA-grouping artifact (`codex_fr17_bf16_bank_batch_best.json`: `state_bf16_bank mismatch_count` 8-58, `pre_round_abs ~1.3e-12`) a CPU reference cannot exhibit. On hardware the tap collapses the spine to 0.0 and leaves only the irreducible sparse single-ULP grouping floor on branch rows — the same floor FR13 already accepted for the FA2-no-copy fork.

---

## 3. ACCUMULATION VERDICT (48-layer)

Qwen3-Next = 64 layers = 48 GDN linear_attn + 16 full_attention (model config). Per-GDN-layer output residual without the tap: ε = 1.22e-4 (1 bf16 ULP).

- COHERENT worst case (same sign every layer, linear add): 48·ε = **5.86e-3** (≈48 ULP).
- INCOHERENT (independent signs, random walk): √48·ε = **8.46e-4** (≈6.9 ULP).
- Spread = √48 ≈ 6.9×.

Cross-layer sign coherence is a live-only unknown (un-measurable from a single captured layer), and the coherent bound is ~7× the incoherent — the open risk before logit argmax.

**The tap makes this MOOT for the spine:** it drives the per-GDN-layer spine output residual to 0.0, so both the coherent and incoherent sums are 0 on the accepted/committed (spine) path regardless of sign correlation. Branch rows retain only the sparse single-ULP MMA floor, which is per-element grouping noise — not a systematic bf16-truncation bias — so it does not accumulate coherently.

---

## 4. APPLY + VALIDATE PLAN FOR CODEX (after the live ladder frees the GPU)

1. **Apply** Edits 1-6c above (6 edits, all old_string blocks copy-paste exact, verified against live source). `FLA_BF16_OUTPUT_SPLIT` / `fla_bf16_output_split` default OFF at every site.
2. **Smoke OFF-path:** with the flag default OFF, re-run the existing WY parity probe and confirm the OFF path is byte-identical to current (the fp32-oracle path must be unchanged). This is the safety check that Edits 2-4 did not perturb the default branch.
3. **Turn the flag ON** (set `fla_bf16_output_split=True` at the launch site used by the live ladder / scan harness) and re-run the SAME live per-layer ladder (`fr10_tree_gdn_scan_l1.pt` / the wy_l1 spine+branch scan that produced `wy_l1_spine_scan_live_fla_bf16.json`).
4. **Confirm on hardware:**
   - GDN output max_abs → **0.0 on spine rows** [0,1,2,4,6,8] vs native chunk_o two-term reference.
   - Branch rows → only the **sparse single-ULP MMA-grouping floor** (the accepted FA2-no-copy class), not a systematic gap.
   - Verify SPINE **and** BRANCH oracle (branch = native-on-branch-path oracle per the GDN-tree branch-losslessness theorem; spine ≡ E5).
5. **Propagate:** continue the input→layer0→…→logits ladder; confirm final-logit drift stays within the self-noise floor and per-depth argmax matches.
6. **Keep default OFF** so the fp32-oracle path stays available; **Gate-2 (regular decode == pristine) untouched** — the tap only touches the tree readout, default-disabled.
7. **Final gate regime (do not skip):** the lossless+superset verdict is **B=4 + CUDA-graph-captured + SWE-Verified 4 tasks** vs **E5** (FLASH_ATTN native MTP-5, `output/fr10_native_mtp5_same8_*`). Re-confirm bit-exact 0.0 there (B=4 changes co-residency) and confirm the kernel still graph-captures with the flag ON (hooks OFF) before any close/pass-fail.

---

## 5. HONESTY: CPU-PROVEN vs GPU-PENDING; reward-hack check

**CPU-PROVEN (this package):**
- The fused-readout gap is exactly 1 bf16 ULP (1.22e-4) and its sole cause is the missing bf16 cast of the intra A·v_new term (isolated by the fp32-vs-bf16 control: fp32 → gap returns, bf16 → 0.0).
- The two-term split with per-node bf16(a_ij) reproduces native chunk_o's `b_o*scale + b_A.to(bf16)@v_new*scale` arithmetic exactly, driving the spine readout to 0.0 against a serial fp32 reference.
- The WY-vs-dense algebraic identity is exact (`sum_k state_update_ij*q_i = a_ij*trans_j`), so the tap is native's arithmetic re-expressed, not an approximation.
- The patch is structurally valid against the live source (all 6 old_string blocks verified to match), the `state_inter_i` snapshot correctly captures inter-only state before the in-place intra fold, and OFF-path is byte-identical (Edits 2-4 gate every new op behind `if FLA_BF16_OUTPUT_SPLIT`).

**NOT YET PROVEN — needs the GPU re-ladder:**
- That spine → 0.0 holds on real GPU MMA tiling (the CPU serial reference cannot exhibit, nor rule out, MMA-grouping effects). The live branch single-ULP floor is from prior artifacts, not from this study.
- That branch rows show only the accepted sparse single-ULP floor and nothing larger once the tap is live.
- Final-logit drift within floor + per-depth argmax match after 48-layer propagation.
- Graph-capture + B=4 co-residency behavior with the flag ON (the final gate regime).
- I am NOT declaring PASS. The CPU study is necessary evidence; the live re-ladder is the verdict.

**Banned-reward-hack check: NONE.** The patch adds a bf16 cast and an inter/intra split inside OUR `_tree_gdn_wy_kernel`. It does not call, splice, or route through native `causal_conv1d_update` / FLA / `chunk_o`; it reconstructs native's per-term arithmetic in our own kernel (the explicit goal: "build native's arithmetic, not a reroute"). No copy/dense/oracle path is introduced into the served kernel; the fp32-oracle path is preserved untouched behind the default-OFF flag.

Relevant paths:
- Patch target: `/home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`
- Monitor commits this text to: `/home/mark/shared/lumoFlyWheel/FR13_WY_OUTPUT_TAP_PATCH.md`
- CPU scratch: `/tmp/fr13_cpu_tap_validate.py`, `/tmp/fr13_output_split_analysis.md`, `/tmp/fr13_patch_proposal.py`
