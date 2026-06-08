# FR13 WY scan STATE-write fix — localization + offline test order (for codex_fr17)

Static localization (read-only, no GPU). Grounded in both kernels at file:line.
**Headline: the fix is already located AND committed (`8a975837`). Offline evidence
confirms it drives state 1.66e-3 → fp32 floor (~6e-8) on spine + branches. The remaining
work is a single live-ladder confirmation, in the order below.** This doc also corrects
a wrong conclusion that was circulating in the prior FINDINGS.

---

## TL;DR

- **Root op (single, high-confidence):** the WY state-store was being assembled from the
  **bf16-tapped OUTPUT-readout basis** (`y_i`/`sk_i`/`tv_i`, bf16-rounded under
  `FLA_BF16_BOUNDARIES`) instead of a **raw fp32 recurrent basis**. The readout taps are
  correct for the OUTPUT (which is already at the 1-bf16-ULP floor), but carrying them into
  the recurrent **state** over-rounds the carry → 1.66e-3, which compounds over 64 layers
  to final-logit 3.32.
- **Exact in-kernel fix (already applied, commit `8a975837`):** split the readout surface
  from the stored-state surface. Keep the bf16-tapped WY path for `out`; add a **separate
  raw-fp32 `_state` track** (`b_k_state`, `kk_state` via `input_precision="ieee"`,
  `system_state`, `solved_state_v/k`, `trans_state_v`, `tv_state_i`, `state_store_i`) and
  store **that** into the state buffer. File:line below.
- **Result (offline, fixed payload, same `fused_sigmoid_gating` reference):**
  state max_abs 0.001657 → **5.96e-8 / 8.94e-8** (spine `original_spine_vs_native_fla`,
  full-replay `original_full_replay_vs_payload`, and branch `reverse_sibling_dfs_full`);
  `out` unchanged at 1.2e-4 bf16-readout floor.
  Artifacts: `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_patch2_spine_state.json`
  and `..._patch2_batch_state.json`.
- **The fp32-oracle (`FLA_BF16_BOUNDARIES` off, ~9.3e-10) is untouched** — the split only
  activates under the flag; off-flag still computes the single fp32 closed form.
- **No copy / no splice / no dense.** The WY kernel still computes the state; the `_state`
  track is a second WY/UT solve in raw fp32, not a reroute through native.

---

## What the divergence actually was (file:line, BOTH kernels)

### OUR kernel — `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`, `_tree_gdn_wy_kernel` (def at :401)

Before `8a975837`, the state buffer at the store (`:649-653`) received `state_i`, and
`state_i` (`:615-642`) was built from `trans_v` (`:600-604`) and `b_k` (`:625`). Those
trans/solve values come from `y_i`/`sk_i`/`tv_i`, which ARE bf16-rounded under the flag:
- `:561-563`  `y_i`, `sk_i` → `.to(bf16).to(f32)`  (tap #4)
- `:584-586`  `y_store_i`, `sk_store_i` → bf16  (tap #4 solved store)
- `:602-603`  `tv_i` → bf16  (tap #5)
- `:527-528`  the WY system solve `kk`/`b_kb` runs in **bf16** (`.to(tl.bfloat16)`, tap)

Those taps are CORRECT for the OUTPUT — they mirror the live chunked FLA readout
(`wy_fast.py:92,94,114,116`, `chunk_delta_h.py:235`). The bug was using the **same
bf16-tapped basis for the recurrent STATE**, which the native state path does NOT match in
that direction at the fp32-reference level (see below).

### Native FLA reference the probe compares against

The probe's `native_reference` is **`vllm.fused_sigmoid_gating_delta_rule_update`** (the
per-token fp32 decode recurrence), `/tmp/vllm_live_019/.../fla/ops/fused_sigmoid_gating.py`:
- `:134` `b_h += tl.load(p_h0).to(tl.float32)` — h0 read fp32
- `:158-165` the recurrence is **entirely fp32**: `b_h *= exp(b_g)`; `b_v -= sum(b_h*b_k)`;
  `b_v *= b_beta`; `b_h += b_v[:,None]*b_k[None,:]` — **no bf16 operand rounds on the state
  carry**
- `:184` final state store is fp32 (`inplace_final_state=False` buffer); the bf16
  quantization of the carry happens once later at the shared ssm-cache write
  (`fr10_phase4_patch_vllm_tree_gdn.py:2686` == `gdn_linear_attn.py:1160-1161`), which BOTH
  kernels go through identically — so it is not the WY-vs-native seam.

Net: against this fp32 state reference, the WY state-store must be **raw fp32**, not
bf16-tapped. Carrying the readout taps into the state is an **over-round** → 1.66e-3.

> **Correction to prior FINDINGS.** The earlier finding #3 concluded "this is a comparator
> artifact — re-measure against the chunked FLA `ht` and the taps collapse; do NOT change
> the kernel." That is **empirically refuted**: `codex_fr17_patch2_spine_state.json` drives
> the state to 6e-8 **against the same `fused_sigmoid_gating` reference** by changing the
> kernel (the readout/state split). The 1.66e-3 was a real kernel over-round in the state
> path, not a reference mismatch. The taps themselves stay (they are right for `out`); the
> fix is to stop feeding them into the state. The earlier finding #5/#1 ("serialize / use a
> raw fp32 state basis matching the native fp32 recurrence", likelihood high) was the
> correct call.

---

## The fix as committed (`8a975837` "FR13 align WY state store with native FLA")

Split the WY readout surface from the stored recurrent-state surface, **gated by
`FLA_BF16_BOUNDARIES`**. Concrete in-kernel changes (all in `_tree_gdn_wy_kernel`):

1. **Raw-fp32 k basis for state** — `:504-512`
   `b_k_state = b_k` captured BEFORE the bf16 q/k taps at `:508-510`, so the state uses the
   un-bf16-rounded l2-normed k.

2. **Raw-fp32 WY system for state** — `:530-535`
   `kk_state = tl.dot(b_k_state, tl.trans(b_k_state), input_precision="ieee")` and
   `system_state = where(m_strict, kk_state * b_beta * decay, 0)` — fp32 solve, in contrast
   to the bf16 `kk`/`system` at `:527-529` used for output.

3. **Raw-fp32 triangular solve + transition for state** — `:543-546`, `:551`, `:556`,
   `:559-560`, `:567`, `:570-581`, `:589-609`
   Separate `solved_state_v/k`, `coeff_state`, `k_state_i`, `y_state_i`, `sk_state_i`,
   `trans_state_v`, `tv_state_i = y_state_i - sum(b_h0*sk_state_i)` (`:601`) — **none**
   bf16-rounded.

4. **State assembled from the raw basis and stored** — `:616-642`, `:649-653`
   `state_store_i` (`:616`) accumulates `state_store_update_ij = trans_state_j * k_state_j *
   decay_ij` (`:630-642`) and is stored to the fp32 `state` buffer at `:649-653` (buffer
   alloc fp32 at `:848`). `state_i` (`:615`, bf16-tapped basis) now feeds ONLY the output
   `out_i` (`:643-648`).

5. **Oracle preserved** — under `FLA_BF16_BOUNDARIES` off, `:512`/`:539` set
   `b_k_state = b_k` and `system_state = system` (the existing fp32 closed form), so the
   off-flag R-correctness path stays at ~9.3e-10. **Do not** add any bf16 round to
   `state_store_i` (that is the already-removed wrong #6; it would over-round the carry and
   double the gap — see `FR13_LADDER_LOG.md:410`).

The state store round point is UNCHANGED and correct: single fp32 store (`:649-653`),
matching native `ht` fp32 (`chunk_delta_h.py:263,317`) and the shared bf16 ssm-cache write.
**No bf16 round is missing or needed at the state store.** The drift was operand-basis, not
store-round.

---

## Offline test order (test-first; fr17 confirms)

The kernel change is already in HEAD. The required confirmations, in order:

### Step 1 — offline fixed-payload state replay  ✅ ALREADY PASSING (re-run to re-bind)
- Runner: named CUDA-entrypoint container, no vLLM/server boot, no `--rm`.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Compare WY scan **STATE** (`state` out of `launch_tree_gdn_prepared`) vs native
  `fused_sigmoid_gating_delta_rule_update` state, `--use-wy --fla-bf16-boundaries
  --max-depth 6`, across the traversal contexts already in the harness
  (`original_full`, `spine_only`, `reverse_sibling_dfs_full`, `spine_first_full`).
- **PASS bar:** state max_abs ≤ ~1e-7 (fp32 replay floor) on **spine AND branches**;
  `out` max_abs unchanged at the ~1.2e-4 bf16-readout floor.
- **Observed (`8a975837`):** spine 8.94e-8, full-replay 5.96e-8, branch traversal collapses
  (`codex_fr17_patch2_*_state.json`). Done.
- Note: the bf16-floor target in the handoff ("→ ~6e-5") is the OUTPUT floor; the STATE
  goes all the way to the **fp32** floor (~6e-8) because the state path is now raw fp32 (no
  bf16 round on the state carry). That is correct and better than the 6e-5 ask — do not add
  a bf16 round to bring it "up" to 6e-5.

### Step 2 — branch-oracle correctness (off-spine nodes)
- The `_state` track is shared across the tree; confirm `reverse_sibling_dfs_full` and
  `spine_first_full` contexts (different `remapped_parent` orderings, already in
  `codex_fr17_patch2_batch_state.json` `contexts`) keep state ≤1e-7 vs the native-on-path
  oracle. Already green in patch2 batch (`reverse_sibling … vs native_fla` state at fp32
  floor). This guards against a spine-only fix that breaks branch ancestry.

### Step 3 — ONE live top-down ladder (only after Steps 1–2 re-bound)
- `FR10_TREE_GDN_WY=1` **and** `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES=1`, `TREE_ATTN`, forked
  FA2, `FR13_FA2_PREFILL_NATIVE=1`, B=1 eager diagnostic capture; native arm
  `FLASH_ATTN`/`naive_mtp` 5-token; one GPU; host recovery between arms; fallback unset.
- Walk the spine ladder: `input_hidden → layer0 → … → final logits`. Before `8a975837` this
  first diverged at **layer 1 linear_attention = 1.22e-4** then blew up
  (`FR13_LADDER_LOG.md` Gate-A table: layer2 0.0156, final-logit 3.32). With the state-store
  fix the per-layer recurrent carry should stay at the fp32 floor, so layer-1+ linear-attn
  hidden should collapse toward 0.0 and the final-logit 3.32 should drop out.
- **PASS bar (lossless gate is e2e, not per-layer):** per-layer ladder is a DEV check; the
  verdict is e2e vs **E5** (`output/fr10_native_mtp5_same8_*`): our dist within E5 self-noise
  floor + accept/event ≥ native. Per-layer 0.0 is the diagnostic that the cross-step carry
  no longer drifts.
- If layer-1 is still nonzero live (co-residency / cross-step carry the single-forward L1
  payload can't see): the next suspect is the **cross-step / cross-layer h0 handoff**, i.e.
  the carried `b_h0` read (`:464-468`) seeing a stale-dtype value vs the bf16 ssm-cache that
  native re-reads (`fused_sigmoid_gating.py:134`). That is a separate surface from this
  single-forward fix and must be chased on a multi-step ladder, NOT the L1 smoke — but only
  if Step 3 shows residual drift after the now-fp32 state store.

---

## Honesty / scope

- This is a static localization corroborated by already-captured offline artifacts. The
  single-forward state gate (Step 1) is **passing** in HEAD. The live multi-step / 64-layer
  confirmation (Step 3) has **not** been run since `8a975837`; that is the one remaining
  bind, and it is where any residual cross-step h0-handoff dtype seam would show.
- No copy/splice/dense/reward-hack: the `_state` track is OUR raw-fp32 WY/UT solve, gated;
  the off-flag path remains the fp32 closed-form oracle (~9.3e-10).
