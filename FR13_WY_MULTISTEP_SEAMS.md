# FR13 WY GDN tree-verify — multi-step / cross-step state seams

For codex_fr17. Pre-analysis of the seams the SINGLE-FORWARD offline test (scan state
0.00166 -> 6e-8 on spine AND branches after state-write fix 8a975837) does NOT exercise,
but which the LIVE multi-step top-down ladder will. Every claim grounded in file:line on
BOTH kernels. Read-only analysis; no GPU/docker. NO copy/splice/dense — alignment only.
The fp32-oracle path stays.

Native source pinned to live container `/tmp/vllm_live_019/vllm`.

---

## VERDICT ON THE FLAGGED SUSPECT (b_h0 cross-step handoff DTYPE)

**Will the cross-step b_h0 carry drift live? NO — it is symmetric with native by
construction. The flagged "h0 read dtype" / "fp32-pure vs bf16" asymmetry does NOT exist.**

Proof — the carry round-trip is byte-for-byte the same pipeline on both arms:

- **Store (round to bf16 exactly ONCE):**
  - WY: `ssm_state.index_copy_(0, spec_state_indices[...], tree_state[:tree_n].to(dtype=ssm_state.dtype))`
    at `scripts/fr10_phase4_patch_vllm_tree_gdn.py:2683-2686`.
  - Native decode: `tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))` at
    `fla/ops/fused_sigmoid_gating.py:180` (and `:184`), with `inplace_final_state=True`
    making `ht == initial_state == ssm_state`.
  - Native chunk/prefill: `tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty))` at
    `fla/ops/chunk_delta_h.py:263`; prefill cache init
    `ssm_state[non_spec_state_indices] = last_recurrent_state.to(ssm_state.dtype)` at
    `mamba/gdn_linear_attn.py:1160-1162`.
  - Both round to the **same** `ssm_state` bank, **same** dtype, **same** single round.

- **Carry dtype = `ssm_state.dtype` = bf16 in BOTH arms.** `mamba_utils.py:61-67`
  `_mamba_state_dtype`: `mamba_cache_dtype=auto` and `mamba_ssm_cache_dtype=auto`
  -> `temporal_state_dtype = conv_state_dtype = get_kv_cache_torch_dtype(auto, model_dtype)`
  = model bf16. WY and native write the SAME tensor (`self_kv_cache[1]`), so they cannot
  disagree on carry precision.

- **Read (promote bf16 -> fp32, identical op):**
  - WY: `b_h0 = tl.load(h0_base + ...).to(tl.float32)` at
    `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:464-468`.
  - Native: `b_h += tl.load(p_h0, ...).to(tl.float32)` at `fused_sigmoid_gating.py:134`.
  - Same bf16 bank, same promotion. Accepted-column index logic also matches:
    WY `tl.load(h0_num_accepted_tokens + ...) - 1` at `:457-462` == native
    `i_t = num_accepted_tokens - 1` at `fused_sigmoid_gating.py:114/:120`.

- **The fp32 `tree_state`/`state` buffer is an IN-STEP intermediate, NOT a fp32-pure
  cross-step carry.** `tree_state_all` alloc fp32 at `fr10_phase4_patch:1945-1953`;
  kernel fp32 store at `fr10_gdn_tree_kernel.py:649-653`; buffer alloc fp32 at `:848`.
  It mirrors native's fp32 `b_h`/`ht` intermediate (`fused_sigmoid_gating.py` pre-store,
  `chunk_delta_h.py:263`). It is **never fed as next-step h0** — next step reads
  `h0 = ssm_state` (bf16) at `fr10_phase4_patch:2399`. `last_recurrent_state = tree_state_all`
  at `:2770` is used only for the merge/non-spec writeback, not as the carry.

**DO NOT add a bf16 round to `state_store_i` in the kernel (`:649-653`).** That double-rounds
(kernel store + cache write) and OVER-rounds the carry vs native's single round, doubling the
gap. This is already-removed wrong change #6 (FR13_WY_STATE_FIX.md / FR13_LADDER_LOG.md:410).

---

## SINGLE MOST-LIKELY CAUSE OF A LIVE LAYER-1 RESIDUAL (do this FIRST)

### S1 — bf16-bucket rounding of the committed carry (priority HIGH)

**The one seam the single-forward fp32 test structurally cannot see.** The offline gate
compares the fp32 `tree_state` vs native fp32 `ht` PRE-round (6e-8 with FLA_BF16_BOUNDARIES,
9.3e-10 on the off-flag fp32 oracle). The LIVE carry is the POST-round bf16 bank row. A 6e-8
fp32 delta that straddles a bf16 bucket boundary rounds to a DIFFERENT bf16 value at a small
fraction of elements. That different bf16 value becomes next step's `b_h0` and compounds
across steps and layers — invisible to the single-forward fp32 comparison.

The round itself is correct (both round exactly once at the cache write,
`fr10_phase4_patch:2686` == `fused_sigmoid_gating.py:180-184`, structurally identical — the
store-round is NOT the seam, FR13_WY_STATE_FIX.md:114-117 is right). The risk is purely the
bucket the 6e-8 fp32 delta lands in.

**FIX IF LIVE LADDER SHOWS A LAYER-1 RESIDUAL (gated FLA_BF16_BOUNDARIES):**
1. Change the offline/ladder gate to compare **POST-bf16-round**: assert the COMMITTED bf16
   bank rows are bit-exact to native's `ht`-in-bank bf16 (`tree_state[:tree_n].to(bf16)` vs
   native committed row), NOT just the pre-round fp32. The staged-delta diag already reads
   the committed row (`fr10_phase4_patch:2757-2768`) — promote it to a hard per-element bit
   equality at the bank, not `max()`.
2. If post-round bank rows differ at any element, drive the fp32 `_state` track tighter so it
   rounds into the IDENTICAL bf16 bucket as native. The off-flag fp32 oracle at 9.3e-10
   already rounds bit-identically; the FLA_BF16_BOUNDARIES `_state` at 6e-8 may not. Tighten
   the `_state` solve op-order (`fr10_gdn_tree_kernel.py:611-642`, the raw-fp32 `_state` track
   `state_store_i`) to match native's `b_h` carry accumulation order
   (`fused_sigmoid_gating.py:158-165`: `b_h *= exp(b_g)`; `b_v -= sum(b_h*b_k)`;
   `b_v *= beta`; `b_h += b_v*b_k`). Alignment only — NO copy/splice; do NOT route through
   native's kernel.

Note the `_state` track already correctly excludes the root bf16-boundary rounding for the
decode-step carry: `state_store_i = state_i` for `i==0` else seeded from `b_h0` raw fp32 at
`fr10_gdn_tree_kernel.py:336-338`, and the `_state` accumulation at `:611-642` uses raw
`b_k_state`/`trans_state` (no `.to(bf16).to(fp32)` boundary applied to the state path, unlike
the output path at `:561-563/:584-586/:602-603`). This is the correct asymmetry — keep it.

---

## CROSS-BRANCH SEAMS (priority HIGH — single-forward replay misses the cross-STEP topology change)

### S2 — accepted branch state becomes next-step h0 for a DIFFERENT topology

The offline test covers branch correctness STATICALLY (FR13_WY_STATE_FIX.md:141-146 Step 2:
reverse_sibling_dfs_full + spine_first_full at fp32 floor vs native-on-path oracle). It does
NOT exercise: across decode STEPS the accepted branch's bf16-committed row
(`fr10_phase4_patch:2685` writes ALL `:tree_n` node rows) is remapped by
`launch_tree_state_linear_remap` (`fr10_gdn_tree_kernel.py:170-194` ->
`_remap_state_rows:123-167` -> `_linear_remap_rows_kernel:90-120`) and becomes next step's
shared `b_h0` for a NEW tree topology. WY computes per-node state in ONE kernel invocation
with shared `b_h0` + shared `cum_g`/`trans_state` accumulators masked by `visible_mask`
(`fr10_gdn_tree_kernel.py:615-642`); native has NO tree counterpart (single linear path,
`fused_sigmoid_gating.py:173-180`), so the branch oracle is native-on-path-rerun (per MEMORY
reference_gdn_tree_branch_oracle_losslessness / SpecInfer Def 4.1).

If a non-accepted sibling's bf16 bank row (also written for ALL nodes at `:2685`) is later
mis-read as h0 due to a remap/accepted-column off-by-one, an off-spine state leaks across
steps. This shows up as a per-depth argmax LAG, not a smooth 6e-8 (per MEMORY: cross-branch
bleed = mask bug seen as per-node argmax lag).

**FIX IF LIVE SHOWS CROSS-BRANCH DRIFT (wiring, not kernel — fix the index plumbing):**
Assert in the live ladder, after `launch_tree_state_linear_remap`, that column k of the bank
holds exactly `accepted_paths[b,k]`'s node state (the remap contract,
`fr10_gdn_tree_kernel.py:180-185`; kernel src/dst index logic `:100-120`), AND that
`h0_use_accepted_column` (`:457-462`) reads `num_accepted-1` = the same column native's `i_t`
reads (`fused_sigmoid_gating.py:114/:120`). Verify spine AND each branch via native-on-path
oracle, per-depth argmax NOT max_abs. Do NOT build a kernel; do NOT add dense/copy.

### S3 — remap index lag vs committer (priority MEDIUM)

`launch_tree_state_linear_remap` (`fr10_gdn_tree_kernel.py:704-712` wiring,
`_linear_remap_rows_kernel:90-120`) gather/scatters accepted node rows into stock linear
columns by `accepted_paths` AFTER the commit — pure bf16->bf16 row copy, no recompute, no
extra round (`tl.load`/`tl.store` of `vals` at `:119-120`). Dtype is fine. The LIVE-only risk:
the remap keys on `_fr10_accepted_paths_tensor`/`_fr10_accepted_lens_tensor` pulled from
globals (`fr10_phase4_patch:619-627`); if these lag the committer by one step, or the
accepted-length used by the remap disagrees with the length used by `h0_use_accepted_column`
(`:457-462`) or native's `num_accepted_tokens`, the next step reads the WRONG node's bf16
state as h0 — a topology/index bug masquerading as numeric drift.

**FIX IF LIVE SHOWS A STEP-2+ ARGMAX LAG (not a smooth 6e-8):** assert remapped column k
bit-equals the committed node row for `accepted_paths[b,k]`, and that the accepted-length used
by the remap == the length used by `h0_use_accepted_column` == native `num_accepted_tokens`.
Wiring fix, not kernel.

---

## CROSS-LAYER (priority MEDIUM — compounding, not a distinct seam)

### S4 — per-layer bf16-commit delta compounds over 48 layers x N steps

Each of the 48 GDN layers owns its own ssm bank slice (`self_kv_cache[1]` per-layer,
`mamba/gdn_linear_attn.py:880`) and runs its own WY tree kernel + its own bf16 commit
(`fr10_phase4_patch:2685-2686`, same code all layers). No layer-index term inside
`_tree_gdn_wy_kernel`; native likewise has no per-layer accumulation
(`fused_sigmoid_gating.py:178-180`). So there is NO distinct per-layer seam.

The cross-layer effect is COMPOUNDING: a per-layer bf16-commit delta (S1) that is ~0 at
layer 1 but nonzero-bf16-bucket at a few elements feeds forward layer-to-layer AND
step-to-step. The single-forward single-layer offline replay cannot see the 48-layer x N-step
accumulation that produced the prior final-logit 3.32 (FR13_WY_STATE_FIX.md:154).

**No new kernel fix here.** The compounding is fully neutralized IFF the per-layer
bf16-committed bank rows are bit-exact to native (the S1 POST-round bank assertion). VERIFY in
the live ladder by walking input -> layer0 -> ... -> logits and confirming each layer's
COMMITTED bf16 ssm row (not just fp32 tree_state) matches native; first nonzero layer is the
root (per MEMORY top_down_per_layer_lossless_gate). Do NOT grant per-layer tolerance — 1e-7
amplifies ~32x via the gate 1/rms and crosses fp8/bf16 buckets over 48 layers -> argmax flip.

---

## SEAMS THAT MATCH NATIVE (no drift — do NOT touch)

### S5 — prefill -> decode h0 seam (NO drift)

Under `FR13_FA2_PREFILL_NATIVE=1` the prefill branch is the UNMODIFIED native
`self.chunk_gated_delta_rule` (`fr10_phase4_patch:2968/2989`), writing its fp32-accumulated
final state into the bf16 bank via native
`ssm_state[non_spec_state_indices] = last_recurrent_state.to(ssm_state.dtype)`
(`gdn_linear_attn.py:1160-1162`). The FIRST tree-decode step reads that bf16-seeded row as h0
(`fr10_gdn_tree_kernel.py:464-468`). Native decode seeds from the SAME native prefill into the
SAME bf16 write, then reads via `fused_sigmoid_gating.py:134`. **Bit-identical in both arms.**
No prefill->decode dtype seam is introduced by the WY kernel (it does not touch the prefill
path). A step-1 layer-1 residual therefore CANNOT come from the prefill handoff — it must come
from the WY scan (6e-8 offline) or the bf16 commit round (S1). Drift can only ENTER from
step 2+ via WY-written state.

### S6 — cross-step dtype resolution (NO drift)

WY rounds the carry to `ssm_state.dtype` (`fr10_phase4_patch:2686`); native keys off the same
tensor. They cannot disagree on carry precision because it is the SAME tensor (`self_kv_cache[1]`).
A misconfig (`mamba_ssm_cache_dtype=float32`) would make BOTH arms carry fp32 and still match.
No WY-specific drift. Optional live-confidence guard only (next section).

### S7 — h0 READ dtype (NO drift — the flagged suspect, debunked)

Covered in the verdict above. Both kernels read the bf16 bank and promote bf16->fp32 with the
same op (`fr10_gdn_tree_kernel.py:464-468` == `fused_sigmoid_gating.py:134`); accepted-column
index matches. Do NOT add or remove any cast here.

---

## OPTIONAL LIVE-CONFIDENCE GUARDS (document, do not change the carry)

- Assert `ssm_state.dtype == torch.bfloat16` once before the `:2686` writeback, to document
  that WY rounds exactly where native rounds (catches a future config that silently makes the
  spec-path bank fp32). No change to the carry itself.
- Promote the existing staged-delta diag (`fr10_phase4_patch:2757-2768`) from `max()` to a
  per-element bf16 bit-equality of the COMMITTED bank row vs native — this is the S1
  detector and the live ladder's go/no-go on the carry.

---

## PRIORITY SUMMARY

| # | Seam | Drift live? | Priority | Fix kind |
|---|------|-------------|----------|----------|
| S1 | bf16-bucket rounding of committed carry | maybe (THE likely cause) | HIGH | tighten `_state` op-order, gate FLA_BF16_BOUNDARIES; assert POST-round bank bit-exact |
| S2 | accepted branch -> next-step h0 for new topology | maybe | HIGH | wiring: assert remap column == accepted_paths[b,k]; per-depth argmax vs native-on-path |
| S3 | remap index lag vs committer | maybe | MEDIUM | wiring: assert accepted-len consistency across remap/h0/native |
| S4 | per-layer bf16-commit compounding | maybe (compounds) | MEDIUM | no new fix; verify POST-round bank per layer top-down |
| S5 | prefill -> decode h0 | no (matches native) | low | none |
| S6 | cross-step dtype resolution | no (matches native) | low | optional guard |
| S7 | h0 READ dtype (flagged suspect) | no (matches native) | low | none |

**Start at S1. The b_h0 cross-step carry DTYPE is symmetric with native (S7) — that is NOT
the seam. The only fp32-invisible cross-step risk is the bf16 BUCKET the 6e-8 fp32 delta lands
in (S1), then its cross-step/cross-layer compounding (S4), then the cross-branch topology
wiring (S2/S3). All fixes are alignment / assertion / index plumbing — NO copy/splice/dense,
fp32-oracle path stays.**
