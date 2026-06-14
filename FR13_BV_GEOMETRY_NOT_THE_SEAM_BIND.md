# FR13 — DECISIVE NEGATIVE: the GDN-scan launch geometry is NOT the seam (D16=D32=0.0)

Workflow `wf_0e55b1be-3f0` (woaiybls4) A/B verify, GB10. Independently confirmed by hand from the
capture. 2026-06-14. **This OVERTURNS the seam-scan's "geometry is the diffuse carrier" hypothesis
and moots the entire BV-match / spill / recompute line.**

## The measurement (RAW max_abs, NOT atol; layer 0 GDN, bit-identical captured inputs)
`out_native` = the REAL native `fused_sigmoid_gating_delta_rule_update` (native BV=32/warps=4),
called directly on the path0 spine inputs in the capture hook (`fr10_phase4_patch_vllm_tree_gdn.py:4172`
— verified genuine, not our kernel mislabeled). Our `_tree_gdn_kernel` replayed at both geometries on
the SAME inputs:

| | D16 (ours BV=16/w8 vs native) | D32 (ours BV=32/w4 vs native) |
|---|---|---|
| N_PAD=1 | **0.0** | **0.0** |
| N_PAD=16 (cat9) | **0.0** | **0.0** |

`negControlD16Nonzero=FALSE`, `geometryIsTheSeam=FALSE`. Our scan kernel is **already bit-exact to
native at BOTH geometries and BOTH tree sizes.** The launch geometry (BV/warps) produces ZERO
divergence — `FR13_BF16_FP32_SEAM_SCAN`'s "geometry reshapes the reduction → ~1-ULP/node" was a
STATIC code-reading hypothesis; the silicon refutes it (the DIM_K=128 reduction is geometry-stable
for our inputs).

## Consequences
1. **BV-match / spill / recompute path is MOOT** — nothing to fix in the scan kernel. The whole
   `FR13_BV_NATIVE_MATCH_BIND` plan (match BV=32/warps=4, the 636 B spill, the recompute-from-spine
   workaround, the SMEM/tiling ranking) is **shelved** — there is no geometry seam to chase.
   (The "measure first / A/B before building" discipline saved building the recompute kernel.)
2. **Every per-forward GDN sub-op matches native**: conv tap matched (seam-scan, both bf16),
   scan matches (this, D16=D32=0), gate/o_proj are unchanged native modules. **So the 22-flip
   carrier is NOT in the per-forward GDN computation.**
3. **The v2 "layer-56 carrier" must be re-read**: it was measured vs the no-spec *sequential-decode*
   oracle (`causal_conv1d_update` + `fused_recurrent` — a DIFFERENT kernel path than the tree-verify
   scan). "Matches native-tree-verify" ≠ "matches no-spec-decode." The layer-56 divergence is vs the
   decode path, not vs native-tree-verify.

## Re-orientation — the carrier is one of (per-forward ruled out)
- **Cross-event h0 state handoff** (our fp32 vs native bf16; the replay route's state-logistics,
  which is KNOWN-BROKEN-LIVE = gate-4, accept/event 2.02→1.58, `FR13_REPLAY_GPU_GATES_BIND`).
  PRIME SUSPECT — it accumulates across forwards, which is exactly how 22 e2e flips arise from a
  bit-exact per-forward kernel.
- **Tree co-residency** (cat9 is a 9-node TREE; native E5 is LINEAR MTP-5 — branch nodes share the
  verify forward, the linear baseline never does).
- **Reference artifact** (the binding gate uses the no-spec DECODE oracle; tree-verify legitimately
  differs from sequential-decode, so part of "22 flips" may be a path difference, not a kernel bug).

## NEXT localization (the corrected target)
A same-boot A/B of the **tree-verify served stream vs the NATIVE tree-verify** (native MTP on the
same prefix) — NOT the decode oracle — to separate "kernel correct" (proven here) from
"handoff/co-residency drift." If our tree-verify == native tree-verify e2e, the 22 flips are a
decode-reference artifact; if not, isolate the cross-event handoff. Pairs with
[[reference_gdn_verify_sequential_dispatch]], [[project_fr13_conv_priorwindow_root]],
[[reference_multispine_not_lossless_closed_nonship]], [[feedback_read_vllm_source_first]],
[[feedback_research_before_deadend]].
