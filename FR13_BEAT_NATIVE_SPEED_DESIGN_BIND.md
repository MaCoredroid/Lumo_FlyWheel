# FR13 — beat-native speed design: GPU-resident committer (OPT-1) is the decisive lossless lever

Date 2026-06-14. Workflow `wrgf8u74v` (`wf_a10860ac-d53`), Verify **holds=True**. Raw:
`research/fr13_workflows/beat_native_speed_design_wrgf8u74v.raw.json`. Goal: lossless structural speed to
beat native s/fwd. All bit-exact (structural only, no math change).

## The dominant residual is NOT graph nodes — it's the committer SYNC killing run-ahead
FIX-3 already deleted ~3,038 graph nodes (the per-node conv emulation) → graph-node count is wall-bounded
<10 ms and is NOT the dominant residual. The real residual: the eager committer's **packed DtoH + sync**
(`fr10_phase4_patch_vllm_tree_gdn.py:5674`) sits on the **MAIN launching thread** — census: chain5 blocks
the main thread in memcpyAsync **91.9% of the window vs native 0.8%** (native waits on the async OUTPUT
thread via cudaEventSynchronize, preserving run-ahead). The path-LCP/accept decision (:5780-5879) is
**pure-Python on host lists**, forcing one DtoH+sync/forward on the critical thread → the tree path loses
native's async run-ahead.

## OPT-1 (the fix): GPU-resident committer + CUDA conditional-node accept
Move the entire accept/path-LCP/bonus-source decision (pure INTEGER work: drafts[node]==parent_targets[node]
compares, parents-walk, LCP scan, earliest-leaf tie-break, bonus_source) to a **Triton committer kernel** +
**CUDA-12.4 graph conditional-node / torch.cond** for the data-dependent accept branch INSIDE the capture,
with on-device next-step metadata (absorbs OPT-4) and the host `.tolist` moved to a non-gating side stream.
Removes the sync → restores native-style run-ahead. **Lossless-safe by construction** (pure integer,
location-only move host-Python→Triton, no float/reduction). Flag-gated default-OFF (FR13_GPU_COMMITTER),
like every prior FIX. Reclaims ~4-6 ms of the 6.6 ms cat9 tax. Effort: large.

## Beat-native arithmetic (Verify-corrected, honest)
native MTP-5 = 218.2 ms/fwd; cat9 = 224.7 ms (1.030x, +6.6 ms); bandwidth floor 98.6 ms (native at 2.2x =
~120 ms of overhead headroom).
- OPT-1 conservative (2.5-4 ms): cat9 → ~220.7-222.2 ms (1.011-1.018x, still ABOVE native).
- OPT-1 optimistic (6 ms) + OPT-2/3/4: ~217-218 ms (AT/just-below native).
- **chain5 (+4.4 ms) crosses below native first** (less reclaim needed).
- **The accept edge (cat9 ~3.18 vs native ~3.07 tok/fwd) makes cat9 FASTER END-TO-END (TPS) even at s/fwd
  parity** — that's the real win. "WAY faster" on s/fwd alone is NOT realistic from the structural pass; it
  comes from accept-edge TPS + (later) a bigger tree (needs the 64-node recompute scaling + the BI lossless
  fix first, since bigger tree = more branches = more flips).

## Secondary opts (after OPT-1, which hides them behind the sync)
OPT-2 fused all-layer tree-conv Triton kernel (~1-2 ms); OPT-3 delete captured conv waste (<1 ms,
byte-identical node deletions, e.g. redundant self-copy :3159); OPT-4 on-device runner metadata (~1-2 ms,
folded into OPT-1); OPT-5 (conditional) TARGETED fp8-GEMM BI override instead of global VLLM_BATCH_INVARIANT
(if the 22-flip BI fix is needed — avoids the global BI speed+memory cost, ties to the branch-flip work).

## FIRST BUILD: OPT-1 (FR13_GPU_COMMITTER, default-OFF), gate = byte A/B (class-10) + s/fwd vs native.
Pairs with [[feedback_flag_gate_metrics_reuse_infra]], [[feedback_build_deliverable_form_once_contract_proven]],
[[reference_fr10_speed_measurement_pitfalls]], [[feedback_no_reroute_reward_hacking]].
