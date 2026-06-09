# FR13 BV / spill / decouple verdict (workflow w921xvgzx, 2026-06-09) — TPS-opt, DEFERRED to the TPS gate

The SEQ scan kernel is ALREADY bit-exact at BV=16 (committed e4a6a2f2). Nothing here changes numerics; it's whether a smaller/rewarped V-tile removes a state-traffic spill. [PROVEN]=code/specs on CPU host; [LIVE]=needs GPU compile/measure.

## 1. DECOUPLE (BLOCK_V=1 8KB + 2-D reduce) = NOT FEASIBLE [PROVEN]
The reduction tile IS the state tile: `state_i` is `[BLOCK_V,DIM_K]`, written by the rank-1 (L341) + cached (L343); the reductions (L339/L342) reduce that 2-D `[BLOCK_V,128]`. You cannot give the reduce M>=2 rows without the state being M>=2 rows. A 1-V-row reduce inherently carries the `[1,128]` collapse class. **8KB/BLOCK_V=1 is unreachable bit-exact — abandon the decouple hope.**

## 2. The spill is REAL for the N_PAD=16 (14-node) family at BV=16 [PROVEN by register arithmetic; ptxas reality is LIVE]
`h_cache = [N_PAD, BV, DIM_K]` fp32 (L277), N_PAD<=16 (families 2,3,6,8,14 -> N_PAD 2,4,8,8,16). At N_PAD=16,BV=16 = 128KB. With default 4 warps (128 lanes; no num_warps override at L548) that's **256 fp32 regs/lane for h_cache alone > GB10 255-reg/thread cap -> spills to LPDDR5X (273 GB/s, the same pool decode saturates)**. 128KB also > 99KB shared cap (can't park in SRAM). For N_PAD<=8: borderline occupancy, not a hard overrun. So the +35.8%-tax risk is real for the 14-node family.

## 3. Smallest non-collapsing BV: >=2 [PROVEN]; smallest RELIABLY bit-exact = 4 [LIVE prediction]
BV=1 -> Triton collapses the degenerate `[1,128]` -> different butterfly tree -> 1.19e-7 (= f3260def). BV>=2 keeps the 2-D `[M,128]` tree. Native `[32,128]` compiles (TTGIR) to `sizePerThread=[1,4] threadsPerWarp=[1,32] warpsPerCTA=[4,1]` = 4 warps on ROWS, 32 lanes on K. Matching that op-order needs leading extent >= num_warps=4, so **predict BV=4 reliable; BV=2 at-risk** (warps may spill onto K -> cross-warp tree). Worst case = N_PAD=1 (linear spine, leading extent = BV).

## 4. RECOMMENDATION (ranked, all LIVE-verify when the TPS gate is reached)
1. **num_warps=8 at launch (L548), keep BV=16.** Halves per-lane register pressure (256->128 regs/lane at N_PAD=16) -> kills the hard spill WITHOUT shrinking the tile (keeps BV=16's proven 0.0). Cheapest; re-run the gate to confirm still bit-exact (same op, expected to hold). Risk: thread-mapping change could perturb the layout.
2. **BV=8.** Largest BV register-resident across ALL families incl N_PAD=16; predicted bit-exact (leading extent 8 >= 4 warps). Smaller blast radius than num_warps surgery.
3. **BV=4.** 4x smaller h_cache, predicted smallest bit-exact; only risk is the op-order prediction (gate at N_PAD=1 AND 16).
4. **Ship BV=16 as-is** if the N_PAD=16 spill costs negligible TPS (14-node family rare / spill hidden).
- BV=2 only if more TPS wanted (expected to fail at N_PAD=1). BV=1 NOT FEASIBLE.

## 5. CRITICAL gate caveat [PROVEN] — read RAW max_abs, NOT the 1e-3 exit code
`scripts/fr10_scan_output_replay_gate.py` has **`atol=1e-3` (L170)** — its exit code / root_pass does NOT certify bit-exact (it passes up to 1e-3 drift). For ANY BV sweep, read the raw `root_out_vs_native_max_abs` / `max_node_out_vs_native` floats and require **== 0.0** by hand + confirm `negative_control_powered==true`. (The gateA LADDER uses threshold 0.0, so the committed L0-3=0.0 result IS genuine; this caveat is for the scan-output replay gate.)

## What the TPS gate (#5) must confirm (none settled on CPU)
(a) raw out_vs_native_max_abs == 0.0 at N_PAD=1 AND N_PAD=16 for the chosen config; (b) ptxas -v / TRITON_KERNEL_DUMP spill-bytes == 0 at N_PAD=16; (c) decode-TPS (metrics OFF, B=4) vs the BV=16 baseline — the whole exercise is justified only if (c) shows measurable TPS. Kernel L17 BV, L277 h_cache, L548 launch (default num_warps). Native ttgir at /tmp/lumo-l0c-fp8-cutlass-run30-triton/.../fused_sigmoid_gating_delta_rule_update_kernel.ttgir.
