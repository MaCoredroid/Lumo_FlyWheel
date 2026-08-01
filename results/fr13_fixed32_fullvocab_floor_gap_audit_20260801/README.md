# FR13 fixed32 full-vocabulary floor-gap kernel audit

This is a CPU/source audit. It did not use the GPU and it does not claim a new
B1 or B4 timing result.

## Floor and measurement state

The required `K=0`, `root=0` event streams 42,025,179,008 mandatory weight
bytes. At 273 GB/s its weight-only floor is 153.938385 ms/event and the 1.15x
cap is 177.029142 ms/event. Only 23.090758 ms/event remains for every activation,
state transfer, launch, synchronization, and implementation inefficiency.

No valid full-vocabulary B1 wall result exists in the inspected artifacts and
B4 has not been timed. The nearest real SWE-Verified timing is the four-task
Hydra27 B1 exact4 K64 arm: 232.779790 ms wall, with SFWD 159.619263 ms, DFWD
36.813368 ms, CFWD 20.677391 ms, and 15.669768 ms outside those GPU phase
timers. It is a useful phase prior only. It is not full-vocabulary acceptance
evidence and produced no paired one-sided U95.

## Kernel finding

The target model contains 256 FP8 projection GEMMs/event across five `(N,K)`
families. At B1 M32 their weight-only arithmetic intensity is about 64 FLOP/B;
at B4 M128 it is about 256 FLOP/B. Both remain weight/wave limited on this
hardware. Their historical real-workload attribution was 112.313 ms/event,
making them the largest quantified kernel group.

The existing `streamk_coop128` implementation uses CUTLASS `Heuristic`
decomposition. On 48 SMs, four of the five projection families have at least a
half-wave tail and therefore select data parallel. Only the 16 calls with
`N=8192` actually select Stream-K. In other words, 240/256 target calls miss the
optimization selected by the mode name.

The immediate candidate is therefore a shape-gated wide-N implementation in
`scripts/fr13_patch_cutlass_fixed32_wave.py`:

* B1 swapped tile: `[256,32,128]`.
* B4 tile: `[128,256,128]`.
* `StreamKScheduler`, forced `StreamK`, one split, deterministic reduction.
* Exact fixed32 row set and exact five production projection shapes only.

That implementation is committed and pushed as `7fa0ac0e4` on
`agent/fixed32-streamk-wide256-f963`; 127 static tests passed and the built
binary SHA-256 is
`b957cf49da2977056661443192fc2725e153adba7f21fb522c07b439c04540ee`. It is
not GPU-gated or timed. The best defensible analytical recovery is only
10.923627 ms/event; the likely range remains 0 to 10.923627 ms until real
exact4 timing. It can regress because of Stream-K fixups, workspace, and the
compiled resource footprint. A rejected candidate has zero production saving.

## Why one kernel is insufficient

The old K64 SFWD phase exceeds the full target-model weight floor by
70.306445 ms. The complete full-vocabulary step can afford only 23.090758 ms
above all weight floors. Even if DFWD, the verifier head, CFWD, and host work
were otherwise perfect, SFWD must lose at least 47.215687 ms. Ideal wide256
recovery still leaves at least 36.292060 ms of SFWD reduction.

The next necessary code-level target is the fixed32 per-layer state path in
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`, together with its FA2 and
vLLM patch wiring. Preserve the two-level `[1,11]` path schedule and B1-B4
batch-folded grids, but fuse compatible GDN ring export, conv gather/writeback,
flags, and attention-side state movement. The objective is invariant launches
and fewer state round trips, not fewer mandatory weight bytes. Cross-layer
fusion is a larger byte-identity project and is not implemented by this audit.

## Launch and bound summary

| Group | B1 launches/event | B4 launches/event | Bound classification |
| --- | ---: | ---: | --- |
| target FP8 projections | 256 | 256 | weight bandwidth + wave occupancy |
| target tree FA2 | 16 | 16 | short-sequence memory + launch |
| target tree GDN | 96 | 96 | state memory + launch |
| conv select/copy/writeback | 48/48/48 | structurally invariant | memory motion + launch |
| MTP FP8 projections | 20 | 20 | weight bandwidth |
| full-vocabulary drafter heads | 5 | 5 | weight bandwidth |
| DFWD unified attention | 4 | 4 expected | short-work memory + launch |
| CFWD GDN state commits | 48 | 48 expected | state memory + launch |
| verifier full head | 1 | 1 | weight bandwidth |

All B4 times remain unmeasured. See `audit.json` for mandatory bytes,
arithmetic-intensity assumptions, stale symbol priors, candidate bounds, and
the required real-task gate order.
