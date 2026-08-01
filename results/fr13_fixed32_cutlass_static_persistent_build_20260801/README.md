# Fixed32 CUTLASS static-persistent projection candidate

Status: source and SM121 compile PASS; real SWE-Verified byte gate and timing
not run.

The candidate replaces CUTLASS's dynamic Blackwell CLC tile allocation with
`StaticPersistentTileScheduler100` only for the fixed32 projection shapes. It
keeps the stock tile shapes, complete K reduction per output tile, mainloop
schedule, scale layouts, FP32 accumulator, and BF16/FP16 epilogue:

- B1, M=32/64: swapped A/B, cooperative 128x32x128.
- B4, M=96/128: normal layout, pingpong 64x128x128.

This is an exact-safe mechanism at source level because no output tile is split
and every output accumulator consumes the same ordered K tiles as stock. The
diagnostic selector `static_persistent_stocktile_byte_ab` still serves the stock
result and records raw-byte comparisons; source reasoning is not a substitute
for that real-task gate.

The measured/stale projection phase is 112.313 ms against an 87.266 ms
mandatory-weight floor, leaving 25.047 ms (1.287x floor). This candidate only
removes CLC scheduler queries. It does not reduce mandatory bytes, and the
compiled resources show no register/shared-memory occupancy improvement.
Therefore it is a bounded follow-up candidate, not a claim that the projection
gap or full-step hardware-floor goal is closed.

Build identity:

- repo source commit: `0adf68ed9`
- vLLM base: `fe9c3d6c5f66c873d196800384ed6880687b9e52`
- CUTLASS: v4.4.2, `da5e086dab31d63815acafdac9a9c5893b1c69e2`
- candidate SO: `/home/mark/fr13_static_scheduler_build/build/_C_stable_libtorch.abi3.so`
- candidate SHA-256: `b1cbd96651ec5e2b35cdacbe7fe6e75105e0c18593feea67c31f8cf8057cac4a`
- candidate bytes/mode: 113080920 / 0555

Verification performed:

- 30 focused Python contract tests passed.
- Pinned source patch and SM121 CUDA compilation passed.
- Candidate strong dynamic exports exactly match the reference extension: 880
  symbols, normalized-list SHA-256
  `eced6cd9fdd5aa2950ffeadd2a3d174023d03ea7b4dce115f4bd16e0e83909d2`.
- Container RUNPATH is
  `/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64`.

Required next gate: one real SWE-Verified B1 task for diagnostic byte equality,
then exact4 B1 timing only if every compared byte passes. B4 must use exact4;
the B4 stack-frame increase makes it lower priority than B1.
