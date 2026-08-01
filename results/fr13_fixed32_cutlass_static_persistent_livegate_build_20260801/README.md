# Fixed32 static-persistent B1 live-gate build

Status: CPU build and gate wiring PASS; real SWE-Verified live gate not run.

This rebuild keeps the stock SM120 projection tile, layout, mainloop, K
iteration order, scale granularity, accumulator, and epilogue. The only kernel
mechanism change is complete-output-tile allocation through CUTLASS
`StaticPersistentTileScheduler100` instead of dynamic CLC queries.

The original static binary `b1cbd966...` stopped its comparator at 256 calls.
The first real MTP `8192x5120` projection occurs after exactly 256 target
projection comparisons, so that binary cannot satisfy the five-shape live
gate. This build raises both the compiled comparator and reducer maximum to
320. The original binary is preserved at:

`/home/mark/fr13_static_scheduler_build/build/preserved/_C_stable_libtorch.limit256.b1cbd966.abi3.so`

Gate-ready binary:

`/home/mark/fr13_static_scheduler_build/build/_C_stable_libtorch.abi3.so`

- SHA-256: `66c37f2593cd38738ed2689e1cabdeaaf8383663597b4b29b46558bbf6bd2cfb`
- Bytes/mode: `113080920` / `0555`
- Diagnostic selector: `static_persistent_stocktile_byte_ab`
- Candidate selector: `static_persistent_stocktile`
- Live schema: `fr13.fixed32.cutlass_static_persistent_live_gate.v1`
- Comparison maximum: `320`

The generated dispatch exactly matches the repo patcher output. The old and
new binaries have identical `cuobjdump --dump-resource-usage` and `nm -D
--defined-only` output, confirming that the bound-only rebuild did not change
the compiled CUDA kernel resource geometry or dynamic export surface.

Verification:

- 76 focused tests passed.
- Four shell launchers passed `bash -n`.
- Pinned vLLM/CUTLASS source validation passed.
- Binary identity verification passed.
- Production RUNPATH is
  `/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64`.

Required next gate: one real SWE-Verified B1 diagnostic task with full
vocabulary and `FR13_STREAMK_GATE_CANDIDATE=static_persistent_stocktile`.
The diagnostic remains production-off and serves the stock result.
