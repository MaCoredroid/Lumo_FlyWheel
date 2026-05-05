# Candidate 001 Blocked

No `mutation.patch` is submitted.

The required warm diagnostic measured the current live stack at `7.624 generated tok/s` and `131.160521 ms/generated token`, with bottleneck hint `decode`. A 20% post-parity speed gate requires about `9.149 tok/s`, or saving roughly `21.86 ms/token`.

The controller timing proxy puts `ffn_linear` at `20.0%` of decode. Mapped to this warm run, that is only `26.232 ms/token`, with `104.928 ms/token` outside FFN. A CUTLASS-only FP8 GEMM mutation would need to eliminate about `83%` of the FFN proxy to clear the gate unless it also changes other linear slices. Even including the listed projection-linear slices, the broad linear proxy is about `38.037 ms/token`, so the required saving is still about `57.5%` of all listed linear time.

The live source path is:

`W8A8BlockFp8LinearOp._run_cutlass -> cutlass_scaled_mm -> torch.ops._C.cutlass_scaled_mm -> dispatch_scaled_mm -> cutlass_scaled_mm_blockwise_sm120_fp8 -> cutlass_gemm_blockwise_sm120_fp8_dispatch`.

For decode shapes, `M <= 256` selects `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>`, `KernelTmaWarpSpecializedBlockwisePingpongSm120`, and scale granularity `(1,128,128)`.

The representative shapes are B-weight-stream dominated:

- `M=1,N=34816,K=5120`: total `178.376 MB`; B weights `178.258 MB`; arithmetic intensity `1.999 FLOP/B`.
- `M=1,N=5120,K=17408`: total `89.179 MB`; B weights `89.129 MB`; arithmetic intensity `1.999 FLOP/B`.

The current microbench command was attempted, but the rebuild was still compiling after more than eight minutes and never reached shape timing; `cutlass_microbench_pre.json` records this as skipped. Nearest prior same-machine microbench records for the same path show `M=1,N=34816,K=5120` at `0.833901 ms`, `213.906 GB/s estimated bandwidth`, and `M=1,N=5120,K=17408` at `0.685587 ms`, `130.077 GB/s estimated bandwidth`.

I also checked the broader byte mechanisms required by the round brief:

- Qwen3.5 already fuses obvious paired projections: `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba` are packed in `Qwen3_5ForCausalLMBase.packed_modules_mapping`.
- FFN down projection cannot reuse the gate/up B stream because it consumes a nonlinear activation and a distinct B matrix. Fusing it would require a new FFN operator signature, not a drop-in `cutlass_scaled_mm` mutation.
- Persistent staging across tokens is not exposed by the per-GEMM public op and would not fit the model's full B-weight working set.
- Prior same-machine trials already rejected the compile-clean adjacent families: workspace/hardware-info caching, zero-workspace, M1 branch/padding, fixed stage counts, epilogue tile, swap-AB, Python quant cache, and several tile/scheduler changes. Larger N64/K256/Stream-K/cooperative/alternate-epilogue ideas compile-blocked.

Therefore the remaining legal CUTLASS surfaces preserve the dominant B-weight bytes and cannot defend a `>=20%` end-to-end warm decode lift while preserving dtype, layout, scale semantics, public signatures, and the CUTLASS-only backend contract.
