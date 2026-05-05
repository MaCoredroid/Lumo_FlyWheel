# Candidate 002 Blocked

No `mutation.patch` is submitted.

The required warm diagnostic measured the current live stack at `7.536 generated tok/s` and `132.695675 ms/generated token`, with bottleneck hint `decode`. A 20% post-parity speed gate requires about `9.043 tok/s`, or saving roughly `22.116 ms/token`.

The controller timing proxy puts `ffn_linear` at `20.0%` of decode. Mapped to this warm run, that is only `26.539 ms/token`, with `106.157 ms/token` outside FFN. Even including the listed projection-linear slices, the broad linear proxy is about `38.482 ms/token`, so the required saving is about `57.5%` of all listed linear time.

The live CUTLASS source path remains:

`W8A8BlockFp8LinearOp._run_cutlass -> cutlass_scaled_mm -> torch.ops._C.cutlass_scaled_mm -> dispatch_scaled_mm -> cutlass_scaled_mm_blockwise_sm120_fp8 -> cutlass_gemm_blockwise_sm120_fp8_dispatch`.

For decode shapes, `M <= 256` selects `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>`, `KernelTmaWarpSpecializedBlockwisePingpongSm120`, `ClusterShape<_1,_1,_1>`, and scale granularity `(1,128,128)`.

The representative shapes are B-weight-stream dominated:

- `M=1,N=34816,K=5120`: total `178.376 MB`; B weights `178.258 MB`; arithmetic intensity `1.999 FLOP/B`.
- `M=1,N=5120,K=17408`: total `89.179 MB`; B weights `89.129 MB`; arithmetic intensity `1.999 FLOP/B`.

The requested microbench rebuild did not reach timing after about five minutes, so `cutlass_microbench_pre.json` records it as skipped and cites the nearest same-machine shape records: `M=1,N=34816,K=5120` at `0.833901 ms`, `213.906 GB/s estimated bandwidth`, and `M=1,N=5120,K=17408` at `0.685587 ms`, `130.077 GB/s estimated bandwidth`.

I checked the broader mechanism requested by the brief instead of repeating a local schedule block. The only plausible remaining caller-level fusion is `SiluAndMul + down_proj` input quantization. The existing helper `silu_mul_per_token_group_quant_fp8_colmajor` is not decode-legal as written because it asserts large M multiples, and a new masked fused activation+quant route would still leave the `89.129 MB` down-projection B stream unchanged. It removes under `0.09 MB` of M1 activation/quant traffic plus launches per dense FFN layer, which does not defend a `22.116 ms/token` end-to-end saving.

Qwen3.5 already fuses same-input projection pairs through `packed_modules_mapping` (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba`). Fusing FFN down with gate/up enough to reduce B streaming would require a new FFN operator boundary that owns two distinct B matrices and the nonlinear activation, not a parity-preserving `cutlass_scaled_mm` mutation.

Therefore the remaining legal CUTLASS surfaces preserve the dominant B-weight bytes and cannot defend a `>=20%` end-to-end warm decode lift while preserving dtype, layout, scale semantics, public signatures, and the CUTLASS-only backend contract.
