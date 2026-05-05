# Candidate 004 Blocked

No `mutation.patch` is submitted.

The required warm diagnostic measured the current live stack at `7.599 generated tok/s` and `131.594103 ms/generated token`, with bottleneck hint `decode`. The 20% post-parity gate requires `9.119 tok/s`, which means saving about `21.932 ms/token`.

The controller timing proxy assigns `ffn_linear` `20.0%` of decode. Mapped to this warm run, that is `26.319 ms/token`, with `105.275 ms/token` outside FFN. A CUTLASS-only FFN mutation would need to remove about `83.3%` of the FFN proxy to pass. Even including the listed projection-linear slices, the required saving is still about `57.5%` of all listed linear time.

The live CUTLASS route is:

`W8A8BlockFp8LinearOp._run_cutlass -> cutlass_scaled_mm -> torch.ops._C.cutlass_scaled_mm -> dispatch_scaled_mm -> cutlass_scaled_mm_blockwise_sm120_fp8 -> cutlass_gemm_blockwise_sm120_fp8_dispatch`.

For decode shapes, `M <= 256` selects `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>`, `ClusterShape<_1,_1,_1>`, `KernelTmaWarpSpecializedBlockwisePingpongSm120`, and scale granularity `(1,128,128)`.

The required microbench completed for the live FFN shapes:

- `M=1,N=34816,K=5120`: `0.796723 ms`, `223.887 GB/s` estimated bandwidth, `1.998672 FLOP/B`.
- `M=1,N=5120,K=17408`: `0.412848 ms`, `216.009 GB/s` estimated bandwidth, `1.998880 FLOP/B`.
- `M=4,N=34816,K=5120`: `0.803834 ms`, slower than M1.
- `M=4,N=5120,K=17408`: `0.414576 ms`, slower than M1.

The byte split is dominated by B weights:

- `M=1,N=34816,K=5120`: total about `178.376 MB`, B weights about `178.258 MB`.
- `M=1,N=5120,K=17408`: total about `89.179 MB`, B weights about `89.129 MB`.

I checked the broader mechanisms required by the brief:

- Qwen3.5 already packs same-input projection pairs: `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba`.
- FFN down projection cannot reuse the gate/up B stream because it consumes a nonlinear activation and a distinct B matrix; reducing that B stream needs a new FFN operator boundary, not a drop-in `cutlass_scaled_mm` mutation.
- Fused `SiluAndMul + quant + down CUTLASS` was already tried in prior candidate 009, passed parity, and was discarded with no material speed gain.
- M1 padded shape-lift was already rejected by the speed gate in prior candidate 024, and this iteration's microbench shows M4 is slightly slower than M1 on the large FFN shapes.
- Persistent B staging across tokens is blocked by the stateless per-GEMM public op and by the full-model B-weight working set on GB10 LPDDR/cache.

Therefore the remaining legal CUTLASS surfaces preserve the dominant B-weight bytes and cannot defend a `>=20%` end-to-end warm decode lift while preserving public signatures, dtype/layout contracts, scale semantics, parity, and the CUTLASS-only backend identity.
