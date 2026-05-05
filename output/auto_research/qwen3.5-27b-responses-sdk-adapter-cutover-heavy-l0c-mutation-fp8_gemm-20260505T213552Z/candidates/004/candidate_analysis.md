# Candidate 004 Analysis

No `mutation.patch` is submitted. The cheap evidence does not support a parity-preserving CUTLASS FP8 GEMM mutation that can clear the current 20% post-parity warm decode speed gate.

## Baseline Timing

- Required warm diagnostic: `candidates/004/warm_pre_mutation.json`.
- Current warm decode rate: `7.599 generated tok/s`, `131.594103 ms/generated token`, bottleneck hint `decode`.
- Per-step warm diagnostic:
  - Request 1: `7.605 decode tok/s`, `131.499873 ms/generated token`, `64` generated tokens, `1255.0` prompt tokens, `62.47011952191235%` cache hit rate.
  - Request 2: `7.594 decode tok/s`, `131.688332 ms/generated token`, `64` generated tokens, `1252.0` prompt tokens, `62.61980830670927%` cache hit rate.
- Aggregate GB10 context: NVIDIA GB10 / DGX Spark, `128 GB unified LPDDR5x`, `273.0 GB/s` theoretical bandwidth, `273000000` stream bytes/ms, `10.111 tok/s` full-model FP8 stream ceiling, `35.925190036753 GB/generated token` bandwidth budget at the observed decode time.
- Current round baseline measurement rows: `baselines/measurement_01..05.json` show decode bottleneck with decode rates around `7.323-7.400 tok/s` and decode ms/token around `135.20-136.55`; the warm diagnostic for this iteration is slightly faster but same regime.
- Strategy timing proxy: `ffn_linear` is the controller-owned CUTLASS/FP8 GEMM proxy at `20.0%` leaf share, `80.597 ms/requested output token` in the P3a breakdown. No lower-level full-model CUTLASS timer is available; the local microbench below is only a shape-level proxy around `torch.ops._C.cutlass_scaled_mm`.

At the current warm rate, a 20% speed gate requires `9.119 tok/s`, or `109.662 ms/token`; the required saving is `21.932 ms/token`. Mapping the strategy proxy to this warm run gives `ffn_linear = 26.319 ms/token` and non-FFN residual `105.275 ms/token`. A CUTLASS-only patch limited to `ffn_linear` would need to remove about `83.3%` of the whole FFN proxy to pass. Even if the two projection-linear rows (`deltanet_projection_linear` 7.0% and `gatedattn_projection_linear` 2.0%) were fully affected, the listed linear slice is `38.162 ms/token`, so the required saving is still `57.5%` of all listed linear time.

7.5 tok/s breakdown line: at `7.5 tok/s`, token time is `133.333 ms/token`. A 27B FP8 full-model stream approximation is `27 GB/token`, implying `202.5 GB/s`, or `74.2%` of the `273 GB/s` GB10 ceiling. The full-model stream ceiling is `10.111 tok/s`. The `ffn_linear` 20.0% proxy is `26.667 ms/token`, and non-FFN residual is `106.667 ms/token`. The legal CUTLASS surfaces below attack schedule/launch/activation overhead, not the dominant B-weight traffic.

## Source And Dispatch Facts

- Live CUTLASS path: `W8A8BlockFp8LinearOp._run_cutlass -> cutlass_scaled_mm -> torch.ops._C.cutlass_scaled_mm -> dispatch_scaled_mm -> cutlass_scaled_mm_blockwise_sm120_fp8 -> cutlass_gemm_blockwise_sm120_fp8_dispatch`.
- Source file/symbol: `cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh`, `cutlass_gemm_blockwise_sm120_fp8_dispatch`.
- Live-shape dispatch-hit proof: dynamic block-FP8 activation and weight scales are 2D, so `dispatch_scaled_mm` selects the blockwise function. Warm decode has tiny `M` from `input.view(-1, hidden)`, so `M <= 256` selects `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>`, `ClusterShape<_1,_1,_1>`, `KernelTmaWarpSpecializedBlockwisePingpongSm120`, and scale granularity `(1,128,128)`.
- Primary source fact: NVIDIA CUTLASS 4.2.1 Blackwell docs state block-scaled GEMMs apply scale factors along GEMM K and list Blackwell SM120 GEMM constraints for cluster size, tensor layout, schedules, epilogue schedule, and tile size. The local vLLM config already follows the legal TN-style row-major A / column-major B convention, `1x1x1` cluster, auto epilogue, and SM120 blockwise scale layout; prior attempts against adjacent legal tile/schedule variants either compile-blocked or measured near baseline.

## Microbench Evidence

Required shape microbench: `candidates/004/cutlass_microbench_pre.json`, compile/jobs `1`, CUDA events around `torch.ops._C.cutlass_scaled_mm`.

| Shape M/N/K | Branch | event_ms_mean | estimated_effective_bandwidth_gb_s | arithmetic_intensity |
|---|---:|---:|---:|---:|
| `1/34816/5120` | `M<=256` | `0.796723` | `223.887480` | `1.998672` |
| `1/5120/17408` | `M<=256` | `0.412848` | `216.009066` | `1.998880` |
| `4/34816/5120` | `M<=256` | `0.803834` | `222.186644` | `7.984629` |
| `4/5120/17408` | `M<=256` | `0.414576` | `215.312723` | `7.987943` |

The fresh M4 results are slightly slower than M1 on the two large live FFN shapes, so the previously considered padded-M shape lift cannot defend the speed gate. The two M1 FFN shapes already estimate about `216-224 GB/s`, so pure caller/cache/schedule overhead is too small relative to the required `21.932 ms/token` saving.

## Compute/Bandwidth Accounting

| Required field | Accounting |
|---|---|
| representative shape(s) as M/N/K | Gate/up packed FFN `M=1,N=34816,K=5120`; down FFN `M=1,N=5120,K=17408`; microbench also checked M4 shape-lift variants. |
| FLOPs per token or per GEMM | Gate/up M1: `2*1*34816*5120 = 356.516 MFLOP`; down M1: `2*1*5120*17408 = 178.258 MFLOP`. |
| estimated bytes moved | Gate/up M1 total `178.376 MB`, B weights `178.258 MB`; down M1 total `89.179 MB`, B weights `89.129 MB`. |
| arithmetic intensity | About `1.999 FLOP/B` for both M1 FFN GEMMs; M4 shape-lift raises launched arithmetic intensity to about `7.99 FLOP/B` but did not improve event time. |
| GB10 roofline/ceiling comparison | At `273 GB/s`, one M1 gate/up B stream is at least `0.653 ms`; microbench is `0.797 ms`. One M1 down B stream is at least `0.326 ms`; microbench is `0.413 ms`. These are bandwidth-sensitive and already a large fraction of GB10 LPDDR ceiling. |
| current `ffn_linear` ms/token proxy | Warm-mapped `26.319 ms/token` using the strategy 20.0% share; P3a source proxy is `80.597 ms/requested output token`. |
| expected changed bytes/FLOPs/overhead | No defended legal mutation changes B-weight bytes, FLOPs, output bytes, dtype/layout, or scale semantics. Schedule/caller/padding/fused-activation variants only change overhead around an unchanged B stream, and prior controller rows measured those near baseline or below the speed gate. |
| expected end-to-end tok/s delta if the hypothesis is right | Best remaining legal overhead-only mechanisms would need to save `21.932 ms/token`, i.e. `83.3%` of the FFN proxy or `57.5%` of all listed linear time. The evidence does not support that, so expected defended delta is below the required `+1.520 tok/s` to reach `9.119 tok/s`. |

## Low-Level Evidence Block

| Required field | Evidence |
|---|---|
| source file/symbol | `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh`, `sm120_blockwise_fp8_config_M64` and `cutlass_gemm_blockwise_sm120_fp8_dispatch`; caller in `c3x/cutlass_gemm_caller.cuh`; Python route in `vllm/model_executor/layers/quantization/utils/fp8_utils.py`, `W8A8BlockFp8LinearOp._run_cutlass`. |
| live-shape dispatch-hit proof | Warm diagnostic is decode/concurrency-1; `W8A8BlockFp8LinearOp.apply` flattens to tiny M; block FP8 2D scales select blockwise CUTLASS; C++ dispatch selects `M <= 256`. |
| before-mutation observation | Warm diagnostic `7.599 tok/s`, `131.594103 ms/token`; microbench M1 live FFN shapes `0.796723 ms` and `0.412848 ms`, estimated `223.887` and `216.009 GB/s`. M4 shape-lift is not faster. |
| byte-component split for A/B weights/scales/output/epilogue | `M=1,N=34816,K=5120`: A about `0.005 MB`, B `178.258 MB`, A scales about `0.00016 MB`, B scales about `0.005 MB`, BF16 output about `0.070 MB`, total `178.376 MB`. `M=1,N=5120,K=17408`: A about `0.017 MB`, B `89.129 MB`, A scales about `0.00054 MB`, B scales about `0.0027 MB`, BF16 output about `0.010 MB`, total `89.179 MB`. Epilogue traffic is included in the output estimate and is negligible compared with B. |
| whether B-weight bytes change | No legal CUTLASS-internal schedule, caller, padding, or scale-placement change reduces B-weight bytes. Reducing B streaming would require changing the operator boundary to own multiple projections/tokens or changing weight dtype/layout/scale semantics, which is forbidden. |
| why the expected lift is at least 20% end-to-end | It is not. The measured and source-level evidence show remaining legal mechanisms do not reduce the dominant B stream and cannot credibly save the required `21.932 ms/token`. |

## Broader Mechanism Check

- Paired projection reuse: Qwen3.5 already packs same-input projections via `packed_modules_mapping`: `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba`.
- FFN down fusion with gate/up: down projection consumes `SiluAndMul(gate, up)` and a distinct B matrix. Reusing or avoiding the down B stream requires a new FFN operator boundary that owns gate/up, activation, and down projection together; that violates the public operator/signature boundary for this CUTLASS FP8 GEMM mutation.
- Fused `SiluAndMul + quant + down CUTLASS`: prior same-machine candidate 009 implemented this broader mechanism, preserved `ops.cutlass_scaled_mm`, passed parity, and was discarded with objective `0.045162`; it did not produce a material warm decode gain.
- M1 padded shape-lift: prior candidate 024 was rejected by the post-parity generation speed gate, and this iteration's microbench shows M4 is slightly slower than M1 on the large FFN shapes.
- Persistent/reuse staging across tokens: the public `cutlass_scaled_mm` op is stateless per GEMM call. The model's FP8 B weights are effectively a full-model stream; GB10 cache capacity cannot hold the many distinct 27B-model B matrices across layers/tokens. A persistent multi-token kernel would need a new runtime scheduling/operator contract, not a local CUTLASS dispatch mutation.
- Launch reduction across adjacent linears: obvious same-input launches are already packed; remaining projection and down paths have data dependencies or target-specific state semantics outside `cutlass_scaled_mm`.

Conclusion: under the required parity and CUTLASS-only constraints, the exact source boundary preventing a 20% mutation is the per-call `cutlass_scaled_mm(A, B, scale_a, scale_b, out_dtype, bias)` contract plus the model-level separation of nonlinear FFN down projection and stateful attention/deltanet paths. Legal edits preserve B-weight bytes, and the live workload is B-stream dominated on GB10 LPDDR.
