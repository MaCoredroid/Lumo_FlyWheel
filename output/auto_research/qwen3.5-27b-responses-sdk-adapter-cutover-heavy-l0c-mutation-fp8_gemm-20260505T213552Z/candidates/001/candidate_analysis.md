# Candidate 001 Analysis

Decision: blocked, no `mutation.patch` submitted.

## Baseline Timing

Current warm diagnostic artifact: `candidates/001/warm_pre_mutation.json`.

- Aggregate warm decode: `7.624 generated tok/s`, `131.160521 ms/generated token`.
- Per-step request 1: `7.626 tok/s`, `131.125860 ms/token`, `prompt_tokens_per_request=1255.0`, `generation_tokens_per_request=64.0`, `prefill_ms_per_kv_token=0.570977`, `cache_hit_rate_pct=62.47011952191235`, bottleneck hint `decode`.
- Per-step request 2: `7.622 tok/s`, `131.195182 ms/token`, `prompt_tokens_per_request=1252.0`, `generation_tokens_per_request=64.0`, `prefill_ms_per_kv_token=0.569991`, `cache_hit_rate_pct=62.61980830670927`, bottleneck hint `decode`.
- Aggregate GB10 context: `128 GB unified LPDDR5x`, `273.0 GB/s` theoretical bandwidth, `10.111 tok/s` full-model FP8 stream ceiling for 27B params, `35806822314.276 bytes/generated token` bandwidth budget at the observed decode time. This is roofline context, not measured DRAM throughput.
- Required 20% post-parity speed gate from this baseline: `9.1488 tok/s`, or about `109.300 ms/token`; required saving is about `21.860 ms/token`.

Strategy-brief CUDA-event proxy:

- `ffn_linear`: `20.0%`, `80.597 ms/requested output token` in the P3a timing source.
- Mapped onto current warm decode, the `ffn_linear` proxy is `26.232 ms/token`; non-FFN residual is `104.928 ms/token`.
- The broader listed projection-linear proxy adds `deltanet_projection_linear=7.0%` and `gatedattn_projection_linear=2.0%`, so even all listed CUTLASS-like linear leaves are about `38.037 ms/token`; the 20% speed gate would require removing about `57.5%` of that broad linear proxy, or `83.3%` of FFN alone.

7.5 tok/s breakdown line:

- `7.5 tok/s` implies `133.333 ms/token`.
- A 27B-param FP8 full-model stream at 7.5 tok/s implies `202.5 GB/s`, `74.2%` of the `273 GB/s` GB10 ceiling.
- The full-model FP8 stream ceiling is `10.111 tok/s`.
- `ffn_linear` at `20.0%` is `26.667 ms/token`; non-FFN residual is `106.667 ms/token`.
- A CUTLASS-only FFN schedule/dispatch patch must attack bandwidth traffic or nearly eliminate FFN overhead to clear a 20% end-to-end gate.

## Structured Compute/Bandwidth Accounting

Representative decode shapes come from the round brief and prior same-machine microbench records. Bytes assume FP8 A/B, FP32 block scales, and BF16 output.

| representative shape M/N/K | FLOPs per GEMM | estimated bytes moved | arithmetic intensity | GB10 roofline/ceiling comparison | current `ffn_linear` proxy | expected changed bytes/FLOPs/overhead | expected end-to-end tok/s delta |
|---|---:|---:|---:|---|---:|---|---|
| `1/34816/5120` gate/up-style | `356.516 MFLOP` | `178.376 MB` | `1.999 FLOP/B` | B-stream dominated; lower bound `0.653 ms` at 273 GB/s | `26.232 ms/token` | Legal tile/schedule/caller edits keep B bytes `178.258 MB`; A, scales, output are under `0.12 MB` | below 20%; roofline-only kernel upside is too small and affects only a slice |
| `1/5120/17408` down-style | `178.258 MFLOP` | `89.179 MB` | `1.999 FLOP/B` | B-stream dominated; lower bound `0.327 ms` at 273 GB/s | `26.232 ms/token` | Legal tile/schedule/caller edits keep B bytes `89.129 MB`; A, scales, output are under `0.05 MB` | below 20%; B bytes dominate |
| `4/34816/5120` batched decode | `1.426 GFLOP` | `178.601 MB` | `7.985 FLOP/B` | still dominated by shared B stream for small M | `26.232 ms/token` | B bytes unchanged | below 20% |
| `4/5120/17408` batched decode | `713.032 MFLOP` | `89.263 MB` | `7.988 FLOP/B` | still dominated by shared B stream for small M | `26.232 ms/token` | B bytes unchanged | below 20% |

Byte split:

- `M=1,N=34816,K=5120`: A `5.120 KB`, B weights `178.258 MB`, A scales `160 B`, B scales `43.520 KB`, output `69.632 KB`, total `178.376 MB`.
- `M=1,N=5120,K=17408`: A `17.408 KB`, B weights `89.129 MB`, A scales `544 B`, B scales `21.760 KB`, output `10.240 KB`, total `89.179 MB`.

The current run's microbench command was attempted but did not reach shape timing after more than eight minutes of rebuild, so `cutlass_microbench_pre.json` records it as skipped. Nearest prior same-machine observations for this exact path reported:

- `M=1,N=34816,K=5120`: `event_ms_mean=0.833901`, `estimated_effective_bandwidth_gb_s=213.906`, `arithmetic_intensity=1.998672`.
- `M=1,N=5120,K=17408`: `event_ms_mean=0.685587`, `estimated_effective_bandwidth_gb_s=130.077`, `arithmetic_intensity=1.998880`.
- `M=4,N=34816,K=5120`: `event_ms_mean=0.886848`, `estimated_effective_bandwidth_gb_s=201.389`, `arithmetic_intensity=7.984629`.
- `M=4,N=5120,K=17408`: `event_ms_mean=0.766227`, `estimated_effective_bandwidth_gb_s=116.497`, `arithmetic_intensity=7.987943`.

No lower-level CUTLASS sub-kernel timer is available in this authoring pass beyond the shape microbench records; the controller-owned `ffn_linear` proxy is the relevant live timing split.

## Low-Level Evidence

| required field | evidence |
|---|---|
| source file/symbol | `vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py::W8A8BlockFp8LinearOp._run_cutlass`; `vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp::dispatch_scaled_mm`; `vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh::cutlass_gemm_blockwise_sm120_fp8_dispatch` |
| live-shape dispatch-hit proof | `_run_cutlass` quantizes activations then calls `cutlass_scaled_mm`; local `cutlass_scaled_mm` passes `B.T` and `Bs.T` to `ops.cutlass_scaled_mm`; `dispatch_scaled_mm` selects blockwise when both scale tensors are 2D and validates `a_scale_group_shape=[1,128]`, `b_scale_group_shape=[128,128]`; `cutlass_gemm_blockwise_sm120_fp8_dispatch` sends all decode `M <= 256` to `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>` |
| before-mutation observation | Current warm diagnostic: `7.624 tok/s`, `131.160521 ms/token`, bottleneck `decode`; prior same-machine microbench shows the exact `M<=256` CUTLASS op on the representative shapes is B-stream dominated |
| byte-component split for A/B/scales/output/epilogue | `M1,N34816,K5120`: A `5.120 KB`, B `178.258 MB`, A scales `160 B`, B scales `43.520 KB`, output/epilogue store `69.632 KB`; `M1,N5120,K17408`: A `17.408 KB`, B `89.129 MB`, A scales `544 B`, B scales `21.760 KB`, output/epilogue store `10.240 KB` |
| whether B-weight bytes change | No legal schedule, tile, stage, workspace, SM-count, or caller dispatch mutation changes B-weight bytes. B remains the dominant compulsory stream. |
| why expected lift is at least 20% end-to-end | It is not. The evidence does not support a 20% end-to-end lift. With `ffn_linear` at `26.232 ms/token`, the 20% speed gate needs `21.860 ms/token` saved, or about `83.3%` of the full FFN proxy. Legal CUTLASS-internal edits preserve the dominant B stream and prior adjacent trials failed speed gates. |

## Broader Mechanisms Checked

- Caller-level fusion and paired-projection reuse: Qwen3.5 already packs `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba` in `Qwen3_5ForCausalLMBase.packed_modules_mapping`; `MergedColumnParallelLinear` and `QKVParallelLinear` concatenate output matrices along N. The obvious paired projections are already fused at the linear layer.
- FFN gate/up plus down fusion: `gate_up_proj` is one GEMM, but `down_proj` consumes a nonlinear `silu/gate` activation with a different weight matrix. Fusing it would require a new FFN operator boundary that takes two B matrices and activation semantics, not the current `cutlass_scaled_mm(A,B,scale_a,scale_b,out_dtype,bias)` contract.
- Persistent/reuse staging across decode tokens: B weights are model parameters far larger than cache, and the public op is called one GEMM at a time. There is no source surface that preserves the public signature while staging a layer's B weights across future token requests.
- Launch reduction: previous same-machine trials measured workspace, hardware-info, zero-workspace, prequantization cache, M1 branch/padding, fixed stage counts, epilogue tile, swap-AB, and tile/scheduler variants. They either compile-blocked or passed parity and missed the speed gate.
- New specialized CUTLASS route: a true GEMV or fused FFN route could only improve the non-B overhead unless it changes B streaming. A route that reduces B bytes would need to change dtype/layout/scale semantics, reuse two projection weights in one public op, or bypass the existing CUTLASS blockwise scaled-mm contract.

Primary/source research facts used:

- NVIDIA CUTLASS docs describe GEMM as a tiled loop over CTA M/N/K and note that small M/N with large K can underutilize hardware, while split-K/parallel reductions add extra reduction/workspace costs: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- The same docs describe persistent cooperative/ping-pong designs as tile-scheduler mechanisms for amortizing launch/prologue overhead, not mechanisms that remove compulsory B operand bytes: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- Local source enforces the live block-FP8 scale contract in `scaled_mm_helper.hpp`: A scales must be `[M, ceil(K/128)]`, B scales `[ceil(K/128), ceil(N/128)]`, and bias is unsupported for blockwise scaled-mm.

## Conclusion

No `mutation.patch` is submitted. A legal CUTLASS-only mutation in the remaining surfaces cannot be defended against the 20% warm speed gate because it does not reduce the compulsory B-weight stream and cannot plausibly remove the necessary `21.860 ms/token`.
