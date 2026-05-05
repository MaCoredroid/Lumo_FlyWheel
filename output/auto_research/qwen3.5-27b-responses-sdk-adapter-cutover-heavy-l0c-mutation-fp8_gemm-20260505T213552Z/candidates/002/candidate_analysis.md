# Candidate 002 Analysis

Decision: blocked, no `mutation.patch` submitted.

## Baseline Timing

Current warm diagnostic artifact: `candidates/002/warm_pre_mutation.json`.

- Aggregate warm decode: `7.536 generated tok/s`, `132.695675 ms/generated token`.
- Per-step request 1: `7.531 tok/s`, `132.782102 ms/token`, `prompt_tokens_per_request=1255.0`, `generation_tokens_per_request=64.0`, `prefill_ms_per_kv_token=0.579466`, `cache_hit_rate_pct=62.47011952191235`, bottleneck hint `decode`.
- Per-step request 2: `7.541 tok/s`, `132.609247 ms/token`, `prompt_tokens_per_request=1252.0`, `generation_tokens_per_request=64.0`, `prefill_ms_per_kv_token=0.581539`, `cache_hit_rate_pct=62.61980830670927`, bottleneck hint `decode`.
- Aggregate GB10 context: `128 GB unified LPDDR5x`, `273.0 GB/s` theoretical bandwidth, `10.111 tok/s` full-model FP8 stream ceiling for 27B params, `36225919139.411 bytes/generated token` bandwidth budget at the observed decode time. This is roofline context, not profiler-measured DRAM bandwidth.
- Required 20% post-parity speed gate from this baseline: `9.0432 tok/s`, or about `110.580 ms/token`; required saving is about `22.116 ms/token`.

Strategy-brief CUDA-event proxy:

- `ffn_linear`: `20.0%`, `80.597 ms/requested output token` in the P3a timing source.
- Mapped onto current warm decode, the `ffn_linear` proxy is `26.539 ms/token`; non-FFN residual is `106.157 ms/token`.
- The broader listed projection-linear proxy adds `deltanet_projection_linear=7.0%` and `gatedattn_projection_linear=2.0%`, so all listed linear leaves are about `38.482 ms/token`. The 20% speed gate would require saving about `57.5%` of that broad linear proxy, or `83.3%` of FFN alone.

7.5 tok/s breakdown line:

- `7.5 tok/s` implies `133.333 ms/token`.
- A 27B-param FP8 full-model stream at `7.5 tok/s` implies `202.5 GB/s`, `74.2%` of the `273 GB/s` GB10 ceiling. The current `7.536 tok/s` implies `203.472 GB/s`, `74.5%` of that ceiling under the same full-model-stream assumption.
- The full-model FP8 stream ceiling is `10.111 tok/s`.
- `ffn_linear` at `20.0%` is `26.667 ms/token`; non-FFN residual is `106.667 ms/token`.
- A CUTLASS-only mutation must either reduce bandwidth traffic materially or erase almost all FFN/projection overhead; local schedule or caller overhead edits do not attack enough residual.

No lower-level live CUTLASS sub-kernel timer is available in this authoring pass. The requested shape microbench was started but was still rebuilding the vLLM `_C` extension after about five minutes, so `candidates/002/cutlass_microbench_pre.json` records a bounded skip. I use the nearest prior same-machine shape microbench records for the same SM120 blockwise path as the low-level proxy.

## Structured Compute/Bandwidth Accounting

Representative decode shapes come from the strategy brief and prior same-machine microbench records. Bytes assume FP8 A/B, FP32 block scales, and BF16 output.

| representative shape M/N/K | FLOPs per GEMM | estimated bytes moved | arithmetic intensity | GB10 roofline/ceiling comparison | current `ffn_linear` ms/token proxy | expected changed bytes/FLOPs/overhead | expected end-to-end tok/s delta |
|---|---:|---:|---:|---|---:|---|---|
| `1/34816/5120` gate/up-style | `356.516 MFLOP` | `178.376 MB` | `1.999 FLOP/B` | B-stream dominated; lower bound `0.653 ms` at 273 GB/s | `26.539 ms/token` | Legal CUTLASS tile/schedule/caller edits keep B bytes `178.258 MB`; A, scales, and output are under `0.12 MB` | below 20%; cannot remove the required `22.116 ms/token` |
| `1/5120/17408` down-style | `178.258 MFLOP` | `89.179 MB` | `1.999 FLOP/B` | B-stream dominated; lower bound `0.327 ms` at 273 GB/s | `26.539 ms/token` | Legal CUTLASS tile/schedule/caller edits keep B bytes `89.129 MB`; A, scales, and output are under `0.05 MB` | below 20%; down-fusion can only remove small activation/quant traffic and launches |
| `4/34816/5120` batched decode | `1.426 GFLOP` | `178.601 MB` | `7.985 FLOP/B` | still dominated by shared B stream for small M | `26.539 ms/token` | B bytes unchanged | below 20% |
| `4/5120/17408` batched decode | `713.032 MFLOP` | `89.263 MB` | `7.988 FLOP/B` | still dominated by shared B stream for small M | `26.539 ms/token` | B bytes unchanged | below 20% |

Byte split:

- `M=1,N=34816,K=5120`: A `5.120 KB`, B weights `178.258 MB`, A scales `160 B`, B scales `43.520 KB`, output/epilogue store `69.632 KB`, total `178.376 MB`.
- `M=1,N=5120,K=17408`: A `17.408 KB`, B weights `89.129 MB`, A scales `544 B`, B scales `21.760 KB`, output/epilogue store `10.240 KB`, total `89.179 MB`.
- FFN activation plus down-quant fusion would additionally avoid roughly `34.816 KB` BF16 activation output write, `34.816 KB` BF16 quant input read, `17.408 KB` FP8 quant output write, and `544 B` scale write for the `M=1,K=17408` down path, plus one or two launches. This is small next to the `89.129 MB` compulsory B stream and does not change GEMM FLOPs.

Nearest prior same-machine shape microbench records for this exact path:

- `M=1,N=34816,K=5120`: `event_ms_mean=0.833901`, `estimated_effective_bandwidth_gb_s=213.906`, `arithmetic_intensity=1.998672`.
- `M=1,N=5120,K=17408`: `event_ms_mean=0.685587`, `estimated_effective_bandwidth_gb_s=130.077`, `arithmetic_intensity=1.998880`.
- `M=4,N=34816,K=5120`: `event_ms_mean=0.886848`, `estimated_effective_bandwidth_gb_s=201.389`, `arithmetic_intensity=7.984629`.
- `M=4,N=5120,K=17408`: `event_ms_mean=0.766227`, `estimated_effective_bandwidth_gb_s=116.497`, `arithmetic_intensity=7.987943`.

## Low-Level Evidence

| required field | evidence |
|---|---|
| source file/symbol | `vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py::W8A8BlockFp8LinearOp._run_cutlass`; `vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp::dispatch_scaled_mm`; `vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh::cutlass_gemm_blockwise_sm120_fp8_dispatch`; broader FFN fusion surface `vllm-source/vllm/model_executor/models/qwen2.py::Qwen2MLP.forward` inherited by `qwen3_5.py` as `Qwen3NextMLP` |
| live-shape dispatch-hit proof | `Qwen3_5DecoderLayer` uses `Qwen3NextMLP`, imported from `Qwen2MoeMLP`, for dense `qwen3_5_text`; dense Qwen MLP uses `gate_up_proj -> SiluAndMul -> down_proj`. The block-FP8 linear path quantizes activations then calls `cutlass_scaled_mm`; local `cutlass_scaled_mm` passes `B.T` and `Bs.T` to `ops.cutlass_scaled_mm`; `dispatch_scaled_mm` selects blockwise when both scale tensors are 2D and validates A scales `[M,ceil(K/128)]`, B scales `[ceil(K/128),ceil(N/128)]`; `cutlass_gemm_blockwise_sm120_fp8_dispatch` sends all decode `M <= 256` to `sm120_blockwise_fp8_config_M64` with `Shape<_64,_128,_128>` |
| before-mutation observation | Current warm diagnostic: `7.536 tok/s`, `132.695675 ms/token`, bottleneck `decode`; prior same-machine microbench shows representative `M<=256` CUTLASS ops are B-stream dominated; current round candidate 001 already blocked adjacent local-schedule and B-stream-neutral ideas |
| byte-component split for A/B weights/scales/output/epilogue | `M1,N34816,K5120`: A `5.120 KB`, B `178.258 MB`, A scales `160 B`, B scales `43.520 KB`, output/epilogue store `69.632 KB`; `M1,N5120,K17408`: A `17.408 KB`, B `89.129 MB`, A scales `544 B`, B scales `21.760 KB`, output/epilogue store `10.240 KB` |
| whether B-weight bytes change | No. Existing legal CUTLASS dispatch, schedule, tile, scale-layout-preserving, activation-quant fusion, and caller-launch edits preserve the compulsory B stream. |
| why the expected lift is at least 20% end-to-end | It is not. The evidence does not support a 20% end-to-end lift. The speed gate needs `22.116 ms/token` saved. FFN is only `26.539 ms/token`; all listed linear leaves are `38.482 ms/token`. Remaining legal mechanisms leave B bytes unchanged and prior adjacent measured trials missed the speed gate. |

## Broader Mechanisms Checked

- Paired-projection reuse: Qwen3.5 already packs `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba` in `Qwen3_5ForCausalLMBase.packed_modules_mapping`. The obvious same-input B streams are already fused at the linear-module level.
- FFN gate/up plus down fusion: `gate_up_proj` is one GEMM, but `down_proj` consumes `SiluAndMul(gate_up)` and a distinct B matrix. Reducing the two B streams would require a new FFN operator that owns both matrices and activation semantics, not a drop-in `cutlass_scaled_mm` mutation.
- Activation plus down-input quant fusion: local `fp8_utils.py::silu_mul_per_token_group_quant_fp8_colmajor` shows the natural fusion surface, but it currently asserts `input.size(0) % 128 == 0` and `M % 8 == 0`, so it is not directly usable for decode `M=1`. A new masked fused activation+quant kernel could preserve the subsequent CUTLASS GEMM, but it would not change the `89.129 MB` down B stream. It only removes under `0.09 MB` of M1 activation/quant traffic plus launches per dense FFN layer, which cannot defend the required `22.116 ms/token` end-to-end saving.
- Persistent/reuse staging across tokens: the public op is one GEMM at a time and the full B-weight working set is much larger than cache; there is no CUTLASS source surface that preserves the public dtype/layout/scale contract while staging a layer's B weights across future token requests.
- Launch reduction: prior same-machine trials measured workspace/hardware-info caching, zero-workspace, M1 branch/padding, fixed stage counts, epilogue tile, swap-AB, Python quant cache, and multiple tile/scheduler variants. Compile-clean variants passed parity but missed the speed gate.
- New specialized CUTLASS route: a true GEMV or fused FFN route that materially reduces B bytes would need a new public operator boundary with two weight matrices or a changed weight/scale layout. That violates the current `cutlass_scaled_mm(A, B, scale_a, scale_b, out_dtype, bias)` boundary and/or the round's public signature/layout/scale constraints.

Primary/source research facts used:

- NVIDIA CUTLASS documentation describes GEMM as hierarchical CTA/warp/instruction tiling and says small M/N with large K can underutilize hardware; split-K/parallel reductions increase parallelism but require an extra reduction kernel/workspace. Source: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- The same CUTLASS documentation frames pipelining/persistent-style schedules as mechanisms to overlap memory/prologue overhead, not mechanisms that remove compulsory operand B bytes. Source: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- NVIDIA CUTLASS Blackwell documentation says SM120 block-scaled narrow-precision GEMMs use the same scale-factor layout as SM100 and fixed `1x1x1` cluster shape on GeForce-series SM120. This matches the local `ClusterShape = Shape<_1,_1,_1>` and scale-layout constraints. Source: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

## Conclusion

No `mutation.patch` is submitted. The only broader CUTLASS-backed mechanism that was not just another schedule knob, activation plus down-input quant fusion, does not reduce B-weight bytes and cannot plausibly beat the `9.0432 tok/s` speed gate. A patch would spend controller validation on a candidate whose best-defended effect is launch/small-activation overhead, while the observed decode residual is dominated by non-FFN work and compulsory model-weight streaming.
