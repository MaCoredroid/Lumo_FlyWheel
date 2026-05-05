# L0c FP8 CUTLASS Auto-Research Closeout - 2026-05-05

## Round

- Round id: `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z`
- Target: `fp8_gemm`, CUTLASS only
- Controller outcome: `ROUND_BLOCKED`
- Terminal condition: `compile_failures_3x`
- Wall clock: 81.83 minutes
- Attempts: 29 total, 1 accepted into paired measurement, 15 rejected or blocked
- Remeasured baseline objective mean: `0.056847`
- Best measured candidate objective mean: `0.042228`
- Conclusion: no candidate beat the remeasured baseline.

Primary artifact directory:

`output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z`

Key artifacts:

- `results.tsv`
- `measurements.tsv`
- `mutations_rejected.tsv`
- `research_memory.tsv`
- `research_memory.md`
- `measurement_trace_combined.json`
- `candidates/*/candidate_analysis.md`
- `candidates/*/BLOCKED.md`

## Controller Changes Validated By This Round

The new post-parity warm-generation speed gate worked as intended. After correctness/parity admission, the controller ran a cheap warm generation diagnostic and discarded candidates before paired measurement unless patched decode throughput exceeded the candidate's warm pre-mutation baseline by 3%.

The gate rejected three candidates before expensive paired measurement:

| Candidate | Baseline tok/s | Candidate tok/s | Required tok/s | Result |
| --- | ---: | ---: | ---: | --- |
| 020 | 7.368 | 7.381 | 7.58904 | rejected |
| 024 | 7.368 | 7.382 | 7.58904 | rejected |
| 026 | 7.369 | 7.387 | 7.59007 | rejected |

This prevented spending full apply-and-test windows on changes that were only `0.18%` to `0.24%` above warm baseline, far below the configured `3%` margin.

The authoring-agent split also worked: candidates 027, 028, and 029 did not run expensive apply-and-test. Candidate 027 attempted a targeted compile preflight, saw the C++ instantiation failure, removed `mutation.patch`, and wrote `BLOCKED.md`. Candidates 028 and 029 blocked without submitting low-evidence patches.

## Candidate Analysis

Measured candidate:

- 003: persistent scheduler plus hardware-info handling reached paired measurement but was discarded. Objective mean was `0.042228` versus baseline `0.056847`. This confirmed that the adjacent schedule/caller surface did not improve the real objective.

Speed-gate failures:

- 020: added an M==1 blockwise CUTLASS branch. Correctness was admitted under the FP8 tier-4 diagnostic policy, but warm decode improved only from `7.368` to `7.381 tok/s`.
- 024: routed SM120 M==1 block-FP8 Python path through padded CUTLASS. Warm decode improved only from `7.368` to `7.382 tok/s`.
- 026: added a `MainloopStageCount` knob and routed the hot decode specialization through `StageCount<3>`. Warm decode improved only from `7.369` to `7.387 tok/s`.

Compile or authoring blocked:

- 027: tried the highest-upside new direction from CUTLASS docs: keep the epilogue on `OpClassTensorOp` while changing the SM120 blockwise mainloop builder tag to `OpClassBlockScaledTensorOp`. Targeted compile failed because the generated `GemmKernel` did not expose the vLLM caller's expected `MainloopArguments` and `EpilogueArguments`. The agent removed the patch and recorded the failure.
- 028: blocked after finding no remaining compile-clean, parity-safe CUTLASS surface with evidence for the required `>=3%` end-to-end lift.
- 029: blocked after confirming the live vLLM FP8 path uses FP32 block scales through the current SM120 blockwise mainloop. The documented MX/NV block-scaled `OpClassBlockScaledTensorOp` path is not a semantics-preserving swap for this FP32-scale contract.

## Bottleneck Read

The warm request diagnostics consistently show decode around `7.36` to `7.39 tok/s`, or about `135.5` to `135.9 ms/token`.

The strategy brief's CUTLASS proxy is `ffn_linear` at 20% of decode time. At the observed warm speed:

- token time: about `135.85 ms/token`
- `ffn_linear` proxy: about `27.17 ms/token`
- non-FFN residual: about `108.68 ms/token`
- 3% speed gate target: about `7.58 to 7.59 tok/s`
- required savings: about `3.95 ms/token`
- implied required improvement if only FFN changes: about `14.5%` of the whole `ffn_linear` proxy

Representative decode GEMMs remain B-weight streaming dominated:

- `M=1, N=34816, K=5120`: about `356.5 MFLOP`, about `178.3 MB` B weights, about `2.0 FLOP/byte`
- `M=1, N=5120, K=17408`: about `178.3 MFLOP`, about `89.1 MB` B weights, about `2.0 FLOP/byte`

On GB10/DGX Spark, the round's diagnostic context is 128 GB unified LPDDR5x and roughly 273 GB/s theoretical bandwidth. At `7.5 tok/s`, a 27 GB/token FP8 stream implies about `202.5 GB/s`, or about 74% of that theoretical ceiling. The observed 7.36 tok/s behavior is therefore consistent with a bandwidth-sensitive decode path, not a host-launch dominated path.

## Online/Source Research Notes

- NVIDIA CUTLASS Blackwell documentation describes `tcgen05.mma` narrow precision and block-scaled GEMMs, including the MX/NV block-scaled forms and `OpClassBlockScaledTensorOp`: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- The same source makes the important distinction that MX/NV block-scaled GEMMs use narrow scale-factor types and scale-vector semantics. The current vLLM blockwise FP8 path in this round uses FP32 scale pointers, so directly swapping to the MX/NV block-scaled operator class is not semantics preserving.
- CUTLASS 4.2.1 release notes mention SM120 blockwise GEMM support and Blackwell SM120 mixed-input blockscaled grouped GEMM support, but that does not by itself provide a compile-clean replacement for vLLM's current FP32-scale path: https://docs.nvidia.com/cutlass/4.2.1/overview.html
- The Karpathy/autoresearch pattern argues for a bounded same-machine loop with a fixed metric and keep/discard gate. This round now follows that more closely: authoring agents produce one candidate or block, cheap gates reject weak candidates, and memory records each failed diff plus the failure class. Reference: https://autoresearch.lol/

## Memory And Guidance Quality

The loop memory now records:

- patch diff excerpts
- failure class: performance, build, preflight safety, authoring, or context
- controller gate: compile, parity, speed gate, or measurement
- speed-gate failures with exact baseline, candidate, and required tok/s
- next-search implications such as `deprioritize_until_speed_gate_hypothesis_changes`

This is aligned with AutoTVM/Ansor/MetaSchedule/OpenTuner style measured-trial memory: each row is a workload-keyed experiment record, not just prose.

The live guidance also successfully forced agents to:

- run warm diagnostics before mutating
- include compute/bandwidth accounting
- include low-level dispatch evidence
- use online/source research before choosing a mutation
- avoid expensive apply-and-test
- own targeted compile preflight and block on compile failure

## Recommendation

Do not spend another near-term loop on the same CUTLASS-only FP32-scale blockwise FP8 surface without adding a new lever. The round exhausted the obvious compile-clean schedule/tile/stage/caller/padding/cache surfaces, and the only new source-backed high-upside operator-class direction compile-blocked or changed scale semantics.

The next useful loop should change the available surface first:

1. Add a real microbenchmark/autotuning harness for the exact vLLM SM120 FP32-scale blockwise FP8 path, including M/N/K distributions from warm decode. This should produce candidate-level CUTLASS event timings without full vLLM restart.
2. Add a semantics-preserving C++ dispatch knob for a small set of known-legal SM120 FP8 blockwise alternatives only if the microbench shows a >=15% `ffn_linear` proxy win.
3. Consider a broader serving-level loop, because the current CUTLASS proxy is only 20% of decode time and the 3% end-to-end target requires a large improvement in that slice.
4. If staying CUTLASS-only, investigate whether a separate FP32-scale Blackwell blockwise collective exists upstream or can be integrated cleanly. Do not retry MX/NV `OpClassBlockScaledTensorOp` as a direct swap for the current FP32-scale path.

More auto-research loops are worthwhile only after one of those surface changes is in place. Running the same current surface again will likely produce more `BLOCKED.md` rows or sub-1% speed-gate failures.
