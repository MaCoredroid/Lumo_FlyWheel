# L0c FP8 CUTLASS Auto-Research Loop Report - 2026-05-05

## Scope

This round followed `docs/reports/auto_research/l0-ffn-gemm-pivot-20260502.md` for CUTLASS-only L0c mutation on the Qwen3.5 27B FP8 responses SDK adapter workload. DeltaNet was kept out of scope.

Round artifact:

`output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z`

## Result

The controller terminated with `ROUND_NULL_RESULT`.

| Metric | Value |
| --- | ---: |
| Terminal condition | `accepted_cap_reached` |
| Total attempts | 40 |
| Accepted candidates | 24 |
| Rejected candidates | 8 |
| Paired baseline objective mean | 0.0568122 |
| Best candidate objective mean | 0.045434 |
| Bundle path | null |

No candidate beat the paired baseline. No tuned config bundle was defensible.

## Candidate Summary

The late-round candidates are the most relevant because they used the strengthened warm decode and compute/bandwidth analysis contract.

| Candidate | Direction | Measurements | Mean | Outcome |
| --- | --- | ---: | ---: | --- |
| 035 | M16 schedule changed to SM120 pingpong `<2>` | 0.034261, 0.056144 | 0.0452025 | discard |
| 036 | Guarded small-M/large-N tile path, `16x256x128` | 0.034120, 0.056315 | 0.0452175 | discard |
| 037 | Claimed cooperative direction, patch actually changed M16/M32/M64 to pingpong `<1>` | 0.034008, 0.056031 | 0.0450195 | discard |
| 038 | M<=4 dispatch to M8/K256 tile | 0.033971, 0.010687 | 0.022329 | discard, performance regression |
| 039 | Persistent scheduler on small-M SM120 custom wrapper | 0.033981, 0.056140 | 0.0450605 | discard |
| 040 | Scalar-scale epilogue specialization for SM120 dense FP8 | 0.029618, 0.048929 | 0.0392735 | discard |

All six passed the current FP8 GEMM parity policy. Tier-4 downstream-logit divergence remained diagnostic only after Tier-3 GEMM-output compare, with the repeated overshoot around `0.35653125`.

## Compute/Bandwidth Finding

The agents did start producing useful warm decode diagnostics, but not yet a strong enough low-level CUTLASS timing breakdown. The repeated warm-path numbers were stable:

- Warm decode throughput stayed near 7.3-7.5 tok/s for one request.
- Token time stayed near 135 ms/token.
- The analysis repeatedly estimated roughly 37 GB/generated token and an LPDDR effective ceiling near 10 tok/s from the full-model stream.
- The reported `ffn_linear` proxy was about 80.6 ms/token in the late candidates.

That is enough to reject many small mutations, but not enough to prove the exact CUTLASS kernel-level bottleneck. The current evidence says B-weight streaming dominates and most tested mutations left the B byte stream unchanged. Schedule, tile, and epilogue edits that only reduce CTA overhead, scale visitor overhead, or A-side reload overhead need a much larger measured per-kernel delta before they can plausibly move end-to-end warm decode.

## Loop Quality

Operationally, the loop is now closer to the intended auto-research design:

- Authoring agents run cheap patch checks, compile-oriented preflight, and warm diagnostics.
- The controller owns expensive apply-and-test, vLLM restart, parity, and measurement.
- Failed candidates are recorded with patch hash and rejection reason.
- Resumed candidates now go through the same `candidate_analysis.md` gate before expensive validation.

Scientifically, the loop is only partly on track. It has better memory and filtering, but the late mutations still leaned too much on plausible CUTLASS template changes rather than measured kernel-level bottleneck removal.

## Online Research Alignment

NVIDIA's CUTLASS documentation confirms that Blackwell SM120 supports pingpong and cooperative schedules, and that valid tile shapes and dispatch policies are constrained by architecture and layout. It also says `KernelScheduleAuto` selects cooperative by default for SM120, which explains why schedule mutation is a real but shape-sensitive surface:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

The CUTLASS 3.x design notes describe the kernel as a composition of mainloop and epilogue, with tile scheduling controlling work distribution. Persistent and Stream-K scheduling are established concepts, but on Blackwell the relevant scheduler story involves Cluster Launch Control:

- https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/
- https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/blackwell.html

Karpathy's AutoResearch loop uses fixed-budget experiments, a single comparable metric, and logs/results as memory. The lesson for this workload is that we need the same ratchet, but with stronger negative memory and cheap low-level measurements before expensive full vLLM rebuilds:

- https://github.com/karpathy/autoresearch

## Recommendation

Run another CUTLASS-only loop only if the prompt and preflight require a stronger measurement package before mutation:

1. Require a before/after low-level CUTLASS microbenchmark or profiler-derived timing for the exact M/N/K path the patch claims to affect.
2. Reject mutations with expected end-to-end lift below 3-5% unless they remove a known correctness blocker or isolate a repeated uncertainty.
3. Require patch-analysis consistency checks. Candidate 037 is the warning case: the written analysis and actual diff diverged.
4. Require the agent to state which prior failures it is avoiding, including 028, 031, 038, 039, and 040.
5. Require byte math that separates A activations, B weights, scale loads, output stores, and epilogue overhead. A mutation that leaves B bytes unchanged needs direct proof that non-B overhead is the bottleneck.

If the goal is a significant increase from the observed 7.5 tok/s on this machine, this round did not produce evidence that small CUTLASS FP8 schedule/tile/epilogue mutations are sufficient. The next loop should either demand stronger CUTLASS kernel-level proof before full validation, or, when allowed, widen the surface beyond CUTLASS because the late breakdown still points to substantial non-FFN time.

## Artifact Index

- Round spec: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/round_spec.yaml`
- Results: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/results.tsv`
- Measurements: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/measurements.tsv`
- Rejections: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/mutations_rejected.tsv`
- Combined trace: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/measurement_trace_combined.json`
- Candidate 040 trace: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z/candidates/040/measurement_trace.json`
