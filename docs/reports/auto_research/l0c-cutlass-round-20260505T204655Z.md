# L0c CUTLASS Auto-Research Round 20260505T204655Z

## Outcome

- Round directory: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z`
- Kernel target: `fp8_gemm`
- Harness: real vLLM
- Outcome: `ROUND_BLOCKED`
- Terminal condition: `compile_failures_3x`
- Accepted candidates: 0
- Rejected candidates: 3
- Baseline objective mean: `0.056913`
- Baseline rollout throughput: about `112.1-112.6` rollout tok/s in measured windows
- Warm decode baseline seen by agents: `7.514-7.603` generated tok/s
- 20% warm speed gate: about `9.017-9.124` generated tok/s, depending on iteration warm diagnostic

No candidate reached controller apply/parity/full measurement because all three authoring agents wrote `BLOCKED.md` and no `mutation.patch`.

## Candidate Summary

| Iteration | Artifact | Warm decode | Result | Main reason |
|---|---:|---:|---|---|
| 001 | `candidates/001` | `7.603` tok/s | blocked, no patch | Local schedule/tile edits could not plausibly remove enough of the 20% `ffn_linear` proxy to clear the 20% end-to-end gate. |
| 002 | `candidates/002` | `7.568` tok/s | blocked, no patch | Same B-weight traffic argument; fallback same-machine microbench showed representative shapes around `209-225 GB/s`. |
| 003 | `candidates/003` | `7.514` tok/s | blocked, no patch | Microbench completed on GB10 and confirmed `M<=256` SM120 blockwise path; legal schedule variants preserve dominant B-weight bytes. |

## Candidate 003 Evidence

The useful new artifact is `candidates/003/cutlass_microbench_pre.json`. It compiled and ran on `NVIDIA GB10` with capability `12.1`.

| Shape M/N/K | Event ms mean | Estimated bandwidth | Arithmetic intensity |
|---|---:|---:|---:|
| `1/34816/5120` | `0.833901` | `213.906 GB/s` | `1.998672` FLOP/B |
| `1/5120/17408` | `0.685587` | `130.077 GB/s` | `1.998880` FLOP/B |
| `4/34816/5120` | `0.886848` | `201.389 GB/s` | `7.984629` FLOP/B |
| `4/5120/17408` | `0.766227` | `116.497 GB/s` | `7.987943` FLOP/B |

The representative decode GEMMs are dominated by B-weight streaming:

- `M=1,N=34816,K=5120`: about `178.376 MB` moved, with about `178.258 MB` B weights.
- `M=1,N=5120,K=17408`: about `89.179 MB` moved, with about `89.129 MB` B weights.

The agent mapped the current warm decode `133.092 ms/token` to the strategy `ffn_linear` proxy at `20%`, or about `26.62 ms/token`. Clearing the 20% gate requires about `22.18 ms/token` saved, so a CUTLASS-only mutation would need to remove most of the FFN proxy unless it also changes a broader linear/fusion/reuse mechanism.

## Online / Prior-Art Notes

- NVIDIA CUTLASS 4.2.1 documents DGX Spark / SM121 support and says SM121 shares major code with Blackwell SM120. It also notes Blackwell SM120 blockwise GEMM support.
- NVIDIA's CUTLASS 3.x design article places tile scheduling at the `GemmUniversal` kernel layer and distinguishes persistent/Stream-K scheduling from basic one-CTA-per-output-tile scheduling. It also notes Blackwell uses Cluster Launch Control schedulers.
- Those sources support schedule mutation as a real CUTLASS mechanism, but this round's local scale constraints and prior compile failures make the high-upside schedule variants invalid for the current vLLM blockwise FP8 path.

Sources:

- https://docs.nvidia.com/cutlass/4.2.1/overview.html
- https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/

## Controller / Prompt Changes Made

- Committed and pushed `64f76b1 Raise L0c speed gate to 20 percent`.
- Committed and pushed `765d447 Guide L0c agents toward byte mechanisms`.
- The live round `strategy_brief.md` was patched to tell later agents not to repeat the same local-schedule byte-limit block unless they first check broader CUTLASS-backed byte mechanisms such as caller-level fusion, persistent/reuse staging, paired-projection reuse, launch reduction, or a specialized route that reduces B-weight streaming.

## Recommendation

Do not immediately run another identical CUTLASS schedule-only loop. With the 20% gate, local schedule/tile/epilogue changes are unlikely to clear the threshold because the measured CUTLASS proxy is too small and B-weight bytes dominate.

Run another loop only after expanding the mutation surface in one of these directions:

1. CUTLASS-backed caller-level fusion or paired-projection reuse that reduces repeated B-weight streaming or launch count.
2. A persistent/reuse mechanism with a clear source boundary and local compile preflight.
3. A custom decode-specialized CUTLASS route that changes the byte mechanism while preserving dtype/layout/scale/output parity.
4. A better cheap preflight that avoids rebuilding full `_C` for every agent and exposes the compiled microbench cache/result to future agents.

The next prompt should make "submit a byte-mechanism patch or identify the exact source contract that prevents it" the required action. Another local CTA-shape sweep is low value under the current evidence.
