# L0c FP8 CUTLASS Auto-Research Round Report

Generated: 2026-05-03

Round:
`output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T021359Z`

## Executive Summary

This was a CUTLASS-only `fp8_gemm` L0c mutation round on the Phase A winner bundle:

`output/tuned_configs/responses-sdk-adapter-cutover-heavy/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260503T0120560000_30eb69ce.yaml`

Terminal outcome:

- `outcome`: `ROUND_BLOCKED`
- `terminal_condition`: `proposer_stuck`
- accepted candidates: `0`
- rejected candidates: `8`
- produced bundle: none

Recommendation: do not run another loop on the current repo-owned CUTLASS Python overlay surface. More loop budget will mostly generate wrapper/source-replacement variants that either fail downstream parity or are correctly preflight-demoted. The next useful work is to change the mutation surface: either implement the cheap isolated FP8 GEMM replay gate from the pivot doc, or move mutation into an actual vLLM/CUTLASS source-build surface where tile/layout/epilogue choices can be changed and measured.

## Baseline

The round measured the CUTLASS Phase A baseline with five real measurements after one cold-discard:

| Measurement | Eval throughput |
|---|---:|
| `baselines/measurement_01.json` | `0.056624` |
| `baselines/measurement_02.json` | `0.055872` |
| `baselines/measurement_03.json` | `0.057043` |
| `baselines/measurement_04.json` | `0.056681` |
| `baselines/measurement_05.json` | `0.056957` |

Mean: `0.0566354`.

Note: final `run_log.json` records `paired_baseline_objective_mean: 0.0`; the durable per-measurement JSON files above contain the valid measured baseline values. Treat the `run_log` baseline mean as a resume/finalization bookkeeping defect, not as the measured baseline.

## Candidate Outcomes

| Iteration | Candidate shape | Outcome |
|---|---|---|
| `001` | Inline `ops.cutlass_scaled_mm(...).view(...)` into one expression. | Failed Tier 4 downstream-logit parity at probe `0`, overshoot `0.369375`. |
| `002` | Alias `ops.cutlass_scaled_mm` to `_lumo_cutlass_scaled_mm` and call alias. | Failed Tier 4 downstream-logit parity at probe `0`, overshoot `0.369375`. |
| `003` | Tried to extend a stale prior candidate patch, after an interrupted dirty source state. | Patch apply failed; recorded as `compile_nvcc_error`. |
| `004` | Add `if len(output_shape) == 2: return output` wrapper shortcut. | Controller preflight-demoted as `fp8_gemm_cutlass_python_wrapper_rewrite`. |
| `005` | Comment-only source replacement in the CUTLASS wrapper. | Controller preflight-demoted. |
| `006` | Remove legacy CUTLASS target module from overlay target list. | Controller preflight-demoted. |
| `007` | Replace legacy target module with duplicate active target module. | Controller preflight-demoted. |
| `008` | Remove active target module, leaving legacy target only. | Controller preflight-demoted; loop stopped as `proposer_stuck`. |

Key observation: even syntactically small Python wrapper edits are not a productive CUTLASS mutation surface. The first two executable rewrites changed Python dispatch/wrapper behavior enough to fail downstream logits immediately. Later non-executable or metadata edits were correctly screened as not valid kernel optimization attempts.

## Controller Corrections Made During The Round

The initial run was not fully aligned with the pivot doc's intended cheap candidate gate: early candidates paid full vLLM restarts. During the run, the controller was corrected so that:

- internal Codex proposal subprocesses use `gpt-5.5` with high reasoning;
- FP8/CUTLASS briefs instruct agents not to run `auto-research apply-and-test`;
- controller preflight handles CUTLASS overlay mutations before expensive vLLM startup;
- diff headers with timestamps are normalized before preflight path matching;
- repeated FP8/CUTLASS preflight demotions count toward `proposer_stuck` instead of burning the full attempt cap.

Focused verification passed:

```text
6 passed
```

## Online Research Notes

Primary-source check:

- vLLM's public README advertises optimized GEMM/MoE kernels using CUTLASS, TRTLLM-GEN, and CuTeDSL, plus FP8/MXFP8/NVFP4 quantization support. That points at backend/kernel implementation as the meaningful optimization surface, not Python wrapper rewriting. Source: https://github.com/vllm-project/vllm
- vLLM releases include Blackwell-specific CUTLASS FP8/GEMM work, including optimized SM120 CUTLASS blockwise FP8 GEMM, SM121/DGX Spark CUTLASS support, and Qwen3.5 FP8 accuracy fixes. This supports treating upstream vLLM/CUTLASS source or version selection as a real lever. Source: https://github.com/vllm-project/vllm/releases
- NVIDIA CUTLASS Blackwell docs describe SM100 `tcgen05.mma` support for FP8/f8f6f4 and block-scaled GEMM, with layout/alignment constraints. Useful mutations need to operate at CUTLASS/CUDA template, layout, scale-factor, tile, or epilogue level. Source: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA CUTLASS quickstart shows Blackwell FP8 GEMM setup through datatypes, tile shapes, schedules, and collective builders. Again, the real mutation knobs are C++/CUDA CUTLASS construction choices, not Python source replacement around `ops.cutlass_scaled_mm`. Source: https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html

## Should We Run More Auto-Research?

Not on this current surface.

Run more only after one of these is true:

1. The Tier 3 isolated FP8 GEMM replay harness exists and can evaluate CUTLASS candidates cheaply against `(A, B, scale_a, scale_b, reference_output)` probes.
2. The mutable target is changed from the repo-owned Python overlay bootstrap to a real vLLM/CUTLASS source-build target.
3. The round is reframed as backend/version selection, for example comparing current CUTLASS against newer vLLM builds that include Blackwell/DGX Spark FP8 fixes.

If none of those is true, the expected outcome of another loop is more `forbidden_mutation_family_demoted` rows, not a throughput improvement.

## Better Agent Guidance

The FP8/CUTLASS agent brief should be stricter:

- Say this round is CUTLASS-only `fp8_gemm`; no DeltaNet history is actionable except as controller behavior precedent.
- Forbid Python wrapper source replacements as optimization candidates unless a cheap replay gate exists.
- Require the agent to either produce a real CUTLASS/CUDA-source mutation against an approved source-build surface or explicitly write `NO_VALID_MUTATION.md`.
- Require each proposal to name the concrete runtime symbol and expected lower-level effect: tile shape, scale layout, epilogue fusion, dispatch policy, or target backend/version.
- Keep `apply-and-test` controller-owned for FP8/CUTLASS; agents should only write `mutation.patch`.

## Artifact Index

Primary round artifacts:

- `round_spec.yaml`
- `run_log.json`
- `mutations_rejected.tsv`
- `filter_hit_review.tsv`
- `baselines/measurement_01.json` through `measurement_05.json`
- `candidates/001` through `candidates/008`

Important candidate artifacts:

- `candidates/<NNN>/mutation.patch`
- `candidates/<NNN>/parity_check.json`
- `candidates/<NNN>/BLOCKED.md`
- `candidates/<NNN>/agent_last_message.txt`
- `candidates/<NNN>/agent_session.jsonl`

Implementation changes made to make the round conform better to the intended loop are in:

- `src/lumo_flywheel_serving/auto_research.py`
- `src/lumo_flywheel_serving/round_driver.py`
- `tests/test_auto_research.py`
