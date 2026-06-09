# FR13 SWE B=4 CUDA-Graph Diagnostic Bind

Date: 2026-06-09

HEAD before bind: `9e2efd5d`

Run root: `output/fr13_swe_verified_b4_diag_20260609T190931Z`

## Scope

This is the user-requested diagnostic run on the SWE-Verified prompt subset using the existing FR12 output-level probe:

- TREE arm: `TREE_ATTN`, forked FA2 tree verify, `tree_mtp`, B=4, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_METRICS=0`.
- Native E5 arm: `FLASH_ATTN`, `naive_mtp`, MTP-5, B=4, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR10_METRICS=0`.
- Probe: `scripts/fr12_deliverable_swe4_probe.py`, `samples_per_prompt=4`, four SWE-Verified prompts, `max_tokens=128`, `temperature=0.6`, `top_p=0.95`, `seed=1313`.
- One GPU, sequential arms, `recover_host_memory()` before each arm.

This probe records completion token IDs/text and vLLM metrics around the measured window. It does not run the full Codex SWE agent/evaluator loop, so it does not produce SWE task grader verdicts.

## CUDA Graph / Hook Status

Both arms were captured, not eager:

- TREE log: `enforce_eager=False`; `cudagraph_mode=<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>`; `Capturing CUDA graphs (decode, FULL): 4/4`.
- Native log: `enforce_eager=False`; `cudagraph_mode=<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>`; `Capturing CUDA graphs (decode, FULL): 4/4`.
- TREE engagement artifacts present: `tree_sampler_debug.jsonl`, `tree_path_lcp.jsonl`, `fr10_mtp_draft_trace.jsonl`.
- No FR13 subkernel/prefill capture hook artifacts were produced in this diagnostic run.

## Artifacts

- TREE probe: `output/fr13_swe_verified_b4_diag_20260609T190931Z/tree_b4_swe4/tree_b4_swe4_probe.json`
- TREE log: `output/fr13_swe_verified_b4_diag_20260609T190931Z/tree_b4_swe4/docker_full.log`
- Native probe: `output/fr13_swe_verified_b4_diag_20260609T190931Z/native_b4_swe4/native_b4_swe4_probe.json`
- Native log: `output/fr13_swe_verified_b4_diag_20260609T190931Z/native_b4_swe4/docker_full.log`
- Raw comparison: `output/fr13_swe_verified_b4_diag_20260609T190931Z/native_vs_tree_swe4_compare.json`

## Results

Wall-consistent speed metrics:

| arm | returned tokens | wall_s | returned_tokens_per_wall_s | request_tps_mean | request_tps_median |
| --- | ---: | ---: | ---: | ---: | ---: |
| TREE | 1856 | 102.950 | 18.028 | 4.902 | 4.966 |
| Native E5 | 2048 | 49.629 | 41.266 | 11.875 | 11.417 |

Spec acceptance:

| arm | accepted/event | accepted/token | accepted | draft tokens | drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| TREE | 2.024 | 0.225 | 1255 | 5580 | 620 |
| Native E5 | 3.794 | 0.422 | 1643 | 3897 | 433 |

Token distribution / output-level comparison:

- Records compared: `16`.
- Exact token sequences: `0/16`.
- Emitted-token bag TV: `0.2194066541`.
- First-token TV: `0.0`.
- Token-count TV: `0.25`.
- First mismatch: prompt `0`, sample `0`, position `15`, native token `1970`, tree token `5759`.
- Prefix match across records: min `1`, median `16`, max `68`.
- TREE returned fewer than `128` tokens for `4/16` records; native returned `128/128` for all records.

Self-noise correction was not established by this two-arm run. A native-vs-native repeat on the same SWE B=4 diagnostic would be required to subtract native self-noise from the raw TREE-vs-native token differences.

## Diagnostic Verdict

This binds the requested SWE B=4 CUDA-graph diagnostic numbers only. It does not self-declare pass/fail.

The measured diagnostic is not lossless at the raw served-token level (`0/16` exact, bag-TV `0.2194`) and TREE is slower than Native E5 on the wall-consistent per-request metrics in this run (`18.03` vs `41.27` returned tokens/s; accept/event `2.024` vs `3.794`). Task-verdict match was not measured by this output-level probe.

Against the deployed-regime B=4 CUDA-captured native E5 baseline, the TREE arm is a clear negative diagnostic: request TPS mean `4.90` vs `11.88` means TREE is `2.42x` slower by wall-consistent per-request TPS, and accept/event `2.024` vs `3.794` means TREE has `1.87x` fewer accepted tokens per draft event. Both arms had `enforce_eager=False` and FULL decode CUDA graph capture verified.
