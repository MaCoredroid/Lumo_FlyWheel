# Prior CUTLASS Auto-Research Memory

- indexed_round_count: 30
- warm_decode_observed_tps: 7.36-7.39 tok/s in May 5 CUTLASS diagnostics
- track_a_surface_status: exhausted_for_2x_target

## Closeout Reports

- `docs/reports/auto_research/l0c-fp8-cutlass-loop-20260505.md`
- `docs/reports/auto_research/l0c-fp8-cutlass-round-20260505-closeout.md`
- `docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md`

## Negative Memory

- CUTLASS schedule/tile/stage/caller edits left B-weight bytes unchanged.
- Warm speed-gate failures were below 0.25% lift, far below the 2x target.
- MX/NV block-scaled OpClassBlockScaledTensorOp is not a semantics-preserving direct swap for vLLM's FP32-scale path.
- Further CUTLASS-only work needs a new low-level timing lever before full vLLM validation.

## Recent Rounds

- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z` outcome=ROUND_BLOCKED terminal=compile_failures_3x
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204103Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z` outcome=ROUND_BLOCKED terminal=compile_failures_3x
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z` outcome=ROUND_NULL_RESULT terminal=accepted_cap_reached
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T114722Z` outcome=ROUND_BLOCKED terminal=proposer_stuck
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T104920Z` outcome=None terminal=None
