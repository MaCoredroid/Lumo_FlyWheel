# L0 FFN GEMM Phase A Result

Date: 2026-05-03

Source plan:
`docs/reports/auto_research/l0-ffn-gemm-pivot-20260502.md`

## Phase A Measurement

Command shape:

```bash
.venv/bin/python -m lumo_flywheel_serving.cli auto-research tune-kernel-select \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --action-space-file kernel_search/phase_a_action_space.yaml \
  --baselines 5 \
  --screen-measurements-per-combo 2 \
  --rescreen-top-k 2 \
  --rescreen-measurements-per-candidate 4 \
  --parallel-instances auto \
  --round-root output/auto_research \
  --harness real \
  --base-stack-resolution vllm_default \
  --round-prefix qwen3.5-27b-fp8-gemm-phase-a \
  --phase-a-screen-method full_vllm
```

Round:
`output/auto_research/qwen3.5-27b-fp8-gemm-phase-a-20260502T233338Z`

Result: `PASS`

Artifact counts from `run_log.json`:

- baseline rows: 5
- screen rows: 4
- rescreen rows: 8
- survivor rows: 2
- eliminated rows: 0
- runtime-unsupported rows: 0

Screen means:

- `combo_001` / `cublas`: `0.015134`
- `combo_002` / `cutlass`: `0.015131`
- `vllm-default`: `0.015122`

Rescreen means:

- `combo_001` / `cublas`: `0.017118`
- `combo_002` / `cutlass`: `0.017183`

The L0a runner maximizes objective value, so `combo_002` won.

Winner bundle:

`output/tuned_configs/responses-sdk-adapter-cutover-heavy/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260503T0120560000_30eb69ce.yaml`

Winner selection:

```yaml
fp8_gemm_kernel: cutlass
attention_backend: vllm-default
deltanet_kernel: triton-chunked-delta-v2
torch_compile_mode: default
cuda_graph_capture: off
```

## Phase B Attempt

The requested L0c auto-research loop was invoked against the Phase A winner
bundle with `--kernel-target fp8_gemm` and `--harness real`.

Observed halt:

```text
HALT_REASON: l0c_fp8_gemm_real_harness_out_of_scope; Phase B fp8_gemm support is bootstrap-only until a Triton FP8 replay/capture harness is implemented
```

This is the expected outcome for the measured Phase A result. The pivot plan
states that a CUTLASS/vendor backend winner ships as the new FP8 GEMM baseline
and Phase B does not run because vendor C++ is outside the current L0c-mutable
surface.

## Verification

- Independent 5.5/high verifier: PASS.
- Focused local suite:
  `.venv/bin/python -m pytest -q tests/test_kernel_activation.py tests/test_auto_research.py tests/test_cli.py tests/test_parity_fixture.py tests/test_build_parity_fixture.py tests/test_l0c_real_apply.py tests/test_l0c_real_run.py`
  returned `179 passed in 50.25s`.
- No active `vllm serve` or `lumo_flywheel_serving.cli auto-research` process remained after cleanup.
