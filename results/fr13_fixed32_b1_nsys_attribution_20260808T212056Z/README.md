# FR13 fixed32 B1 — Nsight per-kernel attribution (post-Qrow stack)

First real Nsight capture of the current fixed32 B1 stack. This is the post-Qrow
symbol table the campaign has been missing: every prior attempt died before the
capture window, so the tables in circulation predate qrow16 production dispatch
and the conv-postprep fusion.

- STAMP: `20260808T212056Z`
- runroot: `output/fr13_fixed32_b1_nsys_20260808T212056Z`
- arm: `tail6_fixed32_b1_nsys_f32_20260808T212056Z`
- mode `tail6_fixed32`, B=1, concurrency 1
- Nsight Systems 2026.2.1.210, `cuda,cuda-sw,nvtx`, delay 1200 s, duration 300 s,
  flush 100 ms, `CuptiUseRawGpuTimestamps=false`
- capture: 1146 `fr13.fixed32.step` NVTX instances, 1,990,552 GPU ops
- source report: 218,692,330 B,
  sha256 `241d4541f5c4767a649fe49968a4af2991346156bf073a71fae6752980f45c48`

`attribution_only=true`, `acceptance_valid=false`: the capture window is a
profiling window, not an acceptance run. Use these numbers for attribution only.
The raw `.nsys-rep` is **not** publishable
(`raw_profiler_artifacts_publishable=false`); the curated reduction here is
(`curated_publishable=true`, `provenance_bound=true`).

## Step envelope

| | ms/step | share |
|---|---:|---:|
| step envelope GPU | 237.248 | 100.0% |
| sum of disjoint phases | 224.134 | 94.5% |
| residual outside phase ranges | 13.114 | 5.5% |

The step range is an envelope over disjoint child phases
(`first_to_last_projected_gpu_operation`), not an additive phase itself.

## Phase carve-up

| phase | ms/step | % of phases | % of envelope | GPU ops |
|---|---:|---:|---:|---:|
| **sfwd** | **155.943** | **69.6%** | **65.7%** | 1,147 |
| dfwd | 35.138 | 15.7% | 14.8% | 148,980 |
| cfwd | 20.704 | 9.2% | 8.7% | 1,370,616 |
| postprocess | 12.350 | 5.5% | 5.2% | 2,292 |
| phases total | 224.134 | 100.0% | 94.5% | |

## SFWD carve-up — the headline

| category | ms/step | % SFWD | % envelope |
|---|---:|---:|---:|
| **target GEMM (CUTLASS fp8 blockwise)** | **114.812** | **73.6%** | **48.4%** |
| FA2 tree attention (qrow16) | 21.369 | 13.7% | 9.0% |
| GDN (`_tree_gdn_path_kernel`) | 12.357 | 7.9% | 5.2% |
| elementwise/other | 3.693 | 2.4% | 1.6% |
| quant | 1.483 | 1.0% | 0.6% |
| conv | 0.542 | 0.3% | 0.2% |
| cuBLAS GEMM/GEMV | 0.390 | 0.3% | 0.2% |
| norm/triton | 0.201 | 0.1% | 0.1% |
| unlisted tail beyond top-20 | 1.096 | 0.7% | 0.5% |
| **SFWD total** | **155.943** | **100.0%** | **65.7%** |

The target GEMM is the whole story: one kernel is 48.4% of the entire step
envelope, and 256.2 instances/step. FA2 and GDN together are 33.7 ms/step,
under a third of the GEMM. **Conv is finished as a target** — 0.542 ms/step,
0.2% of the envelope, so conv-side work can be retired from the optimisation
queue.

Top SFWD kernels:

| ms/step | % SFWD | inst/step | kernel |
|---:|---:|---:|---|
| 114.812 | 73.6% | 256.2 | `cutlass_3x_gemm_fp8_blockwise<bf16,128,1,128, tile<128,32,128>, KernelTmaWarpSpecializedBlockwiseCooperativeSm120>` |
| 21.369 | 13.7% | 16.0 | `flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256,16,64,1>, ...>` |
| 12.357 | 7.9% | 96.1 | `_tree_gdn_path_kernel` |
| 1.170 | 0.8% | 192.1 | `elementwise_kernel CUDAFunctor_add<float>` |
| 0.824 | 0.5% | 64.0 | `silu_and_mul_per_block_quant_kernel<bf16, fp8_e4m3, 128>` |
| 0.431 | 0.3% | 1.0 | `_fr13_conv_col0_pregather_kernel` |
| 0.111 | 0.1% | 48.0 | `_fused_post_conv_kernel` |

## qrow16 engagement (verified before trusting these tables)

- `container_env.txt`: `FR13_FA2_QROW16_PRODUCTION=1`
- `qrow16_engagement.json`: `status=ENGAGED`, `runtime_mode=FULL`,
  `dispatch="qrow16 exact geometry; no fallback"`, `layer_count=16`
- `qrow16_production_pass.json`: `status=PASS`,
  `candidate_so_sha256=1649fbe9…cbb86` matching
  `FR13_FA2_QROW16_SO_SHA256`
- independent corroboration in the trace itself: the FA2 kernel is
  `Flash_fwd_kernel_traits<256, **16**, 64, 1>` at exactly **16.0
  instances/step**, matching the 16 engaged layers one-for-one

## Ingress provenance

`provenance_bound=true`, `real_swe_verified=true`. Proxy ledger finalized,
engine ledger in campaign phase, 49 matched completed attempts across 3 tasks,
0 failed/aborted/zero-attempt requests. `driver_exit_code=15` — the campaign
driver is terminated by the profiler once the capture completes, which is
expected in attribution-only mode.

## Reproduce

Reduction is offline; it reads the report plus run evidence and needs no GPU:

```
PYTHONPATH="$PWD/src" .venv/bin/python scripts/fr13_fixed32_nsys_reduce.py \
  <runroot>/<arm>/logs/fr13_fixed32_b1_real_swe.nsys-rep \
  --output <runroot>/<arm>/logs/fr13_fixed32_b1_nsys_attribution.json \
  --nsys-bin /opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys \
  --expected-report-identity "$(stat -c '%d:%i:%h:%s:%Y:%Z' <report>)" \
  --expected-report-sha256 "$(sha256sum <report> | awk '{print $1}')" \
  --subset config/fr13_fixed32/subset_b4_four.json \
  --mode tail6_fixed32 --batch-size 1 --concurrency 1 --driver-rc 15 \
  --nsys-delay-s 1200 --nsys-duration-s 300 --nsys-flush-ms 100 \
  --nsys-trace cuda,cuda-sw,nvtx \
  --nsys-config-directives CuptiUseRawGpuTimestamps=false \
  --nsys-discard-environment true \
  [--runtime-manifest-* / --external-manifest-* / --process-identity /
   --container-identity / --runtime-attestation / --pretask-zero-traffic /
   --proxy-ledger / --engine-ledger ...]
```

The `PYTHONPATH` prefix is load-bearing: the venv resolves
`lumo_flywheel_serving` through an editable-install `.pth` pointing at another
tree where the package is only a namespace stub, so without it the fixed32
ingress verifier is unavailable and the reduction fails closed.

## Files

| file | what |
|---|---|
| `fr13_fixed32_b1_nsys_attribution.json` | curated reduction, `fr13.fixed32.nsys_attribution.v2` |
| `attribution_tables.txt` | rendered tables (source of the numbers above) |
| `qrow16_engagement.json` | qrow16 production capture, `status=ENGAGED` |
| `qrow16_production_pass.json` | qrow16 production pass sidecar, `status=PASS` |
| `nsys_lifecycle.log` | session bind/collect/generate/latch trace for the capture |
