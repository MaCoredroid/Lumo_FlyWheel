# Round 5 E3 FP8-KV Early Regression Analysis

**Generated:** 2026-05-26  
**Status:** early partial-run analysis; do not treat as final promotion data.  
**Scope:** compare `q36a_E3_fp8kv_b4` against `q36a_E3_b4` and isolate likely
non-KV confounders behind the observed speed regression.

---

## Executive summary

`q36a_E3_fp8kv_b4` did realize FP8 KV cache:

```text
cache_dtype="fp8_e4m3"
block_size="1600"
gpu_memory_utilization="0.88"
num_gpu_blocks="1254"
calculate_kv_scales="False"
kv_cache_dtype_skip_layers="[]"
```

The control `q36a_E3_b4` realized:

```text
cache_dtype="auto"
block_size="800"
gpu_memory_utilization="0.9"
num_gpu_blocks="1291"
calculate_kv_scales="False"
kv_cache_dtype_skip_layers="[]"
```

Therefore the run is a true FP8-KV smoke, but not a clean one-variable speed
experiment. The launch changed at least three speed-sensitive surfaces:

1. KV dtype changed from `auto` to `fp8_e4m3`.
2. `block_size` and `mamba_block_size` doubled from 800 to 1600.
3. `gpu_memory_utilization` dropped from 0.90 to 0.88.

The early speed result is negative after the first 30 minutes:

| Window from run start | `q36a_E3_b4` decode TPS | `q36a_E3_fp8kv_b4` decode TPS | FP8-KV delta |
|---:|---:|---:|---:|
| 1800 s | 11.575 | 11.750 | +1.5% |
| 3600 s | 13.674 | 12.520 | -8.4% |
| 4002 s | 13.895 | 12.564 | -9.6% |
| full available run | 15.058 | 12.564 | not comparable |

The full-run headline is misleading because `q36a_E3_b4` completed 16 tasks and
`q36a_E3_fp8kv_b4` currently has only 7 completed task artifacts. The cleanest
early comparison is same-duration-from-own-start windows. On that basis, the
regression is roughly 8-10% after the first hour, not 15-17%.

---

## Local evidence

### Completion state

`q36a_E3_fp8kv_b4` currently has 7 completed task artifacts:

| Task | FP8-KV verdict | FP8-KV timed out | Control verdict | Control timed out |
|---|---:|---:|---:|---:|
| `astropy__astropy-12907` | resolved | yes | resolved | yes |
| `astropy__astropy-13033` | failed | yes | failed | yes |
| `astropy__astropy-13236` | failed | yes | failed | yes |
| `astropy__astropy-13398` | failed | no | failed | no |
| `astropy__astropy-13453` | resolved | yes | resolved | yes |
| `astropy__astropy-13977` | failed | no | failed | no |
| `astropy__astropy-14096` | failed | yes | resolved | no |

Aggregate:

| Run slice | Resolved | Failed | Timeouts |
|---|---:|---:|---:|
| `q36a_E3_fp8kv_b4`, current 7 tasks | 2/7 | 5/7 | 5 |
| `q36a_E3_b4`, same 7 tasks | 3/7 | 4/7 | 4 |
| `q36a_E3_b4`, full run | 7/16 | 9/16 | 10 |

The only shared-task verdict regression so far is `astropy__astropy-14096`.
That task should be treated as a quality/correctness investigation separately
from decode speed.

### Request-level diagnostics

Request rows are diagnostic only. They use global Prometheus counter deltas and
are overlap-contaminated under `B=4`.

| Run slice | Requests | Prompt tokens | Completion tokens | Wall p50 | Wall p95 | Request accept ratio |
|---|---:|---:|---:|---:|---:|---:|
| `q36a_E3_fp8kv_b4` | 262 | 5,881,747 | 78,153 | 30.644 s | 167.580 s | 0.752 |
| `q36a_E3_b4`, same 7 tasks | 233 | 5,150,809 | 70,796 | 26.442 s | 151.257 s | 0.745 |

This slice suggests the FP8-KV run had heavier prompt traffic and slower
request wall times, but these rows should not be used as the primary speed
metric.

### Steptrace speed

Primary speed uses `dgx_steptrace.jsonl`:

```text
decode_tps = delta(gen) / delta(dec_sum)
```

Same-duration windows from each run's own start:

| Window | Run | Gen tokens | Prompt tokens | Decode sum | Prefill sum | Decode TPS | Accept ratio | Accepted/event | Active pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1800 s | `q36a_E3_b4` | 72,340 | 6,939,530 | 6,249.858 | 535.737 | 11.575 | 0.756 | 2.268 | 97.85% |
| 1800 s | `q36a_E3_fp8kv_b4` | 68,362 | 7,220,121 | 5,817.937 | 853.503 | 11.750 | 0.766 | 2.298 | 98.71% |
| 3600 s | `q36a_E3_b4` | 148,398 | 12,684,101 | 10,852.197 | 918.255 | 13.674 | 0.731 | 2.194 | 98.92% |
| 3600 s | `q36a_E3_fp8kv_b4` | 146,856 | 12,982,072 | 11,729.917 | 1,483.855 | 12.520 | 0.752 | 2.257 | 99.36% |
| 4002 s | `q36a_E3_b4` | 166,024 | 14,281,698 | 11,948.133 | 1,017.139 | 13.895 | 0.736 | 2.209 | 99.03% |
| 4002 s | `q36a_E3_fp8kv_b4` | 163,298 | 14,387,938 | 12,997.135 | 1,626.096 | 12.564 | 0.749 | 2.246 | 99.42% |

Important observations:

- GPU utilization is saturated in both runs: 94-95% average, 96% p50.
- Waiting is effectively zero in both runs.
- Speculative acceptance is not the culprit; FP8-KV acceptance is slightly
  higher over the comparable windows.
- FP8-KV accumulated much more prefill time over the same-duration windows:
  +62% at 3600 s and +60% at 4002 s.

The main speed suspect is not idle time or MTP acceptance. It is the interaction
between FP8-KV attention, doubled cache block sizing, hybrid cache state, and
prefill/decode overlap under a saturated server.

---

## Why "2x smaller KV" did not become "2x faster decode"

FP8 KV halves the cache footprint, but it only produces a large decode speedup
when the attention backend consumes FP8 KV efficiently rather than storing FP8
then paying conversion or fixed overheads that dominate this workload.

Public vLLM guidance is consistent with this interpretation:

- vLLM's FP8 KV path is most compelling for long-context decode-heavy serving
  where attention KV memory traffic dominates.
- The best published cases require an optimized FP8 attention backend such as
  FlashAttention-3 or FlashInfer paths.
- Hybrid-attention models may need per-layer skipping because some layer types
  pay quantization overhead without enough KV-memory benefit.
- `head_dim = 256` is a known caveat: decode ITL can still improve, but prefill
  may regress because the FP8 attention path has higher register pressure.

Qwen3.6-27B-FP8 has exactly the concerning shape:

```text
head_dim = 256
full_attention_interval = 4
layer_types = mostly linear_attention, with full_attention every fourth layer
```

So the expected gain is not "2x decode" for the whole model. The benefit is
limited to the layers and kernels that are truly KV-memory-bound and using the
fast FP8 path.

References:

- vLLM FP8 KV-cache blog:
  <https://vllm.ai/blog/2026-04-22-fp8-kvcache>
- vLLM serve CLI:
  <https://docs.vllm.ai/en/latest/cli/serve/>
- vLLM quantized KV cache docs:
  <https://docs.vllm.ai/en/v0.18.1/features/quantization/quantized_kvcache/>
- Qwen3.6-27B-FP8 config:
  <https://huggingface.co/Qwen/Qwen3.6-27B-FP8/blob/main/config.json>

---

## Regression hypotheses

### H1: FP8 path is not using the fastest attention backend

The run proves storage dtype, not the actual attention-kernel path. The metrics
show `cache_dtype="fp8_e4m3"`, but they do not prove FlashAttention-3/FlashInfer
FP8 kernels were used for all relevant full-attention layers.

Evidence to collect:

- startup logs with selected attention backend;
- kernel/backend markers from vLLM logs;
- local vLLM version and whether it includes the FP8 KV fixes described in the
  2026-04 vLLM blog;
- if available, profiler evidence for FP8 attention kernels rather than
  dequantize-then-BF16 attention.

Decision:

- If backend is generic/dequantizing, do not expect FP8-KV speedup from this
  stack. Upgrade or force the supported FP8 backend before remeasuring.

### H2: `block_size=1600` and `mamba_block_size=1600` hurt scheduler/cache behavior

The FP8 run doubled cache block size. Although token capacity increased, larger
blocks may increase internal fragmentation, page/cache movement granularity, or
hybrid cache bookkeeping cost.

Evidence to collect:

- rerun FP8-KV with block size pinned to 800 if vLLM permits it;
- or rerun control with block size 1600 while keeping `cache_dtype=auto`.

Decision:

- If the control regresses at 1600, block size is a confounder and must be
  pinned for all FP8 comparisons.

### H3: FP8-KV increased prefill cost enough to erase decode gains

The 4002 s window shows:

```text
prefill_sum: 1017.139 s -> 1626.096 s
decode_tps:  13.895 -> 12.564
```

This matches the public large-head-dimension caveat: for `head_dim=256`, FP8
attention may improve decode slope while making prefill slower.

Evidence to collect:

- a decode-only microbench with fixed prefilled contexts and fixed output
  lengths;
- an input-length sweep reporting TTFT and ITL separately;
- same prompt-token volume across control and FP8 runs.

Decision:

- If ITL improves in decode-only but TTFT/prefill regresses, FP8-KV should be
  used only in regimes where enough decode tokens amortize prefill overhead.

### H4: Qwen3.6 hybrid layer mix limits FP8-KV benefit

The model's layer pattern is mostly `linear_attention`, with `full_attention`
every fourth layer. Full KV-cache savings only apply where the serving stack has
large attention KV traffic. Linear/Mamba/GDN state behavior is separate and may
not get the same 2x bandwidth win.

Evidence to collect:

- try `--kv-cache-dtype-skip-layers linear_attention` if the installed vLLM
  accepts attention type names for this model;
- verify the metrics label `kv_cache_dtype_skip_layers`;
- compare full FP8-KV, full-attention-only FP8-KV, and `auto`.

Decision:

- Prefer the lowest ITL/TTFT curve, not maximal FP8 coverage.

### H5: Quality/correctness needs calibration, but speed should be isolated first

Both runs have `calculate_kv_scales="False"`. vLLM docs say this loads
checkpoint scales if available, otherwise defaults to 1.0. Calibration may
affect quality, but enabling dynamic scales adds another runtime variable.

Decision:

- Keep `calculate_kv_scales=False` while isolating speed.
- Only test calibration after the speed path and layer policy are fixed.

---

## Required next experiment matrix

Run these in order. Stop after the first clear falsification.

| ID | Purpose | KV dtype | Block size | Skip layers | Backend requirement | Success signal |
|---|---|---|---:|---|---|---|
| M0 | Current baseline repeat | auto | 800 | none | current | reproduces `q36a_E3_b4` TPS band |
| M1 | One-variable FP8 | fp8_e4m3 | 800 | none | same as M0 | isolates KV dtype effect |
| M2 | Block-size control | auto | 1600 | none | same as M0 | measures block-size confound |
| M3 | Current FP8 shape repeat | fp8_e4m3 | 1600 | none | same as current | reproduces regression |
| M4 | Hybrid-aware FP8 | fp8_e4m3 | 800 | `linear_attention` if accepted | FP8 attention path | tests layer skipping |
| M5 | Upgraded backend FP8 | fp8/e4m3 | 800 | best from M4 | vLLM with FP8 KV fixes | validates real serving gain |

Primary microbench before SWE-Bench:

```text
input lengths: 8k, 16k, 32k, 64k
output tokens: fixed 256 or 512
concurrency: 1, then 4
metrics: TTFT, ITL slope, output tok/s, prefill_sum, decode_sum
```

Promotion gate:

```text
FP8-KV is eligible only if decode-only ITL improves and SWE-Bench B=4
steptrace decode TPS improves by at least 5% without a material resolved-count
regression.
```

Current `q36a_E3_fp8kv_b4` does not meet that gate.

