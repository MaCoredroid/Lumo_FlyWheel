# Qwen3.6 vLLM FP8-KV Optimization Spec

**Generated:** 2026-05-26  
**Status:** implementation and measurement spec.  
**Scope:** make FP8 KV cache useful for Qwen3.6-27B-FP8 in the Track-B vLLM
serving stack without conflating storage savings, kernel selection, hybrid-cache
behavior, and SWE-Bench task noise.

---

## Objective

Use FP8 KV only where it improves end-to-end serving:

```text
decode-heavy speedup >= 5%
no unexplained quality regression
no hidden fallback to auto/BF16 KV
no measurement confound from block size, backend, or prefill mix
```

The goal is not to maximize FP8 labels. The goal is lower inter-token latency
and higher B=4 output throughput on the real Codex/SWE-Bench workload.

---

## Public-source constraints

### vLLM FP8 KV path

Current vLLM guidance says FP8 KV is most useful when:

- long-context decode is memory-bandwidth-bound;
- the backend performs attention in the FP8 path rather than merely storing KV
  compactly;
- hybrid-attention models skip layer types whose KV footprint is bounded or
  whose quantization overhead does not amortize;
- `head_dim=256` models are treated carefully because prefill can regress even
  when decode ITL improves.

Operational flags from the current vLLM serve CLI:

```text
--kv-cache-dtype
--kv-cache-dtype-skip-layers
--calculate-kv-scales / --no-calculate-kv-scales
--kv-cache-memory-bytes
--num-gpu-blocks-override
--block-size
```

`--calculate-kv-scales` is deprecated in the current CLI surface. If false, vLLM
loads scales from the checkpoint when available and otherwise uses 1.0. Treat it
as a quality/calibration variable, not a first speed knob.

References:

- vLLM FP8 KV-cache blog:
  <https://vllm.ai/blog/2026-04-22-fp8-kvcache>
- vLLM serve CLI:
  <https://docs.vllm.ai/en/latest/cli/serve/>
- vLLM quantized KV cache docs:
  <https://docs.vllm.ai/en/v0.18.1/features/quantization/quantized_kvcache/>
- Qwen3.6-27B vLLM recipe:
  <https://recipes.vllm.ai/Qwen/Qwen3.6-27B>

### Qwen3.6-27B-FP8 model shape

The public Qwen3.6-27B-FP8 config reports:

```text
model_type = qwen3_5
head_dim = 256
full_attention_interval = 4
layer_types = mostly linear_attention, with full_attention every fourth layer
```

This means:

- FP8 KV savings are not automatically a 2x whole-model decode win.
- The standard full-attention layers are the most likely beneficiaries.
- Linear-attention / hybrid-state layers may need different cache handling or
  may be better left in the default dtype.

Reference:

- Qwen3.6-27B-FP8 config:
  <https://huggingface.co/Qwen/Qwen3.6-27B-FP8/blob/main/config.json>

---

## Current local baseline

The current partial FP8-KV run:

```text
run tag: q36a_E3_fp8kv_b4
realized KV: fp8_e4m3
block_size: 1600
mamba_block_size: 1600
gpu_memory_utilization: 0.88
kv_cache_dtype_skip_layers: []
calculate_kv_scales: False
tasks complete: 7
resolved: 2/7
```

The control:

```text
run tag: q36a_E3_b4
realized KV: auto
block_size: 800
mamba_block_size: 800
gpu_memory_utilization: 0.90
kv_cache_dtype_skip_layers: []
calculate_kv_scales: False
tasks complete: 16
resolved: 7/16
same 7-task slice: 3/7
```

Same-duration steptrace from own run start:

| Window | Control TPS | FP8-KV TPS | Result |
|---:|---:|---:|---|
| 1800 s | 11.575 | 11.750 | FP8-KV slightly ahead |
| 3600 s | 13.674 | 12.520 | FP8-KV -8.4% |
| 4002 s | 13.895 | 12.564 | FP8-KV -9.6% |

At 4002 s, FP8-KV had similar GPU saturation and acceptance but much higher
prefill accumulation:

```text
control prefill_sum: 1017.139 s
FP8-KV prefill_sum: 1626.096 s
```

Do not promote FP8-KV from the current data. Use it as a regression seed.

---

## Serving-stack requirements

### R1: Prove realized cache dtype

Every run must capture `/metrics` and fail closed unless it reports the intended
cache dtype:

```text
vllm:cache_config_info{cache_dtype="fp8_e4m3", ...}
```

Also record:

```text
block_size
mamba_block_size
gpu_memory_utilization
num_gpu_blocks
kv_cache_dtype_skip_layers
calculate_kv_scales
attention backend
vLLM version / image tag
```

### R2: Prove attention backend

Storage dtype is not enough. The launch logs must prove which attention backend
is active for Qwen3.6 full-attention layers.

Required evidence:

```text
selected backend = FlashAttention-3, FlashInfer, or other explicitly known FP8 path
```

If the backend is unknown or generic, the run is a storage-capacity experiment,
not an FP8-KV speed experiment.

### R3: Keep one speed variable at a time

The FP8-KV comparison must pin:

```text
B
temperature
top_p
max_model_len
max_num_seqs
max_num_batched_tokens
prefix caching
chunked prefill
block_size
mamba_block_size
gpu_memory_utilization
num_gpu_blocks or kv_cache_memory_bytes, where possible
vLLM image/version
attention backend
```

Only one of these should move per ablation.

### R4: Separate TTFT/prefill from ITL/decode

Report both:

```text
TTFT / prefill_sum
ITL / decode_sum / generation_tokens
```

FP8-KV can be a decode win and still lose on SWE-Bench if prefill overhead grows
and the workload does not generate enough tokens to amortize it.

---

## Recommended implementation changes

### Add explicit FP8-KV backend proof to launch artifacts

Extend the run metadata emitted by the serving launcher to include:

```json
{
  "vllm_version": "...",
  "vllm_image": "...",
  "attention_backend_env": "...",
  "selected_attention_backend_log_excerpt": "...",
  "requested_kv_cache_dtype": "...",
  "realized_cache_config": {
    "cache_dtype": "...",
    "block_size": "...",
    "mamba_block_size": "...",
    "kv_cache_dtype_skip_layers": "...",
    "calculate_kv_scales": "..."
  }
}
```

This should be copied into each run directory so the result survives container
cleanup.

### Expose skip-layer policy as a first-class run knob

Add a launcher/runtime knob:

```text
--kv-cache-dtype-skip-layers <patterns...>
```

The first policy to test for Qwen3.6 is:

```text
linear_attention
```

If vLLM rejects that attention type name for this model, record the accepted
surface and fall back to explicit layer indices for non-full-attention layers.

### Pin cache block geometry for clean comparisons

Add explicit run controls for:

```text
--block-size
--num-gpu-blocks-override
--kv-cache-memory-bytes
```

Use whichever of these the installed vLLM accepts reliably. The first clean
comparison should keep `block_size=800` for both control and FP8-KV.

### Do not enable scale calibration in the speed isolation pass

Keep:

```text
calculate_kv_scales=False
```

until a speed-positive FP8 configuration exists. Then run a separate
quality/calibration pass:

```text
no calibration
checkpoint/static scales
dataset-calibrated scales, if available
```

---

## Measurement plan

### Phase A: engine/backend smoke

For each candidate:

```text
auto KV
fp8_e4m3 KV
fp8_e4m3 KV + skip linear_attention
```

Run one short server smoke and collect:

```text
/metrics cache_config_info
startup backend logs
one fixed prompt response
TTFT
ITL
```

Gate:

```text
No candidate advances unless realized dtype and skip-layer labels match request.
```

### Phase B: decode microbench

Use fixed prompts and fixed output lengths before SWE-Bench:

| Dimension | Values |
|---|---|
| input length | 8k, 16k, 32k, 64k |
| output length | 256 or 512 |
| concurrency | 1, then 4 |
| variants | auto, FP8 all layers, FP8 full-attention-only |

Metrics:

```text
median TTFT
median ITL
ITL slope vs input length
output tok/s
prefill_sum
decode_sum
GPU util
selected backend
realized cache config
```

Decision:

```text
Advance only if FP8 improves decode ITL slope and the TTFT penalty has a
measured break-even point inside the real SWE-Bench/Codex context-output mix.
```

### Phase C: 4-task SWE smoke

Use the same E3 `B=4` configuration with a small fixed task slice:

```text
astropy__astropy-12907
astropy__astropy-13453
astropy__astropy-13977
astropy__astropy-14096
```

The slice includes two currently resolved tasks, one fast failed task, and the
observed quality-regression task.

Report:

```text
resolved / 4
timeouts
steptrace decode TPS
prefill_sum
accept ratio
accepted/event
prompt tokens
completion tokens
wall p50/p95
```

### Phase D: 16-task Round-5 remeasure

Only run the full 16-task `q36a_E3_fp8kv_b4` remeasure after phases A-C pass.

Promotion gate:

```text
resolved count >= q36a_E3_b4 - 1, with any drop explained
steptrace decode TPS >= q36a_E3_b4 * 1.05
no material increase in timeout count
backend and realized cache config archived
```

---

## Candidate run tags

Use unambiguous tags:

| Tag | Purpose |
|---|---|
| `q36a_E3_kvauto_block800_b4_r2` | reproduced control |
| `q36a_E3_fp8kv_block800_b4` | one-variable FP8-KV |
| `q36a_E3_kvauto_block1600_b4` | block-size confound |
| `q36a_E3_fp8kv_block1600_b4_r2` | reproduce current shape |
| `q36a_E3_fp8kv_skiplinear_block800_b4` | hybrid-aware FP8-KV |
| `q36a_E3_fp8kv_fa3_block800_b4` | upgraded/forced backend |

Avoid reusing `q36a_E3_fp8kv_b4` for reruns. It already denotes the current
partial run shape with `block_size=1600`.

---

## Expected outcomes

### Outcome 1: FP8 all-layer loses, skip-linear wins

Promote the skip-layer policy and document Qwen3.6 as hybrid-cache sensitive.
Use FP8 only for full-attention layers.

### Outcome 2: FP8 decode wins in microbench but loses SWE-Bench

Keep FP8-KV for long-context/decode-heavy regimes only. Do not use it as the
default Codex/SWE-Bench serving profile unless the prompt/output mix changes.

### Outcome 3: FP8 loses even in decode microbench

Treat the local vLLM/backend as not speed-ready for Qwen3.6 FP8 KV. Upgrade or
change backend before further SWE-Bench runs.

### Outcome 4: FP8 wins speed but hurts `astropy__astropy-14096`

Freeze speed work and run quality investigation:

```text
logprob drift
fixed-seed response comparison
scale calibration
per-layer skip sensitivity
```

Do not attribute correctness loss to FP8-KV until the prompt, stochastic path,
tool errors, and retry behavior are ruled out.

