# LLD-01 · vLLM Serving Layer

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S0 → Implement S0  
> Status: DRAFT v0.4 — updated for HLD v2.3

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial draft |
| v0.2 | Fixed `vllm serve` entry point; corrected `--enable-chunked-prefill` flag; rewrote §8 Codex CLI wiring to `[model_providers]` pattern; added `VLLM_SERVER_DEV_MODE` gate for `/reset_prefix_cache`; promoted `/metrics` to required; added per-task delta-sampling; surfaced Responses API multi-turn risk; revised 122B VRAM claim to conditional |
| v0.3 | **P0 fixes:** (1) `wire_api` locked to `"responses"` — `"chat"` hard-removed from Codex Feb 2026; reframed vllm#33089 as vLLM implementation gap, not config workaround. (2) §9.2 rewritten: throughput formulas now use `_sum` deltas (total model-internal GPU time across all task turns) as denominators, replacing incorrect TTFT-proxy and wall-time formulas; added explicit warning against mean-per-request denominator. (3) §17.3 removes `model_max_output_tokens` (non-functional on responses path, openai/codex#4138); replaces with documented working keys (`model_context_window`, `model_auto_compact_token_limit`); demotes `model_verbosity` to optional experiment. **P1 fixes:** Corrected `vllm:inter_token_latency_seconds` (was `time_per_output_token_seconds`); Sleep Mode levels: Level 1 = same-model reuse (offload to CPU); Level 2 = weight replacement within same architecture (not a documented cross-architecture hot switch — cold restart remains default for Dev-Bench cross-family transitions until Sprint 0 validates specific pairs). **§8.2** qualified with LoRA carve-out: adapter name is the Codex `model` value in Sprint 3, not `--served-model-name`. **New:** §17 Serving Speed — vLLM + Codex config for Sprint 3 trained-model evaluation at maximum throughput. |
| v0.4 | **HLD v2.3 alignment pass (minor — no structural changes to serving design).** (1) §3 Model Registry: comparison slot reframed — practical default is a second Qwen variant; GLM-4.7 and Step 3.5 Flash are aspirational add-ons pending Gates 1b + 6, not a presumptive slot. 27B sprint column updated from "Bench-Train" to "Bench-Control + Codex-Long collection." (2) §14 Sprint 0 checklist: Gate 1b (streamed tool-call event semantics) cross-referenced — proto-fixture requirements, sequential/parallel tool-call tests, SDK replay test. Gate 1c (frontier API) noted as out-of-scope for this LLD. (3) §17 Sprint 3 arithmetic updated: B2 core is 1,100 runs (~73 Spark days), not 1,000 (~69 days), reflecting v2.2 addition of Base-through-SWE-Agent to the B2 roster. |

---

## 1. Purpose & Scope

This document specifies the design of the vLLM serving layer — the root dependency for all inference in Codex-Bench. Every LLD that touches a model (LLD-03 Task Orchestrator, LLD-07 Benchmark Runner, LLD-08 Logprob Proxy, LLD-09 mini-SWE-Agent) targets the endpoint defined here.

**Responsibilities:**

- Load and serve all local benchmark models on the DGX Spark (ARM64 GB10)
- Expose an OpenAI-compatible REST endpoint that Codex CLI targets via a `[model_providers.localvllm]` config block
- Enforce FP8 quantization and a consistent precision baseline across all models
- Enable prefix caching ON by default; expose `/reset_prefix_cache` (dev-mode) for between-task flushes
- Expose `/metrics` as a **required** endpoint — server-authoritative source for all four Contribution A latency metrics
- Support clean model switching (LLD-07 triggers this)
- Accept LLD-08 (Logprob Proxy) sitting in front of it transparently for Phase 2b
- Serve LoRA adapter checkpoints (Sprint 3 evaluation) at maximum throughput with minimum Codex latency

**Out of scope:** Logprob proxy shim (LLD-08), multi-model parallelism, training-time GPU usage (LLD-10/11 use separate processes).

---

## 2. Hardware Target

| Property | Value |
|---|---|
| Machine | NVIDIA DGX Spark |
| SoC / CPU | ARM64 GB10 |
| GPU | Blackwell (GB10) |
| Memory bandwidth | ~273 GB/s (primary decode bottleneck) |
| VRAM | To be validated in Sprint 0 per-model (see §3.1 and §14 gate) |
| OS | Linux (ARM64) |

Single-node, single-GPU-complex setup assumed throughout.

---

## 3. Model Registry

Five local models are served over the project. The frontier API model routes directly through Codex CLI and does not use this layer.

| Model ID | HLD Role | Architecture | Total / Active | Quant | Sprint |
|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` | Primary open baseline | MoE | 35B / 3B | FP8 | S2 Dev-Bench |
| `qwen3-coder-next-80b-a3b` | Coding-specialized | MoE | 80B / 3B | FP8 | S2 Dev-Bench |
| `qwen3.5-122b-a10b` | Open-weight ceiling | MoE | 122B / 10B | FP8 | S2 Dev-Bench |
| `comparison-slot`† | Comparison slot (fallback: 2nd Qwen) | TBD | TBD | FP8 | S2 Dev-Bench (Gates 1b + 6 conditional) |
| `qwen3.5-27b` | RL target / trained model | Dense | 27B / 27B | FP8 | S0 Gate 3 + S2 Bench-Control + Codex-Long + S3 eval |

†**Practical default fallback: a second Qwen variant** (e.g., an additional Qwen3.5 size or Qwen3-Coder variant confirmed compatible in Sprint 0). GLM-4.7 and Step 3.5 Flash are **aspirational add-ons** pending Gate 1b and Gate 6 — not a presumptive slot. Recent vLLM issue traffic shows GLM streaming tool-call failures are an active risk. Do not plan around GLM until it passes Gate 1b. If neither GLM nor Step passes, the slot becomes a second Qwen variant and the benchmark remains internally valid.

For Sprint 3, `qwen3.5-27b` is re-served with LoRA adapters loaded for SFT and DAPO evaluation. See §17.

**Frontier API**: Not routed through this layer. Frontier API compatibility is validated by Gate 1c (separate from Gate 1b), which is out of scope for this LLD.

### 3.1 Model Registry YAML

Sprint 0 must validate and fill in VRAM-sensitive fields for all models before Dev-Bench.

```yaml
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.90

  qwen3.5-35b-a3b:
    hf_repo: Qwen/Qwen3.5-35B-A3B
    local_path: /models/qwen3.5-35b-a3b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.92

  qwen3-coder-next-80b-a3b:
    hf_repo: Qwen/Qwen3-Coder-Next-80B-A3B
    local_path: /models/qwen3-coder-next-80b-a3b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.93

  qwen3.5-122b-a10b:
    hf_repo: Qwen/Qwen3.5-122B-A10B
    local_path: /models/qwen3.5-122b-a10b-fp8
    quantization: fp8
    dtype: auto
    # SPRINT 0 GATE: start at 65536, increase only if OOM-free.
    max_model_len: 65536
    gpu_memory_utilization: 0.95
    sprint0_gate: "Validate max_model_len before S2"

  # ── Comparison slot (resolved after Gates 1b + 6) ──
  # Practical default: second Qwen variant. GLM-4.7 and Step 3.5 Flash are
  # aspirational — do not add an entry here until the model passes Gate 1b.
  #
  # If GLM-4.7 passes Gate 1b:
  #   comparison-slot:
  #     hf_repo: THUDM/GLM-4.7
  #     local_path: /models/glm-4.7-fp8
  #     quantization: fp8
  #     dtype: auto
  #     max_model_len: 131072
  #     gpu_memory_utilization: 0.90
  #     notes: >
  #       Published scores (73.8% SWE-bench V., 41% T-Bench 2) obtained with
  #       Preserved Thinking via SGLang — not available in vLLM. Treat as
  #       capability ceiling only under vLLM. Gate 6 measures actual perf.
  #
  # If GLM fails → test Step 3.5 Flash. If both fail → second Qwen variant.
  # Uncomment and fill the winning entry after Sprint 0 Gate 1b + Gate 6.
```

---

## 4. Quantization Strategy

### 4.1 Baseline: FP8 (Primary Condition)

All benchmark runs use FP8 as the primary quantization. Pre-quantized checkpoints are preferred over on-the-fly quantization; their HuggingFace commit hash is pinned in `model_registry.yaml` at Sprint 0 and treated as immutable.

```
vllm serve $MODEL_PATH \
  --quantization fp8 \
  --dtype auto \
  --kv-cache-dtype fp8_e5m2
```

### 4.2 Experimental: NVFP4 + MTP (Sprint 0.5 Appendix Only)

Never used as the primary benchmark condition. Published as an appendix with the exact vLLM commit hash that stabilises it on ARM64.

| Known blocker | Risk |
|---|---|
| ARM64 GB10 CUDA illegal instruction crash (open vLLM bug) | HIGH |
| Qwen3.5 NVFP4 accuracy degradation (separate report) | HIGH |
| Marlin MoE backend patch not upstream | MEDIUM |

---

## 5. vLLM Launch Configuration

One vLLM process per model. LLD-07 owns shutdown and restart between model switches.

### 5.1 Minimum vLLM Feature Contract

The pinned vLLM version (chosen Sprint 0) must expose all four features. Confirm before committing to a release; record git commit hash.

| Feature | Required for |
|---|---|
| `GET /health` | Readiness gating in LLD-03/07 |
| `POST /v1/chat/completions` | Codex inference fallback validation path |
| `POST /v1/responses` | Codex Responses API (only supported `wire_api` — see §8.3) |
| `GET /metrics` | Contribution A latency anatomy |
| `POST /reset_prefix_cache` (with `VLLM_SERVER_DEV_MODE=1`) | Between-task cache flush |

### 5.2 Base Launch Command

The canonical entry point is `vllm serve` (not `python -m vllm.entrypoints.openai.api_server`).

```bash
VLLM_SERVER_DEV_MODE=1 \
TOKENIZERS_PARALLELISM=false \
TRITON_CACHE_DIR=/tmp/triton_cache/${MODEL_ID} \
vllm serve $MODEL_PATH \
  --served-model-name          $MODEL_ID \
  --host                       127.0.0.1 \
  --port                       $VLLM_PORT \
  --quantization               fp8 \
  --dtype                      auto \
  --kv-cache-dtype             fp8_e5m2 \
  --max-model-len              $MAX_MODEL_LEN \
  --gpu-memory-utilization     $GPU_MEM_UTIL \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens     8192 \
  --max-num-seqs               4 \
  2>&1 | tee /logs/vllm_${MODEL_ID}_${RUN_ID}.log
```

> **`VLLM_SERVER_DEV_MODE=1`** activates `/reset_prefix_cache`. Server is bound to `127.0.0.1` only; never expose dev-mode endpoints externally.

> **CUDA graphs are active by default** (`--enforce-eager` not set). Graph capture at startup adds 30–90s but improves steady-state decode throughput.

> **vLLM V1:** Chunked prefill is always enabled by the scheduler regardless of the flag. `--enable-chunked-prefill` is included for V0 compatibility and is harmless in V1.

### 5.3 Per-Parameter Rationale

| Parameter | Value | Rationale |
|---|---|---|
| `--max-num-seqs` | 4 | Single-stream agentic sessions; low batch keeps TTFT predictable |
| `--enable-chunked-prefill` | — | Prevents prefill stalls on 100K+ context sessions |
| `--max-num-batched-tokens` | 8192 | Per-step token budget. Smaller → smoother ITL; larger → better TTFT. Frozen after Sprint 0 tuning |
| `--enable-prefix-caching` | — | ON by default. Codex re-sends growing shared prefixes each turn; caching converts this to O(1) lookups |
| `--kv-cache-dtype` | fp8_e5m2 | Reduces KV cache memory footprint; extends effective context length |
| `--gpu-memory-utilization` | 0.90–0.95 | Leaves headroom for CUDA graph buffers and OS |

### 5.4 Environment Variables

```bash
VLLM_SERVER_DEV_MODE=1          # Required: activates /reset_prefix_cache
TOKENIZERS_PARALLELISM=false    # Suppress parallelism warnings
TRITON_CACHE_DIR=/tmp/triton_cache/${MODEL_ID}   # ARM64: isolate Triton cache per model
```

---

## 6. Prefix Caching Design

### 6.1 Default Behavior

Prefix caching is ON for all collection and benchmark runs. Codex re-sends the system prompt and growing conversation prefix on every turn. The prefix cache converts repeated prefill of the shared prefix into O(1) KV block lookups, reducing TTFT from turn 2 onward and producing a measurable cache hit rate — one of the four Contribution A metrics.

### 6.2 Between-Task Cache Flush

The prefix cache must be flushed between tasks to prevent cross-task KV contamination. LLD-03 triggers this after task teardown and before the next task starts.

**Endpoint:** `POST /reset_prefix_cache`

Dev-only endpoint, gated by `VLLM_SERVER_DEV_MODE=1`. Open request to promote to public API (vllm#32593, Jan 2026); re-check when pinning vLLM version.

```python
def flush_prefix_cache(host: str, port: int) -> None:
    """Flush vLLM's internal prefix cache between tasks.
    Requires VLLM_SERVER_DEV_MODE=1.
    Does NOT reset /metrics counters — delta-sampling handles per-task attribution (§9.2).
    """
    requests.post(f"http://{host}:{port}/reset_prefix_cache", timeout=10).raise_for_status()
```

**Fallback if endpoint unavailable:** Restart the vLLM process between tasks (§10 restart protocol). Document which approach was used in the run log.

### 6.3 Ablation: Cache OFF

For Contribution A cache-off ablation, restart the server omitting `--enable-prefix-caching` (or with `--no-enable-prefix-caching`). Separate run campaign — not an online toggle.

---

## 7. Chunked Prefill

### 7.1 TTFT vs ITL Tradeoff

| `--max-num-batched-tokens` | TTFT | ITL |
|---|---|---|
| 4096 | Worse | Better |
| **8192 (starting default)** | **Balanced** | **Balanced** |
| 16384 | Better | Worse |

Sprint 0 measures TTFT at several values on a 50K+ token synthetic prompt. The chosen value is frozen for all subsequent benchmark runs.

> **vLLM V1:** Chunked prefill is always active; `--max-num-batched-tokens` is the effective tuning knob.

---

## 8. Codex CLI Endpoint Wiring

### 8.1 Canonical Config Pattern

Codex CLI reads provider configuration from `~/.codex/config.toml` or a project-scoped `.codex/config.toml`. For local vLLM, use a named custom provider under `[model_providers.<id>]`. There is no `[provider]` root block, and `api_key` at the root level is not a valid Codex config key.

```toml
# ~/.codex/config.toml

model          = "qwen3.5-27b"     # must match --served-model-name
model_provider = "localvllm"

[model_providers.localvllm]
name                   = "Local vLLM"
base_url               = "http://127.0.0.1:8000/v1"
env_key                = "VLLM_API_KEY"         # any non-empty string
wire_api               = "responses"             # the only valid value — see §8.3
stream_idle_timeout_ms = 600000                  # 10 min: agentic sessions are slow
request_max_retries    = 2
```

```bash
export VLLM_API_KEY="EMPTY"    # vLLM accepts any non-empty string
```

### 8.2 Model Name Alignment

**Base-model serving (Sprints 0–2):** `model = "..."` in `config.toml` must exactly match `--served-model-name` in the `vllm serve` command. Both are keyed off `model_id` in `model_registry.yaml`.

**LoRA adapter serving (Sprint 3):** When `--lora-modules` is set, vLLM exposes *both* the base model ID and each adapter name separately in `GET /v1/models`. Requests target the adapter by its `--lora-modules` name directly — the `model` value in Codex `config.toml` should be the adapter name (e.g., `"codex-sft-all"`), not the base model's `--served-model-name`. This is documented behavior, not a mismatch. See §17.1 for the Sprint 3 config pattern.

### 8.3 `wire_api`: Only Valid Value Is `"responses"`

**Codex CLI deprecation timeline:**

Codex deprecated `wire_api = "chat"` (Chat Completions API) in late 2025 and **hard-removed it in February 2026** (openai/codex discussion #7782). As of the current Codex release (April 2026), `wire_api = "chat"` produces a hard error; the only valid value is `"responses"`. The official Codex sample config explicitly documents this: `wire_api = "responses" # only supported value`.

**End-to-end Responses API compatibility risk (P0 — two independent root causes):**

The multi-turn 400 failure has been reported from both sides of the stack:

| Root cause | Issue | Status |
|---|---|---|
| **vLLM side** — multi-turn `input[]` format not fully implemented for custom-provider traffic | vllm-project/vllm#33089 (Jan 2026) | Open; vLLM H1 2026 roadmap includes ongoing Responses/tool-calling alignment |
| **Codex side** — multi-turn requests to custom/OSS Responses endpoints can omit required fields on conversation items, yielding 400s from strict servers | openai/codex#12230 | Closed as "not planned" |

Both can independently produce 400 errors on turn 2+. There is no valid `wire_api` value that avoids this — `"chat"` no longer exists.

**Sprint 0 Gate 1 action (blocking):** Run a 5-turn `codex exec --yolo --json` session against local vLLM. Capture raw HTTP request bodies on turns 2+ (e.g., via `mitmproxy`, or on the vLLM side by launching with `--enable-log-requests` and setting `VLLM_LOGGING_LEVEL=DEBUG`) alongside any 400 response bodies. This isolates which failure mode is active: a vLLM schema rejection vs a Codex serialization gap vs both.

**Mitigation paths — in scope for LLD-01 (direct Codex↔vLLM path only):**

1. **Upgrade vLLM version:** Try a newer vLLM release where #33089 may be resolved. Addresses the vLLM-side gap only; does not cover openai/codex#12230.
2. **Patch Codex CLI:** If failure is purely Codex-side (#12230), patch the Codex binary to correctly serialize multi-turn conversation items for custom providers.

**Out of scope for LLD-01 — escalate to LLD-08 redesign if needed:**

A request-normalising proxy between Codex and vLLM has been considered but is not specified here. Such a proxy would need to emulate the full Responses API contract back to Codex (including streaming and multi-turn state), forward translated requests to vLLM's `/v1/chat/completions`, and sit on the baseline serving path from Sprint 0 — making it a transport-adapter component, not a Phase 2b logprob shim. If Gate 1 cannot be resolved by options 1 or 2 above, the decision to add a translation proxy must be made at the HLD and LLD-index level (reassigning or expanding LLD-08's scope), not within this LLD. At that point, §§5.1 and 9.1 would also need a branched feature contract separating proxy-facing endpoints from vLLM-facing endpoints.

**Do not begin any collection until Gate 1 passes end-to-end.** Record the passing mitigation path in the launch log and update the HLD fairness contract accordingly.

### 8.4 Logprob Proxy Wiring (Phase 2b)

When LLD-08 is active, change `base_url` in `config.toml` to the proxy port. Nothing else changes.

```toml
[model_providers.localvllm]
base_url = "http://127.0.0.1:8001/v1"   # proxy port (Phase 2b only)
```

---

## 9. API Surface and Observability Contract

### 9.1 Required Endpoints

| Endpoint | Method | Consumer | Status | Notes |
|---|---|---|---|---|
| `/health` | GET | LLD-03, LLD-07 | **Required** | Readiness gate before every task |
| `/v1/models` | GET | LLD-03 | **Required** | Confirm `served-model-name` / adapter alignment |
| `/v1/responses` | POST | Codex CLI | **Required** | Only `wire_api` after Feb 2026; direct path subject to Gate 1 |
| `/v1/chat/completions` | POST | LLD-09 mini-SWE-Agent | **Required** | LLD-09 calls this directly without Codex. Also validated in Gate 1 smoke test to confirm vLLM is serving correctly at the API level |
| `/metrics` | GET | LLD-04 | **Required** | Server-authoritative latency source |
| `/reset_prefix_cache` | POST | LLD-03 | **Required** | Needs `VLLM_SERVER_DEV_MODE=1` |

### 9.2 Per-Task Metric Attribution — Delta-Sampling Procedure

vLLM's `/metrics` counters and histograms are **cumulative** — they grow from server start and are not reset by `/reset_prefix_cache`. Per-task attribution requires LLD-04 to snapshot `/metrics` before and after each task and compute deltas.

**Metrics consumed from `/metrics`:**

**Metric name version variance — resolve before implementing LLD-04:**

Two counter name schemas coexist across vLLM releases. The v1 design docs and production metrics source code register counters as `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` (with `_total`). The stable production-metrics HTML page and some v1/LLM-D deployments expose `vllm:prompt_tokens` / `vllm:generation_tokens` (without `_total`), because the Prometheus OpenMetrics client library can strip and re-add the suffix depending on version. Since **the vLLM version pin is still open**, the exact form that appears in the endpoint is not yet known.

**Required Sprint 0 action:** After pinning the vLLM version, run `curl -s http://127.0.0.1:$VLLM_PORT/metrics | grep prompt_tokens` and record which name form appears. Update the metric name constants in LLD-04 to match, and log the schema variant alongside the vLLM git hash in the launch log header.

**LLD-04 implementation must use the resolver pattern below** — probe both variants at server startup, record the canonical name, and raise a hard error if neither is present. Using `after.get(key, 0.0)` silently produces bad throughput numbers when a name is wrong; that failure mode is unacceptable for a published benchmark.

| Prometheus metric (canonical form TBD at Sprint 0) | Type | Used for |
|---|---|---|
| `vllm:prompt_tokens_total` OR `vllm:prompt_tokens` | Counter | Total prompt tokens sent per task (includes cache-hit tokens; used for context-size tracking, not throughput numerator) |
| `vllm:generation_tokens_total` OR `vllm:generation_tokens` | Counter | Decode token count per task |
| `vllm:request_prefill_kv_computed_tokens` | Histogram | **Newly computed KV tokens per request — excludes prefix cache hits.** Use this as the prefill throughput numerator. |
| `vllm:prefix_cache_queries` | Counter | Cache queries per task |
| `vllm:prefix_cache_hits` | Counter | Cache hits per task |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT per task (mean across turns) |
| `vllm:request_prefill_time_seconds` | Histogram | Model-internal prefill time — the GPU time actually spent on newly computed tokens |
| `vllm:request_decode_time_seconds` | Histogram | Model-internal decode time — not wall time |
| `vllm:inter_token_latency_seconds` | Histogram | Time between tokens (ITL / TPOT) |

> **Why `request_prefill_kv_computed_tokens` replaces `prompt_tokens` as the prefill numerator:**
> `prompt_tokens` counts every token in the input prompt, including the tokens served from the prefix cache that required no GPU prefill work. When prefix caching is ON (which is the default throughout this project), dividing by total prompt tokens overstates prefill throughput and folds cache efficiency into the throughput metric even though cache-hit rate is already reported separately. `request_prefill_kv_computed_tokens` counts only the tokens for which new KV blocks were actually computed — the tokens that the GPU actually processed during prefill. Dividing these newly computed tokens by `request_prefill_time_seconds_sum` gives a clean prefill compute throughput independent of caching behaviour.
>
> Note: `request_prefill_kv_computed_tokens` is a histogram, so apply the same `_sum` delta procedure used for prefill and decode time. Verify its presence with `curl /metrics | grep prefill_kv` after pinning the vLLM version — add to `resolve_metric_schema()` if naming variants appear.

> **Why throughput uses `_sum` delta, not mean per request:**
> A multi-turn Codex task issues N separate API requests (one per turn). Each request contributes one sample to the `request_prefill_time_seconds` histogram. The `_sum` delta gives the total model-internal prefill time across all N turns of the task — which is the correct denominator for task-level throughput. Dividing by the per-request mean (= `_sum / _count`) and then multiplying by total tokens would compute `total_tokens / mean_time_per_request`, which overstates throughput by a factor of N relative to the task total. **Always divide total task tokens by total task time (`_sum` delta), not by per-request mean time.**

**Sampling protocol (LLD-03 ↔ LLD-04 interface):**

```
1. LLD-04 → GET /metrics           (task_before snapshot)
2. LLD-03   starts codex exec      (task begins)
3. LLD-03   stdout EOF / timeout   (task complete)
4. LLD-04 → GET /metrics           (task_after snapshot)
5. LLD-04   computes per-task deltas (see below)
6. LLD-03   POST /reset_prefix_cache  ← AFTER snapshot step 4, not before
```

The flush in step 6 must come after the snapshot in step 4. The flush clears KV blocks but issues no inference requests and moves no `/metrics` counters.

**Derived Contribution A metrics:**

| HLD Metric | Prometheus source | Formula |
|---|---|---|
| TTFT (ms) | `vllm:time_to_first_token_seconds` histogram | `_sum_delta / _count_delta × 1000` — mean per turn across the task |
| Prefill throughput (tok/s) | `vllm:request_prefill_kv_computed_tokens` `_sum` + `vllm:request_prefill_time_seconds` `_sum` | `kv_computed_tokens_delta / prefill_sum_delta_s` — newly computed tokens only, excluding prefix cache hits |
| Decode throughput (tok/s) | `vllm:request_decode_time_seconds` `_sum` + `generation_tokens` | `gen_tokens_delta / decode_sum_delta_s` |
| Cache hit rate (%) | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | `hits_delta / queries_delta × 100` |

```python
# ── Schema resolver (call once at server startup, after version pin) ──────────
_COUNTER_VARIANTS = {
    "prompt_tokens":    ["vllm:prompt_tokens_total",    "vllm:prompt_tokens"],
    "generation_tokens":["vllm:generation_tokens_total","vllm:generation_tokens"],
}

def resolve_metric_schema(metrics_snapshot: dict) -> dict[str, str]:
    """
    Probe the live /metrics output and return the canonical name for each counter.
    Raises RuntimeError if neither variant is present — never silently defaults to 0.
    Call once after vLLM startup; cache the result for the lifetime of the run.
    """
    resolved = {}
    for logical_name, candidates in _COUNTER_VARIANTS.items():
        found = next((c for c in candidates if c in metrics_snapshot), None)
        if found is None:
            raise RuntimeError(
                f"vLLM /metrics does not expose any of {candidates}. "
                f"Update metric name constants for the pinned vLLM version."
            )
        resolved[logical_name] = found
    return resolved   # e.g. {"prompt_tokens": "vllm:prompt_tokens_total", ...}

def snapshot_metrics(host: str, port: int) -> dict:
    """Read /metrics and return flat {key: value} for tracked fields.
    Histograms are read as separate _sum, _count entries.
    """
    raw = requests.get(f"http://{host}:{port}/metrics", timeout=5).text
    return parse_prometheus_text(raw)   # caller must parse Prometheus exposition format

def compute_task_metrics(before: dict, after: dict, schema: dict[str, str]) -> dict:
    """
    All throughput denominators use histogram _sum deltas (total GPU time across
    all N turns of the task), NOT per-request mean times. Using the per-request mean
    as the denominator would overstate throughput by a factor of N.

    schema: result of resolve_metric_schema(), captures the vLLM-version-specific
    counter name form so name mismatches fail at startup, not silently during runs.
    """
    def delta(key: str) -> float:
        if key not in after:
            raise RuntimeError(f"Expected metric '{key}' not found in /metrics snapshot")
        return after[key] - before[key]

    prompt_tokens      = delta(schema["prompt_tokens"])      # total prompt tokens incl. cache hits; used for context tracking only
    kv_computed_tokens = delta("vllm:request_prefill_kv_computed_tokens_sum")  # newly computed tokens only; prefill throughput numerator
    gen_tokens         = delta(schema["generation_tokens"])
    cache_queries      = delta("vllm:prefix_cache_queries")
    cache_hits         = delta("vllm:prefix_cache_hits")

    # Histogram _sum deltas = total GPU time across all task turns
    ttft_sum           = delta("vllm:time_to_first_token_seconds_sum")
    ttft_count         = delta("vllm:time_to_first_token_seconds_count")
    prefill_sum_s      = delta("vllm:request_prefill_time_seconds_sum")    # total prefill time
    decode_sum_s       = delta("vllm:request_decode_time_seconds_sum")     # total decode time

    return {
        # TTFT: mean per turn (one TTFT per Codex API call)
        "ttft_ms":                (ttft_sum / ttft_count * 1000) if ttft_count > 0 else None,
        # Throughput: total tokens / total GPU time  (NOT total / mean_per_request)
        "prefill_throughput_tps": kv_computed_tokens / prefill_sum_s if prefill_sum_s > 0 else None,
        "decode_throughput_tps":  gen_tokens    / decode_sum_s  if decode_sum_s  > 0 else None,
        # Cache hit rate
        "cache_hit_rate_pct":     (cache_hits / cache_queries * 100) if cache_queries > 0 else None,
        # Raw deltas for LLD-12 storage
        "prompt_tokens":      prompt_tokens,        # total incl. cache hits; for context-size reporting
        "kv_computed_tokens": kv_computed_tokens,   # newly computed; prefill throughput numerator
        "gen_tokens":         gen_tokens,
        "prefill_sum_s":   prefill_sum_s,
        "decode_sum_s":    decode_sum_s,
    }
```

---

## 10. Model Switching Protocol

LLD-07 orchestrates multi-model Dev-Bench campaigns. One model is loaded at a time.

### 10.1 Process-Restart Switching (Default)

```
1. LLD-07: switch_model("qwen3.5-35b-a3b")
2. ModelServer sends SIGTERM to current vllm process
3. Wait for graceful shutdown (timeout 30s → SIGKILL)
4. Poll nvidia-smi until VRAM used < 1 GB (model fully unloaded)
5. Launch new vllm serve with next model params (from model_registry.yaml)
6. Poll GET /health until 200 OK (timeout 300s — CUDA graph capture takes 60–90s)
7. Poll GET /v1/models → confirm served-model-name
8. Signal LLD-07: "ready"
```

### 10.2 Sleep Mode Switching (Sprint 0 evaluation — conditional)

vLLM Sleep Mode keeps the server process alive, preserving CUDA graph infrastructure and avoiding cold-start overhead (18–200× faster switching in benchmarks vs cold restart). Sprint 0 should benchmark switching time and adopt if substantially faster **and confirmed working on the actual model families used**.

**Level semantics:**

| Level | Action | CPU memory | Intended use case |
|---|---|---|---|
| **Level 1** | Offloads weights to CPU RAM; discards KV cache | Must fit weights | **Reusing the same model** (e.g., between task batches for the same model) |
| **Level 2** | Discards weights AND KV cache | Minimal | **Weight update or replacement within the same architecture** |

**Important constraint for Dev-Bench cross-architecture switching:**

The `reload_weights` step in the Level 2 wake sequence (`wake_up(tags=weights)` → `collective_rpc("reload_weights")`) performs in-place weight loading into an already-initialized model architecture. vLLM's LoRA and sleep mode docs describe this as reloading updated weights into the same model instance — not as a documented hot-switch between different model architectures (e.g., 27B dense → 35B/80B/122B MoE). The cold-restart protocol in §10.1 tears down the entire process and relaunches with the new model path, which is safe across any architecture change.

**Decision rule:**
- **Same-model reuse** (e.g., between Bench-Control or Codex-Long collection batches on 27B): Level 1 sleep is appropriate and well-supported.
- **Cross-model switching** (Dev-Bench: 27B dense ↔ 35B/80B/122B MoE): **default is cold restart (§10.1)**. Sleep Mode Level 2 may be adopted only if Sprint 0 explicitly validates it across all model-family transitions used in Dev-Bench. If any transition fails, fall back to cold restart for that pair.

After Level 2 wake-up, these steps are required before any inference:

```bash
curl -X POST localhost:$VLLM_PORT/wake_up?tags=weights
curl -X POST localhost:$VLLM_PORT/collective_rpc \
     -H 'Content-Type: application/json' \
     -d '{"method":"reload_weights"}'
curl -X POST localhost:$VLLM_PORT/wake_up?tags=kv_cache
curl -X POST localhost:$VLLM_PORT/reset_prefix_cache
```

Enable Sleep Mode at launch:

```bash
VLLM_SERVER_DEV_MODE=1 \
vllm serve $MODEL_PATH \
  --enable-sleep-mode \
  ... (other flags as per §5.2)
```

### 10.3 ModelServer Class

```python
class ModelServer:
    def __init__(self, registry_path: str, port: int, use_sleep_mode: bool = False):
        self.registry = yaml.safe_load(open(registry_path))
        self.port = port
        self.use_sleep_mode = use_sleep_mode
        self._proc: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None

    def switch_model(self, model_id: str) -> None:
        if (self.use_sleep_mode
                and self._proc and self._proc.poll() is None
                and self._same_family(self.current_model, model_id)):
            # Sleep Mode: ONLY for same-architecture family pairs validated in Sprint 0.
            # Cross-family switches (dense ↔ MoE, or different param counts) always
            # use cold restart regardless of use_sleep_mode, because reload_weights
            # is an in-place loader for an already-initialised architecture — it is not
            # a documented cross-architecture hot switch.
            self._sleep_level2()
            self._reload_for_model(model_id)   # reload_weights + kv_cache + reset_prefix_cache
        else:
            # Default path: cold restart. Used for all cross-family switches and any
            # same-family switch that was not explicitly validated in Sprint 0.
            self._shutdown_current()
            self._wait_vram_free(threshold_gb=1.0, timeout_s=120)
            self._launch(model_id)
        self._wait_ready(timeout_s=300)
        self.current_model = model_id

    def _same_family(self, model_a: Optional[str], model_b: Optional[str]) -> bool:
        """
        Returns True only if both models belong to a pair that has been explicitly
        validated for Sleep Mode Level 2 switching in Sprint 0. Start with an empty
        allowlist; add pairs as Sprint 0 validates them. This prevents the Sleep Mode
        path from silently being taken for untested cross-architecture transitions.
        """
        # Populated after Sprint 0 validation. Example after validation:
        # VALIDATED_SLEEP_PAIRS = {frozenset({"qwen3.5-27b", "qwen3.5-27b"})}  # same-model
        VALIDATED_SLEEP_PAIRS: set[frozenset] = set()   # empty until Sprint 0 runs
        if model_a is None or model_b is None:
            return False
        return frozenset({model_a, model_b}) in VALIDATED_SLEEP_PAIRS

    def flush_prefix_cache(self) -> None:
        requests.post(
            f"http://127.0.0.1:{self.port}/reset_prefix_cache", timeout=10
        ).raise_for_status()

    def _launch(self, model_id: str) -> None:
        cfg = self.registry["models"][model_id]
        env = {
            **os.environ,
            "VLLM_SERVER_DEV_MODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": f"/tmp/triton_cache/{model_id}",
            "VLLM_API_KEY": "EMPTY",
        }
        cmd = self._build_cmd(model_id, cfg)
        log_path = Path(f"/logs/vllm_{model_id}.log")
        self._proc = subprocess.Popen(cmd, env=env,
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

    def _wait_ready(self, timeout_s: int = 300) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if requests.get(
                    f"http://127.0.0.1:{self.port}/health", timeout=5
                ).status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not ready within {timeout_s}s")
```

---

## 11. Mid-Run Health Monitoring

LLD-03 checks `GET /health` before each task dispatch. On failure:

1. Log: task ID, model, timestamp, error
2. Attempt one restart via `ModelServer.switch_model(current_model)` (same model, cold start)
3. If restart succeeds: retry task (max 2 total attempts)
4. If restart fails: mark task as `crash` in LLD-02 run-state; continue to next task

HLD failure contract: `resolved / failed / no_patch / timeout / crash`.

---

## 12. Launch Log Format

Every vLLM startup writes a structured header to the log. Required for reproducibility.

```
[VLLM-INIT] timestamp=2026-04-13T09:00:00Z
[VLLM-INIT] model_id=qwen3.5-27b  vllm_version=0.x.y  git_hash=abc1234
[VLLM-INIT] quantization=fp8  kv_cache_dtype=fp8_e5m2
[VLLM-INIT] max_model_len=131072  gpu_memory_utilization=0.90
[VLLM-INIT] wire_api=responses  (Gate 1 Responses multi-turn: [pass|vllm-mitigation|codex-mitigation|escalated] — set at Sprint 0)
[VLLM-INIT] dev_mode=true  sleep_mode=[enabled|disabled]
[VLLM-INIT] launch_cmd: vllm serve /models/qwen3.5-27b-fp8 --served-model-name qwen3.5-27b ...
[VLLM-READY] cuda_graph_capture_time=47.3s
[VLLM-READY] /health 200 OK at 2026-04-13T09:00:47Z
```

---

## 13. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **vllm#33089 + codex#12230: Codex multi-turn Responses API 400 errors** (Jan 2026) | **P0** | Gate 1 smoke test (§8.3) isolates which failure mode is active. Mitigations in scope: upgrade vLLM; patch Codex CLI. If neither resolves it, escalate to HLD-level decision about adding a translation proxy (LLD-08 redesign). `wire_api = "chat"` is hard-removed from Codex and is not a mitigation option. |
| ARM64 NVFP4 CUDA crash | HIGH for NVFP4 | NVFP4 kept to Sprint 0.5 appendix only |
| Qwen3.5 NVFP4 accuracy degradation | HIGH for NVFP4 | Same |
| `/reset_prefix_cache` is dev-only (requires `VLLM_SERVER_DEV_MODE=1`) | MEDIUM | Server bound to loopback; fallback is process restart |
| GLM-4.7 Preserved Thinking unsupported in vLLM | MEDIUM | Gate 6 measures actual perf; document asymmetry |
| Responses API logprobs empty (vLLM) | HIGH for Phase 2b | Gate 5 validates; Phase 2b is stretch-only |
| 122B VRAM fit unconfirmed | MEDIUM | Sprint 0 §14 gate resolves |
| CUDA graph capture adds 60–90s per cold start | LOW | 300s timeout in `_wait_ready()` |

---

## 14. Sprint 0 Validation Checklist

### Baseline Serving

- [ ] Qwen3.5-27B loads at FP8 on the Spark without crash
- [ ] Decode tok/s measured (batch 1, short context) — Gate 3 baseline
- [ ] `GET /health` → 200 after startup
- [ ] `GET /v1/models` → correct `served-model-name`
- [ ] `GET /metrics` → Prometheus text includes `vllm:time_to_first_token_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:prefix_cache_queries`, and `vllm:request_prefill_kv_computed_tokens` (the §9.2 prefill-throughput numerator — absence here means the pinned vLLM version predates this metric and the schema resolver must be updated)

### Codex CLI Wiring — Gate 1 (P0 blocker)

- [ ] **5-turn `codex exec --yolo --json` session completes against local vLLM with `wire_api = "responses"`**
- [ ] Capture raw HTTP request bodies on turns 2+ (mitmproxy or vLLM debug logs) to establish baseline
- [ ] Confirm turns 2+ succeed with no 400 errors
- [ ] If turns 2+ fail: inspect response body to determine root cause — vLLM schema rejection (vllm#33089) vs Codex serialization gap (codex#12230) vs both
- [ ] If vLLM-side (vllm#33089): try a newer vLLM release; if resolved, re-pin and re-run Gate 1 to confirm
- [ ] If Codex-side or both sides: patch Codex CLI; if unresolvable by options above, escalate to HLD for LLD-08 transport-adapter decision before proceeding
- [ ] Record whether `model_reasoning_effort` was honoured; update HLD fairness contract if not
- [ ] Confirm `--json` event stream is received and parseable

### Streamed Tool-Call Event Semantics — Gate 1b (P0 blocker, cross-ref)

Gate 1b is primarily an LLD-03 / LLD-07 concern (task orchestrator and benchmark runner own the fixture execution and result evaluation). From the LLD-01 perspective, the serving layer's responsibility is limited to:

- [ ] Confirm `/v1/responses` event stream includes correctly formed function-call events: `type`, `function.name`, `function.arguments` fields, proper `delta` streaming vs. complete object structure
- [ ] Confirm parallel multi-tool-call events (multiple tool calls in a single model turn) are correctly aggregated in the streamed response — this is the exact failure mode recent vLLM bugs target
- [ ] **Repeat for each locally-served model** before that model's Dev-Bench run begins

Proto-fixtures (lightweight committed fixture workspace + prompts) must be authored before Gate 1b runs. These are authored independently of Sprint 0b and require no full Docker environments or verifiers. See HLD §6 Gate 1b for the full fixture specification.

**KILL rules (from HLD):**
- Any locally-served non-27B model where any sub-test fails → excluded from lineup
- **If Qwen3.5-27B fails Gate 1b → Contribution B is killed entirely** (no fallback RL target)

**Gate 1c (frontier API compatibility):** Out of scope for this LLD. The frontier API ceiling model uses a provider-hosted endpoint, not local vLLM. Gate 1c is a separate lighter compatibility check run through Codex CLI against the provider endpoint. See HLD §6 Gate 1c.

### Prefix Caching

- [ ] `VLLM_SERVER_DEV_MODE=1` set; `POST /reset_prefix_cache` returns 200
- [ ] `vllm:prefix_cache_hits` > 0 in `/metrics` after a multi-turn session
- [ ] After flush: re-run same task; confirm TTFT resets to cold-cache value

### Per-Task Delta Sampling

- [ ] Delta-sampling procedure (§9.2) exercised on one real task
- [ ] `vllm:request_prefill_time_seconds` and `vllm:request_decode_time_seconds` histograms populated with non-zero `_sum` and `_count`
- [ ] Derived prefill throughput and decode throughput values are plausible

### Chunked Prefill

- [ ] No prefill stall on 50K+ token synthetic prompt
- [ ] `--max-num-batched-tokens` baseline value chosen and frozen

### Model Switching

- [ ] Cold-restart: `switch_model()` transitions 27B → 35B-A3B and back cleanly; VRAM freed
- [ ] If Sleep Mode enabled: Level 2 sleep → wake → `reload_weights` → inference succeeds
- [ ] Switch time (cold vs sleep) measured and logged; adopt Sleep Mode if materially faster

### 122B VRAM Gate

- [ ] 122B loaded at `max_model_len=65536`, `gpu_memory_utilization=0.95` — no OOM
- [ ] If no OOM: attempt `max_model_len=131072`; record result
- [ ] Update `model_registry.yaml` with confirmed values before S2

### Gate 3 Integration

- [ ] One full SWE-bench task via Codex + local 27B completes
- [ ] Wall time ≤ 150 min (Gate 3 threshold)

---

## 15. Open Questions — Status

| Question | Status |
|---|---|
| `wire_api = "chat"` as fallback | **Resolved (closed):** hard-removed from Codex Feb 2026; not an option |
| Correct vLLM entry point | **Resolved:** `vllm serve <model>` |
| Chunked prefill flag name | **Resolved:** `--enable-chunked-prefill` |
| `/reset_prefix_cache` availability | **Resolved:** dev-only, requires `VLLM_SERVER_DEV_MODE=1` |
| `/metrics` and per-task measurement | **Resolved:** §9.2 uses `request_prefill_time_seconds` / `request_decode_time_seconds` |
| Sleep Mode: Level 1 vs Level 2 | **Resolved:** Level 1 = offload weights to CPU RAM, discard KV cache; use for same-model reuse only. Level 2 = discard weights and KV cache; designed for in-place weight replacement within the same already-initialized architecture — **not** a documented cross-architecture hot switch. Cross-architecture Dev-Bench transitions (e.g. 27B dense ↔ 35B/80B/122B MoE) default to cold restart (§10.1) unless Sprint 0 explicitly validates the specific pair and adds it to `VALIDATED_SLEEP_PAIRS`. |
| Responses API multi-turn compatibility (vllm#33089 + codex#12230) | **OPEN — Sprint 0 Gate 1 (§8.3) diagnoses which failure mode is active and which in-scope mitigation resolves it. If neither upgrade vLLM nor patch Codex succeeds, the outcome is an HLD-level escalation decision — not a local LLD-01 fix.** |
| vLLM version pin | **OPEN — Sprint 0 selects release satisfying §5.1 feature contract** |
| 122B VRAM fit | **OPEN — Sprint 0 §14 gate resolves** |
| Sleep Mode adoption | **OPEN — Sprint 0 benchmarks switching time. Level 1 (same-model reuse) is the only Sleep Mode level considered for adoption without explicit validation. Level 2 (weight replacement) requires Sprint 0 to confirm it on the actual model-family pairs used; `VALIDATED_SLEEP_PAIRS` in `ModelServer._same_family()` starts empty and is populated only by Sprint 0 results. Cross-architecture switches default to cold restart until validated.** |
| Comparison slot model identity | **OPEN — Sprint 0 Gates 1b + 6 resolve. Practical default is a second Qwen variant. GLM-4.7 and Step 3.5 Flash are aspirational; neither is assumed until Gate 1b passes. §3.1 YAML placeholder is commented out until resolved.** |

---

## 16. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-03** Task Orchestrator | Calls `/v1/responses` via Codex CLI; calls `/reset_prefix_cache` after each task; coordinates snapshot timing with LLD-04 |
| **LLD-04** Latency Telemetry | Snapshots `/metrics` before/after each task; computes per-task deltas using §9.2 formulas |
| **LLD-07** Benchmark Runner | Calls `ModelServer.switch_model()`; receives "ready" signal; controls Sleep Mode level |
| **LLD-08** Logprob Proxy | Sits in front for Phase 2b; only `base_url` in Codex config changes |
| **LLD-09** mini-SWE-Agent | Calls `/v1/chat/completions` directly (no Codex layer) on the same host:port |
| **LLD-13** Codex-Long Scenario Framework | No direct interface — LLD-13 defines the scenario envs that LLD-03 launches against this serving layer. Included for completeness; LLD-01 serves identically regardless of task type |

---

## 17. Serving Speed — Maximum Throughput for Sprint 3 Evaluation

Sprint 3 serves the trained Qwen3.5-27B models (base, Codex-SFT-all, Codex-SFT-matched, SWE-Agent-SFT-matched) through Codex for Final-Test evaluation. The B2 core roster is 1,100 runs (800 Codex + 300 SWE-Agent); at ~10 avg tok/s on the Codex runs, this is the dominant timeline bottleneck (~73 Spark days at baseline). Every throughput improvement translates directly to calendar time saved.

This section defines the serving configuration for Sprint 3 and the Codex-side config levers that reduce token generation per session, both of which contribute to end-to-end task time.

### 17.1 LoRA Adapter Serving

The SFT and DAPO checkpoints from LLD-10/11 are LoRA adapters on top of the frozen 27B base. vLLM supports serving multiple LoRA adapters concurrently without reloading the base model.

**Launch command for Sprint 3 (single adapter per run):**

```bash
VLLM_SERVER_DEV_MODE=1 \
vllm serve /models/qwen3.5-27b-fp8 \
  --served-model-name          qwen3.5-27b \
  --enable-lora \
  --lora-modules               codex-sft-all=/checkpoints/codex_sft_all_adapter \
  --max-lora-rank              64 \
  --max-num-seqs               4 \
  --quantization               fp8 \
  --kv-cache-dtype             fp8_e5m2 \
  --gpu-memory-utilization     0.88 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens     8192 \
  2>&1 | tee /logs/vllm_codex_sft_all.log
```

> **`gpu-memory-utilization` is reduced to 0.88** when LoRA is active. Serving LoRA adapters requires vLLM to maintain adapter weight tensors in VRAM alongside the base model weights, plus additional allocator headroom for the per-request LoRA activation paths. The exact VRAM increase depends on rank and the number of target modules. 0.88 is a conservative starting point; tune upward if OOM does not occur in Sprint 3 smoke tests.

> **`--max-lora-rank`** must match the rank used during training (LLD-10 specifies this). If QLoRA used rank 16–32, set accordingly and leave headroom.

**Switching between adapters (e.g., base model → SFT-all → SFT-matched):** Restart the server with the new `--lora-modules` argument. The base model weights are unchanged; only the adapter changes. This is faster than a full model reload but still requires a vLLM restart (approximately 1–2 min) because the LoRA module table is set at startup. Sprint 3's adapter-switch cadence is low (3–4 switches per 300-run block) so this is acceptable.

### 17.2 vLLM Serving Knobs for Speed

The following settings are tuned specifically for the Sprint 3 speed objective (27B × 1,100 B2 core runs through Codex):

| Setting | Sprint 2 default | Sprint 3 target | Rationale |
|---|---|---|---|
| `--max-num-batched-tokens` | 8192 | 16384 (if ITL acceptable) | Longer agentic sessions with large context benefit from larger prefill chunks |
| `--gpu-memory-utilization` | 0.90 | 0.88 (with LoRA) | Leave headroom for adapter memory |
| NVFP4 + MTP | Off (FP8 only) | Enable if ARM64 crash bug is fixed by S3 | Could increase throughput from ~10 to ~18–22 tok/s if stable |
| Prefix caching | ON | ON (unchanged) | Growing repo prefix is re-sent each turn; caching is always beneficial for long-horizon Codex sessions |

### 17.3 Codex config.toml Speed Knobs

Codex agent behavior can be configured to reduce token generation overhead and manage context window usage. The knobs below are documented `config.toml` keys with confirmed behavior on the Responses API path. They are applied in the Codex `config.toml` for Sprint 3 evaluation runs.

```toml
# ~/.codex/config.toml (Sprint 3 evaluation profile)

model          = "codex-sft-all"    # adapter name, matches --lora-modules key (see §8.2)
model_provider = "localvllm"

# Context window management
model_context_window          = 131072   # must match --max-model-len; Codex uses this for KV budget tracking
model_auto_compact_token_limit = 118000  # trigger compaction at ~90% of context; prevents truncation

[model_providers.localvllm]
name                   = "Local vLLM"
base_url               = "http://127.0.0.1:8000/v1"
env_key                = "VLLM_API_KEY"
wire_api               = "responses"
stream_idle_timeout_ms = 600000
request_max_retries    = 2
```

**Per-turn output token cap — enforced via `codex exec -c` flags, not config.toml:**

The HLD fairness contract specifies `max_output_tokens = 8192` per turn. This is set via the `-c` override flag in the `codex exec` invocation (as defined in HLD §5), not as a permanent `config.toml` key. Note: `model_max_output_tokens` appears in the Codex config struct but as of September 2025 it is not forwarded to the Responses API request body (openai/codex#4138) and has no practical effect on the `wire_api = "responses"` path. Do not rely on it as a cap.

| Codex config key | Effect | Status |
|---|---|---|
| `model_context_window` | Tells Codex how much context is available; controls KV budget estimation and compaction timing | **Use: set to match `--max-model-len`** |
| `model_auto_compact_token_limit` | Token count at which Codex triggers automatic context compaction | **Use: set to ~90% of context window** to prevent hard truncation in long sessions |
| `model_reasoning_effort` | Controls thinking token budget (set to `"high"` in HLD fairness contract) | **Active only if Gate 1 Responses API compatibility is confirmed end-to-end**; otherwise silently dropped for local models |
| `model_verbosity` | Verbosity override sent with Responses API requests | **Optional experiment only** — Codex docs describe this as a Responses-side hint, but its effect on arbitrary OSS models behind a custom vLLM provider is not guaranteed. Measure empirically in Sprint 3; do not pin as a throughput assumption. |
| `model_max_output_tokens` | Per-turn output cap | **Non-functional on responses path** (openai/codex#4138). Use `-c` flag in `codex exec` instead. |

### 17.4 Prefix Caching Value in Long Agentic Sessions

For Sprint 3's 27B model evaluation, the majority of each session's cost is in the growing shared prefix (system prompt, AGENTS.md, previous turn tool-call observations). With a 50-turn Codex session producing 60K tokens, the shared prefix by turn 30 may be 30K+ tokens re-sent every turn. Without prefix caching, each turn pays full prefill cost. With prefix caching at the cache hit rates observed in Sprint 2, the effective per-turn prefill cost can be reduced to near-zero for cached blocks.

**Sprint 3 target:** Verify that cache hit rate from Sprint 2 data transfers to the SFT models. If the SFT model produces systematically different tool call patterns that break prefix alignment, cache hit rate may drop. Report per-model cache hit rates in LLD-12.

### 17.5 Sprint 0.5 Optimized Serving as Sprint 3 Accelerator

If Sprint 0.5 autoresearch finds a stable NVFP4 + MTP configuration on ARM64, the throughput benefit (~18–22 tok/s vs ~10 tok/s baseline) would compress the Sprint 3 ~73-day B2 eval estimate to ~37–42 days. This is the single highest-value Sprint 0.5 outcome. The condition for using it in Sprint 3:

- ARM64 NVFP4 crash bug confirmed fixed in the autoresearch-found vLLM version
- Accuracy delta on a held-out 10-task sample is < 2% vs FP8 baseline
- Results published separately from the FP8-baseline primary condition

If both conditions hold, the optimized config becomes the Sprint 3 evaluation stack. Sprint 3 reports it as the serving configuration used, and Contribution A is updated to include an optimized serving appendix.

---

*LLD-01 · vLLM Serving Layer · Draft v0.4 · April 2026*
