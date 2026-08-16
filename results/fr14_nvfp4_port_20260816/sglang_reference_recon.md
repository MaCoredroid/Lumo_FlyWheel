# SGLang Recon: Qwen3.8-27B NVFP4 + MTP

**Evidence base:** sglang @ `ace7314173c8221ecf5f213575302eab98f4e84f` (main, 2026-08-16), sparse clone retained at `/home/mark/shared/tmp-scratch/sglang`. HF configs/safetensors headers fetched live. GitHub issue search (unauthenticated; code search needs auth and was unavailable).

**Headline correction to the operator's premise up front:** there is no `qwen3_8.py` in SGLang, and there is no official Qwen or `nvidia/` NVFP4 build of this model. Qwen3.8-27B is served entirely through the **Qwen3.5 code path**, and SGLang's documented NVFP4 target is a **third-party** checkpoint. "If it works for SGLang it works for us" holds for the *mechanism* but not for the *checkpoint* — details in §5.

---

## 1. WHICH CHECKPOINT

### 1.1 The architecture is Qwen3.5, not a new one

`Qwen/Qwen3.8-27B/config.json`, `Qwen/Qwen3.6-27B/config.json` and `Qwen/Qwen3.5-27B/config.json` **all declare the identical architecture**:

```json
"architectures": ["Qwen3_5ForConditionalGeneration"],
"model_type": "qwen3_5",
"text_config": { "model_type": "qwen3_5_text", ... }
```

That is why searching SGLang for `qwen3_8` / `Qwen3.8` finds no model file. The serving files are:

- `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/models/qwen3_5.py`
- `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/models/qwen3_5_text.py`
- `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/models/qwen3_5_mtp.py`
- `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/configs/qwen3_5.py`

The cookbook states this outright: *"The serving-relevant architecture is identical to Qwen3.6-27B."* (`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx:114-115`). `Qwen3_5ForConditionalGeneration` extends `Qwen3VLForConditionalGeneration` — **this is a vision-language model** and SGLang serves it through the Qwen3-VL path with the vision tower live.

Text config essentials (`Qwen/Qwen3.8-27B`): 64 layers, `full_attention_interval: 4` → 48 Gated-DeltaNet + 16 full-attention; `hidden_size` 5120, `intermediate_size` 17408, GQA 24/4 at `head_dim` 256, `attn_output_gate: true`, `partial_rotary_factor` 0.25, vocab 248320, `max_position_embeddings` 262144, `tie_word_embeddings: false`, **`mtp_num_hidden_layers: 1`**, **`mtp_use_dedicated_embeddings: false`**.

### 1.2 The NVFP4 checkpoint SGLang targets

**`RadixArk/Qwen3.8-27B-NVFP4`** — cited in three independent places:

- `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx:146-149` — model table row, *"NVFP4 W4A4 + FP8 projections"*
- `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx:81` — `"default|nvfp4": "RadixArk/Qwen3.8-27B-NVFP4"`
- Agent-harness examples throughout the cookbook §3

Its `quantization_config` is **ModelOpt**, not compressed-tensors:
```json
"quant_method": "modelopt",
"quant_algo": "MIXED_PRECISION",
"producer": {"name": "modelopt", "version": "0.47.0.dev0"},
"ignore": ["mtp*", "mtp.layers.0*"],
"kv_cache_scheme": {"type": "float", "num_bits": 8, ...}
```

**There is no `nvidia/Qwen3.8-27B-NVFP4`.** I queried the HF API for `author=nvidia&search=Qwen3.8` → **0 results**. (By contrast `nvidia/Qwen3.6-27B-NVFP4` does exist, 1.37M downloads — that is the generation you are on today.) The RadixArk org also ships `RadixArk/Qwen3.8-27B-DSpark` (the separate trained draft model) and `RadixArk/Qwen3.8-2.4T-A95B-NVFP4`.

### 1.3 The other NVFP4 builds, and which one is broken

`unsloth/Qwen3.8-27B-NVFP4` also exists (276k downloads) but is **compressed-tensors**, not ModelOpt:
```json
"quant_method": "compressed-tensors", "format": "mixed-precision",
"version": "0.17.2.a20260716",
"ignore": [... , "re:^mtp.*"]
```
It is **currently broken on SGLang** and works on vLLM — see §4.1. Also present on HF: `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`, `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`. None are referenced by SGLang.

### 1.4 Test coverage — this is the weak point

**There is no registered CI test for Qwen3.8-27B, in any precision.** Grepping `test/` and `python/` for `RadixArk` or `Qwen3.8` returns exactly one unrelated hit (`test_kimi_k3_b300.py:27`, a Kimi DSpark draft).

The only NVFP4 + MTP e2e test in the tree is `/home/mark/shared/tmp-scratch/sglang/test/registered/models_e2e/test_qwen35_fp4_mtp.py`, and it targets a **different model**: `nvidia/Qwen3.5-397B-A17B-NVFP4` (the MoE flagship), `--quantization modelopt_fp4` (pure, not mixed), TP4 on B200 (SM100), `--attention-backend trtllm_mha`. Its gates: GSM8K ≥ 0.95 and `avg_spec_accept_length > 3.3`.

The cookbook's own provenance comment is candid about what "verified" means (`qwen3.8-27b.jsx:19-32`):

> *"Cells are nevertheless marked `verified: true` at the maintainers' direction — the badge there reflects their own unpublished validation, not measured data carried by this page. The DGX Spark cells are the exception and stay unverified: that recipe is unvalidated on SM121 / aarch64."*

---

## 2. QUANT PLUMBING

### 2.1 Config class

`ModelOptMixedPrecisionConfig` — `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:729`, `get_name() == "modelopt_mixed"` (line 760).

`from_config` (line 771) requires `quant_algo == "MIXED_PRECISION"` and a **non-empty `quantized_layers` map**, else raises. It derives `kv_cache_quant_algo` from `kv_cache_scheme` (`float`/8 → `"FP8"`, `float`/4 → `"NVFP4"`) and builds four sub-configs: `fp8_config`, `mxfp8_config`, `nvfp4_config`, `nvfp4a16_config`.

Resolution is **per-layer by name**, not by pattern group: `_resolve_quant_algo(prefix)` (line 871) looks the layer up in `quantized_layers`, with fallbacks for packed/fused modules (raising if shards of one fused layer disagree) and for the `language_model.model.` ↔ `model.language_model.` prefix swap (line 904).

`get_quant_method` (line 921) dispatch order — **note `ParallelLMHead` is checked before `VocabParallelEmbedding`**, with an explicit comment that a tied lm_head *is* the embedding module (line 948-949):

| Layer type | `quant_algo` | Method |
|---|---|---|
| `LinearBase` / `ParallelLMHead` | `FP8` | `ModelOptFp8LinearMethod` |
| " | `NVFP4` | `ModelOptFp4LinearMethod` |
| " | `W4A16_NVFP4` | `ModelOptNvFp4A16LinearMethod` |
| " | excluded / unknown | `UnquantizedLinearMethod` |
| `VocabParallelEmbedding` | `NVFP4` | `ModelOptNvFp4EmbeddingMethod` |
| `RadixAttention` | (kv algo set) | `ModelOptFp8KVCacheMethod` |

### 2.2 What is quantized in `RadixArk/Qwen3.8-27B-NVFP4`

`quantized_layers` has **401 entries: 193 NVFP4, 208 FP8**.

- **NVFP4 W4A4** (`group_size` 16, `tensor_group` strategy, scale dtype `float8_e4m3`): every `model.language_model.layers.{0..63}.mlp.{gate,up,down}_proj` — **and `lm_head`**.
- **FP8 W8A8**: every `self_attn.{q,k,v,o}_proj` and every `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`.
- **Unquantized** (absent from the map, so `UnquantizedLinearMethod`): all norms, `linear_attn.in_proj_a` / `in_proj_b`, conv state, `embed_tokens`, the entire vision tower, and — via `ignore` — **the whole MTP head**.
- **KV cache**: FP8 E4M3, static per-tensor, from checkpoint calibration. Auto-enabled under the default `--kv-cache-dtype auto`; the cookbook is explicit that no KV flag belongs in the recipe (`qwen3.8-27b.jsx:55-60`).

Contrast with the unsloth compressed-tensors build, which makes a **different** precision split: it keeps **layers 56-63 MLP at FP8** (only 0-55 are NVFP4) and puts `lm_head` at FP8 W8A8 per-channel rather than NVFP4.

### 2.3 GEMM kernels and the hardware gate — the GB10 answer

`/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/layers/quantization/fp4_utils.py:145-158`:

```python
if backend == "auto":
    if is_sm100_supported():                                  backend = "flashinfer_cutedsl"
    elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):  backend = "marlin"
    else:                                                     backend = "flashinfer_cutlass"
```

**On GB10 (sm_121) this resolves to `flashinfer_cutlass`** — i.e. FlashInfer's `mm_fp4(..., backend="cutlass")` (`modelopt_quant.py:131-150`). Not CUTLASS-via-sgl-kernel, not TRT-LLM, not Marlin. If flashinfer is missing, `fp4_gemm` raises `RuntimeError("NVFP4 GEMM requires flashinfer's mm_fp4")`. Overridable with `--fp4-gemm-backend`.

The Marlin branch is the SM80–SM90 W4A16 **weight-only** fallback — this is exactly the H200 note in the cookbook (`Qwen3.8-27B.mdx:167-173`): SM90 has no FP4 tensor cores, so the NVFP4 cell is greyed out rather than shipped.

**The hardware gate itself** (`/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/utils/common.py:282-286`):

```python
is_sm120_supported = lru_cache(...)(partial(
    _check_cuda_device_version, device_capability_majors=[12], cuda_version=(12, 8)))
```

It matches on **major == 12 only** — so **`is_sm120_supported()` returns True on sm_121**. GB10 rides the entire SM120 path through the quantization stack with no special-casing. There *is* a dedicated helper (line 315-317):

```python
# GB10 (DGX Spark and OEM equivalents). Not expressible via
# _check_cuda_device_version, which only matches on the major.
@lru_cache(maxsize=1)
def is_sm121() -> bool:
    return is_cuda() and torch.cuda.get_device_capability() == (12, 1)
```

…but grepping the whole `python/` tree shows **`is_sm121` is defined and never called**. There is no SM121-specific gating anywhere in the quant stack. One place where SM120/121 does diverge: `modelopt_quant.py:153` skips registering the `sgl_kernel::scaled_fp4_quant` fake op on major-12 devices.

### 2.4 Launch recipe (DGX Spark / NVFP4 cell, `qwen3.8-27b.jsx:392-407`)

```
--trust-remote-code --model-path RadixArk/Qwen3.8-27B-NVFP4
--mem-fraction-static 0.95 --attention-backend flashinfer
--chunked-prefill-size 8192 --disable-prefill-cuda-graph
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
```
Plus a live-computed `--mamba-full-memory-ratio`. This cell is **not** marked `verified`. See §4.5 — the only real GB10 measurement contradicts the `0.95`.

---

## 3. MTP UNDER NVFP4 — the core mechanism

Primary file: `/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/models/qwen3_5_mtp.py` (436 lines).

### 3.0 How the draft model is selected

`/home/mark/shared/tmp-scratch/sglang/python/sglang/srt/configs/model_config.py:697-719`: when `is_draft_model` and the architecture is `Qwen3_5ForConditionalGeneration` (or the MoE / CausalLM variants), SGLang **rewrites the architecture string** to `Qwen3_5ForCausalLMMTP` and sets `num_nextn_predict_layers = 1`. No `--speculative-draft-model-path` is needed — the draft loads from the same checkpoint directory.

Flags (`Qwen3.8-27B.mdx:174-177`, and the GB300 `high-throughput` cells):
```
--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
`NEXTN` is a **reserved alias of EAGLE**, not a separate algorithm (`python/sglang/srt/speculative/spec_registry.py:169-171`). The worker is `eagle_worker_v2.py`. (`FROZEN_KV_MTP` and `DSPARK` are distinct algorithms and are *not* what this recipe uses.)

### 3.1 (a) Are the MTP weights in the NVFP4 checkpoint? — **Yes, and they are BF16**

I read the safetensors index and the shard header directly. `RadixArk/Qwen3.8-27B-NVFP4` has 2194 tensors (vs 1199 in the BF16 original — the delta is scale tensors), of which **exactly 15 are MTP, with names identical to the BF16 original**:

```
mtp.fc.weight                                 BF16  [5120, 10240]
mtp.pre_fc_norm_embedding.weight              BF16  [5120]
mtp.pre_fc_norm_hidden.weight                 BF16  [5120]
mtp.layers.0.input_layernorm.weight           BF16  [5120]
mtp.layers.0.post_attention_layernorm.weight  BF16  [5120]
mtp.layers.0.self_attn.q_proj.weight          BF16  [12288, 5120]
mtp.layers.0.self_attn.k_proj.weight          BF16  [1024, 5120]
mtp.layers.0.self_attn.v_proj.weight          BF16  [1024, 5120]
mtp.layers.0.self_attn.o_proj.weight          BF16  [5120, 6144]
mtp.layers.0.self_attn.q_norm.weight          BF16  [256]
mtp.layers.0.self_attn.k_norm.weight          BF16  [256]
mtp.layers.0.mlp.{gate,up}_proj.weight        BF16  [17408, 5120]
mtp.layers.0.mlp.down_proj.weight             BF16  [5120, 17408]
mtp.norm.weight                               BF16  [5120]
```

Full unpacked shapes, `BF16` dtype, **zero `_scale` siblings**. Nothing is loaded from a separate BF16 original — the quantized checkpoint carries its own BF16 MTP head. Note `q_proj` is `[12288, 5120]` = 24 heads × 256 × **2** (output gating), and there is **no `mtp.embed_tokens` and no `mtp.lm_head`**.

### 3.2 (b) Are MTP projections quantized? — **No. BF16, enforced twice.**

**Mechanism 1 — checkpoint-side.** `ignore: ["mtp*", "mtp.layers.0*"]` is consumed as `exclude_modules`. `ModelOptQuantConfig.is_layer_excluded` (`modelopt_quant.py:333-363`) converts glob to regex and `re.fullmatch`es — its docstring literally cites `"mtp*"` as the example.

**Mechanism 2 — runtime-side, and this is the load-bearing one.** `_mtp_quant_config()`, `qwen3_5_mtp.py:47-78`:

```python
def _mtp_quant_config(quant_config):
    """The quantization the MTP module itself is built with.

    The MTP module often ships unquantized even though the target checkpoint is
    quantized; the loader's fusion gate has to see the same normalization the
    constructor applies, or it would answer for the target's quantization.
    """
    # Serialized Qwen3.5 ModelOpt checkpoints keep embedded MTP weights in
    # BF16. Disable quantization for those checkpoints; non-serialized
    # modelopt_fp4 still converts MoE expert weights on load.
    if quant_config and (
        quant_config.get_name() == "modelopt_mixed"
        or (quant_config.get_name() == "modelopt_fp4"
            and quant_config.is_checkpoint_nvfp4_serialized)
    ):
        return None
    ...
```

`RadixArk/Qwen3.8-27B-NVFP4` resolves to `modelopt_mixed` → **returns `None`** → the whole MTP module is constructed with `quant_config=None`. The comment at line 68-70 explains why the exclude list alone is insufficient: the constructor must see `None` **so linear/MoE weight loaders allocate BF16 shapes** (regression referenced: sgl-project/sglang#23113).

Applied at construction (`qwen3_5_mtp.py:105`) and to the fusion gate (`:87`).

Additionally, `self.fc` is a **plain `torch.nn.Linear`** (line 112), not an SGLang `LinearBase` — so the concat projection is structurally unquantizable regardless of config.

### 3.3 (c) Does the draft share the quantized backbone? — **No. Fully separate weights, and a different layer type.**

```python
mtp_config = copy.deepcopy(config)
mtp_config.num_hidden_layers = 1
mtp_config.full_attention_interval = 1          # ← qwen3_5_mtp.py:118-120
self.model = Qwen3_5ForCausalLM(mtp_config, quant_config, prefix="mtp", is_nextn=True)
```

`layers_block_type` is a derived property (`python/sglang/srt/configs/qwen3_next.py:259-269`, inherited by `Qwen3_5TextConfig`):
```python
for l in range(self.num_hidden_layers):
    if (l + 1) % self.full_attention_interval == 0:  full_attention
    else:                                            linear_attention
```

With `num_hidden_layers=1, full_attention_interval=1` → `l=0` → `(0+1) % 1 == 0` → **`full_attention`**. `HybridLayerType.full_attention.value == "attention"` → `Qwen3_5AttentionDecoderLayer`.

> **The single MTP draft layer is a Gated-Attention layer (GQA 24/4, head_dim 256, output-gated), NOT a Gated-DeltaNet layer.** The draft needs a KV cache and **no** GDN/mamba state of its own. This is confirmed independently by the checkpoint: the MTP tensors are `self_attn.{q,k,v,o}_proj`, not `linear_attn.*`.

The draft runs on its own `ModelRunner` (`eagle_worker_v2.py:180`, `self.draft_runner = self.draft_worker.model_runner`) with its own weights and KV pool. There is **no trunk sharing and no on-the-fly dequant** — the draft's 1 layer is genuinely separate BF16 weights that happen to live in the same safetensors files. `is_nextn=True` also forces `dcp_size = 1` for the draft (`qwen3_5.py:865-866`: *"Drafts are TP-sharded and do not replicate KV under DCP"*).

### 3.4 (d) Shared embedding / lm_head

`mtp_use_dedicated_embeddings: false`, and the checkpoint has neither `mtp.embed_tokens` nor `mtp.lm_head`. Both come from the target at runtime, via `eagle_worker_v2.py:277-316`:

```python
def init_lm_head(self):
    embed, head = self.target_worker.model_runner.model.get_embed_and_head()
    target_lm_head = getattr(self.target_worker.model_runner.model, "lm_head", None)

    def maybe_share_target_lm_head():
        if (target_lm_head is not None
            and self.hot_token_id is None
            and getattr(self.draft_runner.model, "hot_token_id", None) is None
            and hasattr(self.draft_runner.model, "set_lm_head_from_target")):
            self.draft_runner.model.set_lm_head_from_target(target_lm_head)
    ...
    self.draft_runner.model.set_embed_and_head(embed, head)   # raw weight tensors
    maybe_share_target_lm_head()                              # whole module swap
```

Two distinct steps, and the order matters:
1. `set_embed_and_head(embed, head)` (`qwen3_5_mtp.py:153-160`) deletes the draft's own `embed_tokens.weight` / `lm_head.weight` and rebinds the target's tensors, then `empty_cache()` + `synchronize()`.
2. `set_lm_head_from_target(target_lm_head)` (`:163-167`) replaces the draft's **entire `lm_head` module** with the target's — which for RadixArk is a `ParallelLMHead` carrying `ModelOptFp4LinearMethod` (lm_head is NVFP4 in that checkpoint).

So the draft computes logits through **the target's NVFP4 lm_head**, not a BF16 copy. Since `tie_word_embeddings: false`, step 2 does fire.

**Hazard worth noting:** step 1 assigns the target's *packed uint8 NVFP4* `lm_head.weight` onto the draft's BF16 `ParallelLMHead` before step 2 discards that module. Harmless only because step 2 always follows. If `hot_token_id` were set (a token-map / EAGLE3 scenario), `maybe_share_target_lm_head()` bails and the draft is left holding a packed FP4 tensor in a BF16 head — silently wrong. Not reachable in the documented Qwen3.8 MTP recipe, but a sharp edge to avoid replicating.

### 3.5 Draft forward

`qwen3_5_mtp.py:169-230` — for each draft step:
```
input_embeds = target.embed_tokens(input_ids)          # or mm_input_embeds for VLM
input_embeds = pre_fc_norm_embedding(input_embeds)     # GemmaRMSNorm
hidden       = pre_fc_norm_hidden(spec_info.hidden_states)
hidden       = fc(cat([input_embeds, hidden], dim=-1)) # BF16 nn.Linear, 10240→5120
hidden       = self.model(input_ids, positions, fb, hidden)   # 1 full-attn layer
return logits_processor(input_ids, hidden, self.lm_head, fb)  # target's NVFP4 head
```

Both `pre_fc_norm_*` and `model.norm` are **`GemmaRMSNorm`** — `(1 + w)` scaling, not plain RMSNorm. Easy to get wrong in a port.

Multimodal detail (`:193-204`): on extend with mm inputs and not `draft_extend_v2`, the last position of each sequence has its embedding overwritten with the token embedding.

### 3.6 Weight loading

`load_weights` (`:232-432`) filters to `"mtp" in name`, then remaps: `mtp.` → `model.`, `model.fc` → `fc`, `model.pre_fc` → `pre_fc`, and **strips `.self_attn`** (`:316-317`), so `mtp.layers.0.self_attn.q_proj.weight` → `model.layers.0.q_proj.weight` → fused → `model.layers.0.qkv_proj.weight`. It also skips a list of `ignore_suffixes` (`.weight_scale`, `.input_scale`, `.k_scale`, `.v_scale`, …) when the param doesn't exist — the mechanism that harmlessly absorbs stray scale tensors.

`--speculative-draft-model-quantization` defaults to the target's quantization (`server_args.py:4244-4245`) and `"unquant"` maps to `None` (`:4255-4256`). On CUDA it is then **overridden by `_mtp_quant_config`** regardless; it is only consulted directly on NPU (`qwen3_5_mtp.py:65`).

---

## 4. KNOWN ISSUES

### 4.1 `unsloth/Qwen3.8-27B-NVFP4` is broken on SGLang — and vLLM already has the fix
**Issue #34895** (OPEN, 2026-08-15) — *"compressed-tensors FP8 lm_head weight_scale silently dropped → degenerate repetition with unsloth/Qwen3.8-27B-NVFP4"*. Symptom: the model repeats a phrase forever from the first token (`"need analysis there need analysis there…"`), `finish_reason=length`, empty content. Root cause: `CompressedTensorsConfig.get_quant_method()` dispatches only `LinearBase` and `FusedMoE`; `ParallelLMHead` falls through to `None` → `UnquantizedEmbeddingMethod`, so `lm_head.weight_scale` is never loaded and raw FP8 weights (±448) are consumed as BF16. Top-10 token overlap vs. correct dequant: 2/10.

> The reporter states: **"The same checkpoint serves correctly with vLLM."** vLLM fixed the identical gap in **vllm-project/vllm#37291** (merged), routing all matching schemes including W8A8-FP8 through `CompressedTensorsLinearMethod`.

Repro was on 2× RTX 5090 (sm120), flashinfer 0.6.17, sglang 41cd5a7. Fix PR **#34904** (OPEN). Note this is a **compressed-tensors** bug — the ModelOpt path used by RadixArk handles `ParallelLMHead` correctly (`modelopt_quant.py:933`, with the ordering comment at :948).

### 4.2 DGX Spark / GB10 — the one real measurement contradicts the shipped recipe
**Issue #34872** (CLOSED, 2026-08-14) — *"Qwen3.8-27B-FP8 validated on one DGX Spark at mem-fraction-static 0.70"*. This is the **only** GB10 datapoint in the tree, and it is **FP8, not NVFP4**:

- `Qwen/Qwen3.8-27B-FP8` rev `017b9c7a`, image `lmsysorg/sglang:qwen38-27b` @ `sha256:febfb971…`, sglang `0.0.0.dev0+qwen38.27b.g561c8f3`
- **Native MTP: EAGLE, 3 steps, top-k 1, 4 draft tokens** — 72/72 requests completed, max prompt 32,768, max concurrency 8
- **`--mem-fraction-static 0.70`**, prefill chunk 8192, prefill CUDA graph **left enabled**
- A pre-run attempt at **0.75 blew a sealed 512 MiB swap-growth guard by 26.4 MB during CUDA-graph capture** and was killed
- Configured attention backend FlashInfer, but **"Hybrid GDN kernels selected by the runtime: Triton"**
- The arm64 image pulled and ran fine on GB10 (the cookbook still carries a `TODO: verify an arm64 build`, `qwen3.8-27b.jsx:131-133`)
- Reporter explicitly: *"it does not validate the BF16 or NVFP4 cells."*

The shipped DGX Spark cell says `--mem-fraction-static 0.95 --disable-prefill-cuda-graph`. **Nobody has measured that.**

### 4.3 MTP + FlashInfer on SM120/SM121 needs a recent FlashInfer
`Qwen3.8-27B.mdx:159-163`: *"MTP with the FlashInfer backend requires a FlashInfer build whose prefill `plan` accepts `uniform_q_len` (newer than 0.6.15.post1); otherwise run spec with `--attention-backend triton`."* `trtllm_mha` is SM100-only.

PR **#34670** (*"Pass uniform_q_len to the FlashInfer prefill plan"*, **merged** 2026-08-13) — but **merged into the `qwen38` branch, not main**. Main carries the equivalent as a hardcoded positional `0,  # uniform_q_len` at `python/sglang/srt/layers/attention/flashinfer_backend.py:284`.

### 4.4 Acceptance-rate and accuracy data (the numbers you want)
From `/home/mark/shared/tmp-scratch/sglang/docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` — single **GB300** (SM103, 288GB), `lmsysorg/sglang:dev @ c4271c3fe`, ISL/OSL 1024/1024, 2026-08-14:

| Quant | accept_length (cap 4) | TPOT bs=1, no MTP → MTP | tok/s/gpu bs=1 | tok/s/gpu conc64 |
|---|---|---|---|---|
| **NVFP4** | **~3.31** | 6.4 ms → **2.3 ms** | 155 → **415** | 4316 → 5155 |
| FP8 | ~3.16 | 8.2 → 3.2 ms | 121 → 302 | 3012 → 4599 |
| BF16 | ~3.22 | 10.3 → 4.0 ms | 97 → 245 | 3208 → 3995 |

> **NVFP4's acceptance (3.31) is the highest of the three, above BF16 (3.22).** Quantizing the backbone to NVFP4 while keeping the MTP head BF16 does **not** degrade draft acceptance. The file notes: *"Suspected cap is `--speculative-num-draft-tokens 4`."*

Accuracy (GSM8K full 1319, sgl-eval, sglang `c7c03ec`): NVFP4 with **fp8 KV = 96.44%**, with **bf16 KV = 96.82%**; both stop_rate 100%, truncated 0%. So the checkpoint's auto-enabled FP8 KV costs ~0.38 pt. FP8/BF16 GB300 GSM8K unmeasured (a TP4 B300 sibling scored FP8 at 96.74%).

Cross-check from `test_qwen35_fp4_mtp.py:85`: the 397B MoE asserts `avg_spec_accept_length > 3.3` at the same 3/1/4 settings.

### 4.5 Open work in flight
- **#34859** *"Qwen3.8-27B Model Support"* (OPEN, **CI failing**) — 18 files: SM120 FP8 GEMV + Hopper BF16 GEMV JIT kernels, GDN fused-proj changes, `modelopt_quant.py` +40, `unquant.py` +29, `qwen3_5.py` +35.
- **#34585** *"support qwen 3.8"* (OPEN) — 100 files; the `qwen38` branch. Adds `docker/qwen38/{qwen38_cu12,qwen38_cu13}.Dockerfile`, DeepEP v2, a FlashInfer fallback layer, heavy GDN work, and touches `qwen3_5_mtp.py` (+21/−5), `eagle_worker_v2.py` (+35/−12), `qwen3_5.py` (+247/−11).
- **#34934** *"Fuse prefill norm/act quantization for NVFP4 W4A4 hybrid models (Qwen3.5 family)"* (OPEN) — directly relevant: *"Kernel-level tracing against **vLLM 0.27.1** on the same model showed the GEMM/attention/GDN families at parity while SGLang lost time exclusively in this glue (**vLLM fuses norm+quant and act+quant via torch.compile**)."* Three fusions (SiLU+mul→NVFP4 quant; post-attn GemmaRMSNorm→NVFP4 quant with the `(1+w)` folded into a precomputed weight; GDN gated-RMSNorm→static-FP8). TTFT −5.1% on SM120; GSM8K 0.9635 (off) → 0.9658 (on). Kill switches `SGLANG_DISABLE_{SILU_FP4,POST_LN_FP4,GATED_NORM_FP8}_QUANT_FUSION`.
- **#34966** *"Support compressed-tensors NVFP4 Marlin with BF16 and DSpark"* (OPEN) — makes `unsloth/Qwen3.8-27B-NVFP4` + `RadixArk/Qwen3.8-27B-DSpark` work on RTX 4090/SM89. Also fixes tiny BF16 NVFP4 block-scale underflow in Marlin, *"following the approach in vLLM PR vllm-project/vllm#34577"*, applies CT quant to `ParallelLMHead`, and routes DSpark logits through the quantized head. Real-checkpoint NVFP4 MLP rel-MAE 0.0019 / cos 0.999995; FP8 linear-attn rel-MAE 0.0265 / cos 0.999638.
- **#34918** (OPEN) — community "verified cell" for rtx6000/nvfp4 that swaps MTP for **DSPARK** (`--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark`, `--mamba-full-memory-ratio 8.25`): TTFT 768.61 ms, ITL 15.47 ms, sglang 0.5.17.

### 4.6 SM120/SM121 rough edges (adjacent, not this model)
- **#34192** Llama4 NVFP4 MoE crashes on SM120/SM121 (`apply_router_weight_on_input` unsupported for FlashInfer) — MoE-only, so N/A for dense 27B, but shows sm121 NVFP4 is thinly covered.
- **#28018** Gemma-4 QAT W4A16 compressed-tensors fails `gptq_marlin_repack` on SM121.
- **#28019** Gemma-4 FP8-dynamic fused MoE exceeds SM121 shared memory.
- **#34018** (OPEN) re-enable FP8 `wo_a` GEMM on sm120/sm121; **#34019** (closed) default fused MHC path on SM12x.

### 4.7 GDN state sizing under speculation
`Qwen3.8-27B.mdx:63-91` gives the formula the Spark cells depend on:
```
ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)
```
- `S` = state slots/request: `extra_buffer`=5 (default), `extra_buffer_lazy`=4, `no_buffer`=3, radix cache off =1
- **`D` = `--speculative-num-draft-tokens` (4) under spec, 0 otherwise** — speculation costs extra GDN verify states
- `state_bytes` = 153.9 MB fp32 / 78.4 MB bf16 (48 GDN layers × 48 heads × 128 × 128)
- `kv_bytes_per_token` = 32.8 KB fp8 / 65.5 KB bf16 (16 attn layers × GQA 4 × 256 × K+V)

Weights: NVFP4 ~16.5 GB, FP8 ~28.5 GB, BF16 ~54 GB.

---

## 5. TRANSFER MAP → our vLLM / GB10 port

| # | SGLang finding | Implication for the FR13 port | Action |
|---|---|---|---|
| **T1** | Qwen3.8-27B **is** `Qwen3_5ForConditionalGeneration` / `qwen3_5`, byte-identical serving arch to Qwen3.6-27B | This is **not** a new-architecture port. If your patched vLLM already serves Qwen3.6-27B, the model-side delta to 3.8 is ~zero. The port is a **quantization + checkpoint** exercise, not an architecture one. | Confirm your vLLM registry maps `Qwen3_5ForConditionalGeneration`; diff `Qwen3.6-27B` vs `Qwen3.8-27B` `text_config` — I found them structurally identical. |
| **T2** | No official Qwen or `nvidia/` NVFP4 exists. SGLang points at **`RadixArk/Qwen3.8-27B-NVFP4`** (ModelOpt MIXED_PRECISION) | You are trading `nvidia/Qwen3.6-27B-NVFP4`-grade provenance for a third-party build. This is the single biggest non-technical risk in the migration. | Decide explicitly: RadixArk (ModelOpt, SGLang-blessed) vs unsloth (compressed-tensors, vLLM-working) vs quantize `Qwen/Qwen3.8-27B` yourselves with ModelOpt 0.47+ using RadixArk's exact recipe (§2.2) — which you can now replicate byte-for-byte from the `quantized_layers` map. |
| **T3** | `_mtp_quant_config()` returns `None` for `modelopt_mixed` / serialized `modelopt_fp4` → **MTP module built entirely BF16**; checkpoint `ignore` is belt-and-braces | **Copy this pattern verbatim.** Your drafter must construct its MTP module with `quant_config=None`, not merely rely on the checkpoint's exclude list — otherwise linear layers allocate quantized shapes and weight loading mismatches (the sglang#23113 class of bug). | Add the equivalent gate in your vLLM MTP module constructor, keyed on `quant_method in {modelopt, compressed-tensors}` + MTP-in-ignore. |
| **T4** | MTP weights are **in** the NVFP4 checkpoint: 15 BF16 tensors, unpacked shapes, no scales | No dual-checkpoint loading, no BF16-original fetch. Your loader must accept a mixed-dtype safetensors set and route `mtp.*` around the quant path. | Verify your loader doesn't try to find `mtp.*.weight_scale`; verify `ignore_suffixes`-style tolerance for stray scale tensors. |
| **T5** | **The draft layer is full-attention, not GDN** (`full_attention_interval=1` on the deep-copied MTP config) | If your drafter currently assumes it mirrors the target's hybrid layer mix, it is wrong. The draft needs a **KV cache and no mamba/GDN state**. This is likely the highest-value single fact here. | Assert the draft allocates 0 GDN state slots and a KV pool sized for 1 layer × GQA 4 × 256. |
| **T6** | `lm_head` is **NVFP4** in RadixArk; draft gets the target's **whole quantized module** via `set_lm_head_from_target`, after a raw-tensor `set_embed_and_head` | Do not let the draft hold a BF16 copy of a 248320×5120 head, and do not let it hold a *packed* FP4 tensor in a BF16 module. Swap the module, don't just rebind `.weight`. | Mirror the two-step order and the `hot_token_id is None` guard (§3.4). Add an assert that the draft's final `lm_head` is the same object as the target's. |
| **T7** | GB10 = **`flashinfer_cutlass`** FP4 GEMM (`is_sm120_supported()` matches major==12; `is_sm121()` is dead code) | vLLM's FP4 backend selection is independent — do **not** assume it picks the same kernel on sm_121. A silent fall to Marlin W4A16 would cost you the W4A4 activation path entirely. | Log the resolved FP4 GEMM backend at startup on the GB10 box. Confirm flashinfer `mm_fp4` cutlass is compiled for sm_121 in your image. |
| **T8** | KV cache auto-goes **fp8_e4m3** from `kv_cache_quant_algo`; costs **0.38 pt GSM8K** (96.82 → 96.44) | Free 2× KV memory, small measurable accuracy cost. SGLang deliberately does *not* put a `--kv-cache-dtype` flag in the recipe. | Decide consciously; if you need the 0.38 pt, force bf16 KV and re-derive your mamba ratio (`kv_bytes_per_token` doubles). |
| **T9** | **accept_length ~3.31 at NVFP4 vs ~3.22 BF16** (cap 4, steps 3 / topk 1 / draft 4) | NVFP4 backbone + BF16 MTP head **does not hurt acceptance**. Your acceptance-rate regression budget vs the current fp8 stack should be ~0, and FP8 actually measured *worse* (3.16). | Use **3.3** as your local acceptance acceptance-gate (matches `test_qwen35_fp4_mtp.py:85`). If you measure materially below ~3.1, suspect a wiring bug, not quantization. |
| **T10** | Spec decoding adds `D = num_draft_tokens` GDN state slots per request | On GB10's 128 GB unified memory this changes your concurrency ceiling: at 3/1/4 you pay 4 extra state slots × 78.4-153.9 MB per running request. | Port the `(S+D)` formula into your sizing; prefer `--mamba-ssm-dtype bfloat16` (halves state) and the `extra_buffer_lazy` equivalent (S 5→4). |
| **T11** | **#34895: compressed-tensors quantized `lm_head` is dropped on SGLang; vLLM fixed it (vllm#37291)** | **Inverted risk.** For the unsloth checkpoint you are on the *good* side of this bug. Do not "port" SGLang's compressed-tensors lm_head handling — yours is newer. | Confirm your vLLM includes vllm#37291. If you pick unsloth, this is mandatory. |
| **T12** | **#34934: vLLM 0.27.1 already fuses norm+quant / act+quant via torch.compile; SGLang is catching up** | You likely inherit the prefill fusion win for free. Don't spend effort porting SGLang's hand-fused kernels. | Verify torch.compile is actually enabled on your GB10 path (it often isn't on aarch64/sm_121). If disabled, you lose the ~5% TTFT that PR reclaims. |
| **T13** | Only real GB10 datapoint: **FP8** at `--mem-fraction-static 0.70`, prefill CUDA graph **enabled**; 0.75 OOM'd during graph capture. GDN kernels fell back to **Triton** despite requesting FlashInfer | The cookbook's `0.95 + --disable-prefill-cuda-graph` for Spark is **unvalidated and contradicted**. Expect graph-capture memory spikes on unified memory. | Start at **0.70**, not 0.95. Log which GDN kernel actually gets selected. Budget for Triton-speed GDN on GB10. |
| **T14** | `NEXTN` is a reserved **alias** of EAGLE; recipe is `EAGLE / steps 3 / topk 1 / draft-tokens 4`, no draft-model-path | Tree-decoding with topk>1 is **not** the shipped configuration for this model — SGLang runs a **chain** (topk=1). | If your tree decoder runs topk>1, you are off SGLang's validated path; that acceptance number (3.31) does not transfer. Validate topk=1 chain first, then extend. |
| **T15** | Norms are **`GemmaRMSNorm`** `(1+w)`; `mtp.fc` is `[5120, 10240]` = `cat([norm(embed), norm(hidden)])`; loader strips `.self_attn` | Classic silent-numerics port bugs. | Unit-test the MTP head forward against HF `transformers` 5.8+ reference on a fixed prompt before trusting any acceptance number. |
| **T16** | Zero CI coverage for Qwen3.8-27B; "verified" badges are unpublished maintainer validation; Spark cells explicitly unverified | **The operator's tip overstates the evidence.** SGLang *documents* NVFP4+MTP for this model and has strong adjacent evidence (GB300 benchmarks, a 397B MoE e2e test), but nothing CI-enforced for 27B on sm_121. | Treat SGLang as a **design reference, not a correctness oracle**. Budget for local GSM8K (expect ~96.4% fp8-KV) + acceptance-length validation as your own gate. |

---

## TL;DR

**Load-bearing facts**
1. Qwen3.8-27B **is** `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5` — same serving arch as Qwen3.6-27B and Qwen3.5-27B. No `qwen3_8.py` exists; everything runs through `qwen3_5.py` / `qwen3_5_mtp.py`. It is also a **VLM** (vision tower live).
2. SGLang's NVFP4 target is **`RadixArk/Qwen3.8-27B-NVFP4`** — third-party, **ModelOpt `MIXED_PRECISION`**. **No official Qwen or `nvidia/` NVFP4 build of this model exists** (HF API: 0 results for nvidia+Qwen3.8).
3. Precision split: **NVFP4 W4A4** (group 16) on all 64 `mlp.{gate,up,down}_proj` **and `lm_head`**; **FP8 W8A8** on `self_attn.{q,k,v,o}` and `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`; norms/conv/`in_proj_a`/`in_proj_b`/embeddings/vision unquantized; **KV cache FP8 E4M3** auto-enabled from the checkpoint.
4. **MTP weights ship inside the NVFP4 checkpoint as 15 BF16 tensors** with unpacked shapes and zero scale tensors — verified from the safetensors header, not inferred.
5. MTP is unquantized by **two** mechanisms: checkpoint `ignore: ["mtp*","mtp.layers.0*"]`, and `_mtp_quant_config()` (`qwen3_5_mtp.py:47-78`) returning `None` for `modelopt_mixed`. The second is required so loaders allocate BF16 shapes. `mtp.fc` is a plain `nn.Linear`, unquantizable by construction.
6. **The draft's single layer is full-attention, not GDN** — `mtp_config.full_attention_interval = 1` forces it. Draft needs a KV cache, **no mamba state**. Separate `ModelRunner`, separate weights, no trunk sharing, no on-the-fly dequant.
7. Embedding + lm_head come from the target: `set_embed_and_head(embed, head)` then `set_lm_head_from_target()` **swaps the target's whole NVFP4 `ParallelLMHead` module** into the draft.
8. **GB10/sm_121 has no special gating** — `is_sm120_supported()` matches major==12 so sm_121 rides the SM120 path; `is_sm121()` exists but is dead code. FP4 GEMM auto-resolves to **`flashinfer_cutlass`** (SM100→cutedsl, SM80-90→marlin W4A16).
9. Recipe: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (NEXTN is an alias). **Chain, not tree.** No draft-model-path.
10. **accept_length ~3.31 at NVFP4 > ~3.22 BF16 > ~3.16 FP8** (GB300, cap 4). NVFP4 backbone + BF16 MTP head does not hurt acceptance. GSM8K 96.44% (fp8 KV) / 96.82% (bf16 KV).
11. `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors) is **broken on SGLang** (#34895, lm_head scale dropped) and **works on vLLM** (vllm#37291 merged). You are on the good side of that one.
12. Only GB10 measurement (#34872) is **FP8 with MTP at `--mem-fraction-static 0.70`, prefill CUDA graph enabled** — contradicting the shipped `0.95 + --disable-prefill-cuda-graph` Spark cell. 0.75 OOM'd at graph capture; GDN kernels silently fell back to **Triton**.

**Unknowns / gaps**
- **No CI test and no published measurement of Qwen3.8-27B NVFP4 + MTP on sm_121.** The GB300 NVFP4+MTP numbers are SM103; the GB10 datapoint is FP8. Your target cell is the intersection of two validated axes, itself unvalidated.
- Whether flashinfer `mm_fp4` cutlass is actually built for sm_121 in any given image — SGLang assumes it via the major==12 match but nothing asserts it.
- Whether the RadixArk checkpoint's calibration set/recipe is public (the benchmarks file calls the exact revision "W4A4-0811 (private)").
- Qwen3.8-27B FP8 and BF16 GSM8K on GB300 are `null` in the benchmarks file — only NVFP4 was scored.
- Whether `--speculative-num-draft-tokens > 4` lifts acceptance; the benchmarks file flags 4 as the *suspected* cap but nobody swept it.
- PRs #34859 and #34585 (the actual Qwen3.8 enablement work) are **still open with failing CI** — main's support is via the inherited Qwen3.5 path, so behavior may shift when they land.
- SGLang's `is_sm121()` being unused may be intentional or an oversight; no issue discusses it.