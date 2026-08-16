#!/usr/bin/env python3
"""CPU-only verification that the FR14 lm_head patch resolves the RadixArk
checkpoint's head to ModelOptNvFp4LinearMethod. Runs inside the pinned image
after fr14_patch_nvfp4_lmhead.py. No GPU, no model weights are read."""
import json

out = {}

from vllm.model_executor.layers.quantization.modelopt import (  # noqa: E402
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4LinearMethod,
    ModelOptFp8LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_vl import (  # noqa: E402
    Qwen3VLForConditionalGeneration,
)

raw = json.load(open("/models/qwen3.8-27b-nvfp4-radixark/hf_quant_config.json"))
q = raw["quantization"]

out["override_quantization_method"] = ModelOptMixedPrecisionConfig.override_quantization_method(
    {"quant_method": "modelopt", "quantization": q}, None
)

cfg = ModelOptMixedPrecisionConfig.from_config(raw)
out["config_class"] = type(cfg).__name__
out["get_name"] = cfg.get_name()

mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
cfg.packed_modules_mapping = {}
cfg.apply_vllm_mapper(mapper)
out["exclude_modules_after_mapper"] = list(cfg.exclude_modules)
out["lm_head_key_present_after_mapper"] = "lm_head" in cfg.quantized_layers
out["language_model_lm_head_key_present"] = "language_model.lm_head" in cfg.quantized_layers
out["body_key_sample_after_mapper"] = [
    k for k in cfg.quantized_layers if k.endswith("layers.0.mlp.gate_proj")
]

# G3 demonstration: the runtime prefix misses, the basename hits.
out["resolve__language_model.lm_head"] = cfg._resolve_quant_algo("language_model.lm_head")
out["resolve__lm_head"] = cfg._resolve_quant_algo("lm_head")

# G2 demonstration: dispatch on a real ParallelLMHead instance (constructed
# without __init__, so no distributed/GPU state is needed -- isinstance is all
# get_quant_method looks at).
fake_head = object.__new__(ParallelLMHead)
out["fake_head_is_ParallelLMHead"] = isinstance(fake_head, ParallelLMHead)
out["fake_head_is_VocabParallelEmbedding"] = isinstance(fake_head, VocabParallelEmbedding)

# The two linear-method constructors reach for state that only exists on a GPU
# box inside a live engine: ModelOptNvFp4LinearMethod calls
# init_nvfp4_linear_kernel() (needs a visible CUDA device) and
# ModelOptFp8LinearMethod calls get_current_vllm_config() (needs a
# set_current_vllm_config context). Both are strictly DOWNSTREAM of the
# dispatch decision under test, so we stub them to keep this check CPU-only.
import torch  # noqa: E402
import vllm.model_executor.layers.quantization.modelopt as _mo  # noqa: E402


class _StubVllmConfig:
    class model_config:
        dtype = torch.bfloat16


_mo.init_nvfp4_linear_kernel = lambda *a, **k: None
_mo.get_current_vllm_config = lambda *a, **k: _StubVllmConfig
out["cpu_stubs"] = ["init_nvfp4_linear_kernel", "get_current_vllm_config"]


def probe(layer, prefix):
    return type(cfg.get_quant_method(layer, prefix=prefix)).__name__


out["quant_method__language_model.lm_head"] = probe(fake_head, "language_model.lm_head")
out["quant_method__lm_head"] = probe(fake_head, "lm_head")

# The MTP drafter's own modules must stay unquantized via exclude_modules.
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402

fake_lin = object.__new__(ColumnParallelLinear)
out["quant_method__mtp.fc"] = probe(fake_lin, "mtp.fc")
out["quant_method__mtp.layers.0.mlp.gate_proj"] = probe(
    fake_lin, "mtp.layers.0.mlp.gate_proj"
)
out["quant_method__body_gate_proj"] = probe(
    fake_lin, "language_model.model.layers.0.mlp.gate_proj"
)
out["quant_method__body_out_proj"] = probe(
    fake_lin, "language_model.model.layers.0.linear_attn.out_proj"
)
# The visual tower is BF16 and must stay unquantized.
out["quant_method__visual"] = probe(fake_lin, "visual.blocks.0.attn.qkv")

ok = (
    out["quant_method__language_model.lm_head"] == "ModelOptNvFp4LinearMethod"
    and out["quant_method__lm_head"] == "ModelOptNvFp4LinearMethod"
    and out["quant_method__mtp.fc"] == "UnquantizedLinearMethod"
    and out["quant_method__mtp.layers.0.mlp.gate_proj"] == "UnquantizedLinearMethod"
    and out["quant_method__body_gate_proj"] == "ModelOptNvFp4LinearMethod"
    and out["quant_method__body_out_proj"] == "ModelOptFp8LinearMethod"
)
out["verdict"] = "PASS" if ok else "FAIL"
print(json.dumps(out, indent=1))
raise SystemExit(0 if ok else 1)
