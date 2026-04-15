from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .yaml_utils import load_yaml_file


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hf_repo: str
    local_path: Path
    served_model_name: str
    quantization: str
    dtype: str
    kv_cache_dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    lora_modules: tuple[tuple[str, Path], ...] = ()
    max_lora_rank: int | None = None
    hf_revision: str | None = None
    sprint0_gate: str | None = None


def _model_from_mapping(model_id: str, raw: dict[str, Any]) -> ModelConfig:
    local_path = raw.get("local_path")
    if not local_path:
        raise ValueError(
            f"Model {model_id} is missing local_path. Leave unresolved placeholders commented out until they are real models."
        )
    local_path_obj = Path(local_path)
    if not str(local_path_obj).startswith("/models/"):
        raise ValueError(f"Model {model_id} local_path must be a container path under /models; got {local_path_obj}")
    served_model_name = raw.get("served_model_name", model_id)
    if not isinstance(served_model_name, str) or not served_model_name.strip():
        raise ValueError(f"Model {model_id} served_model_name must be a non-empty string")
    quantization = raw["quantization"]
    if quantization != "fp8":
        raise ValueError(f"Model {model_id} quantization must be 'fp8'; got {quantization!r}")
    dtype = raw["dtype"]
    if dtype != "auto":
        raise ValueError(f"Model {model_id} dtype must be 'auto'; got {dtype!r}")
    kv_cache_dtype = raw.get("kv_cache_dtype", "fp8_e5m2")
    if kv_cache_dtype not in {"fp8_e5m2", "auto"}:
        raise ValueError(
            f"Model {model_id} kv_cache_dtype must be 'fp8_e5m2' or 'auto'; got {kv_cache_dtype!r}"
        )
    lora_modules_raw = raw.get("lora_modules", {})
    if not isinstance(lora_modules_raw, dict):
        raise ValueError(f"Model {model_id} lora_modules must be a mapping of adapter_name -> container_path")
    lora_modules = tuple(
        (adapter_name, Path(adapter_path))
        for adapter_name, adapter_path in lora_modules_raw.items()
    )
    for adapter_name, adapter_path in lora_modules:
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise ValueError(f"Model {model_id} lora_modules adapter names must be non-empty strings")
        if not str(adapter_path):
            raise ValueError(f"Model {model_id} lora_modules[{adapter_name}] must be a non-empty path")
        if adapter_name == served_model_name:
            raise ValueError(
                f"Model {model_id} lora_modules adapter '{adapter_name}' collides with served_model_name "
                f"'{served_model_name}'"
            )
    if lora_modules and raw.get("max_lora_rank") is None:
        raise ValueError(f"Model {model_id} must define max_lora_rank when lora_modules are configured")
    max_model_len = int(raw["max_model_len"])
    if max_model_len < 1:
        raise ValueError(f"Model {model_id} max_model_len must be >= 1; got {max_model_len}")
    gpu_memory_utilization = float(raw["gpu_memory_utilization"])
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError(
            f"Model {model_id} gpu_memory_utilization must be > 0.0 and < 1.0; got {gpu_memory_utilization}"
        )
    max_num_batched_tokens = int(raw.get("max_num_batched_tokens", 8192))
    if max_num_batched_tokens < 1:
        raise ValueError(f"Model {model_id} max_num_batched_tokens must be >= 1; got {max_num_batched_tokens}")
    max_num_seqs = int(raw.get("max_num_seqs", 4))
    if max_num_seqs < 1:
        raise ValueError(f"Model {model_id} max_num_seqs must be >= 1; got {max_num_seqs}")
    max_lora_rank = raw.get("max_lora_rank")
    if max_lora_rank is not None:
        max_lora_rank = int(max_lora_rank)
        if max_lora_rank < 1:
            raise ValueError(f"Model {model_id} max_lora_rank must be >= 1; got {max_lora_rank}")
    return ModelConfig(
        model_id=model_id,
        hf_repo=raw.get("hf_repo", ""),
        hf_revision=raw.get("hf_revision"),
        local_path=local_path_obj,
        served_model_name=served_model_name,
        quantization=quantization,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        lora_modules=lora_modules,
        max_lora_rank=max_lora_rank,
        sprint0_gate=raw.get("sprint0_gate"),
    )


def _exposed_model_ids(config: ModelConfig) -> tuple[str, ...]:
    return (config.served_model_name, *(adapter_name for adapter_name, _adapter_path in config.lora_modules))


def load_registry(path: str | Path) -> dict[str, ModelConfig]:
    registry_path = Path(path)
    raw = load_yaml_file(registry_path)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Registry {registry_path} must be a YAML mapping")
    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"Registry {registry_path} must define 'models' as a mapping")
    registry = {model_id: _model_from_mapping(model_id, mapping) for model_id, mapping in models.items()}

    exposed_ids: dict[str, str] = {}
    for model_id, config in registry.items():
        for exposed_id in _exposed_model_ids(config):
            owner = exposed_ids.get(exposed_id)
            if owner is not None:
                raise ValueError(
                    f"Registry model id collision: exposed model id '{exposed_id}' is declared by both "
                    f"'{owner}' and '{model_id}'. served_model_name and LoRA adapter names must be globally unique."
                )
            exposed_ids[exposed_id] = model_id
    return registry
