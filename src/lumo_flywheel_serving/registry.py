from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    lora_modules_raw = raw.get("lora_modules", {})
    if not isinstance(lora_modules_raw, dict):
        raise ValueError(f"Model {model_id} lora_modules must be a mapping of adapter_name -> container_path")
    lora_modules = tuple(
        (adapter_name, Path(adapter_path))
        for adapter_name, adapter_path in lora_modules_raw.items()
    )
    if lora_modules and raw.get("max_lora_rank") is None:
        raise ValueError(f"Model {model_id} must define max_lora_rank when lora_modules are configured")
    return ModelConfig(
        model_id=model_id,
        hf_repo=raw.get("hf_repo", ""),
        hf_revision=raw.get("hf_revision"),
        local_path=Path(local_path),
        served_model_name=raw.get("served_model_name", model_id),
        quantization=raw["quantization"],
        dtype=raw["dtype"],
        kv_cache_dtype=raw.get("kv_cache_dtype", "fp8_e5m2"),
        max_model_len=int(raw["max_model_len"]),
        gpu_memory_utilization=float(raw["gpu_memory_utilization"]),
        max_num_batched_tokens=int(raw.get("max_num_batched_tokens", 8192)),
        max_num_seqs=int(raw.get("max_num_seqs", 4)),
        lora_modules=lora_modules,
        max_lora_rank=None if raw.get("max_lora_rank") is None else int(raw["max_lora_rank"]),
        sprint0_gate=raw.get("sprint0_gate"),
    )


def load_registry(path: str | Path) -> dict[str, ModelConfig]:
    registry_path = Path(path)
    raw = yaml.safe_load(registry_path.read_text())
    models = raw.get("models", {})
    return {model_id: _model_from_mapping(model_id, mapping) for model_id, mapping in models.items()}
