from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .registry import ModelConfig
from .yaml_utils import load_yaml_file

_STATE_FILENAME = "serving_runtime_state.json"
_ALLOWED_KV_CACHE_DTYPES = {"fp8_e5m2", "auto"}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    value: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {"field": self.field, "message": self.message}
        if self.value is not None:
            payload["value"] = self.value
        return payload


class StructuredValidationError(ValueError):
    def __init__(self, *, message: str, issues: list[ValidationIssue]) -> None:
        super().__init__(message)
        self.message = message
        self.issues = tuple(issues)

    def as_error_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": "validation_error",
                "message": self.message,
                "details": [issue.as_dict() for issue in self.issues],
            }
        }


def _isoformat_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _require_mapping(value: Any, *, field_name: str, issues: list[ValidationIssue]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    issues.append(ValidationIssue(field=field_name, message="must be a mapping", value=value))
    return {}


def _require_string(
    value: Any,
    *,
    field_name: str,
    issues: list[ValidationIssue],
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        issues.append(ValidationIssue(field=field_name, message="must be a string", value=value))
        return ""
    stripped = value.strip()
    if not allow_empty and not stripped:
        issues.append(ValidationIssue(field=field_name, message="must be a non-empty string", value=value))
        return ""
    if value != stripped:
        issues.append(
            ValidationIssue(field=field_name, message="must not include leading or trailing whitespace", value=value)
        )
    return stripped


def _require_int(
    value: Any,
    *,
    field_name: str,
    issues: list[ValidationIssue],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(field=field_name, message="must be an integer", value=value))
        return 0
    if minimum is not None and value < minimum:
        issues.append(ValidationIssue(field=field_name, message=f"must be >= {minimum}", value=value))
    if maximum is not None and value > maximum:
        issues.append(ValidationIssue(field=field_name, message=f"must be <= {maximum}", value=value))
    return value


def _require_float(
    value: Any,
    *,
    field_name: str,
    issues: list[ValidationIssue],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(ValidationIssue(field=field_name, message="must be a number", value=value))
        return 0.0
    rendered = float(value)
    if minimum is not None and rendered < minimum:
        issues.append(ValidationIssue(field=field_name, message=f"must be >= {minimum}", value=value))
    if maximum is not None and rendered > maximum:
        issues.append(ValidationIssue(field=field_name, message=f"must be <= {maximum}", value=value))
    return rendered


def _require_bool(value: Any, *, field_name: str, issues: list[ValidationIssue]) -> bool:
    if isinstance(value, bool):
        return value
    issues.append(ValidationIssue(field=field_name, message="must be a boolean", value=value))
    return False


def _validate_vllm_config(raw: Any, *, field_name: str, issues: list[ValidationIssue]) -> dict[str, Any]:
    mapping = _require_mapping(raw, field_name=field_name, issues=issues)
    kv_cache_dtype = _require_string(
        mapping.get("kv_cache_dtype"),
        field_name=f"{field_name}.kv_cache_dtype",
        issues=issues,
    )
    if kv_cache_dtype and kv_cache_dtype not in _ALLOWED_KV_CACHE_DTYPES:
        issues.append(
            ValidationIssue(
                field=f"{field_name}.kv_cache_dtype",
                message="must be one of fp8_e5m2 or auto",
                value=kv_cache_dtype,
            )
        )
    return {
        "max_num_seqs": _require_int(
            mapping.get("max_num_seqs"),
            field_name=f"{field_name}.max_num_seqs",
            issues=issues,
            minimum=1,
            maximum=64,
        ),
        "max_num_batched_tokens": _require_int(
            mapping.get("max_num_batched_tokens"),
            field_name=f"{field_name}.max_num_batched_tokens",
            issues=issues,
            minimum=1,
            maximum=16384,
        ),
        "enable_chunked_prefill": _require_bool(
            mapping.get("enable_chunked_prefill"),
            field_name=f"{field_name}.enable_chunked_prefill",
            issues=issues,
        ),
        "enable_prefix_caching": _require_bool(
            mapping.get("enable_prefix_caching"),
            field_name=f"{field_name}.enable_prefix_caching",
            issues=issues,
        ),
        "gpu_memory_utilization": _require_float(
            mapping.get("gpu_memory_utilization"),
            field_name=f"{field_name}.gpu_memory_utilization",
            issues=issues,
            minimum=0.0,
            maximum=0.95,
        ),
        "max_model_len": _require_int(
            mapping.get("max_model_len"),
            field_name=f"{field_name}.max_model_len",
            issues=issues,
            minimum=1,
            maximum=131072,
        ),
        "kv_cache_dtype": kv_cache_dtype,
    }


@dataclass(frozen=True)
class TunedConfigBundle:
    bundle_id: str
    produced_at: str
    weight_version_id: str
    model_id: str
    family_id: str
    workload_distribution_id: str
    vllm_config: dict[str, Any]
    request_shaping: dict[str, Any] = field(default_factory=dict)
    kernel_selection: dict[str, Any] = field(default_factory=dict)
    lora_policy: dict[str, Any] = field(default_factory=dict)
    layer_0_deltanet: dict[str, Any] = field(default_factory=dict)
    layer_0_gatedattn: dict[str, Any] = field(default_factory=dict)
    layer_0_fp8_gemm: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    measurement_trace_ref: str = ""
    search_trace_ref: str = ""
    baseline_bundle_id: str | None = None
    regression_guard: dict[str, Any] = field(default_factory=dict)
    safety_rails: dict[str, Any] = field(default_factory=dict)
    round_provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tuned_config_bundle": {
                "bundle_id": self.bundle_id,
                "produced_at": self.produced_at,
                "weight_version_id": self.weight_version_id,
                "model_id": self.model_id,
                "family_id": self.family_id,
                "workload_distribution_id": self.workload_distribution_id,
                "vllm_config": dict(self.vllm_config),
                "request_shaping": dict(self.request_shaping),
                "kernel_selection": dict(self.kernel_selection),
                "lora_policy": dict(self.lora_policy),
                "layer_0_deltanet": dict(self.layer_0_deltanet),
                "layer_0_gatedattn": dict(self.layer_0_gatedattn),
                "layer_0_fp8_gemm": dict(self.layer_0_fp8_gemm),
                "objective": dict(self.objective),
                "measurement_trace_ref": self.measurement_trace_ref,
                "search_trace_ref": self.search_trace_ref,
                "baseline_bundle_id": self.baseline_bundle_id,
                "regression_guard": dict(self.regression_guard),
                "safety_rails": dict(self.safety_rails),
                "round_provenance": dict(self.round_provenance),
            }
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "TunedConfigBundle":
        issues: list[ValidationIssue] = []
        root = _require_mapping(payload, field_name="tuned_config_bundle_root", issues=issues)
        bundle_payload = _require_mapping(root.get("tuned_config_bundle"), field_name="tuned_config_bundle", issues=issues)

        bundle = cls(
            bundle_id=_require_string(bundle_payload.get("bundle_id"), field_name="tuned_config_bundle.bundle_id", issues=issues),
            produced_at=_require_string(
                bundle_payload.get("produced_at"),
                field_name="tuned_config_bundle.produced_at",
                issues=issues,
            ),
            weight_version_id=_require_string(
                bundle_payload.get("weight_version_id"),
                field_name="tuned_config_bundle.weight_version_id",
                issues=issues,
            ),
            model_id=_require_string(bundle_payload.get("model_id"), field_name="tuned_config_bundle.model_id", issues=issues),
            family_id=_require_string(
                bundle_payload.get("family_id"),
                field_name="tuned_config_bundle.family_id",
                issues=issues,
            ),
            workload_distribution_id=_require_string(
                bundle_payload.get("workload_distribution_id"),
                field_name="tuned_config_bundle.workload_distribution_id",
                issues=issues,
            ),
            vllm_config=_validate_vllm_config(
                bundle_payload.get("vllm_config"),
                field_name="tuned_config_bundle.vllm_config",
                issues=issues,
            ),
            request_shaping=_require_mapping(
                bundle_payload.get("request_shaping", {}),
                field_name="tuned_config_bundle.request_shaping",
                issues=issues,
            ),
            kernel_selection=_require_mapping(
                bundle_payload.get("kernel_selection", {}),
                field_name="tuned_config_bundle.kernel_selection",
                issues=issues,
            ),
            lora_policy=_require_mapping(
                bundle_payload.get("lora_policy", {}),
                field_name="tuned_config_bundle.lora_policy",
                issues=issues,
            ),
            layer_0_deltanet=_require_mapping(
                bundle_payload.get("layer_0_deltanet", {}),
                field_name="tuned_config_bundle.layer_0_deltanet",
                issues=issues,
            ),
            layer_0_gatedattn=_require_mapping(
                bundle_payload.get("layer_0_gatedattn", {}),
                field_name="tuned_config_bundle.layer_0_gatedattn",
                issues=issues,
            ),
            layer_0_fp8_gemm=_require_mapping(
                bundle_payload.get("layer_0_fp8_gemm", {}),
                field_name="tuned_config_bundle.layer_0_fp8_gemm",
                issues=issues,
            ),
            objective=_require_mapping(
                bundle_payload.get("objective", {}),
                field_name="tuned_config_bundle.objective",
                issues=issues,
            ),
            measurement_trace_ref=_require_string(
                bundle_payload.get("measurement_trace_ref"),
                field_name="tuned_config_bundle.measurement_trace_ref",
                issues=issues,
            ),
            search_trace_ref=_require_string(
                bundle_payload.get("search_trace_ref"),
                field_name="tuned_config_bundle.search_trace_ref",
                issues=issues,
            ),
            baseline_bundle_id=(
                None
                if bundle_payload.get("baseline_bundle_id") is None
                else _require_string(
                    bundle_payload.get("baseline_bundle_id"),
                    field_name="tuned_config_bundle.baseline_bundle_id",
                    issues=issues,
                )
            ),
            regression_guard=_require_mapping(
                bundle_payload.get("regression_guard", {}),
                field_name="tuned_config_bundle.regression_guard",
                issues=issues,
            ),
            safety_rails=_require_mapping(
                bundle_payload.get("safety_rails", {}),
                field_name="tuned_config_bundle.safety_rails",
                issues=issues,
            ),
            round_provenance=_require_mapping(
                bundle_payload.get("round_provenance", {}),
                field_name="tuned_config_bundle.round_provenance",
                issues=issues,
            ),
        )
        if issues:
            raise StructuredValidationError(message="Invalid tuned-config bundle", issues=issues)
        return bundle

    @property
    def run_slug(self) -> str:
        timestamp = self.produced_at.replace(":", "").replace("+", "").replace("-", "")
        return f"{timestamp}_{self.bundle_id[:8]}"


def load_tuned_config_bundle(path: str | Path) -> TunedConfigBundle:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise StructuredValidationError(
            message="Invalid tuned-config bundle",
            issues=[ValidationIssue(field="bundle_path", message="file does not exist", value=str(bundle_path))],
        )
    return TunedConfigBundle.from_mapping(load_yaml_file(bundle_path))


def _resolve_descriptor_ref(descriptor_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return descriptor_path.parent / path


def compute_workload_distribution_id(descriptor_path: str | Path) -> str:
    descriptor = Path(descriptor_path)
    payload = load_yaml_file(descriptor)
    if not isinstance(payload, dict):
        raise ValueError(f"Workload descriptor must be a mapping: {descriptor}")
    seed_ref = payload.get("seed_trace_ref")
    holdout_ref = payload.get("holdout_trace_ref")
    if not isinstance(seed_ref, str) or not seed_ref.strip():
        raise ValueError("descriptor_missing_seed_trace_ref")
    if not isinstance(holdout_ref, str) or not holdout_ref.strip():
        raise ValueError("descriptor_missing_holdout_trace_ref")
    seed_hash = hashlib.sha256(_resolve_descriptor_ref(descriptor, seed_ref).read_bytes()).hexdigest()
    holdout_hash = hashlib.sha256(_resolve_descriptor_ref(descriptor, holdout_ref).read_bytes()).hexdigest()
    canonical_payload = dict(payload)
    canonical_payload["workload_distribution_id"] = None
    yaml_hash = hashlib.sha256(
        yaml.safe_dump(canonical_payload, sort_keys=True, default_flow_style=False).encode("utf-8")
    ).hexdigest()
    return hashlib.sha256((seed_hash + holdout_hash + yaml_hash).encode("ascii")).hexdigest()


def validate_bundle_load_policy(
    bundle: TunedConfigBundle,
    *,
    bundle_confidence_policy: str = "warn",
) -> list[dict[str, Any]]:
    if bundle_confidence_policy not in {"strict", "warn", "passthrough"}:
        raise ValueError("bundle_confidence_policy must be strict, warn, or passthrough")
    provenance = bundle.round_provenance
    warnings: list[dict[str, Any]] = []
    if provenance.get("round_type") == "l0a_select_only":
        raise StructuredValidationError(
            message="bundle-validity: refused",
            issues=[
                ValidationIssue(
                    field="round_provenance.round_type",
                    message="l0a_select_only bundle is intermediate and cannot be loaded in production mode",
                    value=provenance.get("round_type"),
                )
            ],
        )
    confidence = provenance.get("confidence", "unknown")
    latency_above_slo = bool(provenance.get("latency_above_slo", False))
    if confidence != "defensible":
        warnings.append({"code": "bundle_confidence_not_defensible", "confidence": confidence})
    if latency_above_slo:
        warnings.append({"code": "bundle_latency_above_slo"})
    descriptor_path = provenance.get("workload_descriptor_path")
    if not isinstance(descriptor_path, str) or not descriptor_path.strip():
        raise StructuredValidationError(
            message="bundle-validity: refused",
            issues=[
                ValidationIssue(
                    field="round_provenance.workload_descriptor_path",
                    message="missing_field: workload_descriptor_path",
                    value=descriptor_path,
                )
            ],
        )
    descriptor = Path(descriptor_path)
    if not descriptor.is_file():
        raise StructuredValidationError(
            message="bundle-validity: refused",
            issues=[
                ValidationIssue(
                    field="round_provenance.workload_descriptor_path",
                    message="descriptor_path_not_found",
                    value=descriptor_path,
                )
            ],
        )
    canonical_id = compute_workload_distribution_id(descriptor)
    if bundle.workload_distribution_id != canonical_id:
        raise StructuredValidationError(
            message="bundle-validity: refused",
            issues=[
                ValidationIssue(
                    field="workload_distribution_id",
                    message="mismatching_field: workload_distribution_id",
                    value=bundle.workload_distribution_id,
                )
            ],
        )
    if bundle_confidence_policy == "strict" and confidence != "defensible":
        raise StructuredValidationError(
            message="bundle-validity: refused",
            issues=[
                ValidationIssue(
                    field="round_provenance.confidence",
                    message="bundle confidence policy strict rejected non-defensible bundle",
                    value=confidence,
                )
            ],
        )
    if bundle_confidence_policy == "warn" and warnings:
        _LOGGER.warning(
            "non_defensible_tuned_config_bundle",
            extra={"bundle_id": bundle.bundle_id, "warnings": warnings},
        )
    return warnings


def persist_tuned_config_bundle(bundle: TunedConfigBundle, root: str | Path) -> Path:
    output_root = Path(root) / bundle.family_id / bundle.weight_version_id
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = output_root / f"{bundle.run_slug}.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle.as_dict(), sort_keys=False), encoding="utf-8")
    return bundle_path


def default_weight_version_id(config: ModelConfig) -> str:
    if config.hf_revision:
        return config.hf_revision
    digest = hashlib.sha256(f"{config.model_id}:{config.hf_repo}:{config.local_path}".encode("utf-8")).hexdigest()
    return digest[:40]


def make_tuned_config_bundle(
    *,
    model_id: str,
    family_id: str,
    weight_version_id: str,
    workload_distribution_id: str,
    vllm_config: dict[str, Any],
    objective: dict[str, Any],
    measurement_trace_ref: str,
    search_trace_ref: str,
    baseline_bundle_id: str | None,
    regression_guard: dict[str, Any],
    safety_rails: dict[str, Any],
    round_provenance: dict[str, Any] | None = None,
    request_shaping: dict[str, Any] | None = None,
    kernel_selection: dict[str, Any] | None = None,
    lora_policy: dict[str, Any] | None = None,
    layer_0_deltanet: dict[str, Any] | None = None,
    layer_0_gatedattn: dict[str, Any] | None = None,
    layer_0_fp8_gemm: dict[str, Any] | None = None,
) -> TunedConfigBundle:
    return TunedConfigBundle(
        bundle_id=str(uuid4()),
        produced_at=_isoformat_now(),
        weight_version_id=weight_version_id,
        model_id=model_id,
        family_id=family_id,
        workload_distribution_id=workload_distribution_id,
        vllm_config=dict(vllm_config),
        request_shaping=dict(request_shaping or {}),
        kernel_selection=dict(kernel_selection or {}),
        lora_policy=dict(lora_policy or {}),
        layer_0_deltanet=dict(layer_0_deltanet or {}),
        layer_0_gatedattn=dict(layer_0_gatedattn or {}),
        layer_0_fp8_gemm=dict(layer_0_fp8_gemm or {}),
        objective=dict(objective),
        measurement_trace_ref=measurement_trace_ref,
        search_trace_ref=search_trace_ref,
        baseline_bundle_id=baseline_bundle_id,
        regression_guard=dict(regression_guard),
        safety_rails=dict(safety_rails),
        round_provenance=dict(round_provenance or {}),
    )


def apply_tuned_vllm_config(config: ModelConfig, bundle: TunedConfigBundle) -> ModelConfig:
    if bundle.model_id != config.model_id:
        raise ValueError(
            f"Tuned-config bundle model_id {bundle.model_id!r} does not match registry model {config.model_id!r}"
        )
    tuned = bundle.vllm_config
    return replace(
        config,
        kv_cache_dtype=str(tuned["kv_cache_dtype"]),
        max_model_len=int(tuned["max_model_len"]),
        gpu_memory_utilization=float(tuned["gpu_memory_utilization"]),
        max_num_batched_tokens=int(tuned["max_num_batched_tokens"]),
        max_num_seqs=int(tuned["max_num_seqs"]),
        enable_chunked_prefill=bool(tuned["enable_chunked_prefill"]),
        enable_prefix_caching=bool(tuned["enable_prefix_caching"]),
    )


@dataclass(frozen=True)
class ServingRuntimeState:
    current_model_id: str | None = None
    active_tuned_config_path: str | None = None
    active_tuned_config_id: str | None = None
    current_weight_version_id: str | None = None
    last_known_good_model_id: str | None = None
    last_known_good_weight_version_id: str | None = None
    last_known_good_tuned_config_path: str | None = None
    last_known_good_tuned_config_id: str | None = None
    status: str = "IDLE"
    invalidate_count: int = 0
    invalidated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_model_id": self.current_model_id,
            "active_tuned_config_path": self.active_tuned_config_path,
            "active_tuned_config_id": self.active_tuned_config_id,
            "current_weight_version_id": self.current_weight_version_id,
            "last_known_good_model_id": self.last_known_good_model_id,
            "last_known_good_weight_version_id": self.last_known_good_weight_version_id,
            "last_known_good_tuned_config_path": self.last_known_good_tuned_config_path,
            "last_known_good_tuned_config_id": self.last_known_good_tuned_config_id,
            "status": self.status,
            "invalidate_count": self.invalidate_count,
            "invalidated_at": self.invalidated_at,
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "ServingRuntimeState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            current_model_id=_optional_string(payload.get("current_model_id")),
            active_tuned_config_path=_optional_string(payload.get("active_tuned_config_path")),
            active_tuned_config_id=_optional_string(payload.get("active_tuned_config_id")),
            current_weight_version_id=_optional_string(payload.get("current_weight_version_id")),
            last_known_good_model_id=_optional_string(payload.get("last_known_good_model_id")),
            last_known_good_weight_version_id=_optional_string(payload.get("last_known_good_weight_version_id")),
            last_known_good_tuned_config_path=_optional_string(payload.get("last_known_good_tuned_config_path")),
            last_known_good_tuned_config_id=_optional_string(payload.get("last_known_good_tuned_config_id")),
            status=_optional_string(payload.get("status")) or "IDLE",
            invalidate_count=int(payload.get("invalidate_count", 0) or 0),
            invalidated_at=_optional_string(payload.get("invalidated_at")),
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


class RuntimeStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / _STATE_FILENAME

    def load(self) -> ServingRuntimeState:
        if not self.path.exists():
            return ServingRuntimeState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid runtime state file {self.path}: {exc}") from exc
        return ServingRuntimeState.from_mapping(payload)

    def save(self, state: ServingRuntimeState) -> ServingRuntimeState:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return state

    def activate_bundle(self, bundle_path: str | Path, bundle: TunedConfigBundle) -> ServingRuntimeState:
        state = self.load()
        next_state = replace(
            state,
            active_tuned_config_path=str(Path(bundle_path)),
            active_tuned_config_id=bundle.bundle_id,
            current_model_id=state.current_model_id or bundle.model_id,
            current_weight_version_id=bundle.weight_version_id,
            last_known_good_model_id=bundle.model_id,
            last_known_good_weight_version_id=bundle.weight_version_id,
            last_known_good_tuned_config_path=str(Path(bundle_path)),
            last_known_good_tuned_config_id=bundle.bundle_id,
            status="READY",
        )
        return self.save(next_state)

    def record_start(
        self,
        *,
        model_id: str,
        weight_version_id: str,
        active_bundle_path: str | None,
        active_bundle_id: str | None,
    ) -> ServingRuntimeState:
        state = self.load()
        next_state = replace(
            state,
            current_model_id=model_id,
            current_weight_version_id=weight_version_id,
            active_tuned_config_path=active_bundle_path,
            active_tuned_config_id=active_bundle_id,
            last_known_good_model_id=model_id,
            last_known_good_weight_version_id=weight_version_id,
            last_known_good_tuned_config_path=active_bundle_path,
            last_known_good_tuned_config_id=active_bundle_id,
            status="READY",
        )
        return self.save(next_state)

    def clear_active_bundle(self) -> ServingRuntimeState:
        state = self.load()
        next_state = replace(
            state,
            active_tuned_config_path=None,
            active_tuned_config_id=None,
            status="READY",
        )
        return self.save(next_state)

    def record_invalidate(self, *, weight_version_id: str) -> ServingRuntimeState:
        state = self.load()
        updating = replace(
            state,
            current_weight_version_id=weight_version_id,
            last_known_good_weight_version_id=weight_version_id,
            status="UPDATING",
            invalidate_count=state.invalidate_count + 1,
            invalidated_at=_isoformat_now(),
        )
        saved = self.save(updating)
        ready = replace(saved, status="READY")
        return self.save(ready)
