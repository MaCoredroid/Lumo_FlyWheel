from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import quote

import requests

from .data_pool import SCENARIO_TYPES, _find_manifest_variant, load_codex_long_manifest, make_scenario_id, sha256_file
from .registry import ModelConfig, load_registry
from .yaml_utils import load_yaml_file

logger = logging.getLogger(__name__)

CONTAINER_RESOURCE_LIMITS: dict[str, Any] = {
    "mem_limit": "32g",
    "cpus": 4.0,
    "pids_limit": 1024,
    "ulimits": [
        {"name": "nofile", "soft": 65536, "hard": 65536},
    ],
    "storage_opt": {"size": "50G"},
}

_INFRASTRUCTURE_ERROR_PATTERNS = (
    "CUDA out of memory",
    "vLLM server",
    "docker: Error",
    "No space left on device",
    "codex: internal error",
    "ConnectionRefusedError",
)
_PINNED_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
_PINNED_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$", re.IGNORECASE)
_CODEX_LONG_VARIANT_TIERS = {"small-investigative", "standard", "pro"}
_MILESTONE_PASS_RULES = {"all", "any"}


class OrchestratorError(RuntimeError):
    """Base class for LLD-03 orchestration failures."""


class TaskDispatchError(OrchestratorError):
    """Raised when a task cannot be dispatched safely."""


class HealthCheckError(TaskDispatchError):
    """Raised when the model server is reachable but not ready."""


class CacheFlushError(TaskDispatchError):
    """Raised when the prefix cache cannot be flushed."""


class ConfigError(TaskDispatchError):
    """Raised when required runtime configuration is missing."""


class DuplicateClaimError(TaskDispatchError):
    """Raised when another worker already claimed the requested run slot."""


class ManifestMismatchError(TaskDispatchError):
    """Raised when frozen benchmark artifacts drift from the locked manifest."""

    def __init__(self, message: str, *, affected_artifact: str | None = None) -> None:
        super().__init__(message)
        self.affected_artifact = affected_artifact


@dataclass(frozen=True)
class TaskSpec:
    track: str
    pool_or_split: str
    scenario_id: str
    model_id: str
    harness: str
    seed: int
    repo: str | None = None
    base_commit: str | None = None
    instance_id: str | None = None
    prompt: str | None = None
    family_id: str | None = None
    variant_id: str | None = None
    image_digest: str | None = None
    scenario_type: str | None = None
    dispatch_decision: str = "proceed"
    attempt: int = 1
    regrade_snapshot_ref: str | None = None
    timeout_seconds: int = 0

    def __post_init__(self) -> None:
        if self.track not in {"swe_bench", "codex_long"}:
            raise ValueError(f"Unsupported task track '{self.track}'")
        if self.harness != "codex":
            raise ValueError("LLD-03 only orchestrates the codex harness")
        if self.dispatch_decision not in {"proceed", "retry", "regrade_needed", "rerun_needed"}:
            raise ValueError(f"Unsupported dispatch_decision '{self.dispatch_decision}'")
        if self.attempt < 1:
            raise ValueError("attempt must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.track == "swe_bench" and not self.instance_id:
            object.__setattr__(self, "instance_id", self.scenario_id)
        if self.track == "swe_bench" and self.instance_id != self.scenario_id:
            raise ValueError(
                "SWE-bench tasks must use scenario_id equal to instance_id; "
                f"got scenario_id='{self.scenario_id}' instance_id='{self.instance_id}'"
            )
        if self.track == "codex_long":
            missing = [
                field_name
                for field_name in ("family_id", "variant_id", "scenario_type")
                if getattr(self, field_name) in {None, ""}
            ]
            if missing:
                raise ValueError(f"Codex-Long tasks require non-empty fields: {missing}")
            expected_scenario_id = make_scenario_id(str(self.family_id), str(self.variant_id))
            if self.scenario_id != expected_scenario_id:
                raise ValueError(
                    "Codex-Long tasks must use canonical scenario_id "
                    f"'{expected_scenario_id}', got '{self.scenario_id}'"
                )
            if self.scenario_type not in SCENARIO_TYPES:
                raise ValueError(
                    f"Codex-Long tasks require scenario_type in {sorted(SCENARIO_TYPES)}; "
                    f"got '{self.scenario_type}'"
                )
            if self.dispatch_decision != "regrade_needed" and not self.image_digest:
                raise ValueError(
                    "Codex-Long tasks require image_digest unless dispatch_decision='regrade_needed'"
                )
        if self.dispatch_decision in {"retry", "rerun_needed", "regrade_needed"} and self.attempt == 1:
            raise ValueError(
                f"dispatch_decision='{self.dispatch_decision}' requires attempt > 1"
            )
        if self.dispatch_decision == "regrade_needed":
            if self.track != "codex_long":
                raise ValueError("Only Codex-Long tasks may use dispatch_decision='regrade_needed'")
            if not self.regrade_snapshot_ref:
                raise ValueError("regrade_needed tasks require regrade_snapshot_ref")


@dataclass(frozen=True)
class ContainerContext:
    container_id: str
    container_name: str
    workspace_path: str
    track: str
    family_id: str | None = None
    variant_id: str | None = None


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stderr: str
    timed_out: bool


@dataclass(frozen=True)
class CodexResult:
    trajectory_path: str
    exit_code: int
    wall_time_seconds: float
    timed_out: bool
    stderr: str


@dataclass(frozen=True)
class RunResult:
    task: TaskSpec
    outcome: str
    trajectory_path: str | None
    wall_time_seconds: float | None
    verify_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class VllmConfig:
    bind_host: str
    client_host: str
    port: int


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    subnet: str
    gateway: str
    proxy_port: int


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str
    trajectory_dir: str
    patch_dir: str
    grading_dir: str
    manifest_path: str
    scenario_families_dir: str
    verifiers_dir: str
    verifier_data_dir: str


@dataclass(frozen=True)
class GradingConfig:
    grader_image_tag: str
    phase2_default_timeout: int
    phase3_timeout: int


@dataclass(frozen=True)
class ExecutionConfig:
    swe_bench_timeout: int
    codex_long_timeout: int
    health_check_retries: int
    health_check_delay: float


@dataclass(frozen=True)
class CodexInstallConfig:
    binary_path: str
    node_modules_path: str
    node_binary_path: str


@dataclass(frozen=True)
class OrchestratorConfig:
    vllm: VllmConfig
    network: NetworkConfig
    model_registry_path: str
    model_registry: dict[str, ModelConfig]
    paths: PathsConfig
    grading: GradingConfig
    execution: ExecutionConfig
    codex: CodexInstallConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> OrchestratorConfig:
        raw = load_yaml_file(path) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Orchestrator config {path} must be a YAML mapping")
        registry_path = str(raw["model_registry_path"])
        return cls(
            vllm=VllmConfig(**raw["vllm"]),
            network=NetworkConfig(**raw["network"]),
            model_registry_path=registry_path,
            model_registry=load_registry(registry_path),
            paths=PathsConfig(**raw["paths"]),
            grading=GradingConfig(**raw["grading"]),
            execution=ExecutionConfig(**raw["execution"]),
            codex=CodexInstallConfig(**raw["codex"]),
        )


class LatencyCaptureProtocol(Protocol):
    async def snapshot_before(self, task_id: str) -> None: ...

    async def snapshot_after(self, task_id: str) -> None: ...


@dataclass
class OrchestratorHooks:
    latency_capture: LatencyCaptureProtocol
    setup_swe_bench_container: Callable[[TaskSpec, OrchestratorConfig], Awaitable[ContainerContext]]
    setup_codex_long_container: Callable[[TaskSpec, dict[str, Any], OrchestratorConfig], Awaitable[ContainerContext]]
    invoke_codex: Callable[[ContainerContext, TaskSpec, str], Awaitable[CodexResult]]
    extract_swe_bench_patch: Callable[[ContainerContext, str, TaskSpec], Awaitable[str | None]]
    teardown_swe_bench_container: Callable[[ContainerContext], Awaitable[None]]
    drive_swe_bench_eval: Callable[[str, str, str], Awaitable[str]]
    phase1_snapshot: Callable[[ContainerContext, str], Awaitable[str]]
    load_family_spec: Callable[[str], dict[str, Any]]
    phase2_functional_checks: Callable[[str, TaskSpec, dict[str, Any], str], Awaitable[None]]
    phase3_integrity_verification: Callable[[str, TaskSpec, str, str], Awaitable[dict[str, Any]]]
    cleanup_grading: Callable[[str, str, bool], Awaitable[None]]
    docker_rm: Callable[[str, bool], Awaitable[None]]
    docker_image_exists: Callable[[str], Awaitable[bool]] | None = None
    health_check: Callable[..., Awaitable[None]] = None  # type: ignore[assignment]
    flush_prefix_cache: Callable[..., Awaitable[None]] = None  # type: ignore[assignment]
    verify_pre_run_hashes: Callable[[TaskSpec, dict[str, Any]], None] = None  # type: ignore[assignment]
    verify_pre_grading_hashes: Callable[[TaskSpec, dict[str, Any], str], None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.health_check is None:
            self.health_check = health_check
        if self.flush_prefix_cache is None:
            self.flush_prefix_cache = flush_prefix_cache
        if self.verify_pre_run_hashes is None:
            self.verify_pre_run_hashes = verify_pre_run_hashes
        if self.verify_pre_grading_hashes is None:
            self.verify_pre_grading_hashes = verify_pre_grading_hashes
        if self.docker_image_exists is None:
            self.docker_image_exists = docker_image_exists


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _call_with_supported_kwargs(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters.values()
    accepts_var_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_var_kwargs:
        return func(*args, **kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **supported_kwargs)


def _response_status(response: Any) -> int | None:
    for attr_name in ("status", "status_code"):
        status = getattr(response, attr_name, None)
        if isinstance(status, int):
            return status
    return None


def _response_json(response: Any) -> dict[str, Any]:
    payload = response.json() if callable(getattr(response, "json", None)) else {}
    if not isinstance(payload, dict):
        raise HealthCheckError("Model listing response did not return a JSON object")
    return payload


def _canonical_sha256(value: str) -> str:
    if value.startswith("sha256:"):
        return value
    if "@sha256:" in value:
        return value.split("@", 1)[1]
    return f"sha256:{value}"


def _vllm_api_key() -> str:
    return os.environ.get("VLLM_API_KEY") or "EMPTY"


def _vllm_request_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_vllm_api_key()}"}


def write_text(path: str | Path, content: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding="utf-8")


def find_manifest_entry(manifest: dict[str, Any], family_id: str, variant_id: str) -> dict[str, Any]:
    return _find_manifest_variant(manifest, family_id, variant_id)


def sha256_tree(root: str | Path) -> str:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected directory tree, got: {root_path}")
    import hashlib

    hasher = hashlib.sha256()
    for file_path in sorted(path for path in root_path.rglob("*") if path.is_file()):
        relative = file_path.relative_to(root_path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def sha256_path_set(paths: list[str | Path], *, repo_root: str | Path) -> str:
    root_path = Path(repo_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected directory tree, got: {root_path}")

    normalized_files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else root_path / candidate
        if not path.exists():
            raise FileNotFoundError(f"Declared asset path does not exist: {path}")
        if path.is_dir():
            file_paths = sorted(child for child in path.rglob("*") if child.is_file())
        else:
            file_paths = [path]
        for file_path in file_paths:
            if file_path in seen:
                continue
            seen.add(file_path)
            normalized_files.append(file_path)

    import hashlib

    hasher = hashlib.sha256()
    for file_path in sorted(normalized_files):
        relative = file_path.relative_to(root_path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def _render_variant_path_template(value: str, *, family_id: str, variant_id: str) -> str:
    rendered = value
    replacements = {
        "<family>": family_id,
        "<family_id>": family_id,
        "<variant>": variant_id,
        "<variant_id>": variant_id,
    }
    for needle, replacement in replacements.items():
        rendered = rendered.replace(needle, replacement)
    return rendered


def _normalize_declared_verifier_data_path(
    value: str,
    *,
    family_id: str,
    variant_id: str,
) -> str:
    rendered = _render_variant_path_template(value, family_id=family_id, variant_id=variant_id).strip()
    if not rendered:
        raise ValueError("path must be non-empty")
    if re.search(r"<[^>]+>", rendered):
        raise ValueError(f"path contains an unsupported template placeholder: {rendered}")

    candidate = Path(rendered)
    if candidate.is_absolute():
        raise ValueError(f"path must be relative to the repo root: {rendered}")

    normalized_parts: list[str] = []
    for part in candidate.parts:
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError(f"path must not escape verifier_data/: {rendered}")
        normalized_parts.append(part)

    if len(normalized_parts) < 2 or normalized_parts[0] != "verifier_data":
        raise ValueError(f"path must resolve under verifier_data/: {rendered}")
    return Path(*normalized_parts).as_posix()


def _merged_mapping(defaults: Any, override: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(defaults, dict):
        merged.update(defaults)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def get_variant_spec(family_spec: dict[str, Any], variant_id: str) -> dict[str, Any]:
    variants = family_spec.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Family spec variants must be a list")
    for variant in variants:
        if isinstance(variant, dict) and variant.get("variant_id") == variant_id:
            return variant
    raise KeyError(f"Variant '{variant_id}' not found in family spec")


def get_variant_quality_contract(
    family_spec: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    variant = get_variant_spec(family_spec, variant_id)
    return {
        "variant": variant,
        "tier": variant.get("tier", family_spec.get("tier")),
        "oracle": _merged_mapping(family_spec.get("oracle"), variant.get("oracle")),
        "hidden_tests": _merged_mapping(family_spec.get("hidden_tests"), variant.get("hidden_tests")),
        "red_team": _merged_mapping(family_spec.get("red_team"), variant.get("red_team")),
        "calibration": _merged_mapping(family_spec.get("calibration"), variant.get("calibration")),
    }


def resolve_milestone_test_nodes(
    family_spec: dict[str, Any],
    *,
    family_id: str,
    variant_id: str,
) -> dict[str, tuple[str, ...]]:
    contract = get_variant_quality_contract(family_spec, variant_id)
    hidden_tests = contract["hidden_tests"]
    milestone_map = hidden_tests.get("milestone_map") if isinstance(hidden_tests, dict) else None
    milestones = family_spec.get("milestones")
    if not isinstance(milestones, list):
        raise ValueError("Family spec milestones must be a list")

    resolved: dict[str, tuple[str, ...]] = {}
    for milestone in milestones:
        if not isinstance(milestone, dict):
            raise ValueError("Family spec milestones entries must be mappings")
        milestone_id = str(milestone.get("id", "")).strip()
        raw_nodes = milestone.get("test_nodes")
        if raw_nodes == "variant_scoped":
            if not isinstance(milestone_map, dict):
                raise ValueError(
                    f"Family '{family_id}' milestone '{milestone_id}' requires hidden_tests.milestone_map "
                    f"for variant '{variant_id}'"
                )
            raw_nodes = milestone_map.get(milestone_id)
            if raw_nodes is None:
                raise ValueError(
                    f"Family '{family_id}' milestone '{milestone_id}' is missing hidden_tests.milestone_map "
                    f"for variant '{variant_id}'"
                )
        if raw_nodes is None:
            continue
        if isinstance(raw_nodes, str):
            nodes = [raw_nodes]
        elif isinstance(raw_nodes, list):
            nodes = raw_nodes
        else:
            raise ValueError(
                f"Family '{family_id}' milestone '{milestone_id}' must use a string, list, or "
                "'variant_scoped' test_nodes contract"
            )
        resolved_nodes = []
        for index, node in enumerate(nodes):
            if not isinstance(node, str) or not node.strip():
                raise ValueError(
                    f"Family '{family_id}' milestone '{milestone_id}' has invalid test_nodes[{index}] "
                    f"for variant '{variant_id}'"
                )
            resolved_nodes.append(node)
        resolved[milestone_id] = tuple(resolved_nodes)
    return resolved


def milestone_contract_hash(
    family_spec: dict[str, Any],
    *,
    family_id: str,
    variant_id: str,
    milestone_id: str,
) -> str:
    milestones = family_spec.get("milestones")
    if not isinstance(milestones, list):
        raise ValueError("Family spec milestones must be a list")
    resolved_nodes = resolve_milestone_test_nodes(
        family_spec,
        family_id=family_id,
        variant_id=variant_id,
    )
    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        if str(milestone.get("id", "")).strip() != milestone_id:
            continue
        payload = {
            "id": milestone_id,
            "pass_rule": milestone.get("pass_rule", "all"),
            "test_nodes": list(resolved_nodes.get(milestone_id, ())),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"
    raise KeyError(f"Milestone '{milestone_id}' not found in family spec")


def collect_declared_verifier_data_paths(task: TaskSpec, family_spec: dict[str, Any]) -> tuple[str, ...]:
    family_id = str(task.family_id or "")
    variant_id = str(task.variant_id or "")
    contract = get_variant_quality_contract(family_spec, variant_id)
    paths: set[str] = set()

    def maybe_add(raw_path: Any) -> None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        try:
            normalized = _normalize_declared_verifier_data_path(
                raw_path.strip(),
                family_id=family_id,
                variant_id=variant_id,
            )
        except ValueError:
            return
        paths.add(normalized)

    for key in ("hidden_tests", "red_team", "calibration"):
        section = contract.get(key)
        if isinstance(section, dict):
            maybe_add(section.get("path"))

    difficulty_estimate = family_spec.get("difficulty_estimate")
    if isinstance(difficulty_estimate, dict):
        maybe_add(difficulty_estimate.get("evidence_path"))

    shortcut_resistance = family_spec.get("shortcut_resistance")
    if isinstance(shortcut_resistance, dict):
        maybe_add(shortcut_resistance.get("generated_from"))

    interactive = family_spec.get("interactive")
    if isinstance(interactive, dict):
        rounds = interactive.get("rounds")
        if isinstance(rounds, int) and rounds > 0:
            for round_index in range(1, rounds + 1):
                round_cfg = interactive.get(f"round_{round_index}")
                if not isinstance(round_cfg, dict):
                    continue
                maybe_add(round_cfg.get("brief_source"))
                maybe_add(round_cfg.get("grader_between_rounds"))

    return tuple(sorted(paths))


def _sync_http_get(url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
    return requests.get(url, headers=headers, timeout=10)


def _sync_http_post(url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
    return requests.post(url, headers=headers, timeout=10)


async def health_check(
    vllm_host: str,
    vllm_port: int,
    expected_model: str,
    max_retries: int = 5,
    retry_delay_seconds: float = 10.0,
    http_get: Callable[[str], Any] | None = None,
) -> None:
    getter = http_get or _sync_http_get
    headers = _vllm_request_headers()
    for attempt in range(max_retries):
        try:
            health_resp = await _maybe_await(
                _call_with_supported_kwargs(
                    getter,
                    f"http://{vllm_host}:{vllm_port}/health",
                    headers=headers,
                )
            )
            health_status = _response_status(health_resp)
            if health_status != 200:
                raise HealthCheckError(f"/health returned {health_status}")

            models_resp = await _maybe_await(
                _call_with_supported_kwargs(
                    getter,
                    f"http://{vllm_host}:{vllm_port}/v1/models",
                    headers=headers,
                )
            )
            models_status = _response_status(models_resp)
            if models_status != 200:
                raise HealthCheckError(f"/v1/models returned {models_status}")
            model_ids = [item["id"] for item in _response_json(models_resp).get("data", []) if isinstance(item, dict)]
            if expected_model not in model_ids:
                raise HealthCheckError(
                    f"Expected model '{expected_model}' not in served models: {model_ids}"
                )
            return
        except (OSError, requests.RequestException, HealthCheckError) as exc:
            if attempt == max_retries - 1:
                raise TaskDispatchError(
                    f"vLLM health check failed after {max_retries} attempts. "
                    f"Last error: {exc}. Task: {expected_model}"
                ) from exc
            logger.warning(
                "Health check attempt %s/%s failed: %s. Retrying in %ss.",
                attempt + 1,
                max_retries,
                exc,
                retry_delay_seconds,
            )
            await asyncio.sleep(retry_delay_seconds)


async def flush_prefix_cache(
    vllm_host: str,
    vllm_port: int,
    http_post: Callable[[str], Any] | None = None,
) -> None:
    poster = http_post or _sync_http_post
    response = await _maybe_await(
        _call_with_supported_kwargs(
            poster,
            f"http://{vllm_host}:{vllm_port}/reset_prefix_cache",
            headers=_vllm_request_headers(),
        )
    )
    status = _response_status(response)
    if status == 405:
        raise ConfigError(
            "/reset_prefix_cache returned 405 — VLLM_SERVER_DEV_MODE may not be set. "
            "Set VLLM_SERVER_DEV_MODE=1 in vLLM launch environment."
        )
    if status != 200:
        raise CacheFlushError(f"/reset_prefix_cache returned {status}")


def _entry_served_model_name(model_entry: dict[str, Any] | ModelConfig, *, fallback: str) -> str:
    if isinstance(model_entry, ModelConfig):
        return model_entry.served_model_name
    if isinstance(model_entry, dict):
        value = model_entry.get("served_model_name")
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def _entry_lora_adapter_names(model_entry: dict[str, Any] | ModelConfig) -> tuple[str, ...]:
    if isinstance(model_entry, ModelConfig):
        return tuple(adapter_name for adapter_name, _adapter_path in model_entry.lora_modules)
    if isinstance(model_entry, dict):
        raw_lora_modules = model_entry.get("lora_modules", {})
        if isinstance(raw_lora_modules, dict):
            return tuple(
                adapter_name
                for adapter_name in raw_lora_modules
                if isinstance(adapter_name, str) and adapter_name.strip()
            )
    return ()


def _resolve_task_model_reference(
    model_registry: dict[str, Any],
    model_id: str,
) -> tuple[dict[str, Any] | ModelConfig, str]:
    if model_id in model_registry:
        model_entry = model_registry[model_id]
        return model_entry, _entry_served_model_name(model_entry, fallback=model_id)

    matches: list[tuple[dict[str, Any] | ModelConfig, str]] = []
    for registry_key, model_entry in model_registry.items():
        served_model_name = _entry_served_model_name(model_entry, fallback=registry_key)
        exposed_ids = {served_model_name, *_entry_lora_adapter_names(model_entry)}
        if model_id in exposed_ids:
            matches.append((model_entry, model_id))

    if len(matches) > 1:
        raise ConfigError(
            f"Model surface id '{model_id}' is ambiguous across multiple registry entries"
        )
    if matches:
        return matches[0]
    raise ConfigError(
        f"Model '{model_id}' not found in model registry keys or exposed serving surface ids"
    )


def generate_codex_config(
    task: TaskSpec,
    proxy_host: str,
    proxy_port: int,
    model_registry: dict[str, Any],
    config_root: str | Path = "/tmp/codex-bench/configs",
) -> Path:
    task_slug = quote(make_run_id(task), safe="")
    config_dir = Path(config_root) / task_slug
    config_path = config_dir / "config.toml"
    model_entry, request_model_name = _resolve_task_model_reference(model_registry, task.model_id)
    if isinstance(model_entry, ModelConfig):
        context_window = model_entry.max_model_len
    else:
        context_window = int(model_entry.get("max_model_len", 131072))
    compact_limit = int(context_window * 0.9)
    proxy_url = f"http://{proxy_host}:{proxy_port}/v1"
    content = (
        f'model          = "{request_model_name}"\n'
        'model_provider = "localvllm"\n\n'
        f"model_context_window           = {context_window}\n"
        f"model_auto_compact_token_limit = {compact_limit}\n\n"
        "[model_providers.localvllm]\n"
        'name                   = "Local vLLM"\n'
        f'base_url               = "{proxy_url}"\n'
        'env_key                = "VLLM_API_KEY"\n'
        'wire_api               = "responses"\n'
        "stream_idle_timeout_ms = 600000\n"
        "request_max_retries    = 2\n"
    )
    write_text(config_path, content)
    return config_path


def get_codex_harness_mounts(
    config_path: str | Path,
    codex_binary_path: str,
    codex_node_modules: str,
    node_binary_path: str,
) -> dict[str, dict[str, str]]:
    wrapper_path = Path(config_path).parent / "codex"
    write_text(
        wrapper_path,
        "#!/bin/sh\n"
        "exec /usr/local/bin/node /usr/local/lib/node_modules/@openai/codex/bin/codex.js \"$@\"\n",
    )
    wrapper_path.chmod(0o755)
    return {
        str(wrapper_path): {"bind": "/usr/local/bin/codex", "mode": "ro"},
        str(codex_node_modules): {
            "bind": "/usr/local/lib/node_modules/@openai/codex",
            "mode": "ro",
        },
        str(node_binary_path): {"bind": "/usr/local/bin/node", "mode": "ro"},
        str(config_path): {"bind": "/root/.codex/config.toml", "mode": "ro"},
    }


def get_codex_harness_env(task: TaskSpec) -> dict[str, str]:
    return {
        "VLLM_API_KEY": _vllm_api_key(),
        "CODEX_SEED": str(task.seed),
    }


def _cmd_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()  # noqa: S603


def get_bridge_gateway_ip(network_name: str) -> str:
    return _cmd_output(
        [
            "docker",
            "network",
            "inspect",
            network_name,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ]
    )


def get_local_image_digest(image_ref: str) -> str:
    digest = _cmd_output(
        [
            "docker",
            "image",
            "inspect",
            image_ref,
            "--format",
            "{{.Id}}",
        ]
    )
    return _canonical_sha256(digest)


def get_agents_md_from_image(image_ref: str) -> Path:
    container_name = f"agents-md-{int(time.time() * 1000)}"
    temp_dir = Path(tempfile.mkdtemp(prefix="codex-bench-agents-"))
    try:
        subprocess.run(["docker", "create", "--name", container_name, image_ref, "true"], check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            ["docker", "cp", f"{container_name}:/workspace/AGENTS.md", str(temp_dir / "AGENTS.md")],
            check=True,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)  # noqa: S603
    return temp_dir / "AGENTS.md"


async def docker_image_exists(image_ref: str) -> bool:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        image_ref,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await process.wait() == 0


def verify_pre_run_hashes(
    task: TaskSpec,
    manifest: dict[str, Any],
    *,
    image_digest_resolver: Callable[[str], str] = get_local_image_digest,
    agents_md_resolver: Callable[[str], Path] = get_agents_md_from_image,
    scenario_families_dir: str | Path = "scenario_families",
) -> None:
    entry = find_manifest_entry(manifest, task.family_id or "", task.variant_id or "")
    try:
        actual_image_digest = _canonical_sha256(image_digest_resolver(task.image_digest or ""))
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ManifestMismatchError(
            f"Locked image missing for {task.scenario_id}: {task.image_digest}",
            affected_artifact="image",
        ) from exc
    if actual_image_digest != _canonical_sha256(entry["image_digest"]):
        raise ManifestMismatchError(
            f"Image digest mismatch for {task.scenario_id}: "
            f"expected {entry['image_digest']}, got {actual_image_digest}",
            affected_artifact="image",
        )

    agents_md_path = agents_md_resolver(task.image_digest or "")
    try:
        try:
            actual_agents_md_hash = _canonical_sha256(sha256_file(agents_md_path))
        except FileNotFoundError as exc:
            raise ManifestMismatchError(
                f"AGENTS.md missing for {task.scenario_id}: {agents_md_path}",
                affected_artifact="agents_md",
            ) from exc
    finally:
        if agents_md_path.name == "AGENTS.md" and agents_md_path.parent.name.startswith("codex-bench-agents-"):
            shutil.rmtree(agents_md_path.parent, ignore_errors=True)
    if actual_agents_md_hash != _canonical_sha256(entry["agents_md_hash"]):
        raise ManifestMismatchError(
            f"AGENTS.md hash mismatch for {task.scenario_id}: "
            f"expected {entry['agents_md_hash']}, got {actual_agents_md_hash}",
            affected_artifact="agents_md",
        )

    family_spec_path = Path(scenario_families_dir) / str(task.family_id) / "family.yaml"
    try:
        actual_spec_hash = _canonical_sha256(sha256_file(family_spec_path))
    except FileNotFoundError as exc:
        raise ManifestMismatchError(
            f"Family spec missing for {task.scenario_id}: {family_spec_path}",
            affected_artifact="family_spec",
        ) from exc
    if actual_spec_hash != _canonical_sha256(entry["family_spec_hash"]):
        raise ManifestMismatchError(
            f"Family spec hash mismatch for {task.scenario_id}: "
            f"expected {entry['family_spec_hash']}, got {actual_spec_hash}",
            affected_artifact="family_spec",
        )


def verify_pre_grading_hashes(
    task: TaskSpec,
    manifest: dict[str, Any],
    grader_image_ref: str,
    *,
    image_digest_resolver: Callable[[str], str] = get_local_image_digest,
    verifiers_dir: str | Path = "verifiers",
    verifier_data_dir: str | Path = "verifier_data",
    scenario_families_dir: str | Path | None = None,
) -> None:
    entry = find_manifest_entry(manifest, task.family_id or "", task.variant_id or "")
    family_spec: dict[str, Any] | None = None

    try:
        actual_grader_digest = _canonical_sha256(image_digest_resolver(grader_image_ref))
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ManifestMismatchError(
            f"Grader image missing for {task.scenario_id}: {grader_image_ref}",
            affected_artifact="grader_image",
        ) from exc
    expected_grader_digest = _canonical_sha256(manifest["grader_image_digest"])
    if actual_grader_digest != expected_grader_digest:
        raise ManifestMismatchError(
            f"Grader image digest mismatch: expected {expected_grader_digest}, got {actual_grader_digest}.",
            affected_artifact="grader_image",
        )

    if scenario_families_dir is not None:
        family_spec_path = Path(scenario_families_dir) / str(task.family_id) / "family.yaml"
        try:
            actual_family_spec_hash = _canonical_sha256(sha256_file(family_spec_path))
        except FileNotFoundError as exc:
            raise ManifestMismatchError(
                f"Family spec missing for {task.scenario_id}: {family_spec_path}",
                affected_artifact="family_spec",
            ) from exc
        if actual_family_spec_hash != _canonical_sha256(entry["family_spec_hash"]):
            raise ManifestMismatchError(
                f"Family spec hash mismatch for {task.scenario_id}: "
                f"expected {entry['family_spec_hash']}, got {actual_family_spec_hash}",
                affected_artifact="family_spec",
            )
        family_spec = load_family_spec(str(task.family_id), scenario_families_dir)
        validate_family_spec(task, family_spec)

    family_verifier_dir = Path(verifiers_dir) / str(task.family_id)
    verifier_path = family_verifier_dir / "verify.sh"
    try:
        actual_verifier_hash = _canonical_sha256(sha256_file(verifier_path))
    except FileNotFoundError as exc:
        raise ManifestMismatchError(
            f"Verifier script missing for {task.scenario_id}: {verifier_path}",
            affected_artifact="verifier",
        ) from exc
    if not os.access(verifier_path, os.X_OK):
        raise ManifestMismatchError(
            f"Verifier script is not executable for {task.scenario_id}: {verifier_path}",
            affected_artifact="verifier",
        )
    verifier_tree_hash = entry.get("verifier_tree_hash")
    if verifier_tree_hash is not None:
        actual_verifier_tree_hash = sha256_tree(family_verifier_dir)
        if actual_verifier_tree_hash != _canonical_sha256(verifier_tree_hash):
            raise ManifestMismatchError(
                f"Verifier tree hash mismatch for {task.scenario_id}: "
                f"expected {verifier_tree_hash}, got {actual_verifier_tree_hash}",
                affected_artifact="verifier",
            )
    else:
        if actual_verifier_hash != _canonical_sha256(entry["verifier_hash"]):
            raise ManifestMismatchError(
                f"Verifier hash mismatch for {task.scenario_id}: "
                f"expected {entry['verifier_hash']}, got {actual_verifier_hash}",
                affected_artifact="verifier",
            )

        milestone_hashes = entry.get("milestone_hashes", {})
        milestones_dir = family_verifier_dir / "milestones"
        expected_milestone_files: set[str] = set()
        if family_spec is not None:
            try:
                milestones = family_spec.get("milestones")
                if isinstance(milestones, list):
                    for milestone in milestones:
                        if not isinstance(milestone, dict):
                            continue
                        milestone_id = str(milestone.get("id", "")).strip()
                        if milestone_id and milestone.get("check_script") is not None:
                            expected_milestone_files.add(f"{milestone_id}.sh")
            except Exception:
                expected_milestone_files = {f"{milestone_id}.sh" for milestone_id in milestone_hashes}
        else:
            expected_milestone_files = {f"{milestone_id}.sh" for milestone_id in milestone_hashes}

        actual_milestone_files = (
            {path.name for path in milestones_dir.iterdir() if path.is_file()}
            if milestones_dir.exists()
            else set()
        )
        if actual_milestone_files != expected_milestone_files:
            missing = sorted(expected_milestone_files - actual_milestone_files)
            unexpected = sorted(actual_milestone_files - expected_milestone_files)
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            rendered = ", ".join(details) if details else "layout mismatch"
            raise ManifestMismatchError(
                f"Milestone file set mismatch for {task.scenario_id}: {rendered}",
                affected_artifact="milestone",
            )

        for milestone_id, expected_hash in milestone_hashes.items():
            milestone_path = milestones_dir / f"{milestone_id}.sh"
            if milestone_path.exists():
                if not os.access(milestone_path, os.X_OK):
                    raise ManifestMismatchError(
                        f"Milestone helper is not executable for {task.scenario_id}/{milestone_id}: {milestone_path}",
                        affected_artifact="milestone",
                    )
                actual_hash = _canonical_sha256(sha256_file(milestone_path))
            elif family_spec is not None:
                try:
                    actual_hash = milestone_contract_hash(
                        family_spec,
                        family_id=str(task.family_id or ""),
                        variant_id=str(task.variant_id or ""),
                        milestone_id=milestone_id,
                    )
                except (KeyError, ValueError) as exc:
                    raise ManifestMismatchError(
                        f"Milestone contract missing for {task.scenario_id}/{milestone_id}: {exc}",
                        affected_artifact="milestone",
                    ) from exc
            else:
                raise ManifestMismatchError(
                    f"Milestone helper missing for {task.scenario_id}/{milestone_id}: {milestone_path}",
                    affected_artifact="milestone",
                )
            if actual_hash != _canonical_sha256(expected_hash):
                raise ManifestMismatchError(
                    f"Milestone hash mismatch for {task.scenario_id}/{milestone_id}: "
                    f"expected {expected_hash}, got {actual_hash}",
                    affected_artifact="milestone",
                )

        allowed_top_level_entries = {"verify.sh"}
        if expected_milestone_files:
            allowed_top_level_entries.add("milestones")
        actual_top_level_entries = {path.name for path in family_verifier_dir.iterdir()}
        unexpected_top_level_entries = sorted(actual_top_level_entries - allowed_top_level_entries)
        if unexpected_top_level_entries:
            raise ManifestMismatchError(
                "Verifier tree contains untracked files for "
                f"{task.scenario_id}: {unexpected_top_level_entries}",
                affected_artifact="verifier",
            )

        unexpected_milestone_dirs = (
            sorted(path.name for path in milestones_dir.iterdir() if path.is_dir())
            if milestones_dir.exists()
            else []
        )
        if unexpected_milestone_dirs:
            raise ManifestMismatchError(
                "Milestones directory contains unexpected subdirectories for "
                f"{task.scenario_id}: {unexpected_milestone_dirs}",
                affected_artifact="milestone",
            )

    repo_root = Path(verifier_data_dir).resolve().parent
    declared_verifier_data_paths = (
        collect_declared_verifier_data_paths(task, family_spec)
        if family_spec is not None
        else ()
    )
    if declared_verifier_data_paths:
        try:
            verifier_data_hash = sha256_path_set(list(declared_verifier_data_paths), repo_root=repo_root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ManifestMismatchError(
                f"Verifier data missing for {task.scenario_id}: {exc}",
                affected_artifact="verifier_data",
            ) from exc
    else:
        verifier_data_path = Path(verifier_data_dir) / str(task.family_id)
        try:
            verifier_data_hash = sha256_tree(verifier_data_path)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ManifestMismatchError(
                f"Verifier data missing for {task.scenario_id}: {verifier_data_path}",
                affected_artifact="verifier_data",
            ) from exc
    if verifier_data_hash != _canonical_sha256(entry["verifier_data_hash"]):
        raise ManifestMismatchError(
            f"Verifier data hash mismatch for {task.scenario_id}: "
            f"expected {entry['verifier_data_hash']}, got {verifier_data_hash}",
            affected_artifact="verifier_data",
        )

    if family_spec is not None and "calibration_hash" in entry:
        calibration = get_variant_quality_contract(family_spec, str(task.variant_id or "")).get("calibration")
        calibration_path = calibration.get("path") if isinstance(calibration, dict) else None
        if not isinstance(calibration_path, str) or not calibration_path.strip():
            raise ManifestMismatchError(
                f"Calibration path missing from family spec for {task.scenario_id}",
                affected_artifact="verifier_data",
            )
        rendered = _render_variant_path_template(
            calibration_path,
            family_id=str(task.family_id or ""),
            variant_id=str(task.variant_id or ""),
        )
        actual_calibration_hash = _canonical_sha256(sha256_file(repo_root / rendered))
        if actual_calibration_hash != _canonical_sha256(entry["calibration_hash"]):
            raise ManifestMismatchError(
                f"Calibration hash mismatch for {task.scenario_id}: "
                f"expected {entry['calibration_hash']}, got {actual_calibration_hash}",
                affected_artifact="verifier_data",
            )


def _get_prompt(task: TaskSpec) -> str:
    if task.track == "swe_bench":
        return (
            "Read AGENTS.md for the problem description. Fix the reported issue "
            "in this repository. Make the minimal changes necessary to resolve "
            "the failing test case described in the issue."
        )
    return (
        "Read AGENTS.md for the task description. Complete the task described "
        "there. The repository is at /workspace."
    )


def build_codex_command(task: TaskSpec) -> str:
    prompt = _get_prompt(task)
    return (
        "codex exec "
        "--skip-git-repo-check "
        "--yolo "
        "--json "
        "-c 'web_search=\"disabled\"' "
        "-c 'model_reasoning_effort=\"high\"' "
        "-c 'personality=\"pragmatic\"' "
        "-C /workspace "
        f'"{prompt}"'
    )


def determine_outcome(
    codex_result: CodexResult,
    track: str,
    patch_path: str | None = None,
    verify_result: dict[str, Any] | None = None,
) -> str:
    if codex_result.exit_code not in (0, 1) and not codex_result.timed_out:
        if _is_infrastructure_error(codex_result.stderr):
            return "crash"
    if codex_result.timed_out:
        return "timeout"
    if track == "codex_long":
        if verify_result is None:
            return "crash"
        return "resolved" if bool(verify_result.get("pass")) else "failed"
    if track == "swe_bench":
        if patch_path is None:
            return "no_patch"
        raise ValueError("SWE-bench outcomes are determined by the evaluation harness wrapper")
    raise ValueError(f"Unexpected track in determine_outcome: {track}")


def _is_infrastructure_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(pattern.lower() in lowered for pattern in _INFRASTRUCTURE_ERROR_PATTERNS)


async def capture_event_stream(
    process: asyncio.subprocess.Process,
    trajectory_path: str | Path,
    task_id: str,
) -> None:
    path = Path(trajectory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        header = {
            "type": "run_metadata",
            "task_id": task_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "codex_flags": "--yolo --json",
        }
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        if process.stdout is None:
            return
        async for line in process.stdout:
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                handle.write(decoded + "\n")
                handle.flush()


async def docker_exec_with_timeout(
    container_id: str,
    command: str,
    timeout_seconds: int,
    stdout_sink: str | Path,
) -> ExecResult:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container_id,
        "bash",
        "-lc",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def drain_stderr() -> str:
        if process.stderr is None:
            return ""
        data = await process.stderr.read()
        return data.decode("utf-8", errors="replace")

    try:
        stdout_task = asyncio.create_task(capture_event_stream(process, stdout_sink, container_id))
        stderr_task = asyncio.create_task(drain_stderr())
        wait_task = asyncio.create_task(process.wait())
        await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, wait_task), timeout=timeout_seconds)
        return ExecResult(returncode=process.returncode or 0, stderr=stderr_task.result(), timed_out=False)
    except asyncio.TimeoutError:
        logger.warning("Task timed out after %ss in container %s", timeout_seconds, container_id)
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        return ExecResult(returncode=-1, stderr="TIMEOUT", timed_out=True)


async def drive_swe_bench_eval(
    instance_id: str,
    patch_path: str,
    output_dir: str,
    runner: Callable[[list[str], int], Any] | None = None,
) -> str:
    eval_output = Path(output_dir) / "swe_eval" / instance_id
    eval_output.mkdir(parents=True, exist_ok=True)
    command = [
        "codex-bench-eval-swe",
        "--instance-id",
        instance_id,
        "--patch-path",
        patch_path,
        "--output-dir",
        str(eval_output),
        "--dataset-name",
        "princeton-nlp/SWE-bench_Verified",
    ]
    if runner is None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        returncode = await asyncio.wait_for(process.wait(), timeout=900)
    else:
        result = await _maybe_await(runner(command, 900))
        returncode = int(getattr(result, "returncode", result))
    if returncode == 0:
        return "resolved"
    if returncode == 1:
        return "failed"
    logger.warning("SWE-bench eval infrastructure error for %s: exit code %s", instance_id, returncode)
    return "crash"


def load_family_spec(family_id: str, scenario_families_dir: str | Path = "scenario_families") -> dict[str, Any]:
    family_path = Path(scenario_families_dir) / family_id / "family.yaml"
    payload = load_yaml_file(family_path) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Family spec {family_path} must be a YAML mapping")
    return payload


def validate_family_spec(task: TaskSpec, family_spec: dict[str, Any]) -> None:
    def _require_non_empty_string(value: Any, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define non-empty {field_name}",
                affected_artifact="family_spec",
            )
        return value

    def _require_pinned_digest(value: Any, *, field_name: str) -> str:
        digest = _require_non_empty_string(value, field_name=field_name)
        if not _PINNED_DIGEST_RE.fullmatch(digest):
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define {field_name} as a pinned sha256 digest",
                affected_artifact="family_spec",
            )
        return digest

    def _require_target_solve_rate(value: Any) -> str:
        solve_rate = _require_non_empty_string(
            value,
            field_name="difficulty_estimate.target_solve_rate",
        )
        match = re.fullmatch(r"\s*(\d{1,3})(?:\s*[–-]\s*(\d{1,3}))?\s*%\s*", solve_rate)
        if not match:
            raise ManifestMismatchError(
                "Family spec for "
                f"{task.scenario_id} must define difficulty_estimate.target_solve_rate as "
                "a percentage or range such as '30-50%'",
                affected_artifact="family_spec",
            )
        bounds = [int(group) for group in match.groups() if group is not None]
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define target_solve_rate with ascending bounds",
                affected_artifact="family_spec",
            )
        if any(bound < 20 or bound > 80 for bound in bounds):
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must keep target_solve_rate within the signed-off 20-80% band",
                affected_artifact="family_spec",
            )
        return solve_rate

    def _require_declared_verifier_data_path(value: Any, *, field_name: str) -> str:
        path = _require_non_empty_string(value, field_name=field_name)
        try:
            return _normalize_declared_verifier_data_path(
                path,
                family_id=expected_family_id,
                variant_id=expected_variant_id,
            )
        except ValueError as exc:
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define {field_name} as a repo-relative path "
                f"under verifier_data/ without unsupported placeholders or '..' segments: {exc}",
                affected_artifact="family_spec",
            ) from exc

    def _is_phase2_only_invariant(check_id: str, description: str) -> bool:
        combined = f"{check_id} {description}".lower()
        explicit_phase2_markers = (
            "phase 2",
            "phase2",
            "functional check",
            "functional_checks",
            "exit 0",
            "round 1",
            "round1",
            "round 2",
            "round2",
        )
        test_runner_markers = (
            "pytest",
            "npm test",
            "cargo test",
            "go test",
            "gradle test",
            "mvn test",
            "hidden test",
            "hidden tests",
            "test suite passes",
            "tests pass",
        )
        return any(marker in combined for marker in (*explicit_phase2_markers, *test_runner_markers))

    def _is_spoofed_functional_success_exploit(exploit: str) -> bool:
        combined = exploit.lower()
        strong_markers = (
            "spoofed functional success",
            "functional success",
            "shadow pytest",
            "shadow npm",
            "shadow cargo",
            "shadow go test",
            "shadow gradle",
            "shadow mvn",
            "shadow test runner",
            "exit 0 without",
        )
        if any(marker in combined for marker in strong_markers):
            return True
        return "shadow" in combined and any(
            runner in combined
            for runner in ("pytest", "npm", "cargo", "go test", "gradle", "mvn", "test suite")
        )

    expected_family_id = task.family_id or ""
    expected_variant_id = task.variant_id or ""
    expected_scenario_type = task.scenario_type or ""

    family_id = _require_non_empty_string(
        family_spec.get("family_id"),
        field_name="family_id",
    )
    if family_id != expected_family_id:
        raise ManifestMismatchError(
            f"Family spec family_id mismatch for {task.scenario_id}: expected '{expected_family_id}', got '{family_id}'",
            affected_artifact="family_spec",
        )

    scenario_type = _require_non_empty_string(
        family_spec.get("scenario_type"),
        field_name="scenario_type",
    )
    if scenario_type not in SCENARIO_TYPES:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must use one of {sorted(SCENARIO_TYPES)} for scenario_type",
            affected_artifact="family_spec",
        )
    if scenario_type != expected_scenario_type:
        raise ManifestMismatchError(
            f"Family spec scenario_type mismatch for {task.scenario_id}: "
            f"expected '{expected_scenario_type}', got '{scenario_type}'",
            affected_artifact="family_spec",
        )
    _require_non_empty_string(
        family_spec.get("description"),
        field_name="description",
    )

    repo_pattern = family_spec.get("repo_pattern")
    if not isinstance(repo_pattern, dict):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define repo_pattern as a mapping",
            affected_artifact="family_spec",
        )
    _require_non_empty_string(
        repo_pattern.get("language"),
        field_name="repo_pattern.language",
    )
    _require_non_empty_string(
        repo_pattern.get("framework"),
        field_name="repo_pattern.framework",
    )
    _require_non_empty_string(
        repo_pattern.get("structure"),
        field_name="repo_pattern.structure",
    )
    base_image = _require_non_empty_string(
        repo_pattern.get("base_image"),
        field_name="repo_pattern.base_image",
    )
    if not _PINNED_IMAGE_RE.fullmatch(base_image):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must pin repo_pattern.base_image with @sha256:<digest>",
            affected_artifact="family_spec",
        )

    breakage_class = family_spec.get("breakage_class")
    if not isinstance(breakage_class, dict):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define breakage_class as a mapping",
            affected_artifact="family_spec",
        )
    _require_non_empty_string(
        breakage_class.get("injection_method"),
        field_name="breakage_class.injection_method",
    )
    _require_non_empty_string(
        breakage_class.get("description"),
        field_name="breakage_class.description",
    )
    breakage_surfaces = breakage_class.get("surfaces")
    if not isinstance(breakage_surfaces, list) or not breakage_surfaces:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define non-empty breakage_class.surfaces",
            affected_artifact="family_spec",
        )
    if len(breakage_surfaces) < 3:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define at least 3 breakage_class.surfaces",
            affected_artifact="family_spec",
        )
    for index, surface in enumerate(breakage_surfaces):
        _require_non_empty_string(
            surface,
            field_name=f"breakage_class.surfaces[{index}]",
        )

    grading_invariant = family_spec.get("grading_invariant")
    if not isinstance(grading_invariant, dict):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define grading_invariant as a mapping",
            affected_artifact="family_spec",
        )
    grading_type = _require_non_empty_string(
        grading_invariant.get("type"),
        field_name="grading_invariant.type",
    )
    if grading_type not in {"state_based", "hybrid"}:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must set grading_invariant.type to "
            "'state_based' or 'hybrid'",
            affected_artifact="family_spec",
        )
    _require_non_empty_string(
        grading_invariant.get("description"),
        field_name="grading_invariant.description",
    )
    verifier_script = _require_non_empty_string(
        grading_invariant.get("verifier_script"),
        field_name="grading_invariant.verifier_script",
    )
    expected_verifier_script = f"verifiers/{expected_family_id}/verify.sh"
    if verifier_script != expected_verifier_script:
        raise ManifestMismatchError(
            f"Family spec verifier_script mismatch for {task.scenario_id}: "
            f"expected '{expected_verifier_script}', got '{verifier_script}'",
            affected_artifact="family_spec",
        )

    functional_checks = grading_invariant.get("functional_checks")
    if not isinstance(functional_checks, list) or not functional_checks:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define non-empty grading_invariant.functional_checks",
            affected_artifact="family_spec",
        )
    seen_functional_check_ids: set[str] = set()
    for index, check in enumerate(functional_checks):
        if not isinstance(check, dict):
            raise ManifestMismatchError(
                f"Family spec functional_checks[{index}] for {task.scenario_id} must be a mapping",
                affected_artifact="family_spec",
            )
        check_id = _require_non_empty_string(
            check.get("id"),
            field_name=f"grading_invariant.functional_checks[{index}].id",
        )
        if check_id in seen_functional_check_ids:
            raise ManifestMismatchError(
                f"Family spec functional_checks for {task.scenario_id} contains duplicate id '{check_id}'",
                affected_artifact="family_spec",
            )
        seen_functional_check_ids.add(check_id)
        _require_non_empty_string(
            check.get("command"),
            field_name=f"grading_invariant.functional_checks[{index}].command",
        )
        timeout_seconds = check.get("timeout_seconds")
        try:
            timeout_seconds = int(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ManifestMismatchError(
                f"Family spec functional_checks[{index}] for {task.scenario_id} must define integer timeout_seconds",
                affected_artifact="family_spec",
            ) from exc
        if timeout_seconds <= 0:
            raise ManifestMismatchError(
                f"Family spec functional_checks[{index}] for {task.scenario_id} must use timeout_seconds > 0",
                affected_artifact="family_spec",
            )

    expected_final_state = grading_invariant.get("expected_final_state")
    if not isinstance(expected_final_state, list) or not expected_final_state:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define non-empty grading_invariant.expected_final_state",
            affected_artifact="family_spec",
        )
    trusted_phase3_invariants = 0
    for index, expected_state in enumerate(expected_final_state):
        if isinstance(expected_state, str):
            description = _require_non_empty_string(
                expected_state,
                field_name=f"grading_invariant.expected_final_state[{index}]",
            )
            if not _is_phase2_only_invariant(description, description):
                trusted_phase3_invariants += 1
            continue
        if isinstance(expected_state, dict) and len(expected_state) == 1:
            check_id, description = next(iter(expected_state.items()))
            check_id = _require_non_empty_string(
                check_id,
                field_name=f"grading_invariant.expected_final_state[{index}] key",
            )
            description = _require_non_empty_string(
                description,
                field_name=f"grading_invariant.expected_final_state[{index}].{check_id}",
            )
            if not _is_phase2_only_invariant(check_id, description):
                trusted_phase3_invariants += 1
            continue
        raise ManifestMismatchError(
            "Family spec grading_invariant.expected_final_state entries must be non-empty strings "
            "or single-entry mappings",
            affected_artifact="family_spec",
        )
    if trusted_phase3_invariants < 1:
        raise ManifestMismatchError(
            "Family spec for "
            f"{task.scenario_id} must define at least one trusted Phase 3 expected_final_state "
            "invariant that can reject spoofed functional-check success independently of Phase 2",
            affected_artifact="family_spec",
        )

    difficulty_estimate = family_spec.get("difficulty_estimate")
    if not isinstance(difficulty_estimate, dict):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define difficulty_estimate as a mapping",
            affected_artifact="family_spec",
        )
    if "target_solve_rate" in difficulty_estimate:
        _require_target_solve_rate(difficulty_estimate.get("target_solve_rate"))
    elif "evidence_path" in difficulty_estimate:
        _require_declared_verifier_data_path(
            difficulty_estimate.get("evidence_path"),
            field_name="difficulty_estimate.evidence_path",
        )
    else:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define either "
            "difficulty_estimate.target_solve_rate or difficulty_estimate.evidence_path",
            affected_artifact="family_spec",
        )

    interactive = family_spec.get("interactive")
    if interactive is not None:
        if not isinstance(interactive, dict):
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define interactive as a mapping",
                affected_artifact="family_spec",
            )
        rounds = interactive.get("rounds")
        if not isinstance(rounds, int) or rounds < 1:
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define interactive.rounds as an integer >= 1",
                affected_artifact="family_spec",
            )
        for round_index in range(1, rounds + 1):
            round_key = f"round_{round_index}"
            round_cfg = interactive.get(round_key)
            if not isinstance(round_cfg, dict):
                raise ManifestMismatchError(
                    f"Family spec for {task.scenario_id} must define interactive.{round_key} as a mapping",
                    affected_artifact="family_spec",
                )
            _require_non_empty_string(
                round_cfg.get("brief_source"),
                field_name=f"interactive.{round_key}.brief_source",
            )
            grader_between_rounds = round_cfg.get("grader_between_rounds")
            if grader_between_rounds is not None:
                grader_between_rounds = _require_non_empty_string(
                    grader_between_rounds,
                    field_name=f"interactive.{round_key}.grader_between_rounds",
                )
                if grader_between_rounds.startswith(("verifier_data/", "./verifier_data/", "../")) or "<" in grader_between_rounds:
                    _require_declared_verifier_data_path(
                        grader_between_rounds,
                        field_name=f"interactive.{round_key}.grader_between_rounds",
                    )
            inject_timing = round_cfg.get("inject_timing")
            if inject_timing is not None:
                _require_non_empty_string(
                    inject_timing,
                    field_name=f"interactive.{round_key}.inject_timing",
                )
            inject_mechanism = round_cfg.get("inject_mechanism")
            if inject_mechanism is not None:
                _require_non_empty_string(
                    inject_mechanism,
                    field_name=f"interactive.{round_key}.inject_mechanism",
                )

    milestones = family_spec.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define a non-empty milestones list",
            affected_artifact="family_spec",
        )
    if len(milestones) < 3:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define at least 3 milestones",
            affected_artifact="family_spec",
        )
    partial_credit_total = 0.0
    previous_partial_credit = 0.0
    seen_milestone_ids: set[str] = set()
    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            raise ManifestMismatchError(
                f"Family spec milestones[{index}] for {task.scenario_id} must be a mapping",
                affected_artifact="family_spec",
            )
        milestone_id = _require_non_empty_string(
            milestone.get("id"),
            field_name=f"milestones[{index}].id",
        )
        if milestone_id in seen_milestone_ids:
            raise ManifestMismatchError(
                f"Family spec milestones for {task.scenario_id} contains duplicate id '{milestone_id}'",
                affected_artifact="family_spec",
            )
        seen_milestone_ids.add(milestone_id)
        _require_non_empty_string(
            milestone.get("description"),
            field_name=f"milestones[{index}].description",
        )
        check_script = milestone.get("check_script")
        test_nodes = milestone.get("test_nodes")
        if check_script is None and test_nodes is None:
            raise ManifestMismatchError(
                f"Family spec milestones[{index}] for {task.scenario_id} must define check_script "
                "or test_nodes",
                affected_artifact="family_spec",
            )
        if check_script is not None:
            expected_check_script = f"verifiers/{expected_family_id}/milestones/{milestone_id}.sh"
            check_script = _require_non_empty_string(
                check_script,
                field_name=f"milestones[{index}].check_script",
            )
            if check_script != expected_check_script:
                raise ManifestMismatchError(
                    f"Family spec milestone check_script mismatch for {task.scenario_id}/{milestone_id}: "
                    f"expected '{expected_check_script}', got '{check_script}'",
                    affected_artifact="family_spec",
                )
        if test_nodes is not None:
            if test_nodes == "variant_scoped":
                pass
            elif isinstance(test_nodes, str):
                _require_non_empty_string(
                    test_nodes,
                    field_name=f"milestones[{index}].test_nodes",
                )
            elif isinstance(test_nodes, list) and test_nodes:
                for node_index, node in enumerate(test_nodes):
                    _require_non_empty_string(
                        node,
                        field_name=f"milestones[{index}].test_nodes[{node_index}]",
                    )
            else:
                raise ManifestMismatchError(
                    f"Family spec milestones[{index}] for {task.scenario_id} must define test_nodes as "
                    "a non-empty list, a non-empty string, or 'variant_scoped'",
                    affected_artifact="family_spec",
                )
            pass_rule = milestone.get("pass_rule", "all")
            if pass_rule not in _MILESTONE_PASS_RULES:
                raise ManifestMismatchError(
                    f"Family spec milestones[{index}] for {task.scenario_id} must use pass_rule "
                    f"in {sorted(_MILESTONE_PASS_RULES)}",
                    affected_artifact="family_spec",
                )
        partial_credit = milestone.get("partial_credit")
        if not isinstance(partial_credit, (int, float)):
            raise ManifestMismatchError(
                f"Family spec milestones[{index}] for {task.scenario_id} must define numeric partial_credit",
                affected_artifact="family_spec",
            )
        if partial_credit < 0:
            raise ManifestMismatchError(
                f"Family spec milestones[{index}] for {task.scenario_id} must use partial_credit >= 0",
                affected_artifact="family_spec",
            )
        if index > 0 and partial_credit < previous_partial_credit:
            raise ManifestMismatchError(
                f"Family spec milestones for {task.scenario_id} must use monotonically non-decreasing "
                "partial_credit weights across task progression",
                affected_artifact="family_spec",
            )
        partial_credit_total += float(partial_credit)
        previous_partial_credit = float(partial_credit)
    if partial_credit_total > 1.0 + 1e-9:
        raise ManifestMismatchError(
            f"Family spec milestone partial_credit sum exceeds 1.0 for {task.scenario_id}: {partial_credit_total}",
            affected_artifact="family_spec",
        )

    shortcut_resistance = family_spec.get("shortcut_resistance")
    if not isinstance(shortcut_resistance, dict):
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define shortcut_resistance as a mapping",
            affected_artifact="family_spec",
        )
    if "known_exploits_tested" in shortcut_resistance:
        _require_non_empty_string(
            shortcut_resistance.get("notes"),
            field_name="shortcut_resistance.notes",
        )
        known_exploits = shortcut_resistance.get("known_exploits_tested")
        if not isinstance(known_exploits, list) or len(known_exploits) < 3:
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must list at least 3 known_exploits_tested entries",
                affected_artifact="family_spec",
            )
        has_spoofed_functional_success = False
        for index, exploit in enumerate(known_exploits):
            exploit_text = _require_non_empty_string(
                exploit,
                field_name=f"shortcut_resistance.known_exploits_tested[{index}]",
            )
            has_spoofed_functional_success = (
                has_spoofed_functional_success
                or _is_spoofed_functional_success_exploit(exploit_text)
            )
        if not has_spoofed_functional_success:
            raise ManifestMismatchError(
                "Family spec for "
                f"{task.scenario_id} must explicitly cover spoofed functional-check success "
                "in shortcut_resistance.known_exploits_tested",
                affected_artifact="family_spec",
            )
    else:
        _require_declared_verifier_data_path(
            shortcut_resistance.get("generated_from"),
            field_name="shortcut_resistance.generated_from",
        )
        min_exploits = shortcut_resistance.get("min_exploits")
        if not isinstance(min_exploits, int) or min_exploits < 1:
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define shortcut_resistance.min_exploits "
                "as an integer >= 1",
                affected_artifact="family_spec",
            )
        mutation_score_floor = shortcut_resistance.get("mutation_score_floor")
        if not isinstance(mutation_score_floor, (int, float)) or not (0 < float(mutation_score_floor) <= 1):
            raise ManifestMismatchError(
                f"Family spec for {task.scenario_id} must define shortcut_resistance.mutation_score_floor "
                "in the interval (0, 1]",
                affected_artifact="family_spec",
            )

    variants = family_spec.get("variants")
    if not isinstance(variants, list) or len(variants) < 3:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} must define at least 3 variants",
            affected_artifact="family_spec",
        )
    variant_ids: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ManifestMismatchError(
                f"Family spec variants[{index}] for {task.scenario_id} must be a mapping",
                affected_artifact="family_spec",
            )
        variant_id = _require_non_empty_string(
            variant.get("variant_id"),
            field_name=f"variants[{index}].variant_id",
        )
        if variant_id in variant_ids:
            raise ManifestMismatchError(
                f"Family spec variants for {task.scenario_id} contains duplicate variant_id '{variant_id}'",
                affected_artifact="family_spec",
            )
        variant_ids.add(variant_id)
        _require_non_empty_string(
            variant.get("injected_breakage"),
            field_name=f"variants[{index}].injected_breakage",
        )
        _require_non_empty_string(
            variant.get("env_dockerfile"),
            field_name=f"variants[{index}].env_dockerfile",
        )
        _require_pinned_digest(
            variant.get("base_image_digest"),
            field_name=f"variants[{index}].base_image_digest",
        )
        repo_source = _require_non_empty_string(
            variant.get("repo_source", "authored"),
            field_name=f"variants[{index}].repo_source",
        )
        variant_surfaces = variant.get("surfaces")
        if variant_surfaces is not None:
            if not isinstance(variant_surfaces, list) or not variant_surfaces:
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define surfaces as "
                    "a non-empty list when present",
                    affected_artifact="family_spec",
                )
            for surface_index, surface in enumerate(variant_surfaces):
                _require_non_empty_string(
                    surface,
                    field_name=f"variants[{index}].surfaces[{surface_index}]",
                )
        if repo_source != "authored" and not repo_source.startswith("derived:"):
            raise ManifestMismatchError(
                f"Family spec variants[{index}] for {task.scenario_id} must use repo_source "
                "equal to 'authored' or 'derived:<source_repo>'",
                affected_artifact="family_spec",
            )
        if repo_source.startswith("derived:"):
            provenance = variant.get("provenance")
            if not isinstance(provenance, dict):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define provenance for derived repos",
                    affected_artifact="family_spec",
                )
            _require_non_empty_string(
                provenance.get("source_repo"),
                field_name=f"variants[{index}].provenance.source_repo",
            )
            _require_non_empty_string(
                provenance.get("license"),
                field_name=f"variants[{index}].provenance.license",
            )
            if not isinstance(provenance.get("redistribution_ok"), bool):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define boolean provenance.redistribution_ok",
                    affected_artifact="family_spec",
                )
            if (
                variant_id == expected_variant_id
                and task.pool_or_split == "public_dev"
                and provenance["redistribution_ok"] is False
            ):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} cannot set "
                    "provenance.redistribution_ok=false in Public-Dev",
                    affected_artifact="family_spec",
                )
            _require_non_empty_string(
                provenance.get("modification_notice"),
                field_name=f"variants[{index}].provenance.modification_notice",
            )
        contract = get_variant_quality_contract(family_spec, variant_id)
        tier = contract.get("tier")
        if tier is not None:
            tier_value = _require_non_empty_string(
                tier,
                field_name=f"variants[{index}].tier",
            )
            if tier_value not in _CODEX_LONG_VARIANT_TIERS:
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must use tier in "
                    f"{sorted(_CODEX_LONG_VARIANT_TIERS)}",
                    affected_artifact="family_spec",
                )

        oracle = contract.get("oracle")
        if oracle:
            if not isinstance(oracle, dict):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define oracle as a mapping",
                    affected_artifact="family_spec",
                )
            _require_non_empty_string(
                oracle.get("path"),
                field_name=f"variants[{index}].oracle.path",
            )
            for optional_field in ("followup_path", "source_commit"):
                if optional_field in oracle:
                    _require_non_empty_string(
                        oracle.get(optional_field),
                        field_name=f"variants[{index}].oracle.{optional_field}",
                    )

        hidden_tests = contract.get("hidden_tests")
        if hidden_tests:
            if not isinstance(hidden_tests, dict):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define hidden_tests as a mapping",
                    affected_artifact="family_spec",
                )
            _require_declared_verifier_data_path(
                hidden_tests.get("path"),
                field_name=f"variants[{index}].hidden_tests.path",
            )
            if "entrypoint" in hidden_tests:
                _require_non_empty_string(
                    hidden_tests.get("entrypoint"),
                    field_name=f"variants[{index}].hidden_tests.entrypoint",
                )
            milestone_map = hidden_tests.get("milestone_map")
            if milestone_map is not None:
                if not isinstance(milestone_map, dict) or not milestone_map:
                    raise ManifestMismatchError(
                        f"Family spec variants[{index}] for {task.scenario_id} must define hidden_tests.milestone_map "
                        "as a non-empty mapping when present",
                        affected_artifact="family_spec",
                    )
                for milestone_id, raw_nodes in milestone_map.items():
                    _require_non_empty_string(
                        milestone_id,
                        field_name=f"variants[{index}].hidden_tests.milestone_map key",
                    )
                    if isinstance(raw_nodes, str):
                        nodes = [raw_nodes]
                    elif isinstance(raw_nodes, list) and raw_nodes:
                        nodes = raw_nodes
                    else:
                        raise ManifestMismatchError(
                            f"Family spec variants[{index}] for {task.scenario_id} must map milestone "
                            f"'{milestone_id}' to a non-empty string or list of strings",
                            affected_artifact="family_spec",
                        )
                    for node_index, node in enumerate(nodes):
                        _require_non_empty_string(
                            node,
                            field_name=(
                                f"variants[{index}].hidden_tests.milestone_map[{milestone_id}]"
                                f"[{node_index}]"
                            ),
                        )

        red_team = contract.get("red_team")
        if red_team:
            if not isinstance(red_team, dict):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define red_team as a mapping",
                    affected_artifact="family_spec",
                )
            _require_declared_verifier_data_path(
                red_team.get("path"),
                field_name=f"variants[{index}].red_team.path",
            )
            exploits_required = red_team.get("exploits_required")
            if exploits_required is not None and (not isinstance(exploits_required, int) or exploits_required < 1):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define red_team.exploits_required "
                    "as an integer >= 1 when present",
                    affected_artifact="family_spec",
                )

        calibration = contract.get("calibration")
        if calibration:
            if not isinstance(calibration, dict):
                raise ManifestMismatchError(
                    f"Family spec variants[{index}] for {task.scenario_id} must define calibration as a mapping",
                    affected_artifact="family_spec",
                )
            _require_declared_verifier_data_path(
                calibration.get("path"),
                field_name=f"variants[{index}].calibration.path",
            )
    if expected_variant_id not in variant_ids:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} does not declare variant_id '{expected_variant_id}'",
            affected_artifact="family_spec",
        )
    try:
        resolve_milestone_test_nodes(
            family_spec,
            family_id=expected_family_id,
            variant_id=expected_variant_id,
        )
    except (KeyError, ValueError) as exc:
        raise ManifestMismatchError(
            f"Family spec for {task.scenario_id} has invalid milestone test-node wiring: {exc}",
            affected_artifact="family_spec",
        ) from exc


def _call_verify_pre_run_hashes(
    verifier: Callable[..., None],
    task: TaskSpec,
    manifest: dict[str, Any],
    *,
    scenario_families_dir: str | Path,
) -> None:
    _call_with_supported_kwargs(
        verifier,
        task,
        manifest,
        scenario_families_dir=scenario_families_dir,
    )


def _call_verify_pre_grading_hashes(
    verifier: Callable[..., None],
    task: TaskSpec,
    manifest: dict[str, Any],
    grader_image_ref: str,
    *,
    verifiers_dir: str | Path,
    verifier_data_dir: str | Path,
    scenario_families_dir: str | Path,
) -> None:
    _call_with_supported_kwargs(
        verifier,
        task,
        manifest,
        grader_image_ref,
        verifiers_dir=verifiers_dir,
        verifier_data_dir=verifier_data_dir,
        scenario_families_dir=scenario_families_dir,
    )


def _load_family_spec_with_configured_path(
    loader: Callable[..., dict[str, Any]],
    family_id: str,
    *,
    scenario_families_dir: str | Path,
) -> dict[str, Any]:
    return _call_with_supported_kwargs(
        loader,
        family_id,
        scenario_families_dir=scenario_families_dir,
    )


def _load_and_validate_family_spec(
    loader: Callable[..., dict[str, Any]],
    task: TaskSpec,
    *,
    scenario_families_dir: str | Path,
) -> dict[str, Any]:
    family_spec = _load_family_spec_with_configured_path(
        loader,
        task.family_id or "",
        scenario_families_dir=scenario_families_dir,
    )
    validate_family_spec(task, family_spec)
    return family_spec


def _grading_dir_for_run(grading_root: str | Path, run_id: str) -> str:
    return str(Path(grading_root) / quote(run_id, safe=""))


class ManifestState:
    def __init__(self, manifest_path: str | Path, grader_image_tag: str) -> None:
        self.manifest_path = str(manifest_path)
        self.grader_image_tag = grader_image_tag
        self.manifest: dict[str, Any] = {}
        self.manifest_version: int = 0
        self.grader_image_ref: str = ""
        self.reload()

    def reload(self) -> None:
        self.manifest = load_codex_long_manifest(self.manifest_path)
        self.manifest_version = int(self.manifest["manifest_version"])
        self.grader_image_ref = self.grader_image_tag
        logger.info("Manifest loaded: version %s", self.manifest_version)


def make_run_id(task: TaskSpec) -> str:
    return f"{task.scenario_id}/{task.model_id}/{task.harness}/seed{task.seed}/attempt{task.attempt}"


@dataclass
class TaskOrchestrator:
    hooks: OrchestratorHooks

    async def execute_task(
        self,
        task: TaskSpec,
        pool_manager: Any,
        manifest_state: ManifestState,
        config: OrchestratorConfig,
    ) -> RunResult:
        run_id = make_run_id(task)
        container: ContainerContext | None = None
        snapshot_ref: str | None = None
        grading_dir: str | None = None
        codex_result: CodexResult | None = None
        telemetry_started = False
        launch_manifest_ver: int | None = None

        if task.dispatch_decision == "regrade_needed":
            return await self._execute_regrade_path(task, pool_manager, manifest_state, config)

        _model_entry, request_model_name = _resolve_task_model_reference(config.model_registry, task.model_id)
        await self.hooks.flush_prefix_cache(config.vllm.client_host, config.vllm.port)
        await self.hooks.health_check(
            config.vllm.client_host,
            config.vllm.port,
            request_model_name,
            max_retries=config.execution.health_check_retries,
            retry_delay_seconds=config.execution.health_check_delay,
        )

        if task.track == "codex_long":
            manifest_state.reload()
            _call_verify_pre_run_hashes(
                self.hooks.verify_pre_run_hashes,
                task,
                manifest_state.manifest,
                scenario_families_dir=config.paths.scenario_families_dir,
            )
            _load_and_validate_family_spec(
                self.hooks.load_family_spec,
                task,
                scenario_families_dir=config.paths.scenario_families_dir,
            )

        claimed = pool_manager.claim_run(
            track=task.track,
            pool_or_split=task.pool_or_split,
            scenario_id=task.scenario_id,
            model_id=task.model_id,
            harness=task.harness,
            seed=task.seed,
            attempt=task.attempt,
            family_id=task.family_id if task.track == "codex_long" else None,
            scenario_type=task.scenario_type if task.track == "codex_long" else None,
            launch_manifest_ver=manifest_state.manifest_version if task.track == "codex_long" else None,
        )
        if not claimed:
            raise DuplicateClaimError(
                f"Run slot already claimed for {task.scenario_id} "
                f"model={task.model_id} seed={task.seed} attempt={task.attempt}."
            )

        launch_manifest_ver = manifest_state.manifest_version if task.track == "codex_long" else None

        try:
            await self.hooks.latency_capture.snapshot_before(task_id=task.scenario_id)
            telemetry_started = True
            if task.track == "swe_bench":
                container = await self.hooks.setup_swe_bench_container(task, config)
            else:
                container = await self.hooks.setup_codex_long_container(task, manifest_state.manifest, config)

            codex_result = await self.hooks.invoke_codex(container, task, config.paths.output_dir)
            verify_result: dict[str, Any] | None = None

            if task.track == "swe_bench":
                patch_path = await self.hooks.extract_swe_bench_patch(container, config.paths.output_dir, task)
                await self.hooks.teardown_swe_bench_container(container)
                container = None

                if codex_result.timed_out:
                    outcome = "timeout"
                elif _is_infrastructure_error(codex_result.stderr):
                    outcome = "crash"
                elif patch_path is None:
                    outcome = "no_patch"
                else:
                    outcome = await self.hooks.drive_swe_bench_eval(
                        task.instance_id or task.scenario_id,
                        patch_path,
                        config.paths.output_dir,
                    )
            else:
                snapshot_ref = await self.hooks.phase1_snapshot(container, run_id)

                manifest_state.reload()
                _call_verify_pre_grading_hashes(
                    self.hooks.verify_pre_grading_hashes,
                    task,
                    manifest_state.manifest,
                    manifest_state.grader_image_ref,
                    verifiers_dir=config.paths.verifiers_dir,
                    verifier_data_dir=config.paths.verifier_data_dir,
                    scenario_families_dir=config.paths.scenario_families_dir,
                )

                family_spec = _load_and_validate_family_spec(
                    self.hooks.load_family_spec,
                    task,
                    scenario_families_dir=config.paths.scenario_families_dir,
                )
                grading_dir = _grading_dir_for_run(config.paths.grading_dir, run_id)
                await self.hooks.phase2_functional_checks(snapshot_ref, task, family_spec, grading_dir)
                verify_result = await self.hooks.phase3_integrity_verification(
                    snapshot_ref,
                    task,
                    grading_dir,
                    manifest_state.grader_image_ref,
                )
                outcome = determine_outcome(codex_result, "codex_long", verify_result=verify_result)
                await self.hooks.cleanup_grading(grading_dir, snapshot_ref, True)
                grading_dir = None

            pool_manager.finish_run(
                track=task.track,
                pool_or_split=task.pool_or_split,
                scenario_id=task.scenario_id,
                model_id=task.model_id,
                harness=task.harness,
                seed=task.seed,
                attempt=task.attempt,
                outcome=outcome,
                wall_time_seconds=codex_result.wall_time_seconds,
                trajectory_path=codex_result.trajectory_path,
                grading_manifest_ver=manifest_state.manifest_version if task.track == "codex_long" else None,
                snapshot_image_ref=snapshot_ref,
                codex_long_pass=verify_result.get("pass") if verify_result else None,
                milestone_results=verify_result.get("milestones") if verify_result else None,
            )

            return RunResult(
                task=task,
                outcome=outcome,
                trajectory_path=codex_result.trajectory_path,
                wall_time_seconds=codex_result.wall_time_seconds,
                verify_result=verify_result,
            )
        except ManifestMismatchError:
            pool_manager.finish_run(
                track=task.track,
                pool_or_split=task.pool_or_split,
                scenario_id=task.scenario_id,
                model_id=task.model_id,
                harness=task.harness,
                seed=task.seed,
                attempt=task.attempt,
                outcome="crash",
                wall_time_seconds=codex_result.wall_time_seconds if codex_result else 0.0,
                trajectory_path=codex_result.trajectory_path if codex_result else None,
                grading_manifest_ver=launch_manifest_ver if task.track == "codex_long" else None,
                snapshot_image_ref=snapshot_ref,
            )
            raise
        except Exception:
            pool_manager.finish_run(
                track=task.track,
                pool_or_split=task.pool_or_split,
                scenario_id=task.scenario_id,
                model_id=task.model_id,
                harness=task.harness,
                seed=task.seed,
                attempt=task.attempt,
                outcome="crash",
                wall_time_seconds=codex_result.wall_time_seconds if codex_result else 0.0,
                trajectory_path=codex_result.trajectory_path if codex_result else None,
                grading_manifest_ver=manifest_state.manifest_version if task.track == "codex_long" else None,
                snapshot_image_ref=snapshot_ref,
            )
            raise
        finally:
            if telemetry_started:
                try:
                    await self.hooks.latency_capture.snapshot_after(task_id=task.scenario_id)
                except Exception:
                    logger.warning("snapshot_after failed for %s", task.scenario_id, exc_info=True)
            if container is not None:
                try:
                    await self.hooks.docker_rm(container.container_id, True)
                except Exception:
                    logger.warning("Failed to clean up agent container %s", container.container_id, exc_info=True)
            if grading_dir is not None and os.path.exists(grading_dir):
                try:
                    shutil.rmtree(grading_dir)
                except Exception:
                    logger.warning("Failed to clean up grading workspace %s", grading_dir, exc_info=True)

    async def _execute_regrade_path(
        self,
        task: TaskSpec,
        pool_manager: Any,
        manifest_state: ManifestState,
        config: OrchestratorConfig,
    ) -> RunResult:
        manifest_state.reload()
        snapshot_ref = task.regrade_snapshot_ref
        if not snapshot_ref:
            raise TaskDispatchError(
                f"Regrade requested for {task.scenario_id} but no snapshot_image_ref is available. Cannot regrade."
            )
        if not await self.hooks.docker_image_exists(snapshot_ref):
            raise TaskDispatchError(
                f"Retained snapshot image '{snapshot_ref}' not found. Snapshot may have been pruned. Full rerun required."
            )
        claimed = pool_manager.claim_run(
            track=task.track,
            pool_or_split=task.pool_or_split,
            scenario_id=task.scenario_id,
            model_id=task.model_id,
            harness=task.harness,
            seed=task.seed,
            attempt=task.attempt,
            family_id=task.family_id,
            scenario_type=task.scenario_type,
            launch_manifest_ver=None,
        )
        if not claimed:
            raise DuplicateClaimError(
                f"Run slot already claimed for {task.scenario_id} "
                f"model={task.model_id} seed={task.seed} attempt={task.attempt}."
            )

        run_id = make_run_id(task)
        grading_dir = _grading_dir_for_run(config.paths.grading_dir, run_id)
        try:
            _call_verify_pre_grading_hashes(
                self.hooks.verify_pre_grading_hashes,
                task,
                manifest_state.manifest,
                manifest_state.grader_image_ref,
                verifiers_dir=config.paths.verifiers_dir,
                verifier_data_dir=config.paths.verifier_data_dir,
                scenario_families_dir=config.paths.scenario_families_dir,
            )
            family_spec = _load_and_validate_family_spec(
                self.hooks.load_family_spec,
                task,
                scenario_families_dir=config.paths.scenario_families_dir,
            )
            await self.hooks.phase2_functional_checks(snapshot_ref, task, family_spec, grading_dir)
            verify_result = await self.hooks.phase3_integrity_verification(
                snapshot_ref,
                task,
                grading_dir,
                manifest_state.grader_image_ref,
            )
            await self.hooks.cleanup_grading(grading_dir, snapshot_ref, True)
            grading_dir = None
            outcome = "resolved" if verify_result.get("pass") else "failed"
            pool_manager.finish_run(
                track=task.track,
                pool_or_split=task.pool_or_split,
                scenario_id=task.scenario_id,
                model_id=task.model_id,
                harness=task.harness,
                seed=task.seed,
                attempt=task.attempt,
                outcome=outcome,
                wall_time_seconds=None,
                trajectory_path=None,
                grading_manifest_ver=manifest_state.manifest_version,
                snapshot_image_ref=snapshot_ref,
                codex_long_pass=bool(verify_result.get("pass")),
                milestone_results=verify_result.get("milestones"),
            )
            return RunResult(
                task=task,
                outcome=outcome,
                trajectory_path=None,
                wall_time_seconds=None,
                verify_result=verify_result,
            )
        except Exception:
            pool_manager.finish_run(
                track=task.track,
                pool_or_split=task.pool_or_split,
                scenario_id=task.scenario_id,
                model_id=task.model_id,
                harness=task.harness,
                seed=task.seed,
                attempt=task.attempt,
                outcome="crash",
                wall_time_seconds=0.0,
                trajectory_path=None,
                grading_manifest_ver=manifest_state.manifest_version,
                snapshot_image_ref=snapshot_ref,
            )
            raise
        finally:
            if grading_dir is not None and os.path.exists(grading_dir):
                shutil.rmtree(grading_dir, ignore_errors=True)


__all__ = [
    "CONTAINER_RESOURCE_LIMITS",
    "CacheFlushError",
    "CodexInstallConfig",
    "CodexResult",
    "ConfigError",
    "ContainerContext",
    "DuplicateClaimError",
    "ExecResult",
    "ExecutionConfig",
    "GradingConfig",
    "HealthCheckError",
    "ManifestMismatchError",
    "ManifestState",
    "NetworkConfig",
    "OrchestratorConfig",
    "OrchestratorHooks",
    "PathsConfig",
    "RunResult",
    "TaskDispatchError",
    "TaskOrchestrator",
    "TaskSpec",
    "VllmConfig",
    "build_codex_command",
    "capture_event_stream",
    "determine_outcome",
    "docker_exec_with_timeout",
    "drive_swe_bench_eval",
    "find_manifest_entry",
    "flush_prefix_cache",
    "generate_codex_config",
    "get_bridge_gateway_ip",
    "get_codex_harness_env",
    "get_codex_harness_mounts",
    "health_check",
    "load_family_spec",
    "make_run_id",
    "sha256_tree",
    "validate_family_spec",
    "verify_pre_grading_hashes",
    "verify_pre_run_hashes",
]
