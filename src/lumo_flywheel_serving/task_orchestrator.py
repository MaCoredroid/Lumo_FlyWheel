from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import quote

import requests

from .data_pool import _find_manifest_variant, load_codex_long_manifest, make_scenario_id, sha256_file
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
            if self.dispatch_decision != "regrade_needed" and not self.image_digest:
                raise ValueError(
                    "Codex-Long tasks require image_digest unless dispatch_decision='regrade_needed'"
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
    return {
        str(codex_binary_path): {"bind": "/usr/local/bin/codex", "mode": "ro"},
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
    if image_ref.startswith("sha256:"):
        return image_ref
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
    actual_image_digest = _canonical_sha256(image_digest_resolver(task.image_digest or ""))
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

    actual_grader_digest = _canonical_sha256(image_digest_resolver(grader_image_ref))
    expected_grader_digest = _canonical_sha256(manifest["grader_image_digest"])
    if actual_grader_digest != expected_grader_digest:
        raise ManifestMismatchError(
            f"Grader image digest mismatch: expected {expected_grader_digest}, got {actual_grader_digest}.",
            affected_artifact="grader_image",
        )

    family_verifier_dir = Path(verifiers_dir) / str(task.family_id)
    verifier_path = family_verifier_dir / "verify.sh"
    try:
        actual_verifier_hash = _canonical_sha256(sha256_file(verifier_path))
    except FileNotFoundError as exc:
        raise ManifestMismatchError(
            f"Verifier script missing for {task.scenario_id}: {verifier_path}",
            affected_artifact="verifier",
        ) from exc
    if actual_verifier_hash != _canonical_sha256(entry["verifier_hash"]):
        raise ManifestMismatchError(
            f"Verifier hash mismatch for {task.scenario_id}: "
            f"expected {entry['verifier_hash']}, got {actual_verifier_hash}",
            affected_artifact="verifier",
        )

    milestone_hashes = entry.get("milestone_hashes", {})
    milestones_dir = family_verifier_dir / "milestones"
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
        actual_hash = _canonical_sha256(sha256_file(milestone_path))
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

    unexpected_milestone_dirs = sorted(path.name for path in milestones_dir.iterdir() if path.is_dir()) if milestones_dir.exists() else []
    if unexpected_milestone_dirs:
        raise ManifestMismatchError(
            "Milestones directory contains unexpected subdirectories for "
            f"{task.scenario_id}: {unexpected_milestone_dirs}",
            affected_artifact="milestone",
        )

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
                container = None

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

                family_spec = _load_family_spec_with_configured_path(
                    self.hooks.load_family_spec,
                    task.family_id or "",
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
            family_spec = _load_family_spec_with_configured_path(
                self.hooks.load_family_spec,
                task.family_id or "",
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
    "verify_pre_grading_hashes",
    "verify_pre_run_hashes",
]
