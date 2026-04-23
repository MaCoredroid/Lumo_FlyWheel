from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

from .metrics import parse_prometheus_text, resolve_metric_schema
from .registry import ModelConfig, load_registry
from .tuned_config import (
    RuntimeStateStore,
    StructuredValidationError,
    TunedConfigBundle,
    ValidationIssue,
    apply_tuned_vllm_config,
    default_weight_version_id,
    load_tuned_config_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNTIME_ENV_FILENAME = ".lumo.local.env"
DEFAULT_STATE_ROOT = REPO_ROOT / "output" / "serving_state"
DEFAULT_VLLM_BASE_IMAGE = "nvcr.io/nvidia/pytorch:26.01-py3"
DEFAULT_VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"
DEFAULT_VLLM_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.nvidia-vllm"
DEFAULT_INFERENCE_PROXY_PORT = 8001
VLLM_ENABLE_RESPONSES_API_STORE = "1"
QWEN_CHAT_TEMPLATE_HOST_PATH = REPO_ROOT / "docker" / "chat_templates" / "qwen3-openai-codex.jinja"
QWEN_CHAT_TEMPLATE_CONTAINER_PATH = Path("/opt/lumo/chat_templates/qwen3-openai-codex.jinja")
MIN_GPU_MEMORY_UTILIZATION = max(0.05, float(os.environ.get("LUMO_MIN_GPU_MEMORY_UTILIZATION", "0.05")))
MIN_KV_CACHE_STARTUP_UTILIZATION = 0.30
UNSUPPORTED_NVIDIA_SMI_GRACE_S = 20
LOW_FREE_MEMORY_GRACE_RETRIES = 2
LOW_FREE_MEMORY_GRACE_SLEEP_S = 45
CUDA_MEMORY_RECOVERY_MARGIN_GIB = 1.0
HOST_MEMORY_RECOVERY_COMMAND = "sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true"
GPU_MEMORY_ERROR_RE = re.compile(
    r"Free memory on device cuda:\d+ \((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\)"
)
LOCAL_ENV_ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$")


class ModelServer:
    def __init__(
        self,
        registry_path: str | Path,
        port: int = 8000,
        image: str = DEFAULT_VLLM_IMAGE,
        container_name: str = "lumo-vllm",
        logs_root: str | Path = "/logs",
        triton_cache_root: str | Path = "/tmp/triton_cache",
        use_sleep_mode: bool = False,
        proxy_port: int = DEFAULT_INFERENCE_PROXY_PORT,
        state_root: str | Path = DEFAULT_STATE_ROOT,
    ) -> None:
        self.registry_path = Path(registry_path).resolve()
        self._load_local_runtime_env()
        self.registry = load_registry(self.registry_path)
        self.port = port
        self.image = image
        self.container_name = container_name
        self.logs_root = Path(logs_root).resolve()
        self.triton_cache_root = Path(triton_cache_root).resolve()
        self.use_sleep_mode = use_sleep_mode
        self.proxy_port = proxy_port
        self.state_store = RuntimeStateStore(Path(state_root).resolve())
        self.current_model = self.state_store.load().current_model_id

    def ensure_runtime_scaffolding(self) -> None:
        for path in (Path("/models"), self.logs_root, self.triton_cache_root):
            path.mkdir(parents=True, exist_ok=True)

    def proxy_base_url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.proxy_port}"

    def _proxy_pid_path(self) -> Path:
        return self.logs_root / "codex_inference_proxy.pid"

    def _proxy_log_path(self) -> Path:
        return self.logs_root / "codex_inference_proxy.log"

    def _start_proxy(self) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self._stop_proxy(missing_ok=True)
        pid_path = self._proxy_pid_path()
        log_path = self._proxy_log_path()
        command = [
            sys.executable,
            "-m",
            "lumo_flywheel_serving.inference_proxy",
            "--listen-host",
            "0.0.0.0",
            "--listen-port",
            str(self.proxy_port),
            "--upstream-base-url",
            f"http://127.0.0.1:{self.port}",
            "--pid-file",
            str(pid_path),
            "--log-path",
            str(log_path),
            "--registry-path",
            str(self.registry_path),
            "--state-root",
            str(self.state_store.root),
        ]
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=handle,
                stderr=handle,
                start_new_session=True,
                text=True,
                env=os.environ.copy(),
            )
        try:
            self._wait_proxy_ready(timeout_s=10)
        except Exception:
            process.poll()
            if process.returncode is None:
                process.terminate()
            raise

    def _wait_proxy_ready(self, timeout_s: int = 10) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex(("127.0.0.1", self.proxy_port)) == 0:
                    return
            time.sleep(0.2)
        raise RuntimeError(f"Inference proxy not ready within {timeout_s}s")

    def _stop_proxy(self, missing_ok: bool = False) -> None:
        pid_path = self._proxy_pid_path()
        if not pid_path.exists():
            if missing_ok:
                return
            raise RuntimeError("Inference proxy pid file is not present")
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid_path.unlink(missing_ok=True)
            if missing_ok:
                return
            raise RuntimeError(f"Invalid inference proxy pid file: {pid_path}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            return
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pid_path.unlink(missing_ok=True)
                return
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid_path.unlink(missing_ok=True)

    def _load_local_runtime_env(self) -> None:
        for env_path in self._local_runtime_env_candidates():
            if not env_path.is_file():
                continue
            for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                match = LOCAL_ENV_ASSIGNMENT_RE.fullmatch(line)
                if match is None:
                    raise RuntimeError(f"Invalid local runtime env line in {env_path}:{line_number}")
                key = match.group("key")
                if key in os.environ:
                    continue
                os.environ[key] = self._parse_local_runtime_env_value(match.group("value"))
            return

    def _local_runtime_env_candidates(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        override = os.environ.get("LUMO_LOCAL_RUNTIME_ENV")
        if override:
            candidates.append(Path(override))
        registry_candidate = self.registry_path.resolve().parent / LOCAL_RUNTIME_ENV_FILENAME
        candidates.append(registry_candidate)
        repo_candidate = REPO_ROOT / LOCAL_RUNTIME_ENV_FILENAME
        if repo_candidate not in candidates:
            candidates.append(repo_candidate)
        return tuple(candidates)

    @staticmethod
    def _parse_local_runtime_env_value(raw_value: str) -> str:
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    def active_tuned_config_bundle(self, model_id: str) -> tuple[str | None, TunedConfigBundle | None]:
        state = self.state_store.load()
        bundle_path = state.active_tuned_config_path
        if not bundle_path:
            return None, None
        bundle = load_tuned_config_bundle(bundle_path)
        if bundle.model_id != model_id:
            return None, None
        return bundle_path, bundle

    def resolved_model_config(self, model_id: str) -> tuple[ModelConfig, str | None, TunedConfigBundle | None]:
        config = self.registry[model_id]
        bundle_path, bundle = self.active_tuned_config_bundle(model_id)
        if bundle is None:
            return config, None, None
        return apply_tuned_vllm_config(config, bundle), bundle_path, bundle

    def load_tuned_config(self, bundle_path: str | Path) -> TunedConfigBundle:
        bundle = load_tuned_config_bundle(bundle_path)
        if bundle.model_id not in self.registry:
            raise RuntimeError(
                f"Tuned-config bundle model_id {bundle.model_id!r} is not present in registry {self.registry_path}"
            )
        self.state_store.activate_bundle(bundle_path, bundle)
        return bundle

    def invalidate(self, *, weight_version_id: str) -> None:
        stripped = weight_version_id.strip()
        if not stripped:
            raise StructuredValidationError(
                message="Invalid invalidate payload",
                issues=[ValidationIssue(field="weight_version_id", message="must be a non-empty string")],
            )
        self.state_store.record_invalidate(weight_version_id=stripped)
        try:
            self.flush_prefix_cache()
        except requests.RequestException:
            return

    def resume_last_known_good(self, *, from_baseline: bool, enable_request_logging: bool = False) -> dict[str, str | None]:
        state = self.state_store.load()
        model_id = state.last_known_good_model_id or state.current_model_id
        if not model_id:
            raise RuntimeError("No last-known-good model is recorded; serve or load a tuned config first.")
        if from_baseline:
            self.state_store.clear_active_bundle()
        elif state.last_known_good_tuned_config_path:
            self.load_tuned_config(state.last_known_good_tuned_config_path)
        self.start(model_id=model_id, enable_request_logging=enable_request_logging)
        refreshed = self.state_store.load()
        return {
            "model_id": refreshed.current_model_id,
            "weight_version_id": refreshed.current_weight_version_id,
            "tuned_config_path": refreshed.active_tuned_config_path,
        }

    def start(self, model_id: str, enable_request_logging: bool = False) -> None:
        self.ensure_runtime_scaffolding()
        self._ensure_image_present()
        config, active_bundle_path, active_bundle = self.resolved_model_config(model_id)
        weight_version_id = (
            active_bundle.weight_version_id if active_bundle is not None else default_weight_version_id(self.registry[model_id])
        )
        self._reset_log(model_id)
        self.stop(missing_ok=True)
        self._recover_host_memory()
        # A prior stop can return before GB10 unified-memory pressure fully drains.
        # Only wait for the minimum viable floor before the first launch so the
        # low-VRAM retry ladder can still engage when the configured target is
        # temporarily unattainable on the host.
        self._wait_vram_free(timeout_s=120, required_utilization=MIN_GPU_MEMORY_UTILIZATION)
        kv_cache_dtype = self._initial_kv_cache_dtype(config)
        gpu_memory_utilization = config.gpu_memory_utilization
        enforce_eager = False
        low_memory_grace_retries = 0
        while True:
            self._run(
                self._build_run_command(
                    model_id,
                    config,
                    enable_request_logging,
                    kv_cache_dtype=kv_cache_dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    tuned_config_id=active_bundle.bundle_id if active_bundle is not None else None,
                    weight_version_id=weight_version_id,
                ),
                capture_output=False,
            )
            try:
                self._wait_ready(model_id=model_id)
                break
            except RuntimeError as exc:
                self.stop(missing_ok=True)
                error_text = str(exc)
                incompatible_fp8_kv = "fp8_e5m2 kv-cache is not supported with fp8 checkpoints."
                insufficient_memory = "Free memory on device cuda:0"
                no_kv_cache_memory = "No available memory for the cache blocks."
                fp8_cutlass_internal_error = "cutlass_scaled_mm"
                retry_utilization = gpu_memory_utilization
                if incompatible_fp8_kv in error_text and kv_cache_dtype != "auto":
                    kv_cache_dtype = "auto"
                    self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                    continue
                if insufficient_memory in error_text:
                    if self._should_retry_low_free_memory(
                        error_text,
                        retries=low_memory_grace_retries,
                        current=gpu_memory_utilization,
                    ):
                        low_memory_grace_retries += 1
                        time.sleep(LOW_FREE_MEMORY_GRACE_SLEEP_S)
                        self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                        continue
                    next_gpu_memory_utilization = self._next_gpu_memory_utilization(
                        error_text,
                        current=gpu_memory_utilization,
                    )
                    if next_gpu_memory_utilization is not None:
                        low_memory_grace_retries = 0
                        gpu_memory_utilization = next_gpu_memory_utilization
                        retry_utilization = next_gpu_memory_utilization
                        self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                        continue
                if no_kv_cache_memory in error_text:
                    next_gpu_memory_utilization = self._next_gpu_memory_utilization_for_kv_cache_startup(current=gpu_memory_utilization)
                    if next_gpu_memory_utilization is not None:
                        low_memory_grace_retries = 0
                        gpu_memory_utilization = next_gpu_memory_utilization
                        retry_utilization = next_gpu_memory_utilization
                        self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                        continue
                if fp8_cutlass_internal_error in error_text and not enforce_eager:
                    enforce_eager = True
                    self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                    continue
                raise
        self._start_proxy()
        self.current_model = model_id
        self.state_store.record_start(
            model_id=model_id,
            weight_version_id=weight_version_id,
            active_bundle_path=active_bundle_path,
            active_bundle_id=active_bundle.bundle_id if active_bundle is not None else None,
        )

    def switch_model(self, model_id: str, enable_request_logging: bool = False) -> None:
        if self.use_sleep_mode and self.current_model == model_id and self._is_serving_model(model_id):
            return
        self.stop(missing_ok=True)
        self._wait_vram_free(timeout_s=120, required_utilization=MIN_GPU_MEMORY_UTILIZATION)
        self.start(model_id=model_id, enable_request_logging=enable_request_logging)

    def stop(self, missing_ok: bool = False) -> None:
        result = self._run(
            ["docker", "ps", "-a", "--filter", f"name=^{self.container_name}$", "--format", "{{.Names}}"]
        )
        if self.container_name not in result.stdout.splitlines():
            if missing_ok:
                self._stop_proxy(missing_ok=True)
                self._recover_host_memory()
                return
            raise RuntimeError(f"Container {self.container_name} is not present")

        self._run(["docker", "stop", "-t", "30", self.container_name], capture_output=False, check=False)
        self._run(["docker", "rm", "-f", self.container_name], capture_output=False, check=False)
        self._stop_proxy(missing_ok=True)
        self.current_model = None
        self._recover_host_memory()

    def flush_prefix_cache(self) -> None:
        response = requests.post(
            f"http://127.0.0.1:{self.port}/reset_prefix_cache",
            headers=self._request_headers(),
            timeout=10,
        )
        response.raise_for_status()

    def health(self) -> requests.Response:
        response = requests.get(
            f"http://127.0.0.1:{self.port}/health",
            headers=self._request_headers(),
            timeout=5,
        )
        response.raise_for_status()
        return response

    def models(self) -> requests.Response:
        response = requests.get(
            f"http://127.0.0.1:{self.port}/v1/models",
            headers=self._request_headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response

    def metrics(self) -> requests.Response:
        response = requests.get(
            f"http://127.0.0.1:{self.port}/metrics",
            headers=self._request_headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response

    def logs_path(self, model_id: str) -> Path:
        return self.logs_root / f"vllm_{model_id}.log"

    def record_launch_metadata(self, model_id: str, **metadata: str | float | int | bool) -> None:
        if not metadata:
            return
        rendered = " ".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in sorted(metadata.items())
        )
        self._append_log_text(model_id, f"[VLLM-META] {rendered}\n")

    @staticmethod
    def _api_key() -> str:
        return os.environ.get("VLLM_API_KEY") or "EMPTY"

    @staticmethod
    def _request_headers() -> dict[str, str]:
        return {"Authorization": f"Bearer {ModelServer._api_key()}"}

    def _recover_host_memory(self) -> None:
        if os.environ.get("LUMO_HOST_MEMORY_RECOVERY", "1").lower() in {"0", "false", "no"}:
            return

        password = os.environ.get("LUMO_SUDO_PASSWORD")
        if password:
            command = (
                f"printf '%s\\n' {shlex.quote(password)} | sudo -S "
                + shlex.join(["bash", "-lc", HOST_MEMORY_RECOVERY_COMMAND])
            )
            result = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        else:
            result = subprocess.run(
                ["sudo", "-n", "bash", "-lc", HOST_MEMORY_RECOVERY_COMMAND],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )

        # Recovery is a best-effort machine hygiene step. Keep startup/teardown
        # working when sudo is unavailable, but surface unexpected failures.
        if result.returncode != 0:
            stderr = (result.stderr or "").lower()
            if "a password is required" in stderr or "password was provided" in stderr:
                return
            if "sudo:" in stderr and "permission" in stderr:
                return
            if "a terminal is required" in stderr:
                return
            raise RuntimeError(
                "Host memory recovery failed before/after vLLM lifecycle event:\n"
                f"{result.stdout}{result.stderr}"
            )

    def _append_log_text(self, model_id: str, text: str) -> None:
        log_path = self.logs_path(model_id)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
            return
        except PermissionError:
            pass

        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.container_name,
                "python3",
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "Path(sys.argv[1]).open('a', encoding='utf-8').write(sys.stdin.read())"
                ),
                str(log_path),
            ],
            input=text,
            text=True,
            check=False,
            capture_output=True,
            env=os.environ.copy(),
        )

    def _reset_log(self, model_id: str) -> None:
        log_path = self.logs_path(model_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_path.write_text("", encoding="utf-8")
            return
        except PermissionError:
            pass

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.logs_root}:{self.logs_root}",
                "--entrypoint",
                "python3",
                self.image,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('', encoding='utf-8')",
                str(log_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    @staticmethod
    def _gpu_memory_snapshot(error_text: str) -> tuple[float, float] | None:
        match = GPU_MEMORY_ERROR_RE.search(error_text)
        if match is None:
            return None
        return float(match.group("free")), float(match.group("total"))

    @classmethod
    def _should_retry_low_free_memory(cls, error_text: str, retries: int, current: float) -> bool:
        if retries >= LOW_FREE_MEMORY_GRACE_RETRIES:
            return False
        snapshot = cls._gpu_memory_snapshot(error_text)
        if snapshot is None:
            return False
        free_gib, total_gib = snapshot
        if total_gib <= 0:
            return False
        # If the host cannot satisfy even the minimum viable utilization floor,
        # extra grace sleeps cannot change the launcher decision.
        if free_gib < (total_gib * MIN_GPU_MEMORY_UTILIZATION):
            return False
        return (free_gib / total_gib) < 0.5

    @staticmethod
    def _next_gpu_memory_utilization(error_text: str, current: float) -> float | None:
        snapshot = ModelServer._gpu_memory_snapshot(error_text)
        if snapshot is not None:
            free_gib, total_gib = snapshot
            usable_free_gib = max(0.0, free_gib - 1.0)
            candidate = max(MIN_GPU_MEMORY_UTILIZATION, int((usable_free_gib / total_gib) * 100) / 100)
            if candidate < current:
                return candidate

        for fallback in (0.85, 0.65):
            if current > fallback:
                return fallback

        stepped = round(current - 0.05, 2)
        if stepped >= MIN_GPU_MEMORY_UTILIZATION:
            return stepped
        return None

    @staticmethod
    def _next_gpu_memory_utilization_for_kv_cache_startup(current: float) -> float | None:
        if current < MIN_KV_CACHE_STARTUP_UTILIZATION:
            return MIN_KV_CACHE_STARTUP_UTILIZATION
        stepped = round(current + 0.05, 2)
        if stepped <= 0.95:
            return stepped
        return None

    def _build_run_command(
        self,
        model_id: str,
        config: ModelConfig,
        enable_request_logging: bool,
        kv_cache_dtype: str,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        tuned_config_id: str | None = None,
        weight_version_id: str | None = None,
    ) -> list[str]:
        log_path = self.logs_path(model_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        triton_cache_dir = self.triton_cache_root / model_id
        triton_cache_dir.mkdir(parents=True, exist_ok=True)
        chat_template_path = self._chat_template_container_path(config)

        vllm_args = [
            "vllm",
            "serve",
            str(config.local_path),
            "--served-model-name",
            config.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--quantization",
            config.quantization,
            "--dtype",
            config.dtype,
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--max-model-len",
            str(config.max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--max-num-seqs",
            str(config.max_num_seqs),
        ]
        if config.enable_prefix_caching:
            vllm_args.append("--enable-prefix-caching")
        if config.enable_chunked_prefill:
            vllm_args.append("--enable-chunked-prefill")
        if config.lora_modules:
            vllm_args.extend(["--enable-lora", "--max-lora-rank", str(config.max_lora_rank), "--lora-modules"])
            vllm_args.extend(self._format_lora_modules(config))
        if chat_template_path is not None:
            vllm_args.extend(["--chat-template", str(chat_template_path)])
        if self._uses_qwen_openai_parsers(config):
            vllm_args.extend(
                [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "qwen3_xml",
                    "--reasoning-parser",
                    "qwen3",
                ]
            )
        if enforce_eager:
            vllm_args.append("--enforce-eager")
        if self.use_sleep_mode:
            vllm_args.append("--enable-sleep-mode")
        if enable_request_logging:
            vllm_args.append("--enable-log-requests")

        header = self._header_script(
            model_id=model_id,
            config=config,
            log_path=log_path,
            vllm_args=vllm_args,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tuned_config_id=tuned_config_id,
            weight_version_id=weight_version_id,
        )
        shell_cmd = (
            "set -euo pipefail\n"
            + header
            + "\n"
            + shlex.join(vllm_args)
            + " 2>&1 | tee -a "
            + shlex.quote(str(log_path))
        )

        return [
            "docker",
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--network",
            "host",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--gpus",
            "all",
            "-e",
            "VLLM_SERVER_DEV_MODE=1",
            "-e",
            f"VLLM_ENABLE_RESPONSES_API_STORE={VLLM_ENABLE_RESPONSES_API_STORE}",
            "-e",
            "TOKENIZERS_PARALLELISM=false",
            "-e",
            f"TRITON_CACHE_DIR={triton_cache_dir}",
            "-e",
            f"VLLM_API_KEY={self._api_key()}",
            "-e",
            f"VLLM_LOGGING_LEVEL={'DEBUG' if enable_request_logging else 'INFO'}",
            "-v",
            "/models:/models",
            "-v",
            f"{self.logs_root}:{self.logs_root}",
            "-v",
            f"{self.triton_cache_root}:{self.triton_cache_root}",
            "-v",
            f"{QWEN_CHAT_TEMPLATE_HOST_PATH}:{QWEN_CHAT_TEMPLATE_CONTAINER_PATH}:ro",
            "--entrypoint",
            "bash",
            self.image,
            "-lc",
            shell_cmd,
        ]

    def _ensure_image_present(self) -> None:
        inspect = self._run(["docker", "image", "inspect", self.image], check=False)
        if inspect.returncode == 0:
            return
        if self.image != DEFAULT_VLLM_IMAGE:
            return
        raise RuntimeError(
            f"Docker image {self.image!r} is not present locally. "
            "Build it first with `lumoserve build-image` or `make bootstrap-runtime`."
        )

    def _header_script(
        self,
        model_id: str,
        config: ModelConfig,
        log_path: Path,
        vllm_args: list[str],
        kv_cache_dtype: str,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        tuned_config_id: str | None,
        weight_version_id: str | None,
    ) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_id": model_id,
            "served_model_name": config.served_model_name,
            "quantization": config.quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "wire_api": "responses",
            "responses_api_store": VLLM_ENABLE_RESPONSES_API_STORE == "1",
            "dev_mode": True,
            "sleep_mode": self.use_sleep_mode,
            "lora_modules": [name for name, _path in config.lora_modules],
            "max_lora_rank": config.max_lora_rank,
            "tuned_config_id": tuned_config_id or "baseline",
            "weight_version_id": weight_version_id or default_weight_version_id(config),
            "launch_cmd": shlex.join(vllm_args),
        }
        encoded = json.dumps(payload)
        return (
            "python3 - <<'PY'\n"
            "import importlib.metadata\n"
            "import json\n"
            "import vllm\n"
            f"payload = json.loads({encoded!r})\n"
            "version = getattr(vllm, '__version__', None) or importlib.metadata.version('vllm')\n"
            "git_hash = getattr(vllm, '__commit__', None) or 'unknown'\n"
            f"log_path = {str(log_path)!r}\n"
            "with open(log_path, 'a', encoding='utf-8') as handle:\n"
            "    handle.write(f\"[VLLM-INIT] timestamp={payload['timestamp']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] model_id={payload['model_id']} served_model_name={payload['served_model_name']} vllm_version={version} git_hash={git_hash}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] quantization={payload['quantization']} kv_cache_dtype={payload['kv_cache_dtype']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] max_model_len={payload['max_model_len']} gpu_memory_utilization={payload['gpu_memory_utilization']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] enforce_eager={str(payload['enforce_eager']).lower()}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] tuned_config_id={payload['tuned_config_id']} weight_version_id={payload['weight_version_id']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] wire_api={payload['wire_api']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] responses_api_store={str(payload['responses_api_store']).lower()}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] dev_mode={str(payload['dev_mode']).lower()} sleep_mode={'enabled' if payload['sleep_mode'] else 'disabled'}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] lora_modules={','.join(payload['lora_modules']) or 'none'} max_lora_rank={payload['max_lora_rank']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] launch_cmd: {payload['launch_cmd']}\\n\")\n"
            "PY"
        )

    @staticmethod
    def _chat_template_container_path(config: ModelConfig) -> Path | None:
        if config.hf_repo.lower().startswith("qwen/"):
            return QWEN_CHAT_TEMPLATE_CONTAINER_PATH
        return None

    @staticmethod
    def _uses_qwen_openai_parsers(config: ModelConfig) -> bool:
        return config.hf_repo.lower().startswith("qwen/")

    @staticmethod
    def _expected_served_model_ids(config: ModelConfig) -> set[str]:
        expected = {config.served_model_name}
        expected.update(adapter_name for adapter_name, _adapter_path in config.lora_modules)
        return expected

    @staticmethod
    def _served_model_surface_matches(expected_model_ids: set[str], served_model_ids: list[str]) -> bool:
        return len(served_model_ids) == len(set(served_model_ids)) and set(served_model_ids) == expected_model_ids

    @staticmethod
    def _request_model_name(config: ModelConfig) -> str:
        # When LoRA is enabled, Codex targets the adapter name rather than the
        # base served model id. Smoke tests should probe that surface directly.
        if config.lora_modules:
            return config.lora_modules[0][0]
        return config.served_model_name

    @staticmethod
    def _initial_kv_cache_dtype(config: ModelConfig) -> str:
        # vLLM 0.19.0 rejects fp8_e5m2 KV cache for fp8 checkpoints during
        # engine init, so use a valid first-launch default instead of relying on
        # a guaranteed crash-and-retry cycle.
        if config.quantization == "fp8" and config.kv_cache_dtype == "fp8_e5m2":
            return "auto"
        return config.kv_cache_dtype

    @staticmethod
    def _format_lora_modules(config: ModelConfig) -> list[str]:
        formatted: list[str] = []
        for adapter_name, adapter_path in config.lora_modules:
            container_path = PurePosixPath(str(adapter_path))
            if (
                container_path.anchor != "/"
                or any(part in {".", ".."} for part in container_path.parts)
                or not str(container_path).startswith("/models/")
            ):
                raise ValueError(
                    f"LoRA adapter '{adapter_name}' must use a container path under /models; got {adapter_path}"
                )
            formatted.append(f"{adapter_name}={adapter_path}")
        return formatted

    def _wait_ready(self, model_id: str, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        log_path = self.logs_path(model_id)
        start = time.time()
        config = self.registry[model_id]
        expected_model_ids = self._expected_served_model_ids(config)
        while time.time() < deadline:
            inspect = self._run(
                ["docker", "inspect", "-f", "{{.State.Status}}", self.container_name], check=False
            )
            if inspect.returncode != 0 or inspect.stdout.strip() == "exited":
                logs = self._run(["docker", "logs", "--tail", "200", self.container_name], check=False)
                log_path_text = ""
                if log_path.exists():
                    log_path_text = log_path.read_text(encoding="utf-8")
                raise RuntimeError(
                    "vLLM container failed during startup:\n"
                    f"{logs.stdout}{logs.stderr}"
                    f"{log_path_text}"
                )
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.port}/health",
                    headers=self._request_headers(),
                    timeout=5,
                )
                if response.status_code == 200:
                    served_model_ids = self._served_model_ids()
                    if not self._served_model_surface_matches(expected_model_ids, served_model_ids):
                        time.sleep(5)
                        continue
                    metrics_response = self.metrics()
                    resolve_metric_schema(parse_prometheus_text(metrics_response.text))
                    self._append_log_text(
                        model_id,
                        f"[VLLM-READY] cuda_graph_capture_time={time.time() - start:.1f}s\n"
                        f"[VLLM-READY] /health 200 OK at {datetime.now(UTC).isoformat()}\n"
                        f"[VLLM-READY] expected_models={','.join(sorted(expected_model_ids))}\n"
                        f"[VLLM-READY] served_models={','.join(served_model_ids)}\n",
                    )
                    return
            except (requests.RequestException, RuntimeError, ValueError, TypeError):
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not ready within {timeout_s}s")

    def _served_model_ids(self) -> list[str]:
        payload = self.models().json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("vLLM /v1/models payload missing data list")
        model_ids: list[str] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"vLLM /v1/models data[{index}] must be an object with a string id")
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError(f"vLLM /v1/models data[{index}] must include a non-empty string id")
            model_ids.append(model_id)
        if not model_ids:
            raise ValueError("vLLM /v1/models returned no served model ids")
        return model_ids

    def _is_serving_model(self, model_id: str) -> bool:
        try:
            self.health()
            expected_model_ids = self._expected_served_model_ids(self.registry[model_id])
            return self._served_model_surface_matches(expected_model_ids, self._served_model_ids())
        except (requests.RequestException, ValueError, TypeError):
            return False

    def _cuda_mem_get_info_gib(self) -> tuple[float, float] | None:
        probe = self._run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "python3",
                self.image,
                "-c",
                (
                    "import torch; "
                    "free, total = torch.cuda.mem_get_info(); "
                    "print(f'{free / 1024**3:.2f} {total / 1024**3:.2f}')"
                ),
            ],
            check=False,
        )
        if probe.returncode != 0:
            return None
        parts = probe.stdout.strip().split()
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _wait_vram_free(self, timeout_s: int = 120, required_utilization: float | None = None) -> None:
        if os.environ.get("LUMO_SKIP_VRAM_WAIT", "0").lower() in {"1", "true", "yes"}:
            return
        deadline = time.time() + timeout_s
        last_snapshot: tuple[float, float] | None = None
        last_busy_pids: list[str] = []
        while time.time() < deadline:
            remaining_s = max(0.0, deadline - time.time())
            result = self._run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                check=False,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines == ["[Not Supported]"]:
                    memory_snapshot = self._cuda_mem_get_info_gib()
                    last_snapshot = memory_snapshot
                    if memory_snapshot is None:
                        time.sleep(min(remaining_s, UNSUPPORTED_NVIDIA_SMI_GRACE_S))
                        continue
                    if self._has_required_free_memory(memory_snapshot, required_utilization):
                        return
                    time.sleep(min(remaining_s, UNSUPPORTED_NVIDIA_SMI_GRACE_S))
                    continue
                pids = [line for line in lines if line != "[Not Supported]"]
                if not pids:
                    last_busy_pids = []
                    memory_snapshot = self._cuda_mem_get_info_gib()
                    last_snapshot = memory_snapshot
                    if memory_snapshot is None:
                        return
                    if self._has_required_free_memory(memory_snapshot, required_utilization):
                        return
                    time.sleep(5)
                    continue
                last_busy_pids = pids
                last_snapshot = self._cuda_mem_get_info_gib()
            else:
                memory_snapshot = self._cuda_mem_get_info_gib()
                last_snapshot = memory_snapshot
                if memory_snapshot is None:
                    time.sleep(min(remaining_s, UNSUPPORTED_NVIDIA_SMI_GRACE_S))
                    continue
                if self._has_required_free_memory(memory_snapshot, required_utilization):
                    return
                time.sleep(min(remaining_s, UNSUPPORTED_NVIDIA_SMI_GRACE_S))
                continue
            time.sleep(2)
        if last_busy_pids:
            message = "Timed out waiting for active GPU compute processes to exit before vLLM launch"
            if last_snapshot is not None:
                free_gib, total_gib = last_snapshot
                message += f": active PIDs {', '.join(last_busy_pids)}; last free VRAM {free_gib:.2f}/{total_gib:.2f} GiB"
            else:
                message += f": active PIDs {', '.join(last_busy_pids)}; VRAM probe unavailable"
            raise RuntimeError(message)
        if last_snapshot is not None and not self._has_required_free_memory(last_snapshot, MIN_GPU_MEMORY_UTILIZATION):
            free_gib, total_gib = last_snapshot or (0.0, 0.0)
            minimum_required_gib = (total_gib * MIN_GPU_MEMORY_UTILIZATION) + CUDA_MEMORY_RECOVERY_MARGIN_GIB
            raise RuntimeError(
                "Timed out waiting for minimum free VRAM before vLLM launch: "
                f"{free_gib:.2f}/{total_gib:.2f} GiB free; need at least {minimum_required_gib:.2f} GiB "
                f"for gpu_memory_utilization>={MIN_GPU_MEMORY_UTILIZATION:.2f}."
            )

    @staticmethod
    def _has_required_free_memory(
        snapshot: tuple[float, float] | None, required_utilization: float | None
    ) -> bool:
        if snapshot is None:
            return False
        free_gib, total_gib = snapshot
        if total_gib <= 0:
            return True
        target_utilization = required_utilization if required_utilization is not None else MIN_GPU_MEMORY_UTILIZATION
        required_free_gib = (total_gib * target_utilization) + CUDA_MEMORY_RECOVERY_MARGIN_GIB
        return free_gib >= required_free_gib

    @staticmethod
    def _run(
        cmd: list[str], capture_output: bool = True, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            check=check,
            text=True,
            env=os.environ.copy(),
        )
