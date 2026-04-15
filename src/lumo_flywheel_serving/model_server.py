from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from .registry import ModelConfig, load_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLLM_BASE_IMAGE = "nvcr.io/nvidia/pytorch:26.01-py3"
DEFAULT_VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"
DEFAULT_VLLM_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.nvidia-vllm"
QWEN_CHAT_TEMPLATE_HOST_PATH = REPO_ROOT / "docker" / "chat_templates" / "qwen3-openai-codex.jinja"
QWEN_CHAT_TEMPLATE_CONTAINER_PATH = Path("/opt/lumo/chat_templates/qwen3-openai-codex.jinja")
MIN_GPU_MEMORY_UTILIZATION = 0.15
UNSUPPORTED_NVIDIA_SMI_GRACE_S = 20
LOW_FREE_MEMORY_GRACE_RETRIES = 2
LOW_FREE_MEMORY_GRACE_SLEEP_S = 45
CUDA_MEMORY_RECOVERY_MARGIN_GIB = 1.0
HOST_MEMORY_RECOVERY_COMMAND = "sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true"
GPU_MEMORY_ERROR_RE = re.compile(
    r"Free memory on device cuda:\d+ \((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\)"
)


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
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry = load_registry(registry_path)
        self.port = port
        self.image = image
        self.container_name = container_name
        self.logs_root = Path(logs_root)
        self.triton_cache_root = Path(triton_cache_root)
        self.use_sleep_mode = use_sleep_mode
        self.current_model: str | None = None

    def ensure_runtime_scaffolding(self) -> None:
        for path in (Path("/models"), self.logs_root, self.triton_cache_root):
            path.mkdir(parents=True, exist_ok=True)

    def start(self, model_id: str, enable_request_logging: bool = False) -> None:
        self.ensure_runtime_scaffolding()
        self._ensure_image_present()
        config = self.registry[model_id]
        self._reset_log(model_id)
        self.stop(missing_ok=True)
        self._recover_host_memory()
        # A prior stop can return before GB10 unified-memory pressure fully drains.
        # Wait before the first launch attempt so fresh starts after a recent stop
        # do not immediately spiral through the low-VRAM fallback ladder.
        self._wait_vram_free(timeout_s=120, required_utilization=config.gpu_memory_utilization)
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
                if fp8_cutlass_internal_error in error_text and not enforce_eager:
                    enforce_eager = True
                    self._wait_vram_free(timeout_s=120, required_utilization=retry_utilization)
                    continue
                raise
        self.current_model = model_id

    def switch_model(self, model_id: str, enable_request_logging: bool = False) -> None:
        if self.use_sleep_mode and self.current_model == model_id:
            return
        self.stop(missing_ok=True)
        self._wait_vram_free(timeout_s=120, required_utilization=self.registry[model_id].gpu_memory_utilization)
        self.start(model_id=model_id, enable_request_logging=enable_request_logging)

    def stop(self, missing_ok: bool = False) -> None:
        result = self._run(
            ["docker", "ps", "-a", "--filter", f"name=^{self.container_name}$", "--format", "{{.Names}}"]
        )
        if self.container_name not in result.stdout.splitlines():
            if missing_ok:
                self._recover_host_memory()
                return
            raise RuntimeError(f"Container {self.container_name} is not present")

        self._run(["docker", "stop", "-t", "30", self.container_name], capture_output=False, check=False)
        self._run(["docker", "rm", "-f", self.container_name], capture_output=False, check=False)
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

    def _build_run_command(
        self,
        model_id: str,
        config: ModelConfig,
        enable_request_logging: bool,
        kv_cache_dtype: str,
        gpu_memory_utilization: float,
        enforce_eager: bool,
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
            model_id,
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
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--max-num-seqs",
            str(config.max_num_seqs),
        ]
        if chat_template_path is not None:
            vllm_args.extend(["--chat-template", str(chat_template_path)])
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
    ) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_id": model_id,
            "quantization": config.quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "wire_api": "responses",
            "dev_mode": True,
            "sleep_mode": self.use_sleep_mode,
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
            "    handle.write(f\"[VLLM-INIT] model_id={payload['model_id']} vllm_version={version} git_hash={git_hash}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] quantization={payload['quantization']} kv_cache_dtype={payload['kv_cache_dtype']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] max_model_len={payload['max_model_len']} gpu_memory_utilization={payload['gpu_memory_utilization']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] enforce_eager={str(payload['enforce_eager']).lower()}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] wire_api={payload['wire_api']}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] dev_mode={str(payload['dev_mode']).lower()} sleep_mode={'enabled' if payload['sleep_mode'] else 'disabled'}\\n\")\n"
            "    handle.write(f\"[VLLM-INIT] launch_cmd: {payload['launch_cmd']}\\n\")\n"
            "PY"
        )

    @staticmethod
    def _chat_template_container_path(config: ModelConfig) -> Path | None:
        if config.hf_repo.lower().startswith("qwen/"):
            return QWEN_CHAT_TEMPLATE_CONTAINER_PATH
        return None

    @staticmethod
    def _initial_kv_cache_dtype(config: ModelConfig) -> str:
        # vLLM 0.19.0 rejects fp8_e5m2 KV cache for fp8 checkpoints during
        # engine init, so use a valid first-launch default instead of relying on
        # a guaranteed crash-and-retry cycle.
        if config.quantization == "fp8" and config.kv_cache_dtype == "fp8_e5m2":
            return "auto"
        return config.kv_cache_dtype

    def _wait_ready(self, model_id: str, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        log_path = self.logs_path(model_id)
        start = time.time()
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
                    if model_id not in served_model_ids:
                        time.sleep(5)
                        continue
                    self._append_log_text(
                        model_id,
                        f"[VLLM-READY] cuda_graph_capture_time={time.time() - start:.1f}s\n"
                        f"[VLLM-READY] /health 200 OK at {datetime.now(UTC).isoformat()}\n"
                        f"[VLLM-READY] served_models={','.join(served_model_ids)}\n",
                    )
                    return
            except (requests.RequestException, ValueError, TypeError):
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not ready within {timeout_s}s")

    def _served_model_ids(self) -> list[str]:
        payload = self.models().json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("vLLM /v1/models payload missing data list")
        model_ids = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if not model_ids:
            raise ValueError("vLLM /v1/models returned no served model ids")
        return model_ids

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
        deadline = time.time() + timeout_s
        last_snapshot: tuple[float, float] | None = None
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
                    memory_snapshot = self._cuda_mem_get_info_gib()
                    last_snapshot = memory_snapshot
                    if memory_snapshot is None:
                        return
                    if self._has_required_free_memory(memory_snapshot, required_utilization):
                        return
                    time.sleep(5)
                    continue
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
