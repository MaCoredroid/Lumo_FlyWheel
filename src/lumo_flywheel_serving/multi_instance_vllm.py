from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import requests

from .model_server import DEFAULT_STATE_ROOT, DEFAULT_VLLM_IMAGE, ModelServer
from .registry import ModelConfig
from .yaml_utils import load_yaml_file


P2_INSUFFICIENT_MEMORY = "multi_instance_insufficient_memory"
P2_CONCURRENT_DIVERGES = "multi_instance_concurrent_diverges"


@dataclass(frozen=True)
class VllmInstanceSpec:
    index: int
    port: int
    proxy_port: int
    container_name: str
    base_url: str
    gpu_memory_utilization: float
    logs_root: str
    triton_cache_root: str


class P2GateBlocked(RuntimeError):
    def __init__(self, halt_reason: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.halt_reason = halt_reason
        self.details = details or {}


class MultiInstanceVllmDriver:
    def __init__(
        self,
        *,
        registry_path: str | Path,
        image: str = DEFAULT_VLLM_IMAGE,
        container_prefix: str = "lumo-vllm-p2",
        base_port: int = 8100,
        base_proxy_port: int = 9100,
        logs_root: str | Path = "/logs",
        triton_cache_root: str | Path = "/tmp/triton_cache",
        state_root: str | Path = DEFAULT_STATE_ROOT,
    ) -> None:
        self.registry_path = Path(registry_path).resolve()
        self.image = image
        self.container_prefix = container_prefix
        self.base_port = int(base_port)
        self.base_proxy_port = int(base_proxy_port)
        self.logs_root = Path(logs_root)
        self.triton_cache_root = Path(triton_cache_root)
        self.state_root = Path(state_root)

    def instance_specs(self, *, count: int, gpu_memory_utilization: float) -> list[VllmInstanceSpec]:
        return [
            VllmInstanceSpec(
                index=index,
                port=self.base_port + index,
                proxy_port=self.base_proxy_port + index,
                container_name=f"{self.container_prefix}-{index}",
                base_url=f"http://127.0.0.1:{self.base_port + index}/v1",
                gpu_memory_utilization=float(gpu_memory_utilization),
                logs_root=str((self.logs_root / f"{self.container_prefix}-{index}").resolve()),
                triton_cache_root=str((self.triton_cache_root / f"{self.container_prefix}-{index}").resolve()),
            )
            for index in range(count)
        ]

    def start(
        self,
        *,
        model_id: str,
        count: int = 4,
        gpu_memory_utilization: float = 0.2,
        enable_request_logging: bool = False,
        bind_retries: int = 3,
    ) -> list[VllmInstanceSpec]:
        if count < 1:
            raise ValueError("count must be >= 1")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be > 0.0 and < 1.0")
        if bind_retries < 1:
            raise ValueError("bind_retries must be >= 1")

        specs = self.instance_specs(count=count, gpu_memory_utilization=gpu_memory_utilization)
        self.stop(count=count, missing_ok=True)
        first_server = self._server_for(specs[0])
        first_server._recover_host_memory()
        first_server._wait_vram_free(timeout_s=120, required_utilization=gpu_memory_utilization)

        started: list[VllmInstanceSpec] = []
        errors: list[dict[str, Any]] = []
        for spec in specs:
            server = self._server_for(spec)
            try:
                self._start_one(
                    server=server,
                    spec=spec,
                    model_id=model_id,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enable_request_logging=enable_request_logging,
                    bind_retries=bind_retries,
                )
            except Exception as exc:
                errors.append({"instance": spec.index, "container_name": spec.container_name, "error": str(exc)})
                self.stop(count=count, missing_ok=True)
                raise P2GateBlocked(
                    P2_INSUFFICIENT_MEMORY,
                    f"Only {len(started)} of {count} vLLM instances started at gpu_memory_utilization={gpu_memory_utilization}.",
                    details={"started": [instance.__dict__ for instance in started], "errors": errors},
                ) from exc
            started.append(spec)
        return started

    def stop(self, *, count: int = 4, missing_ok: bool = True) -> list[dict[str, Any]]:
        stopped: list[dict[str, Any]] = []
        for spec in self.instance_specs(count=count, gpu_memory_utilization=0.2):
            server = self._server_for(spec)
            try:
                server.stop(missing_ok=missing_ok)
                stopped.append({"container_name": spec.container_name, "status": "stopped_or_absent"})
            except Exception as exc:
                if not missing_ok:
                    raise
                stopped.append({"container_name": spec.container_name, "status": "error", "error": str(exc)})
        return stopped

    def _server_for(self, spec: VllmInstanceSpec) -> ModelServer:
        return ModelServer(
            registry_path=self.registry_path,
            port=spec.port,
            image=self.image,
            container_name=spec.container_name,
            logs_root=spec.logs_root,
            triton_cache_root=spec.triton_cache_root,
            proxy_port=spec.proxy_port,
            state_root=(self.state_root / spec.container_name).resolve(),
        )

    def _start_one(
        self,
        *,
        server: ModelServer,
        spec: VllmInstanceSpec,
        model_id: str,
        gpu_memory_utilization: float,
        enable_request_logging: bool,
        bind_retries: int,
    ) -> None:
        server.ensure_runtime_scaffolding()
        server._ensure_image_present()
        config = self._instance_config(server.registry[model_id], gpu_memory_utilization=gpu_memory_utilization)
        kv_cache_dtype = server._initial_kv_cache_dtype(config)
        last_error: Exception | None = None
        for attempt in range(1, bind_retries + 1):
            server._reset_log(model_id)
            server.stop(missing_ok=True)
            try:
                server._run(
                    server._build_run_command(
                        model_id,
                        config,
                        enable_request_logging,
                        kv_cache_dtype=kv_cache_dtype,
                        gpu_memory_utilization=gpu_memory_utilization,
                        enforce_eager=False,
                        weight_version_id=None,
                    ),
                    capture_output=False,
                )
                server._wait_ready(model_id=model_id)
                server.record_launch_metadata(
                    model_id,
                    p2_multi_instance="started",
                    instance_index=spec.index,
                    port=spec.port,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
                return
            except Exception as exc:
                last_error = exc
                server.stop(missing_ok=True)
                if attempt < bind_retries and self._looks_like_bind_failure(str(exc)):
                    time.sleep(2)
                    continue
                break
        assert last_error is not None
        raise last_error

    @staticmethod
    def _instance_config(config: ModelConfig, *, gpu_memory_utilization: float) -> ModelConfig:
        return replace(config, gpu_memory_utilization=float(gpu_memory_utilization))

    @staticmethod
    def _looks_like_bind_failure(error_text: str) -> bool:
        lowered = error_text.lower()
        return "address already in use" in lowered or "failed to bind" in lowered or "errno 98" in lowered


class MultiInstanceP2Verifier:
    def __init__(self, driver: MultiInstanceVllmDriver) -> None:
        self.driver = driver

    def run(
        self,
        *,
        model_id: str,
        workload_file: str | Path,
        count: int = 4,
        gpu_memory_utilization: float = 0.2,
        bind_retries: int = 3,
        request_timeout_s: int = 240,
        keep_running: bool = False,
        enable_request_logging: bool = False,
    ) -> dict[str, Any]:
        instances: list[VllmInstanceSpec] = []
        try:
            instances = self.driver.start(
                model_id=model_id,
                count=count,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_request_logging=enable_request_logging,
                bind_retries=bind_retries,
            )
            if len(instances) != count:
                raise P2GateBlocked(
                    P2_INSUFFICIENT_MEMORY,
                    f"Only {len(instances)} of {count} vLLM instances started.",
                    details={"instances": [instance.__dict__ for instance in instances]},
                )

            fixture_requests = self._fixture_requests(
                workload_file=Path(workload_file),
                model_name=self._request_model_name(model_id),
                count=count,
            )
            serial_results = [
                self._post_fixture(instances[0], fixture_request, timeout_s=request_timeout_s)
                for fixture_request in fixture_requests
            ]
            with ThreadPoolExecutor(max_workers=count) as executor:
                futures = [
                    executor.submit(self._post_fixture, instances[index], fixture_requests[index], timeout_s=request_timeout_s)
                    for index in range(count)
                ]
                concurrent_results = [future.result() for future in futures]

            mismatches = [
                {
                    "fixture_id": serial["fixture_id"],
                    "serial": serial["normalized_result"],
                    "concurrent": concurrent["normalized_result"],
                    "concurrent_instance": concurrent["instance_index"],
                }
                for serial, concurrent in zip(serial_results, concurrent_results, strict=True)
                if serial["normalized_result"] != concurrent["normalized_result"]
            ]
            if mismatches:
                raise P2GateBlocked(
                    P2_CONCURRENT_DIVERGES,
                    "Concurrent multi-instance fixture results differed from sequential single-instance results.",
                    details={"mismatches": mismatches},
                )
            return {
                "pass": True,
                "status": "IMPLEMENTED_VERIFIED",
                "model_id": model_id,
                "workload_file": str(workload_file),
                "instance_count": len(instances),
                "gpu_memory_utilization": gpu_memory_utilization,
                "instances": [instance.__dict__ for instance in instances],
                "serial_results": serial_results,
                "concurrent_results": concurrent_results,
            }
        except P2GateBlocked as exc:
            return {
                "pass": False,
                "status": "BLOCKED_NEEDS_USER_HELP",
                "halt_reason": exc.halt_reason,
                "error": str(exc),
                **exc.details,
            }
        finally:
            if not keep_running:
                self.driver.stop(count=count, missing_ok=True)

    def _request_model_name(self, model_id: str) -> str:
        config = self.driver._server_for(
            self.driver.instance_specs(count=1, gpu_memory_utilization=0.2)[0]
        ).registry[model_id]
        return ModelServer._request_model_name(config)

    @staticmethod
    def _fixture_requests(*, workload_file: Path, model_name: str, count: int) -> list[dict[str, Any]]:
        descriptor = load_yaml_file(workload_file)
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"Workload descriptor must be a mapping: {workload_file}")
        seed_ref = Path(str(descriptor.get("seed_trace_ref", "")))
        seed_path = seed_ref if seed_ref.is_absolute() else workload_file.parent / seed_ref
        rows: list[dict[str, Any]] = []
        for raw_line in seed_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                row = json.loads(raw_line)
                if isinstance(row, dict):
                    rows.append(row)
        if len(rows) < count:
            raise RuntimeError(f"P2 fixture needs {count} seed rows; found {len(rows)} in {seed_path}")

        requests_payloads: list[dict[str, Any]] = []
        for index, row in enumerate(rows[:count]):
            label = str(row.get("capture_prompt_label") or f"turn-{row.get('turn_index', index)}")
            expected = f"P2-FIXTURE-{index}-{label}"
            requests_payloads.append(
                {
                    "fixture_id": expected,
                    "payload": {
                        "model": model_name,
                        "input": (
                            "This is a deterministic P2 multi-instance serving fixture. "
                            f"Reply with exactly this token and no other text: {expected}"
                        ),
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 20260425 + index,
                        "max_output_tokens": 32,
                    },
                }
            )
        return requests_payloads

    @staticmethod
    def _post_fixture(
        instance: VllmInstanceSpec,
        fixture_request: dict[str, Any],
        *,
        timeout_s: int,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{instance.base_url}/responses",
            headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY') or 'EMPTY'}"},
            json=fixture_request["payload"],
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Responses fixture returned a non-object JSON payload")
        return {
            "fixture_id": fixture_request["fixture_id"],
            "instance_index": instance.index,
            "endpoint": instance.base_url,
            "normalized_result": MultiInstanceP2Verifier._normalize_responses_result(payload),
        }

    @staticmethod
    def _normalize_responses_result(payload: dict[str, Any]) -> dict[str, Any]:
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        output_tokens = _int_value(usage.get("output_tokens"))
        reasoning_tokens = _int_value(usage.get("reasoning_tokens"))
        details = usage.get("output_tokens_details", {})
        if reasoning_tokens == 0 and isinstance(details, dict):
            reasoning_tokens = _int_value(details.get("reasoning_tokens"))
        return {
            "text": _responses_text(payload).strip(),
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _responses_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    for key in ("text", "output_text", "reasoning_text"):
                        value = part.get(key)
                        if isinstance(value, str):
                            chunks.append(value)
    return "".join(chunks)
