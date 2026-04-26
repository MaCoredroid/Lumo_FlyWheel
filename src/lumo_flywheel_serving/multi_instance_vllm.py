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
P2_FIXTURE_REQUEST_FAILED = "multi_instance_fixture_request_failed"


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

    def cuda_memory_snapshot(self) -> dict[str, float] | None:
        specs = self.instance_specs(count=1, gpu_memory_utilization=0.2)
        snapshot = self._server_for(specs[0])._cuda_mem_get_info_gib()
        if snapshot is None:
            return None
        free_gib, total_gib = snapshot
        return {"free_gib": free_gib, "total_gib": total_gib}

    def log_tails(
        self,
        *,
        model_id: str,
        count: int,
        gpu_memory_utilization: float,
        lines: int = 80,
    ) -> list[dict[str, Any]]:
        tails: list[dict[str, Any]] = []
        for spec in self.instance_specs(count=count, gpu_memory_utilization=gpu_memory_utilization):
            log_path = self._server_for(spec).logs_path(model_id)
            entry: dict[str, Any] = {
                "instance": spec.index,
                "container_name": spec.container_name,
                "log_path": str(log_path),
            }
            try:
                if log_path.exists():
                    entry["tail"] = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-lines:])
                else:
                    entry["tail"] = ""
                    entry["status"] = "missing"
            except Exception as exc:
                entry["tail"] = ""
                entry["status"] = "error"
                entry["error"] = str(exc)
            tails.append(entry)
        return tails

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
        startup_evidence = self._startup_evidence(count=count, gpu_memory_utilization=gpu_memory_utilization)
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
            if count == 1:
                first_result = self._post_fixture(instances[0], fixture_requests[0], timeout_s=request_timeout_s)
                second_result = self._post_fixture(instances[0], fixture_requests[0], timeout_s=request_timeout_s)
                if first_result["normalized_result"] != second_result["normalized_result"]:
                    raise P2GateBlocked(
                        P2_CONCURRENT_DIVERGES,
                        "Single-instance P2 fixture result was not deterministic across repeated requests.",
                        details={
                            "mismatches": [
                                {
                                    "fixture_id": first_result["fixture_id"],
                                    "serial": first_result["normalized_result"],
                                    "repeat": second_result["normalized_result"],
                                    "concurrent_instance": second_result["instance_index"],
                                }
                            ]
                        },
                    )
                return {
                    "pass": True,
                    "status": "serial_only",
                    "model_id": model_id,
                    "workload_file": str(workload_file),
                    "instance_count": len(instances),
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "instances": [instance.__dict__ for instance in instances],
                    "startup_evidence": {**startup_evidence, "started_instances": [instance.__dict__ for instance in instances]},
                    "serial_results": [first_result, second_result],
                    "concurrent_results": [],
                }

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
                "startup_evidence": {**startup_evidence, "started_instances": [instance.__dict__ for instance in instances]},
                "serial_results": serial_results,
                "concurrent_results": concurrent_results,
            }
        except P2GateBlocked as exc:
            return {
                "pass": False,
                "status": "BLOCKED_NEEDS_USER_HELP",
                "halt_reason": exc.halt_reason,
                "error": str(exc),
                "startup_evidence": startup_evidence,
                "log_tails": self._log_tails(
                    model_id=model_id,
                    count=count,
                    gpu_memory_utilization=gpu_memory_utilization,
                ),
                **exc.details,
            }
        except Exception as exc:
            return {
                "pass": False,
                "status": "BLOCKED_NEEDS_USER_HELP",
                "halt_reason": P2_FIXTURE_REQUEST_FAILED,
                "error": str(exc),
                "startup_evidence": startup_evidence,
                "log_tails": self._log_tails(
                    model_id=model_id,
                    count=count,
                    gpu_memory_utilization=gpu_memory_utilization,
                ),
            }
        finally:
            if not keep_running:
                self.driver.stop(count=count, missing_ok=True)

    def discover_max_viable_fanout(
        self,
        *,
        model_id: str,
        workload_file: str | Path,
        candidate_fanouts: list[int],
        gpu_memory_utilization: float = 0.2,
        bind_retries: int = 3,
        request_timeout_s: int = 240,
        keep_running: bool = False,
        enable_request_logging: bool = False,
    ) -> dict[str, Any]:
        candidates = self._normalize_candidate_fanouts(candidate_fanouts)
        attempts: list[dict[str, Any]] = []
        for fanout in candidates:
            payload = self.run(
                model_id=model_id,
                workload_file=workload_file,
                count=fanout,
                gpu_memory_utilization=gpu_memory_utilization,
                bind_retries=bind_retries,
                request_timeout_s=request_timeout_s,
                keep_running=keep_running,
                enable_request_logging=enable_request_logging,
            )
            details = self._attempt_details(payload)
            attempt = {
                "fanout": fanout,
                "pass": bool(payload.get("pass")),
                "status": payload.get("status"),
                "halt_reason": payload.get("halt_reason"),
                "error": payload.get("error"),
                "startup_evidence": payload.get("startup_evidence"),
                "log_tails": payload.get("log_tails", []),
                "details": details,
            }
            attempts.append(attempt)
            if payload.get("pass"):
                max_viable_fanout = fanout
                return {
                    **payload,
                    "pass": True,
                    "status": "serial_only" if max_viable_fanout == 1 else "IMPLEMENTED_VERIFIED",
                    "max_viable_fanout": max_viable_fanout,
                    "candidate_fanouts": candidates,
                    "fanout_attempts": attempts,
                }

        return {
            "pass": False,
            "status": "BLOCKED_NEEDS_USER_HELP",
            "halt_reason": attempts[-1].get("halt_reason") if attempts else P2_INSUFFICIENT_MEMORY,
            "error": attempts[-1].get("error") if attempts else "No P2 fanout candidates were supplied.",
            "max_viable_fanout": 0,
            "candidate_fanouts": candidates,
            "fanout_attempts": attempts,
        }

    def _request_model_name(self, model_id: str) -> str:
        config = self.driver._server_for(
            self.driver.instance_specs(count=1, gpu_memory_utilization=0.2)[0]
        ).registry[model_id]
        return ModelServer._request_model_name(config)

    def _startup_evidence(self, *, count: int, gpu_memory_utilization: float) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "fanout": count,
            "gpu_memory_utilization": gpu_memory_utilization,
            "pre_start_cuda_memory": self._cuda_memory_snapshot(),
        }
        try:
            evidence["instance_specs"] = [
                spec.__dict__ for spec in self.driver.instance_specs(count=count, gpu_memory_utilization=gpu_memory_utilization)
            ]
        except Exception as exc:
            evidence["instance_specs_error"] = str(exc)
        return evidence

    def _cuda_memory_snapshot(self) -> dict[str, float] | None:
        snapshot = getattr(self.driver, "cuda_memory_snapshot", None)
        if not callable(snapshot):
            return None
        try:
            return snapshot()
        except Exception:
            return None

    def _log_tails(self, *, model_id: str, count: int, gpu_memory_utilization: float) -> list[dict[str, Any]]:
        log_tails = getattr(self.driver, "log_tails", None)
        if not callable(log_tails):
            return []
        try:
            return log_tails(model_id=model_id, count=count, gpu_memory_utilization=gpu_memory_utilization)
        except Exception as exc:
            return [{"status": "error", "error": str(exc)}]

    @staticmethod
    def _normalize_candidate_fanouts(candidate_fanouts: list[int]) -> list[int]:
        normalized: list[int] = []
        for fanout in candidate_fanouts:
            if fanout < 1:
                raise ValueError("candidate fanouts must be >= 1")
            if fanout not in normalized:
                normalized.append(fanout)
        return sorted(normalized, reverse=True)

    @staticmethod
    def _attempt_details(payload: dict[str, Any]) -> dict[str, Any]:
        omitted = {
            "pass",
            "status",
            "model_id",
            "workload_file",
            "instance_count",
            "gpu_memory_utilization",
            "instances",
            "startup_evidence",
            "serial_results",
            "concurrent_results",
            "candidate_fanouts",
            "fanout_attempts",
            "halt_reason",
            "error",
            "log_tails",
        }
        return {key: value for key, value in payload.items() if key not in omitted}

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
