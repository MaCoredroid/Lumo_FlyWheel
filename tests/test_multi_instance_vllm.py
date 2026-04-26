from __future__ import annotations

import json
from pathlib import Path

import requests

from lumo_flywheel_serving.multi_instance_vllm import (
    P2_CONCURRENT_DIVERGES,
    MultiInstanceP2Verifier,
    MultiInstanceVllmDriver,
    VllmInstanceSpec,
)
from lumo_flywheel_serving.registry import ModelConfig


def test_instance_specs_allocate_isolated_ports_and_runtime_roots(tmp_path: Path) -> None:
    driver = MultiInstanceVllmDriver(
        registry_path=tmp_path / "model_registry.yaml",
        container_prefix="p2-test",
        base_port=8200,
        base_proxy_port=9200,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )

    specs = driver.instance_specs(count=4, gpu_memory_utilization=0.2)

    assert [spec.port for spec in specs] == [8200, 8201, 8202, 8203]
    assert [spec.proxy_port for spec in specs] == [9200, 9201, 9202, 9203]
    assert [spec.container_name for spec in specs] == ["p2-test-0", "p2-test-1", "p2-test-2", "p2-test-3"]
    assert all(spec.gpu_memory_utilization == 0.2 for spec in specs)
    assert str(tmp_path / "logs" / "p2-test-0") == specs[0].logs_root
    assert str(tmp_path / "triton" / "p2-test-3") == specs[3].triton_cache_root


def test_instance_config_forces_p2_gpu_memory_utilization() -> None:
    config = ModelConfig(
        model_id="qwen3.5-27b",
        hf_repo="Qwen/Qwen3.5-27B-FP8",
        local_path=Path("/models/qwen3.5-27b-fp8"),
        served_model_name="qwen3.5-27b",
        quantization="fp8",
        dtype="auto",
        kv_cache_dtype="fp8_e5m2",
        max_model_len=131072,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )

    p2_config = MultiInstanceVllmDriver._instance_config(config, gpu_memory_utilization=0.2)

    assert p2_config.gpu_memory_utilization == 0.2
    assert config.gpu_memory_utilization == 0.9
    assert p2_config.max_model_len == config.max_model_len


def test_fixture_requests_use_workload_seed_rows(tmp_path: Path) -> None:
    workload = tmp_path / "workload.yaml"
    seed = tmp_path / "seed_trace.jsonl"
    workload.write_text("seed_trace_ref: seed_trace.jsonl\n", encoding="utf-8")
    seed.write_text(
        "\n".join(
            json.dumps({"turn_index": index, "capture_prompt_label": f"label-{index}"})
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    fixtures = MultiInstanceP2Verifier._fixture_requests(
        workload_file=workload,
        model_name="qwen3.5-27b",
        count=4,
    )

    assert [fixture["fixture_id"] for fixture in fixtures] == [
        "P2-FIXTURE-0-label-0",
        "P2-FIXTURE-1-label-1",
        "P2-FIXTURE-2-label-2",
        "P2-FIXTURE-3-label-3",
    ]
    assert all(fixture["payload"]["model"] == "qwen3.5-27b" for fixture in fixtures)
    assert all(fixture["payload"]["temperature"] == 0 for fixture in fixtures)


class _FakeDriver:
    def __init__(self) -> None:
        self.stopped = False

    def instance_specs(self, *, count: int, gpu_memory_utilization: float) -> list[VllmInstanceSpec]:
        return [
            VllmInstanceSpec(
                index=index,
                port=8200 + index,
                proxy_port=9200 + index,
                container_name=f"p2-{index}",
                base_url=f"http://127.0.0.1:{8200 + index}/v1",
                gpu_memory_utilization=gpu_memory_utilization,
                logs_root="/logs",
                triton_cache_root="/tmp/triton_cache",
            )
            for index in range(count)
        ]

    def start(self, **kwargs):
        return self.instance_specs(count=kwargs["count"], gpu_memory_utilization=kwargs["gpu_memory_utilization"])

    def stop(self, **kwargs):
        self.stopped = True
        return []

    def cuda_memory_snapshot(self):
        return {"free_gib": 92.0, "total_gib": 128.0}

    def log_tails(self, **kwargs):
        return [{"instance": 0, "container_name": "p2-0", "tail": "fake vllm tail"}]


class _Response:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_p2_verifier_blocks_when_concurrent_result_diverges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = tmp_path / "workload.yaml"
    seed = tmp_path / "seed_trace.jsonl"
    workload.write_text("seed_trace_ref: seed_trace.jsonl\n", encoding="utf-8")
    seed.write_text(
        "\n".join(
            json.dumps({"turn_index": index, "capture_prompt_label": f"label-{index}"})
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    driver = _FakeDriver()
    verifier = MultiInstanceP2Verifier(driver)  # type: ignore[arg-type]
    monkeypatch.setattr(verifier, "_request_model_name", lambda model_id: model_id)

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        expected = json["input"].rsplit(": ", 1)[1]
        text = "DIFFERENT" if url.startswith("http://127.0.0.1:8202/") else expected
        return _Response({"output_text": text, "usage": {"output_tokens": 1, "reasoning_tokens": 0}})

    monkeypatch.setattr(requests, "post", fake_post)

    payload = verifier.run(model_id="qwen3.5-27b", workload_file=workload, count=4)

    assert payload["pass"] is False
    assert payload["status"] == "BLOCKED_NEEDS_USER_HELP"
    assert payload["halt_reason"] == P2_CONCURRENT_DIVERGES
    assert payload["mismatches"][0]["fixture_id"] == "P2-FIXTURE-2-label-2"
    assert payload["startup_evidence"]["pre_start_cuda_memory"] == {"free_gib": 92.0, "total_gib": 128.0}
    assert payload["log_tails"][0]["tail"] == "fake vllm tail"
    assert driver.stopped is True


def test_p2_verifier_count_one_returns_serial_only_after_repeat_determinism(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = tmp_path / "workload.yaml"
    seed = tmp_path / "seed_trace.jsonl"
    workload.write_text("seed_trace_ref: seed_trace.jsonl\n", encoding="utf-8")
    seed.write_text(json.dumps({"turn_index": 0, "capture_prompt_label": "label-0"}) + "\n", encoding="utf-8")
    driver = _FakeDriver()
    verifier = MultiInstanceP2Verifier(driver)  # type: ignore[arg-type]
    monkeypatch.setattr(verifier, "_request_model_name", lambda model_id: model_id)

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        expected = json["input"].rsplit(": ", 1)[1]
        return _Response({"output_text": expected, "usage": {"output_tokens": 1, "reasoning_tokens": 0}})

    monkeypatch.setattr(requests, "post", fake_post)

    payload = verifier.run(model_id="qwen3.5-27b", workload_file=workload, count=1)

    assert payload["pass"] is True
    assert payload["status"] == "serial_only"
    assert payload["instance_count"] == 1
    assert len(payload["serial_results"]) == 2
    assert payload["concurrent_results"] == []
    assert driver.stopped is True


def test_p2_discovery_returns_highest_viable_fanout_and_attempt_evidence(monkeypatch) -> None:
    verifier = MultiInstanceP2Verifier(_FakeDriver())  # type: ignore[arg-type]
    seen_counts: list[int] = []

    def fake_run(**kwargs):
        count = kwargs["count"]
        seen_counts.append(count)
        if count > 2:
            return {
                "pass": False,
                "status": "BLOCKED_NEEDS_USER_HELP",
                "halt_reason": "startup_failed",
                "error": f"fanout {count} failed",
                "startup_evidence": {"fanout": count, "pre_start_cuda_memory": {"free_gib": 10.0, "total_gib": 20.0}},
                "log_tails": [{"instance": 0, "tail": "oom"}],
            }
        return {
            "pass": True,
            "status": "IMPLEMENTED_VERIFIED",
            "instance_count": count,
            "startup_evidence": {"fanout": count, "pre_start_cuda_memory": {"free_gib": 10.0, "total_gib": 20.0}},
        }

    monkeypatch.setattr(verifier, "run", fake_run)

    payload = verifier.discover_max_viable_fanout(
        model_id="qwen3.5-27b",
        workload_file=Path("workload.yaml"),
        candidate_fanouts=[1, 4, 3, 2, 4],
    )

    assert payload["pass"] is True
    assert payload["status"] == "IMPLEMENTED_VERIFIED"
    assert payload["max_viable_fanout"] == 2
    assert payload["candidate_fanouts"] == [4, 3, 2, 1]
    assert seen_counts == [4, 3, 2]
    assert payload["fanout_attempts"][0]["halt_reason"] == "startup_failed"
    assert payload["fanout_attempts"][0]["startup_evidence"]["pre_start_cuda_memory"]["free_gib"] == 10.0
