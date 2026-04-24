from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from lumo_flywheel_serving import cli


class _Response:
    def __init__(self, *, payload: dict | None = None, text: str = "", status_code: int = 200) -> None:
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self) -> dict:
        return self._payload


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        registry="model_registry.yaml",
        port=8000,
        image="lumo-flywheel-vllm:test",
        container_name="lumo-vllm",
        logs_root="/logs",
        triton_cache_root="/tmp/triton_cache",
        use_sleep_mode=False,
        model_id="qwen3.5-27b",
        enable_request_logging=False,
        keep_running=False,
        output_dir="/tmp/lumo-smoke",
        pool_or_split="public_dev",
        smoke_task_id="smoke-test/qwen3.5-27b/unit",
        smoke_seed=0,
        smoke_attempt=1,
    )


def test_annotate_log_records_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    server = SimpleNamespace(
        record_launch_metadata=lambda model_id, **metadata: seen.update({"model_id": model_id, "metadata": metadata})
    )
    monkeypatch.setattr(cli, "_server", lambda args: server)

    args = _args()
    args.entries = ["gate1_responses_status=pass", "metric_schema_variant=openmetrics_total"]

    assert cli.cmd_annotate_log(args) == 0
    assert seen == {
        "model_id": "qwen3.5-27b",
        "metadata": {
            "gate1_responses_status": "pass",
            "metric_schema_variant": "openmetrics_total",
        },
    }


def test_annotate_log_rejects_invalid_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_server", lambda args: SimpleNamespace(record_launch_metadata=lambda *a, **k: None))

    args = _args()
    args.entries = ["broken-entry"]

    with pytest.raises(RuntimeError, match="Expected key=value"):
        cli.cmd_annotate_log(args)


def test_smoke_test_requires_prefix_cache_hits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    events: list[str] = []
    telemetry_events: list[str] = []

    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: events.append(f"start:{model_id}"),
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: events.append(f"meta:{model_id}:{metadata}"),
        flush_prefix_cache=lambda: events.append("flush"),
        stop=lambda missing_ok=True: events.append(f"stop:{missing_ok}"),
    )

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            telemetry_events.append(f"init:{host}:{port}:{output_dir}:{model_id}:{pool_or_split}")
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            telemetry_events.append("resolve")

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            telemetry_events.append(f"before:{task_id}:{seed}:{attempt}")

        async def snapshot_after(self, task_id: str):
            telemetry_events.append(f"after:{task_id}")
            return SimpleNamespace(
                anomalies=[],
                cache_hits=3.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
                ttft_ms=500.0,
                prefill_throughput_tps=12.0,
                decode_throughput_tps=18.0,
                cache_hit_rate_pct=75.0,
                gen_tokens=18.0,
                prefill_sum_s=2.0,
                decode_sum_s=1.0,
                ttft_count=5,
                wall_clock_s=4.5,
            )

    requests_seen: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        requests_seen.append({"url": url, "json": json})
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        if len(requests_seen) == 1:
            return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})
        return _Response(payload={"id": "chat-2", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(cli, "load_telemetry", lambda telemetry_dir: [{"telemetry_dir": telemetry_dir}])
    monkeypatch.setattr(
        cli,
        "aggregate_by_model",
        lambda records, reportable_runs: [
            SimpleNamespace(model_id="qwen3.5-27b", pool_or_split="public_dev", n_tasks=1)
        ],
    )
    monkeypatch.setattr(requests, "post", fake_post)

    assert cli.cmd_smoke_test(_args()) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["direct_api_smoke_status"] == "pass"
    assert output["prefix_cache_hits_delta"] == 3.0
    assert output["telemetry_task_id"] == "smoke-test/qwen3.5-27b/unit"
    assert output["telemetry_summary"] == {
        "model_id": "qwen3.5-27b",
        "pool_or_split": "public_dev",
        "n_tasks": 1,
    }
    assert output["telemetry_record"]["ttft_count"] == 5
    assert output["telemetry_record"]["prefill_sum_s"] == 2.0
    assert output["telemetry_record"]["decode_sum_s"] == 1.0
    assert output["telemetry_record"]["wall_clock_s"] == 4.5
    assert events == [
        "start:qwen3.5-27b",
        "meta:qwen3.5-27b:{'direct_api_smoke_status': 'pass', 'metric_schema_variant': 'legacy_no_total', 'prefix_cache_hits_delta': 3.0, 'codex_tool_probe_status': 'pass'}",
        "flush",
        "stop:True",
    ]
    assert telemetry_events == [
        "init:127.0.0.1:8000:/tmp/lumo-smoke:qwen3.5-27b:public_dev",
        "resolve",
        "before:smoke-test/qwen3.5-27b/unit:0:1",
        "after:smoke-test/qwen3.5-27b/unit",
    ]
    assert len(requests_seen) == 5
    assert requests_seen[0]["url"].endswith("/v1/chat/completions")
    assert requests_seen[1]["url"].endswith("/v1/chat/completions")
    assert requests_seen[2]["url"].endswith("/v1/responses")
    assert requests_seen[3]["url"].endswith("/v1/responses")
    assert requests_seen[4]["url"] == "http://127.0.0.1:8001/v1/responses"
    second_messages = requests_seen[1]["json"]["messages"]
    assert second_messages[2] == {"role": "assistant", "content": "OK"}
    assert second_messages[3]["role"] == "user"
    assert requests_seen[3]["json"]["previous_response_id"] == "resp-1"
    assert output["responses_ids"] == ["resp-1", "resp-2"]
    assert output["tool_call_probe_status"] == "pass"


def test_chat_message_text_falls_back_when_chat_content_is_null() -> None:
    assert cli._chat_message_text({"choices": [{"message": {"content": None}}]}) == "OK"


def test_smoke_test_fails_when_prefix_cache_hits_do_not_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: events.append(f"start:{model_id}"),
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: events.append(f"meta:{model_id}:{metadata}"),
        flush_prefix_cache=lambda: events.append("flush"),
        stop=lambda missing_ok=True: events.append(f"stop:{missing_ok}"),
    )

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=0.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
            )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Expected prefix cache hits"):
        cli.cmd_smoke_test(_args())

    assert events == [
        "start:qwen3.5-27b",
        "meta:qwen3.5-27b:{'direct_api_smoke_status': 'escalated', 'direct_api_smoke_error': 'Expected prefix cache hits after repeated-prefix chat turns, but /metrics did not increase.'}",
        "stop:True",
    ]


def test_smoke_test_uses_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )
    seen_headers: list[dict[str, str]] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        seen_headers.append(headers)
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
                ttft_ms=500.0,
                prefill_throughput_tps=12.0,
                decode_throughput_tps=18.0,
                cache_hit_rate_pct=25.0,
                gen_tokens=18.0,
                prefill_sum_s=2.0,
                decode_sum_s=1.0,
                ttft_count=5,
                wall_clock_s=4.5,
            )

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(cli, "load_telemetry", lambda telemetry_dir: [{"telemetry_dir": telemetry_dir}])
    monkeypatch.setattr(
        cli,
        "aggregate_by_model",
        lambda records, reportable_runs: [
            SimpleNamespace(model_id="qwen3.5-27b", pool_or_split="public_dev", n_tasks=1)
        ],
    )
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("VLLM_API_KEY", "custom-token")

    assert cli.cmd_smoke_test(_args()) == 0
    assert seen_headers == [{"Authorization": "Bearer custom-token"}] * 5


def test_smoke_test_targets_served_model_override_and_lora_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        registry={
            "qwen3.5-27b": SimpleNamespace(
                served_model_name="qwen3.5-27b-served",
                lora_modules=(("codex-sft-all", "/models/adapters/codex-sft-all"),),
            )
        },
        _request_model_name=lambda config: "codex-sft-all",
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b-served"}, {"id": "codex-sft-all"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )
    seen_models: list[str] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        seen_models.append(json["model"])
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
                ttft_ms=500.0,
                prefill_throughput_tps=12.0,
                decode_throughput_tps=18.0,
                cache_hit_rate_pct=25.0,
                gen_tokens=18.0,
                prefill_sum_s=2.0,
                decode_sum_s=1.0,
                ttft_count=5,
                wall_clock_s=4.5,
            )

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(cli, "load_telemetry", lambda telemetry_dir: [{"telemetry_dir": telemetry_dir}])
    monkeypatch.setattr(
        cli,
        "aggregate_by_model",
        lambda records, reportable_runs: [
            SimpleNamespace(model_id="qwen3.5-27b", pool_or_split="public_dev", n_tasks=1)
        ],
    )
    monkeypatch.setattr(requests, "post", fake_post)

    assert cli.cmd_smoke_test(_args()) == 0
    assert seen_models == ["codex-sft-all", "codex-sft-all", "codex-sft-all", "codex-sft-all", "codex-sft-all"]


def test_smoke_test_rejects_ttft_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
                ttft_ms=500.0,
                prefill_throughput_tps=12.0,
                decode_throughput_tps=18.0,
                cache_hit_rate_pct=25.0,
                gen_tokens=18.0,
                prefill_sum_s=2.0,
                decode_sum_s=1.0,
                ttft_count=4,
                wall_clock_s=4.5,
            )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="expected ttft_count=5"):
        cli.cmd_smoke_test(_args())


def test_smoke_test_rejects_gpu_time_exceeding_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
                ttft_ms=500.0,
                prefill_throughput_tps=12.0,
                decode_throughput_tps=18.0,
                cache_hit_rate_pct=25.0,
                gen_tokens=18.0,
                prefill_sum_s=3.0,
                decode_sum_s=2.0,
                ttft_count=5,
                wall_clock_s=4.0,
            )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="prefill_sum_s \\+ decode_sum_s < wall_clock_s"):
        cli.cmd_smoke_test(_args())


def test_smoke_test_requires_responses_follow_up_id(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            return _Response(payload={})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
            )

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="did not return a response id"):
        cli.cmd_smoke_test(_args())


def test_smoke_test_requires_structured_tool_call_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: None,
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: None,
        flush_prefix_cache=lambda: None,
        stop=lambda missing_ok=True: None,
    )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "message", "role": "assistant"}]})
            response_id = "resp-1" if "previous_response_id" not in json else "resp-2"
            return _Response(payload={"id": response_id})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
            )

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="did not return a structured function_call item"):
        cli.cmd_smoke_test(_args())


def test_smoke_test_escalates_when_responses_follow_up_id_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: events.append(f"start:{model_id}"),
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        record_launch_metadata=lambda model_id, **metadata: events.append(f"meta:{model_id}:{metadata}"),
        flush_prefix_cache=lambda: events.append("flush"),
        stop=lambda missing_ok=True: events.append(f"stop:{missing_ok}"),
    )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            if "tools" in json:
                return _Response(payload={"output": [{"type": "function_call", "name": "codex_tool_probe"}]})
            if "previous_response_id" in json:
                return _Response(
                    payload={
                        "error": {
                            "message": "Response with id 'resp-1' not found.",
                            "type": "invalid_request_error",
                            "param": "response_id",
                            "code": 404,
                        }
                    },
                    text="""{"error":{"message":"Response with id 'resp-1' not found."}}""",
                    status_code=404,
                )
            return _Response(payload={"id": "resp-1"})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    class _FakeLatencyCapture:
        def __init__(self, host: str, port: int, output_dir: str, model_id: str, pool_or_split: str) -> None:
            self.writer_path = f"{output_dir}/telemetry/latency_{model_id}_{pool_or_split}.jsonl"

        @property
        def resolved_schema(self) -> dict[str, str]:
            return {"cache_hits": "cache_hits"}

        async def resolve_schema(self) -> None:
            return None

        async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
            return None

        async def snapshot_after(self, task_id: str):
            return SimpleNamespace(
                anomalies=[],
                cache_hits=1.0,
                kv_computed_tokens=24.0,
                prompt_tokens=60.0,
                cache_queries=4.0,
            )

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "LatencyCapture", _FakeLatencyCapture)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="did not persist response ids for follow-up chaining"):
        cli.cmd_smoke_test(_args())

    assert events == [
        "start:qwen3.5-27b",
        "meta:qwen3.5-27b:{'direct_api_smoke_status': 'escalated', 'direct_api_smoke_error': \"Responses API follow-up turn failed: backend did not persist response ids for follow-up chaining. vLLM returned 404: Response with id 'resp-1' not found.\"}",
        "stop:True",
    ]


def test_download_model_rejects_unpinned_hf_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3-coder-next-80b-a3b:
    hf_repo: Qwen/Qwen3-Coder-Next-80B-A3B
    local_path: /models/qwen3-coder-next-80b-a3b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.93
    sprint0_gate: "Confirm upstream FP8 checkpoint identity before Dev-Bench"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    args = _args()
    args.registry = str(registry)
    args.model_id = "qwen3-coder-next-80b-a3b"
    args.env_file = None

    with pytest.raises(RuntimeError, match="missing hf_revision"):
        cli.cmd_download_model(args)


def test_metric_schema_variant_detects_openmetrics_total() -> None:
    assert cli._metric_schema_variant(
        {
            "prompt_tokens": "vllm:prompt_tokens_total",
            "generation_tokens": "vllm:generation_tokens_total",
            "cache_queries": "vllm:prefix_cache_queries_total",
            "cache_hits": "vllm:prefix_cache_hits_total",
            "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
            "ttft": "vllm:time_to_first_token_seconds",
            "prefill_time": "vllm:request_prefill_time_seconds",
            "decode_time": "vllm:request_decode_time_seconds",
            "itl": "vllm:inter_token_latency_seconds",
        }
    ) == "openmetrics_total"


def _write_auto_research_registry(path: Path) -> None:
    path.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    hf_revision: 2e1b21350ce589fcaafbb3c7d7eac526a7aed582
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.90
    max_num_batched_tokens: 8192
    max_num_seqs: 4
""",
        encoding="utf-8",
    )


def _write_auto_research_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_tokens": 4096, "output_tokens": 1200, "thinking_tokens": 0, "turn_index": 0}),
                json.dumps({"prompt_tokens": 2048, "output_tokens": 600, "thinking_tokens": 0, "turn_index": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_auto_research_thinking_probe(repo: Path) -> None:
    capture_date = datetime.now(UTC)
    report = repo / "reports" / f"thinking-probe-{capture_date.strftime('%Y%m%d')}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# Serving Thinking Probe\n\n- capture_date: {capture_date.isoformat().replace('+00:00', 'Z')}\n- outcome: row-3\n",
        encoding="utf-8",
    )


def _init_auto_research_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_auto_research_registry(repo / "model_registry.yaml")
    workload_dir = repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment"
    _write_auto_research_trace(workload_dir / "seed_trace.jsonl")
    _write_auto_research_trace(workload_dir / "holdout_trace.jsonl")
    (workload_dir / "serving_workload.yaml").write_text(
        """
family_id: proposal-ranking-manager-judgment
workload_distribution_id: prmj-v1-live
latency_ceiling_ms: 35000
tpot_ceiling_ms: 80
turn_latency_ceiling_ms: 35000
p99_context_tokens: 24576
avg_prompt_tokens: 4096
avg_output_tokens: 1200
rollout_baseline: 10.0
measurement_window_minutes: 25
gpu_memory_utilization_cap: 0.08
seed_trace_ref: seed_trace.jsonl
holdout_trace_ref: holdout_trace.jsonl
""",
        encoding="utf-8",
    )
    _write_auto_research_thinking_probe(repo)
    fixture_src = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst = repo / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    fixture_dst.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def test_auto_research_help_only_registered(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["auto-research", "status", "--help-only", "--round-id", "round-1"])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out) == {"subcommand": "status", "status": "registered"}


def test_auto_research_help_only_registered_without_required_args(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["auto-research", "measure", "--help-only"])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out) == {"subcommand": "measure", "status": "registered"}


def test_auto_research_measure_requires_arguments_after_help_only_check() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["auto-research", "measure"])
    with pytest.raises(RuntimeError, match="--round-id, --candidate"):
        args.func(args)


def test_auto_research_status_reports_missing_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    repo = _init_auto_research_repo(tmp_path)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--registry",
            str(repo / "model_registry.yaml"),
            "--tuned-config-root",
            str(repo / "output" / "tuned_configs"),
            "auto-research",
            "status",
            "--round-id",
            "does-not-exist",
        ]
    )

    assert args.func(args) == 1
    assert json.loads(capsys.readouterr().out) == {"round_id": "does-not-exist", "phase": "missing"}


def test_auto_research_run_uses_non_agent_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _init_auto_research_repo(tmp_path)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")

    def fake_run_non_agent(self, **kwargs):
        return {
            "round_id": "round-123",
            "round_dir": str(repo / "output" / "auto_research" / "round-123"),
            "bundle_path": str(repo / "output" / "tuned_configs" / "bundle.yaml"),
            "rows_created": [],
            "finalize_commit_sha": "synthetic-sha",
            "kwargs": kwargs,
        }

    monkeypatch.setattr(cli.AutoResearchRoundManager, "run_non_agent", fake_run_non_agent)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--registry",
            str(repo / "model_registry.yaml"),
            "--tuned-config-root",
            str(repo / "output" / "tuned_configs"),
            "auto-research",
            "run",
            "qwen3.5-27b",
            "--family-id",
            "proposal-ranking-manager-judgment",
            "--workload-file",
            str(repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"),
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["round_id"] == "round-123"
    assert payload["kwargs"]["model_id"] == "qwen3.5-27b"
    assert payload["kwargs"]["harness_type"] == "real"


def test_auto_research_run_accepts_synthetic_harness_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _init_auto_research_repo(tmp_path)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")

    def fake_run_non_agent(self, **kwargs):
        return {"round_id": "round-123", "kwargs": kwargs}

    monkeypatch.setattr(cli.AutoResearchRoundManager, "run_non_agent", fake_run_non_agent)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--registry",
            str(repo / "model_registry.yaml"),
            "--tuned-config-root",
            str(repo / "output" / "tuned_configs"),
            "auto-research",
            "run",
            "qwen3.5-27b",
            "--family-id",
            "proposal-ranking-manager-judgment",
            "--workload-file",
            str(repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"),
            "--harness",
            "synthetic",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kwargs"]["harness_type"] == "synthetic"
