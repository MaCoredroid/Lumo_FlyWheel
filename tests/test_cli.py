from __future__ import annotations

import argparse
import json
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
        return None

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
    )


def test_smoke_test_requires_prefix_cache_hits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    events: list[str] = []
    metrics_iter = iter(["before", "after"])

    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: events.append(f"start:{model_id}"),
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        metrics=lambda: _Response(text=next(metrics_iter)),
        flush_prefix_cache=lambda: events.append("flush"),
        stop=lambda missing_ok=True: events.append(f"stop:{missing_ok}"),
    )

    requests_seen: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        requests_seen.append({"url": url, "json": json})
        if url.endswith("/v1/responses"):
            return _Response(payload={"id": "resp-1"})
        if len(requests_seen) == 1:
            return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})
        return _Response(payload={"id": "chat-2", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "parse_prometheus_text", lambda raw: {"marker": raw})
    monkeypatch.setattr(cli, "resolve_metric_schema", lambda snapshot: {"prefix_cache_hits": "cache_hits"})
    monkeypatch.setattr(requests, "post", fake_post)

    cache_hits = iter([{"cache_hits": 0.0}, {"cache_hits": 3.0}])
    monkeypatch.setattr(cli, "parse_prometheus_text", lambda raw: next(cache_hits))

    assert cli.cmd_smoke_test(_args()) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["prefix_cache_hits_delta"] == 3.0
    assert events == ["start:qwen3.5-27b", "flush", "stop:True"]
    assert len(requests_seen) == 3
    assert requests_seen[0]["url"].endswith("/v1/chat/completions")
    assert requests_seen[1]["url"].endswith("/v1/chat/completions")
    assert requests_seen[2]["url"].endswith("/v1/responses")
    second_messages = requests_seen[1]["json"]["messages"]
    assert second_messages[2] == {"role": "assistant", "content": "OK"}
    assert second_messages[3]["role"] == "user"


def test_smoke_test_fails_when_prefix_cache_hits_do_not_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    metrics_iter = iter([{"cache_hits": 4.0}, {"cache_hits": 4.0}])

    server = SimpleNamespace(
        start=lambda model_id, enable_request_logging=False: events.append(f"start:{model_id}"),
        health=lambda: _Response(status_code=200),
        models=lambda: _Response(payload={"data": [{"id": "qwen3.5-27b"}]}),
        metrics=lambda: _Response(text="ignored"),
        flush_prefix_cache=lambda: events.append("flush"),
        stop=lambda missing_ok=True: events.append(f"stop:{missing_ok}"),
    )

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _Response:
        if url.endswith("/v1/responses"):
            return _Response(payload={"id": "resp-1"})
        return _Response(payload={"id": "chat-1", "choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(cli, "_server", lambda args: server)
    monkeypatch.setattr(cli, "parse_prometheus_text", lambda raw: next(metrics_iter))
    monkeypatch.setattr(cli, "resolve_metric_schema", lambda snapshot: {"prefix_cache_hits": "cache_hits"})
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Expected prefix cache hits"):
        cli.cmd_smoke_test(_args())

    assert events == ["start:qwen3.5-27b", "stop:True"]
