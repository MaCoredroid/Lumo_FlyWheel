from __future__ import annotations

from typing import Any

import scripts.build_parity_fixture as build_parity_fixture


def test_request_completion_makes_required_state_checkpoint_non_terminal(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Response:
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(build_parity_fixture.requests, "post", fake_post)

    result = build_parity_fixture._request_completion(
        endpoint="http://127.0.0.1:8100/v1",
        api_key="EMPTY",
        model="qwen3.5-27b",
        probe={"prompt": "probe", "output_token_count": 1024},
        timeout_s=30.0,
        minimum_completion_tokens=1025,
    )

    assert result == {"id": "cmpl-test"}
    assert captured["json"]["max_tokens"] == 1025
    assert captured["json"]["min_tokens"] == 1025
    assert captured["json"]["ignore_eos"] is True


def test_request_completion_does_not_extend_gatedattn_only_probe(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Response:
        captured.update({"json": json})
        return Response()

    monkeypatch.setattr(build_parity_fixture.requests, "post", fake_post)

    build_parity_fixture._request_completion(
        endpoint="http://127.0.0.1:8100/v1",
        api_key="EMPTY",
        model="qwen3.5-27b",
        probe={"prompt": "probe", "output_token_count": 16},
        timeout_s=30.0,
    )

    assert captured["json"]["max_tokens"] == 16
    assert "min_tokens" not in captured["json"]
    assert "ignore_eos" not in captured["json"]
