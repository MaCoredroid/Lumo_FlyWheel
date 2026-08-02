from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
import threading
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.inference_proxy import (  # noqa: E402
    FIXED32_ENGINE_BEGIN_PATH,
    FIXED32_ENGINE_FINALIZE_PATH,
    FIXED32_INGRESS_BEGIN_SCHEMA,
    FIXED32_INGRESS_FINALIZE_SCHEMA,
    FIXED32_TASK_KEY_HEADER,
    Fixed32DigestLedger,
    Fixed32EngineIngressMiddleware,
    Fixed32IngressAdmissionError,
    Fixed32IngressError,
    Fixed32ProxyIngress,
    Fixed32TrackedResponse,
    build_proxy_handler,
    derive_fixed32_task_bearer,
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    load_fixed32_ingress_secrets,
    verify_fixed32_ingress_ledger,
)


TASK_IDS = ("astropy__astropy-12907", "astropy__astropy-13033")
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT16_TASK_IDS = (
    *EXACT4_TASK_IDS,
    "astropy__astropy-13453",
    "astropy__astropy-13579",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14309",
    "astropy__astropy-14365",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14539",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
)


@pytest.fixture(autouse=True)
def _exact_vllm_request_id_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def _write_secret(path: Path) -> tuple[str, str]:
    task_seed = "42" * 32
    engine_bearer = "fr13-engine-" + "9a" * 32
    path.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-ingress-secrets-v1",
                "task_hmac_key_hex": task_seed,
                "engine_bearer": engine_bearer,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return task_seed, engine_bearer


def _begin_payload(task_ids: tuple[str, ...] = TASK_IDS) -> dict[str, Any]:
    return {
        "schema": FIXED32_INGRESS_BEGIN_SCHEMA,
        "canonical_task_count": len(task_ids),
        "canonical_task_set_sha256": fixed32_canonical_task_set_sha256(task_ids),
    }


def _finalize_payload() -> dict[str, str]:
    return {"schema": FIXED32_INGRESS_FINALIZE_SCHEMA}


def _proxy_ingress(tmp_path: Path) -> tuple[Fixed32ProxyIngress, Path, Path]:
    secret_path = tmp_path / "ingress-secret.json"
    _write_secret(secret_path)
    ledger_path = tmp_path / "proxy-ingress.jsonl"
    return (
        Fixed32ProxyIngress(
            secret_file=secret_path,
            canonical_task_ids=TASK_IDS,
            ledger_path=ledger_path,
        ),
        secret_path,
        ledger_path,
    )


def _start_server(
    handler: type[BaseHTTPRequestHandler],
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_secret_file_is_strict_and_task_bearers_are_canonical(tmp_path: Path) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    loaded = load_fixed32_ingress_secrets(secret_path)
    bearer, key_id = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    assert loaded.engine_bearer == engine_bearer
    assert bearer.startswith(f"fr13t1.{key_id}.")
    assert key_id == fixed32_task_key_id(TASK_IDS[0])
    assert bearer != engine_bearer

    secret_path.chmod(0o644)
    with pytest.raises(Fixed32IngressError, match="mode 0600"):
        load_fixed32_ingress_secrets(secret_path)
    secret_path.chmod(0o600)
    symlink = tmp_path / "secret-link.json"
    symlink.symlink_to(secret_path)
    with pytest.raises(Fixed32IngressError, match="non-symlink"):
        load_fixed32_ingress_secrets(symlink)


def test_proxy_rejects_unauthorized_wrong_and_noncanonical_task_keys(
    tmp_path: Path,
) -> None:
    ingress, secret_path, ledger_path = _proxy_ingress(tmp_path)
    canonical_bearer, _ = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    noncanonical_bearer, _ = derive_fixed32_task_bearer(
        secret_path, "django__django-99999"
    )
    for bearer, reason in (
        (None, "missing_bearer"),
        ("wrong", "malformed_bearer"),
        (noncanonical_bearer, "unknown_task_key"),
    ):
        with pytest.raises(Fixed32IngressAdmissionError) as caught:
            ingress.start_logical(
                authorization_bearer=bearer,
                route="chat",
            )
        assert caught.value.reason == reason
    ingress.begin(_begin_payload())
    logical = ingress.start_logical(
        authorization_bearer=canonical_bearer,
        route="chat",
    )
    attempt = ingress.begin_attempt(logical)
    ingress.finish_attempt(attempt, status_code=200, exception=False)
    ingress.finish_logical(logical, aborted=False)
    report = ingress.finalize(_finalize_payload())
    ingress.ledger.close()

    assert report["preflight_rejected_requests"] == 3
    assert report["campaign_rejected_requests"] == 0
    assert report["completed_logical_requests"] == 1
    raw = ledger_path.read_bytes()
    assert canonical_bearer.encode() not in raw
    assert noncanonical_bearer.encode() not in raw
    assert TASK_IDS[0].encode() not in raw
    engine_bearer = load_fixed32_ingress_secrets(secret_path).engine_bearer
    assert engine_bearer.encode() not in raw
    serialized_report = json.dumps(report)
    assert canonical_bearer not in serialized_report
    assert noncanonical_bearer not in serialized_report
    assert engine_bearer not in serialized_report
    assert TASK_IDS[0] not in serialized_report
    assert verify_fixed32_ingress_ledger(
        ledger_path, expected_role="proxy", require_finalized=True
    )["active_requests"] == 0


def test_attempt_ids_are_unique_route_marked_and_digest_only(tmp_path: Path) -> None:
    ingress, secret_path, ledger_path = _proxy_ingress(tmp_path)
    bearer, _ = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    ingress.begin(_begin_payload())

    chat = ingress.start_logical(authorization_bearer=bearer, route="chat")
    chat_attempts = [ingress.begin_attempt(chat), ingress.begin_attempt(chat)]
    assert chat_attempts[0].wire_id != chat_attempts[1].wire_id
    for attempt in chat_attempts:
        assert attempt.wire_id.startswith("fr13-chat-")
        assert attempt.engine_request_id == f"chatcmpl-{attempt.wire_id}"
        ingress.finish_attempt(attempt, status_code=200, exception=False)
    ingress.finish_logical(chat, aborted=False)

    responses = ingress.start_logical(
        authorization_bearer=bearer, route="responses"
    )
    response_attempt = ingress.begin_attempt(responses)
    assert response_attempt.wire_id.startswith("fr13-responses-")
    assert response_attempt.engine_request_id == f"{response_attempt.wire_id}_0"
    ingress.finish_attempt(response_attempt, status_code=200, exception=False)
    ingress.finish_logical(responses, aborted=False)
    ingress.finalize(_finalize_payload())
    ingress.ledger.close()

    ledger = ledger_path.read_text(encoding="utf-8")
    for attempt in (*chat_attempts, response_attempt):
        assert attempt.wire_id not in ledger
        assert attempt.engine_request_id not in ledger
        assert attempt.wire_id_sha256 in ledger
        assert attempt.engine_request_id_sha256 in ledger


def test_ledger_detects_tampering_order_and_active_finalization(
    tmp_path: Path,
) -> None:
    ingress, secret_path, ledger_path = _proxy_ingress(tmp_path)
    bearer, _ = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    ingress.begin(_begin_payload())
    logical = ingress.start_logical(authorization_bearer=bearer, route="chat")
    with pytest.raises(Fixed32IngressAdmissionError, match="active_requests"):
        ingress.finalize(_finalize_payload())
    ingress.finish_logical(logical, aborted=False)
    ingress.finalize(_finalize_payload())
    ingress.ledger.close()
    with pytest.raises(Fixed32IngressError, match="incomplete logical work"):
        verify_fixed32_ingress_ledger(
            ledger_path, expected_role="proxy", require_finalized=True
        )

    rows = ledger_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[0])
    tampered["outcome"] = "changed"
    rows[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(Fixed32IngressError, match="digest mismatch"):
        verify_fixed32_ingress_ledger(ledger_path, expected_role="proxy")

    unordered_path = tmp_path / "unordered.jsonl"
    unordered = Fixed32DigestLedger(unordered_path, role="proxy")
    unordered.append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256="6" * 64,
    )
    unordered.append(
        phase="campaign",
        event="attempt_result",
        route="chat",
        task_key_id="2" * 64,
        logical_id_sha256="3" * 64,
        outcome="response",
        wire_id_sha256="1" * 64,
        engine_request_id_sha256="4" * 64,
        evidence_sha256="5" * 64,
        status_code=200,
    )
    unordered.close()
    with pytest.raises(Fixed32IngressError, match="attempt result has no begin"):
        verify_fixed32_ingress_ledger(unordered_path, expected_role="proxy")


def test_ledger_write_failure_poisoning_is_fail_closed(tmp_path: Path) -> None:
    ingress, _secret_path, _ledger_path = _proxy_ingress(tmp_path)
    os.close(ingress.ledger._fd)
    with pytest.raises(Fixed32IngressError, match="append failed"):
        ingress.begin(_begin_payload())
    with pytest.raises(Fixed32IngressError, match="poisoned"):
        ingress.begin(_begin_payload())


class _FakeResponse:
    status_code = 200

    def iter_content(self, *_args: Any, **_kwargs: Any) -> Any:
        yield b"first"
        yield b"second"

    def close(self) -> None:
        return


def test_tracked_stream_generator_close_finishes_attempt_as_failure(
    tmp_path: Path,
) -> None:
    ingress, secret_path, _ledger_path = _proxy_ingress(tmp_path)
    bearer, _ = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    ingress.begin(_begin_payload())
    logical = ingress.start_logical(authorization_bearer=bearer, route="chat")
    attempt = ingress.begin_attempt(logical)
    tracked = Fixed32TrackedResponse(
        _FakeResponse(), ingress=ingress, attempt=attempt
    )
    chunks = tracked.iter_content()
    assert next(chunks) == b"first"
    chunks.close()
    ingress.finish_logical(logical, aborted=False)
    report = ingress.finalize(_finalize_payload())
    assert report["failed_attempts"] == 1
    assert report["aborted_logical_requests"] == 1


async def _asgi_call(
    app: Any,
    *,
    path: str,
    body: bytes,
    headers: list[tuple[bytes, bytes]],
    method: str = "POST",
) -> tuple[int, dict[str, Any]]:
    received = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {"type": "http", "method": method, "path": path, "headers": headers},
        receive,
        send,
    )
    status = next(item["status"] for item in sent if item["type"] == "http.response.start")
    response_body = b"".join(
        item.get("body", b"")
        for item in sent
        if item["type"] == "http.response.body"
    )
    return status, json.loads(response_body) if response_body else {}


def test_engine_middleware_rejects_before_generation_and_replays_exact_body(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    ledger_path = tmp_path / "engine.jsonl"
    calls: list[bytes] = []

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        message = await receive()
        calls.append(message["body"])
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"ok":true}',
                "more_body": False,
            }
        )

    middleware = Fixed32EngineIngressMiddleware(
        inner_app,
        secret_file=secret_path,
        canonical_task_ids=TASK_IDS,
        ledger_path=ledger_path,
    )
    duplicate_control = (
        b'{"schema":"fr13-fixed32-ingress-begin-v1",'
        b'"schema":"fr13-fixed32-ingress-begin-v1",'
        b'"canonical_task_count":2,'
        b'"canonical_task_set_sha256":"'
        + fixed32_canonical_task_set_sha256(TASK_IDS).encode()
        + b'"}'
    )
    status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path=FIXED32_ENGINE_BEGIN_PATH,
            body=duplicate_control,
            headers=[(b"authorization", f"Bearer {engine_bearer}".encode())],
        )
    )
    assert status == 400
    for auth in (None, b"Bearer wrong"):
        headers = [] if auth is None else [(b"authorization", auth)]
        status, _ = asyncio.run(
            _asgi_call(
                middleware,
                path="/v1/chat/completions",
                body=b'{"secret_body":"must-not-leak"}',
                headers=headers,
            )
        )
        assert status == 401
    assert calls == []

    control_headers = [(b"authorization", f"Bearer {engine_bearer}".encode())]
    status, begin = asyncio.run(
        _asgi_call(
            middleware,
            path=FIXED32_ENGINE_BEGIN_PATH,
            body=json.dumps(_begin_payload()).encode(),
            headers=control_headers,
        )
    )
    assert status == 200
    assert begin["preflight_rejected_requests"] == 2

    task_key_id = fixed32_task_key_id(TASK_IDS[0])
    wire_id = "fr13-chat-" + "ab" * 16
    valid_headers = [
        (b"authorization", f"Bearer {engine_bearer}".encode()),
        (FIXED32_TASK_KEY_HEADER.lower().encode(), task_key_id.encode()),
        (b"x-request-id", wire_id.encode()),
    ]
    request_body = b'{"secret_body":"must-not-leak"}'
    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=request_body,
            headers=valid_headers,
        )
    )
    assert status == 200
    assert payload == {"ok": True}
    assert calls == [request_body]

    duplicate_headers = [*valid_headers, (b"authorization", b"Bearer duplicate")]
    status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=request_body,
            headers=duplicate_headers,
        )
    )
    assert status == 400
    assert calls == [request_body]

    status, final = asyncio.run(
        _asgi_call(
            middleware,
            path=FIXED32_ENGINE_FINALIZE_PATH,
            body=json.dumps(_finalize_payload()).encode(),
            headers=control_headers,
        )
    )
    assert status == 200
    assert final["active_requests"] == 0
    middleware.ingress.ledger.close()
    raw = ledger_path.read_bytes()
    assert request_body not in raw
    assert engine_bearer.encode() not in raw
    assert wire_id.encode() not in raw
    verify_fixed32_ingress_ledger(
        ledger_path, expected_role="engine", require_finalized=True
    )


@pytest.mark.parametrize(
    "enabled_name",
    (
        "fr13_fixed32_batch_gdn_byte_ab.enabled",
        "fr13_fixed32_batch_gdn_graph_byte_ab.enabled",
    ),
)
def test_engine_middleware_arms_batch_gdn_from_authenticated_exact4_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled_name: str,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    enabled = logs / enabled_name
    enabled.write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH", str(marker)
    )
    observed: list[bytes] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        observed.append(marker.read_bytes())
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=EXACT4_TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    middleware.ingress.begin(_begin_payload(EXACT4_TASK_IDS))
    wire_id = "fr13-chat-" + "ab" * 16
    unauthorized_status, unauthorized_payload = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=b'{"messages":[]}',
            headers=[
                (b"authorization", f"Bearer {engine_bearer}".encode()),
                (FIXED32_TASK_KEY_HEADER.lower().encode(), b"f" * 64),
                (b"x-request-id", wire_id.encode()),
            ],
        )
    )
    assert unauthorized_status == 401
    assert unauthorized_payload == {"error": {"code": "invalid_task_key"}}
    assert not marker.exists()
    assert observed == []

    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=b'{"messages":[]}',
            headers=[
                (b"authorization", f"Bearer {engine_bearer}".encode()),
                (
                    FIXED32_TASK_KEY_HEADER.lower().encode(),
                    fixed32_task_key_id(EXACT4_TASK_IDS[0]).encode(),
                ),
                (b"x-request-id", wire_id.encode()),
            ],
        )
    )
    assert status == 200
    assert payload == {}
    expected = b"swe_verified:astropy__astropy-12907\n"
    assert observed == [expected]
    assert marker.read_bytes() == expected
    info = os.lstat(marker)
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o400

    second_wire_id = "fr13-chat-" + "cd" * 16
    second_status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=b'{"messages":[]}',
            headers=[
                (b"authorization", f"Bearer {engine_bearer}".encode()),
                (
                    FIXED32_TASK_KEY_HEADER.lower().encode(),
                    fixed32_task_key_id(EXACT4_TASK_IDS[1]).encode(),
                ),
                (b"x-request-id", second_wire_id.encode()),
            ],
        )
    )
    assert second_status == 200
    assert observed == [expected, expected]
    assert marker.read_bytes() == expected


def test_engine_middleware_arms_cutlass_b4_from_authenticated_exact4_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fr13_fixed32_cutlass_b4_byte_ab.enabled").write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_cutlass_b4_byte_ab.real_event.arm"
    monkeypatch.setenv(
        "FR13_FIXED32_CUTLASS_B4_BYTE_AB_REAL_EVENT_PATH", str(marker)
    )
    observed: list[bytes] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        observed.append(marker.read_bytes())
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=EXACT4_TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    middleware.ingress.begin(_begin_payload(EXACT4_TASK_IDS))
    wire_id = "fr13-chat-" + "12" * 16
    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=b'{"messages":[]}',
            headers=[
                (b"authorization", f"Bearer {engine_bearer}".encode()),
                (
                    FIXED32_TASK_KEY_HEADER.lower().encode(),
                    fixed32_task_key_id(EXACT4_TASK_IDS[0]).encode(),
                ),
                (b"x-request-id", wire_id.encode()),
            ],
        )
    )
    assert status == 200
    assert payload == {}
    expected = b"swe_verified:astropy__astropy-12907\n"
    assert observed == [expected]
    assert marker.read_bytes() == expected
    info = os.lstat(marker)
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o400


def test_engine_middleware_arms_sfwd_only_after_all_exact4_are_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    enabled = logs / "fr13_fixed32_sfwd_state_fusion_byte_ab.enabled"
    enabled.write_bytes(b"1\n")
    enabled.chmod(0o400)
    marker = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH", str(marker)
    )
    observed: list[bytes | None] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        observed.append(marker.read_bytes() if marker.exists() else None)
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=EXACT4_TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    middleware.ingress.begin(_begin_payload(EXACT4_TASK_IDS))
    for index, task_id in enumerate(EXACT4_TASK_IDS):
        wire_id = f"fr13-chat-{index + 1:032x}"
        status, payload = asyncio.run(
            _asgi_call(
                middleware,
                path="/v1/chat/completions",
                body=b'{"messages":[]}',
                headers=[
                    (b"authorization", f"Bearer {engine_bearer}".encode()),
                    (
                        FIXED32_TASK_KEY_HEADER.lower().encode(),
                        fixed32_task_key_id(task_id).encode(),
                    ),
                    (b"x-request-id", wire_id.encode()),
                ],
            )
        )
        assert status == 200
        assert payload == {}

    expected = (
        "\n".join(f"swe_verified:{task_id}" for task_id in EXACT4_TASK_IDS)
        + "\n"
    ).encode("ascii")
    assert observed == [None, None, None, expected]
    assert marker.read_bytes() == expected
    info = os.lstat(marker)
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o444


@pytest.mark.parametrize("task_ids", (EXACT4_TASK_IDS, EXACT16_TASK_IDS))
def test_engine_arms_grouped_simd_only_after_complete_authenticated_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_ids: tuple[str, ...],
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    enabled = logs / "fr13_fixed32_gdn_parent_group_simd_b4_graph.enabled"
    enabled.write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_gdn_parent_group_simd_b4.real_event.arm"
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_REAL_EVENT_PATH", str(marker)
    )
    observed: list[bytes | None] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        observed.append(marker.read_bytes() if marker.exists() else None)
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=task_ids,
        ledger_path=tmp_path / "engine.jsonl",
    )
    middleware.ingress.begin(_begin_payload(task_ids))
    for index, task_id in enumerate(task_ids):
        wire_id = f"fr13-chat-{index + 1:032x}"
        status, payload = asyncio.run(
            _asgi_call(
                middleware,
                path="/v1/chat/completions",
                body=b'{"messages":[]}',
                headers=[
                    (b"authorization", f"Bearer {engine_bearer}".encode()),
                    (
                        FIXED32_TASK_KEY_HEADER.lower().encode(),
                        fixed32_task_key_id(task_id).encode(),
                    ),
                    (b"x-request-id", wire_id.encode()),
                ],
            )
        )
        assert status == 200
        assert payload == {}

    expected = (
        "\n".join(f"swe_verified:{task_id}" for task_id in task_ids) + "\n"
    ).encode("ascii")
    assert observed[:-1] == [None] * (len(task_ids) - 1)
    assert observed[-1] == expected
    assert marker.read_bytes() == expected
    info = os.lstat(marker)
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o444


def test_engine_middleware_rejects_inexact_or_ambiguous_cutlass_b4_arm(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fr13_fixed32_cutlass_b4_byte_ab.enabled").write_bytes(b"1\n")
    b4_marker = logs / "fr13_fixed32_cutlass_b4_byte_ab.real_event.arm"

    with pytest.raises(Fixed32IngressError, match="canonical exact4 tasks"):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=TASK_IDS,
            ledger_path=tmp_path / "engine-inexact.jsonl",
            cutlass_b4_real_event_arm=b4_marker,
        )

    (logs / "fr13_fixed32_batch_gdn_byte_ab.enabled").write_bytes(b"1\n")
    gdn_marker = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
    with pytest.raises(Fixed32IngressError, match="mutually exclusive"):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=EXACT4_TASK_IDS,
            ledger_path=tmp_path / "engine-ambiguous.jsonl",
            batch_gdn_real_event_arm=gdn_marker,
            cutlass_b4_real_event_arm=b4_marker,
        )
def test_engine_middleware_rejects_batch_gdn_arm_injected_after_boot(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fr13_fixed32_batch_gdn_byte_ab.enabled").write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
    calls: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["path"])

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=EXACT4_TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
        batch_gdn_real_event_arm=marker,
    )
    middleware.ingress.begin(_begin_payload(EXACT4_TASK_IDS))
    marker.write_bytes(b"swe_verified:astropy__astropy-12907\n")
    marker.chmod(0o400)
    wire_id = "fr13-chat-" + "ef" * 16
    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/chat/completions",
            body=b'{"messages":[]}',
            headers=[
                (b"authorization", f"Bearer {engine_bearer}".encode()),
                (
                    FIXED32_TASK_KEY_HEADER.lower().encode(),
                    fixed32_task_key_id(EXACT4_TASK_IDS[0]).encode(),
                ),
                (b"x-request-id", wire_id.encode()),
            ],
        )
    )
    assert status == 503
    assert payload == {"error": {"code": "ingress_evidence_failure"}}
    assert calls == []
    assert middleware.ingress._task_counts[
        fixed32_task_key_id(EXACT4_TASK_IDS[0])
    ]["accepted_engine_requests"] == 0


def test_engine_middleware_rejects_inexact_batch_gdn_arm_configuration(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fr13_fixed32_batch_gdn_byte_ab.enabled").write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"

    with pytest.raises(Fixed32IngressError, match="canonical exact4 tasks"):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=TASK_IDS,
            ledger_path=tmp_path / "engine-inexact.jsonl",
            batch_gdn_real_event_arm=marker,
        )

    marker.write_bytes(b"swe_verified:astropy__astropy-12907\n")
    with pytest.raises(Fixed32IngressError, match="must be new at engine boot"):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=EXACT4_TASK_IDS,
            ledger_path=tmp_path / "engine-prearmed.jsonl",
            batch_gdn_real_event_arm=marker,
        )


def test_engine_middleware_rejects_both_batch_gdn_gate_sidecars(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in (
        "fr13_fixed32_batch_gdn_byte_ab.enabled",
        "fr13_fixed32_batch_gdn_graph_byte_ab.enabled",
    ):
        (logs / name).write_bytes(b"1\n")
    marker = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"

    with pytest.raises(Fixed32IngressError, match="exactly one eager or graph"):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=EXACT4_TASK_IDS,
            ledger_path=tmp_path / "engine.jsonl",
            batch_gdn_real_event_arm=marker,
        )


@pytest.mark.parametrize("binding_value", (None, "0", "true"))
def test_engine_middleware_requires_exact_vllm_request_id_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_value: str | None,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    ledger_path = tmp_path / "engine.jsonl"
    if binding_value is None:
        monkeypatch.delenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION")
    else:
        monkeypatch.setenv(
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION",
            binding_value,
        )

    with pytest.raises(
        Fixed32IngressError,
        match="requires exact vLLM request ID binding",
    ):
        Fixed32EngineIngressMiddleware(
            object(),
            secret_file=secret_path,
            canonical_task_ids=TASK_IDS,
            ledger_path=ledger_path,
        )
    assert not ledger_path.exists()


def test_engine_responses_requires_matching_body_and_header_request_id(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    calls: list[bytes] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        calls.append((await receive())["body"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    middleware.ingress.begin(_begin_payload())
    wire_id = "fr13-responses-" + "cd" * 16
    headers = [
        (b"authorization", f"Bearer {engine_bearer}".encode()),
        (
            FIXED32_TASK_KEY_HEADER.lower().encode(),
            fixed32_task_key_id(TASK_IDS[0]).encode(),
        ),
        (b"x-request-id", wire_id.encode()),
    ]
    status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/responses",
            body=json.dumps({"request_id": "wrong"}).encode(),
            headers=headers,
        )
    )
    assert status == 400
    assert calls == []
    duplicate_body = (
        b'{"request_id":"' + wire_id.encode() + b'",'
        b'"request_id":"' + wire_id.encode() + b'"}'
    )
    status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/responses",
            body=duplicate_body,
            headers=headers,
        )
    )
    assert status == 400
    assert calls == []
    valid_body = json.dumps({"request_id": wire_id, "input": []}).encode()
    status, _ = asyncio.run(
        _asgi_call(
            middleware,
            path="/v1/responses",
            body=valid_body,
            headers=headers,
        )
    )
    assert status == 204
    assert calls == [valid_body]


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/reset_prefix_cache"),
        ("POST", "/start_profile"),
        ("GET", "/v1/chat/completions"),
        ("POST", "/health"),
    ),
)
def test_engine_middleware_denies_every_non_allowlisted_http_route(
    tmp_path: Path,
    method: str,
    path: str,
) -> None:
    secret_path = tmp_path / "secret.json"
    _task_seed, engine_bearer = _write_secret(secret_path)
    calls: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["path"])

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path=path,
            body=b'{"model":"qwen"}',
            headers=[(b"authorization", f"Bearer {engine_bearer}".encode())],
            method=method,
        )
    )
    assert status == 403
    assert payload == {"error": {"code": "fixed32_route_not_allowed"}}
    assert calls == []
    assert middleware.ingress.ledger.records == 0


@pytest.mark.parametrize("path", ("/health", "/metrics", "/v1/models"))
def test_engine_middleware_allows_only_explicit_non_inference_gets(
    tmp_path: Path,
    path: str,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    calls: list[tuple[str, str]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        calls.append((scope["method"], scope["path"]))
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"ok":true}',
                "more_body": False,
            }
        )

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )
    status, payload = asyncio.run(
        _asgi_call(
            middleware,
            path=path,
            body=b"",
            headers=[],
            method="GET",
        )
    )
    assert status == 200
    assert payload == {"ok": True}
    assert calls == [("GET", path)]
    assert middleware.ingress.ledger.records == 0


def test_engine_middleware_passes_lifespan_and_denies_websockets(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secret.json"
    _write_secret(secret_path)
    calls: list[str] = []
    lifespan_sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["type"])
        message = await receive()
        assert message["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.complete"})

    middleware = Fixed32EngineIngressMiddleware(
        inner,
        secret_file=secret_path,
        canonical_task_ids=TASK_IDS,
        ledger_path=tmp_path / "engine.jsonl",
    )

    async def lifespan_receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def lifespan_send(message: dict[str, Any]) -> None:
        lifespan_sent.append(message)

    asyncio.run(
        middleware(
            {"type": "lifespan"},
            lifespan_receive,
            lifespan_send,
        )
    )
    assert calls == ["lifespan"]
    assert lifespan_sent == [{"type": "lifespan.startup.complete"}]

    async def websocket_receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    for path in (
        "/v1/realtime",
        "/v1/chat/completions",
        FIXED32_ENGINE_BEGIN_PATH,
    ):
        websocket_sent: list[dict[str, Any]] = []

        async def websocket_send(message: dict[str, Any]) -> None:
            websocket_sent.append(message)

        asyncio.run(
            middleware(
                {"type": "websocket", "path": path},
                websocket_receive,
                websocket_send,
            )
        )
        assert websocket_sent == [
            {
                "type": "websocket.close",
                "code": 1008,
                "reason": "fixed32 exclusive ingress",
            }
        ]
        assert calls == ["lifespan"]
    assert middleware.ingress.ledger.records == 0


def test_proxy_overwrites_authorization_and_injects_unique_attempt_ids(
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            captures.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "task_key_id": self.headers.get(FIXED32_TASK_KEY_HEADER),
                    "request_id": self.headers.get("X-Request-ID"),
                    "body": body,
                }
            )
            if len(captures) == 1:
                status = 400
                payload = {"error": {"message": "Unterminated string"}}
            elif self.path == "/v1/responses":
                status = 200
                payload = {
                    "id": "resp-upstream",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            else:
                status = 200
                payload = {"id": "chatcmpl-upstream", "choices": []}
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    upstream_server, upstream_thread, upstream_url = _start_server(Upstream)
    ingress, secret_path, _ledger_path = _proxy_ingress(tmp_path)
    ingress.begin(_begin_payload())
    task_bearer, task_key_id = derive_fixed32_task_bearer(
        secret_path, TASK_IDS[0]
    )
    engine_bearer = load_fixed32_ingress_secrets(secret_path).engine_bearer
    proxy_server, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(
            upstream_url,
            state_root=tmp_path / "state",
            fixed32_ingress=ingress,
        )
    )
    try:
        chat_response = requests.post(
            f"{proxy_url}/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": False},
            headers={"Authorization": f"Bearer {task_bearer}"},
            timeout=10,
        )
        assert chat_response.status_code == 200
        responses_response = requests.post(
            f"{proxy_url}/v1/responses",
            json={"model": "qwen", "input": [], "stream": False},
            headers={"Authorization": f"Bearer {task_bearer}"},
            timeout=10,
        )
        assert responses_response.status_code == 200
    finally:
        _stop_server(proxy_server, proxy_thread)
        _stop_server(upstream_server, upstream_thread)

    assert len(captures) == 3
    assert all(
        item["authorization"] == f"Bearer {engine_bearer}" for item in captures
    )
    assert all(item["task_key_id"] == task_key_id for item in captures)
    wire_ids = [item["request_id"] for item in captures]
    assert len(set(wire_ids)) == 3
    assert wire_ids[0].startswith("fr13-chat-")
    assert wire_ids[1].startswith("fr13-chat-")
    assert wire_ids[2].startswith("fr13-responses-")
    responses_body = json.loads(captures[2]["body"])
    assert responses_body["request_id"] == wire_ids[2]
    report = ingress.finalize(_finalize_payload())
    assert report["accepted_logical_requests"] == 2
    assert report["accepted_attempts"] == 3


def test_proxy_control_endpoints_authenticate_before_strict_body_parse(
    tmp_path: Path,
) -> None:
    ingress, secret_path, _ledger_path = _proxy_ingress(tmp_path)
    engine_bearer = load_fixed32_ingress_secrets(secret_path).engine_bearer
    task_bearer, task_key_id = derive_fixed32_task_bearer(
        secret_path, TASK_IDS[0]
    )
    proxy_server, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(
            "http://127.0.0.1:1",
            state_root=tmp_path / "state",
            fixed32_ingress=ingress,
        )
    )
    try:
        unauthorized = requests.post(
            f"{proxy_url}/admin/fixed32/ingress/begin",
            data=b'{"schema":',
            timeout=10,
        )
        assert unauthorized.status_code == 401
        duplicate = requests.post(
            f"{proxy_url}/admin/fixed32/ingress/begin",
            data=b'{"schema":"x","schema":"y"}',
            headers={
                "Authorization": f"Bearer {engine_bearer}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        assert duplicate.status_code == 400
        begun = requests.post(
            f"{proxy_url}/admin/fixed32/ingress/begin",
            json=_begin_payload(),
            headers={"Authorization": f"Bearer {engine_bearer}"},
            timeout=10,
        )
        assert begun.status_code == 200
        evidence = requests.post(
            f"{proxy_url}/admin/fixed32/ingress/task-evidence",
            json={},
            headers={"Authorization": f"Bearer {task_bearer}"},
            timeout=10,
        )
        assert evidence.status_code == 200
        assert evidence.json()["task_key_id"] == task_key_id
        finalized = requests.post(
            f"{proxy_url}/admin/fixed32/ingress/finalize",
            json=_finalize_payload(),
            headers={"Authorization": f"Bearer {engine_bearer}"},
            timeout=10,
        )
        assert finalized.status_code == 200
        assert finalized.json()["active_requests"] == 0
    finally:
        _stop_server(proxy_server, proxy_thread)


def test_proxy_disables_legacy_admin_before_body_or_state_mutation(
    tmp_path: Path,
) -> None:
    ingress, secret_path, _ledger_path = _proxy_ingress(tmp_path)
    task_bearer, _task_key_id = derive_fixed32_task_bearer(
        secret_path, TASK_IDS[0]
    )
    state_root = tmp_path / "state"
    proxy_server, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(
            "http://127.0.0.1:1",
            state_root=state_root,
            fixed32_ingress=ingress,
        )
    )
    try:
        for path, body in (
            ("/admin/load_tuned_config", b'{"unterminated":'),
            (
                "/admin/invalidate",
                b'{"weight_version_id":"must-not-be-recorded"}',
            ),
        ):
            response = requests.post(
                f"{proxy_url}{path}",
                data=body,
                headers={
                    "Authorization": f"Bearer {task_bearer}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            assert response.status_code == 403
            assert response.json() == {
                "error": {
                    "code": "fixed32_admin_disabled",
                    "message": "Legacy admin endpoint is disabled during fixed32",
                }
            }
            assert response.headers["Connection"].lower() == "close"
    finally:
        _stop_server(proxy_server, proxy_thread)

    assert not state_root.exists()
    assert ingress.phase == "preflight"
    assert ingress.ledger.records == 0


@pytest.mark.parametrize(
    "failure", ("invalid_json", "duplicate_json", "queue_full", "network")
)
def test_proxy_early_failures_abort_logical_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    ingress, secret_path, _ledger_path = _proxy_ingress(tmp_path)
    ingress.begin(_begin_payload())
    bearer, _ = derive_fixed32_task_bearer(secret_path, TASK_IDS[0])
    if failure == "queue_full":
        from lumo_flywheel_serving import inference_proxy

        monkeypatch.setattr(
            inference_proxy.AdmissionController,
            "acquire",
            lambda _self, _request_class: None,
        )
    if failure == "network":
        monkeypatch.setenv("LUMO_PROXY_UPSTREAM_MAX_RETRIES", "1")
        monkeypatch.setenv("LUMO_PROXY_UPSTREAM_BACKOFF_S", "0")
        monkeypatch.setenv("LUMO_PROXY_UPSTREAM_TIMEOUT_S", "0.05")
    proxy_server, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(
            "http://127.0.0.1:1",
            state_root=tmp_path / "state",
            fixed32_ingress=ingress,
        )
    )
    try:
        if failure in {"invalid_json", "duplicate_json"}:
            response = requests.post(
                f"{proxy_url}/v1/chat/completions",
                data=(
                    b"{"
                    if failure == "invalid_json"
                    else b'{"model":"qwen","model":"other"}'
                ),
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            assert response.status_code == 400
        else:
            response = requests.post(
                f"{proxy_url}/v1/chat/completions",
                json={"model": "qwen", "messages": [], "stream": False},
                headers={"Authorization": f"Bearer {bearer}"},
                timeout=10,
            )
            assert response.status_code == (429 if failure == "queue_full" else 504)
    finally:
        _stop_server(proxy_server, proxy_thread)
    report = ingress.finalize(_finalize_payload())
    assert report["completed_logical_requests"] == 0
    assert report["aborted_logical_requests"] == 1
    assert report["failed_attempts"] == (1 if failure == "network" else 0)


def _load_runner() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_swe_bench_q36_a.py"
    spec = importlib.util.spec_from_file_location("fixed32_ingress_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_evidence(
    task_key_id: str,
    *,
    logical: int,
    attempts: int,
    records: int,
) -> dict[str, Any]:
    return {
        "schema": "fr13-fixed32-task-auth-evidence-v1",
        "task_key_id": task_key_id,
        "completed_logical_model_requests": logical,
        "aborted_logical_requests": 0,
        "accepted_attempts": attempts,
        "completed_attempts": attempts,
        "failed_attempts": 0,
        "phase": "campaign",
        "ledger_records": records,
        "ledger_chain_head_sha256": f"{records:064x}",
    }


def _fixed32_agent_meta(runner: Any, task_dir: Path) -> dict[str, Any]:
    workspace = task_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    observation = {
        "qwen_code_version": runner._FIXED32_QWEN_CODE_VERSION,
        "bundle_tree": json.loads(
            json.dumps(runner._FIXED32_QWEN_BUNDLE_TREE_EXPECTED)
        ),
    }
    attestation = runner._build_fixed32_qwen_runtime_attestation(
        bundle_observation=observation,
        host_mode="remote",
    )
    digest = runner._persist_fixed32_qwen_runtime_attestation(
        workspace=workspace,
        attestation=attestation,
    )
    post_digest = runner._persist_fixed32_qwen_runtime_attestation(
        workspace=workspace,
        attestation=attestation,
        filename="qwen_runtime_attestation_post.json",
    )
    pinned_image = runner._FIXED32_AGENT_IMAGE_IDENTITIES[TASK_IDS[0]]
    image = pinned_image["repo_digest"].split("@", 1)[0] + ":latest"
    image_identity = runner._validate_fixed32_agent_image_observation(
        {
            "instance_id": TASK_IDS[0],
            "image": image,
            "id": pinned_image["id"],
            "repo_digest": pinned_image["repo_digest"],
            "architecture": "amd64",
            "os": "linux",
        },
        instance_id=TASK_IDS[0],
        expected_image=image,
    )
    image_digest = runner._fixed32_canonical_json_sha256(image_identity)
    placement = runner._validate_fixed32_agent_placement_observation(
        json.loads(json.dumps(runner._FIXED32_AGENT_HOST_IDENTITY)),
        measured_observation=json.loads(
            json.dumps(runner._FIXED32_MEASURED_HOST_IDENTITY)
        ),
        remote_host="alienware",
    )
    placement_digest = runner._fixed32_canonical_json_sha256(placement)
    remote_settings_observation = {
        **runner._fixed32_expected_remote_settings_observation(),
        "file_identity_sha256": "1" * 64,
    }
    remote_settings_digest = runner._fixed32_canonical_json_sha256(
        remote_settings_observation
    )
    mounted_proof = {
        "schema": runner._FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": observation,
        },
        "system_settings": {
            "container_path": runner._FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **remote_settings_observation,
        },
    }
    mounted_proof_digest = (
        runner._validate_fixed32_mounted_runtime_proof(
            mounted_proof,
            expected_bundle_observation=observation,
        )
    )
    mounted_proof_path = (
        task_dir / runner._FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
    )
    mounted_proof_path.write_text(
        json.dumps(
            mounted_proof,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    return {
        "exit_code": 0,
        "timed_out": False,
        "offloaded": True,
        "network_drop": False,
        "agent_env": "instance_image",
        "instance_image": image,
        "instance_image_identity": image_identity,
        "instance_image_identity_sha256": image_digest,
        "instance_image_postrun_identity_sha256": image_digest,
        "instance_image_run_reference": pinned_image["repo_digest"],
        "agent_placement": placement,
        "agent_placement_sha256": placement_digest,
        "agent_postrun_placement_sha256": placement_digest,
        "qwen_bundle_snapshot": attestation["bundle_snapshot"],
        "qwen_remote_settings_observation": remote_settings_observation,
        "qwen_remote_settings_observation_sha256": remote_settings_digest,
        "qwen_remote_settings_postrun_observation_sha256": (
            remote_settings_digest
        ),
        "qwen_mounted_runtime_proof": mounted_proof,
        "qwen_mounted_runtime_proof_sha256": mounted_proof_digest,
        "qwen_mounted_runtime_proof_file_sha256": (
            runner.hashlib.sha256(mounted_proof_path.read_bytes()).hexdigest()
        ),
        "qwen_runtime_attestation": attestation,
        "qwen_runtime_attestation_sha256": digest,
        "qwen_runtime_postrun_attestation_sha256": post_digest,
    }


def test_runner_v3_counts_only_terminal_qwen_assistant_records(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    task_key_id = "a" * 64
    trace = tmp_path / "qwen_trace.jsonl"
    events = [
        {
            "type": "system",
            "subtype": "init",
            "qwen_code_version": "0.19.4",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "id": "resp-1",
                "content": "partial reasoning",
                "stop_reason": None,
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "id": "resp-1",
                "content": "tool call",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "id": "resp-2",
                "content": "partial answer",
                "stop_reason": None,
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "id": "resp-2",
                "content": "done",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        },
    ]
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_IDS[0],
        trace_path=trace,
        agent_meta=_fixed32_agent_meta(runner, tmp_path),
        task_key_id=task_key_id,
        task_auth_before=_task_evidence(
            task_key_id, logical=3, attempts=4, records=10
        ),
        task_auth_after=_task_evidence(
            task_key_id, logical=5, attempts=6, records=20
        ),
    )
    assert provenance["schema"] == "fr13-fixed32-real-task-provenance-v3"
    assert provenance["trace_completed_logical_model_requests"] == 2
    assert provenance["completed_logical_model_requests"] == 2
    assert provenance["accepted_attempts"] == 2
    serialized = json.dumps(provenance)
    assert "resp-1" not in serialized
    assert "resp-2" not in serialized


def test_runner_passes_only_the_derived_bearer_to_agent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setenv(
        runner.FIXED32_INGRESS_SECRET_FILE_ENV, "/private/ingress-secret"
    )
    monkeypatch.setenv("LUMO_PROXY_FIXED32_SECRET_FILE", "/remote/proxy-secret")
    monkeypatch.setenv("LUMO_PROXY_FIXED32_LEDGER_PATH", "/private/proxy-ledger")
    agent_env = runner._agent_subprocess_env("derived-task-bearer")
    assert agent_env["OPENAI_API_KEY"] == "derived-task-bearer"
    assert runner.FIXED32_INGRESS_SECRET_FILE_ENV not in agent_env
    assert "LUMO_PROXY_FIXED32_SECRET_FILE" not in agent_env
    assert "LUMO_PROXY_FIXED32_LEDGER_PATH" not in agent_env
    for template in (
        runner.CODEX_TEMPLATE,
        runner.QWEN_CODE_TEMPLATE,
    ):
        assert "-e OPENAI_API_KEY " in template
        assert "OPENAI_API_KEY=EMPTY" not in template
        assert "derived-task-bearer" not in template


@pytest.mark.parametrize(
    "trace_line",
    (
        '{"type":"assistant","type":"assistant"}\n',
        '{"type":"assistant","value":NaN}\n',
    ),
)
def test_runner_v3_rejects_ambiguous_trace_json(
    tmp_path: Path,
    trace_line: str,
) -> None:
    runner = _load_runner()
    task_key_id = "b" * 64
    trace = tmp_path / "ambiguous.jsonl"
    trace.write_text(trace_line, encoding="utf-8")
    with pytest.raises(runner.Fixed32BoundaryError, match="invalid JSON"):
        runner._fixed32_real_task_provenance(
            instance_id=TASK_IDS[0],
            trace_path=trace,
            agent_meta=_fixed32_agent_meta(runner, tmp_path),
            task_key_id=task_key_id,
            task_auth_before=_task_evidence(
                task_key_id, logical=0, attempts=0, records=1
            ),
            task_auth_after=_task_evidence(
                task_key_id, logical=0, attempts=0, records=1
            ),
        )


def test_duplicate_proxy_critical_headers_are_detected() -> None:
    from lumo_flywheel_serving.inference_proxy import (
        _fixed32_has_duplicate_critical_headers,
    )

    headers = Message()
    headers.add_header("Authorization", "Bearer first")
    headers.add_header("authorization", "Bearer second")
    assert _fixed32_has_duplicate_critical_headers(headers)
