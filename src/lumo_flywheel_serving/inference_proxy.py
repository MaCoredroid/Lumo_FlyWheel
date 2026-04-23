from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

from .registry import load_registry
from .tuned_config import RuntimeStateStore, StructuredValidationError, ValidationIssue, load_tuned_config_bundle

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
INFERENCE_PATHS = {"/v1/responses", "/v1/chat/completions"}
ADMIN_PATHS = {"/admin/load_tuned_config", "/admin/invalidate"}


def is_inference_path(path: str) -> bool:
    return path in INFERENCE_PATHS


def normalize_responses_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    tools = normalized.get("tools")
    if not isinstance(tools, list):
        return normalized
    normalized_tools: list[Any] = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and "name" not in tool
        ):
            function = dict(tool["function"])
            flattened = {"type": "function", **function}
            normalized_tools.append(flattened)
            continue
        normalized_tools.append(tool)
    normalized["tools"] = normalized_tools
    return normalized


def _filtered_headers(headers: Any) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered


def _write_json_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message}}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_json_payload(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_chunked_stream(handler: BaseHTTPRequestHandler, upstream: requests.Response) -> None:
    try:
        for chunk in upstream.iter_content(chunk_size=8192):
            if not chunk:
                continue
            handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
            handler.wfile.write(chunk)
            handler.wfile.write(b"\r\n")
            handler.wfile.flush()
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        # Codex occasionally abandons an HTTP stream after it already has the
        # terminal event. Treat that as a cancelled client, not a proxy crash.
        return
    finally:
        upstream.close()


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredValidationError(
            message="Invalid JSON request body",
            issues=[ValidationIssue(field="body", message=str(exc))],
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredValidationError(
            message="Invalid JSON request body",
            issues=[ValidationIssue(field="body", message="must decode to a JSON object", value=payload)],
        )
    return payload


def _validate_load_tuned_config_payload(payload: dict[str, Any]) -> str:
    bundle_path = payload.get("bundle_path")
    issues: list[ValidationIssue] = []
    if not isinstance(bundle_path, str) or not bundle_path.strip():
        issues.append(ValidationIssue(field="bundle_path", message="must be a non-empty string", value=bundle_path))
    if issues:
        raise StructuredValidationError(message="Invalid load_tuned_config payload", issues=issues)
    return bundle_path.strip()


def _validate_invalidate_payload(payload: dict[str, Any]) -> str:
    weight_version_id = payload.get("weight_version_id")
    issues: list[ValidationIssue] = []
    if not isinstance(weight_version_id, str) or not weight_version_id.strip():
        issues.append(
            ValidationIssue(
                field="weight_version_id",
                message="must be a non-empty string",
                value=weight_version_id,
            )
        )
    if issues:
        raise StructuredValidationError(message="Invalid invalidate payload", issues=issues)
    return weight_version_id.strip()


def build_proxy_handler(
    upstream_base_url: str,
    *,
    state_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    state_store = RuntimeStateStore(state_root or Path.cwd() / "output" / "serving_state")
    registry = load_registry(registry_path) if registry_path is not None else {}

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            _write_json_error(self, 403, "Blocked by codex-bench-proxy: inference paths only")

        def do_POST(self) -> None:  # noqa: N802
            if self.path in ADMIN_PATHS:
                self._handle_admin_request()
                return
            if not is_inference_path(self.path):
                _write_json_error(self, 403, "Blocked by codex-bench-proxy: inference paths only")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = raw_body
            headers = _filtered_headers(self.headers)
            if self.path == "/v1/responses":
                try:
                    request_json = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    _write_json_error(self, 400, "Invalid JSON request body")
                    return
                payload = json.dumps(normalize_responses_request_payload(request_json)).encode("utf-8")
                headers["Content-Type"] = "application/json"
            try:
                upstream = requests.post(
                    f"{upstream_base_url}{self.path}",
                    data=payload,
                    headers=headers,
                    timeout=600,
                    stream=True,
                )
            except requests.RequestException as exc:
                _write_json_error(self, 502, f"Upstream inference request failed: {exc}")
                return

            self.send_response(upstream.status_code)
            response_headers = _filtered_headers(upstream.headers)
            if upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
                response_headers["Transfer-Encoding"] = "chunked"
                response_headers.pop("Content-Length", None)
            else:
                response_headers["Content-Length"] = str(len(upstream.content))
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()

            if response_headers.get("Transfer-Encoding") == "chunked":
                _write_chunked_stream(self, upstream)
                return

            self.wfile.write(upstream.content)
            self.wfile.flush()

        def _handle_admin_request(self) -> None:
            try:
                payload = _read_json_body(self)
                if self.path == "/admin/load_tuned_config":
                    bundle_path = _validate_load_tuned_config_payload(payload)
                    bundle = load_tuned_config_bundle(bundle_path)
                    if registry and bundle.model_id not in registry:
                        raise StructuredValidationError(
                            message="Invalid load_tuned_config payload",
                            issues=[
                                ValidationIssue(
                                    field="bundle_path",
                                    message=f"bundle model_id {bundle.model_id!r} is not present in registry",
                                    value=bundle_path,
                                )
                            ],
                        )
                    state_store.activate_bundle(bundle_path, bundle)
                    _write_json_payload(
                        self,
                        200,
                        {
                            "status": "loaded",
                            "bundle_id": bundle.bundle_id,
                            "model_id": bundle.model_id,
                            "weight_version_id": bundle.weight_version_id,
                            "bundle_path": str(Path(bundle_path)),
                        },
                    )
                    return
                if self.path == "/admin/invalidate":
                    weight_version_id = _validate_invalidate_payload(payload)
                    state = state_store.record_invalidate(weight_version_id=weight_version_id)
                    flush_status = "not_attempted"
                    try:
                        response = requests.post(
                            f"{upstream_base_url}/reset_prefix_cache",
                            timeout=10,
                            headers=_filtered_headers(self.headers),
                        )
                        response.raise_for_status()
                        flush_status = "flushed"
                    except requests.RequestException:
                        flush_status = "unreachable"
                    _write_json_payload(
                        self,
                        200,
                        {
                            "status": "invalidated",
                            "weight_version_id": weight_version_id,
                            "state": state.status,
                            "invalidate_count": state.invalidate_count,
                            "flush_prefix_cache": flush_status,
                        },
                    )
                    return
            except StructuredValidationError as exc:
                _write_json_payload(self, 400, exc.as_error_payload())
                return
            except Exception as exc:
                _write_json_payload(
                    self,
                    500,
                    {
                        "error": {
                            "code": "internal_error",
                            "message": str(exc),
                        }
                    },
                )
                return
            _write_json_error(self, 404, "Unknown admin endpoint")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return ProxyHandler


def run_proxy_server(
    *,
    listen_host: str,
    listen_port: int,
    upstream_base_url: str,
    pid_file: Path | None = None,
    state_root: Path | None = None,
    registry_path: Path | None = None,
) -> None:
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    server = ThreadingHTTPServer(
        (listen_host, listen_port),
        build_proxy_handler(upstream_base_url, state_root=state_root, registry_path=registry_path),
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference-only proxy for Codex -> vLLM traffic")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[PROXY-INIT] pid={os.getpid()} listen={args.listen_host}:{args.listen_port} "
                f"upstream={args.upstream_base_url}\n"
            )
    run_proxy_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_base_url=args.upstream_base_url,
        pid_file=args.pid_file,
        state_root=args.state_root,
        registry_path=args.registry_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
