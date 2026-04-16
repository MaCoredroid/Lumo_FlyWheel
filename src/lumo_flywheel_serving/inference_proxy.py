from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

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


def build_proxy_handler(upstream_base_url: str) -> type[BaseHTTPRequestHandler]:
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            _write_json_error(self, 403, "Blocked by codex-bench-proxy: inference paths only")

        def do_POST(self) -> None:  # noqa: N802
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
                try:
                    for chunk in upstream.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    upstream.close()
                return

            self.wfile.write(upstream.content)
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return ProxyHandler


def run_proxy_server(
    *,
    listen_host: str,
    listen_port: int,
    upstream_base_url: str,
    pid_file: Path | None = None,
) -> None:
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    server = ThreadingHTTPServer((listen_host, listen_port), build_proxy_handler(upstream_base_url))
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
