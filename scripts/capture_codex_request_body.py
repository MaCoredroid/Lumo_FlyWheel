#!/usr/bin/env python3
"""One-shot recorder proxy: accept one Codex /v1/responses request, dump the
verbatim request_json to disk, forward to upstream vLLM, return the response,
exit. Used by Track B Round 4a Phase 1 to capture the static Codex CLI system
prompt for decomposition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests


CAPTURED: dict[str, Any] = {"event": threading.Event(), "path_out": None}


class Handler(BaseHTTPRequestHandler):
    upstream_base_url = ""
    capture_path: Path | None = None

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet default access log; we have richer prints.
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        if self.path == "/v1/responses" and Handler.capture_path is not None and not CAPTURED["event"].is_set():
            try:
                request_json = json.loads(body.decode("utf-8"))
            except Exception:
                request_json = None
            if isinstance(request_json, dict):
                Handler.capture_path.write_text(
                    json.dumps(request_json, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                CAPTURED["path_out"] = str(Handler.capture_path)
                CAPTURED["event"].set()
                print(f"[recorder] captured request body to {Handler.capture_path}", file=sys.stderr)

        # Forward to upstream
        url = f"{Handler.upstream_base_url}{self.path}"
        forward_headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
        try:
            upstream = requests.post(url, data=body, headers=forward_headers, timeout=600, stream=True)
        except requests.RequestException as exc:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"upstream error: {exc}".encode())
            return
        self.send_response(upstream.status_code)
        for k, v in upstream.headers.items():
            if k.lower() in {"transfer-encoding", "content-encoding", "content-length"}:
                continue
            self.send_header(k, v)
        # Force chunked transfer for SSE
        if upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    if upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    else:
                        self.wfile.write(chunk)
            if upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
                self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        # /v1/models -> 403 like the live proxy, so codex skips model refresh.
        if self.path.startswith("/v1/models"):
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            body = b'{"error": "Blocked by codex-bench-recorder: inference paths only"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Pass-through GET (health, metrics) — useful for codex's startup probes.
        url = f"{Handler.upstream_base_url}{self.path}"
        try:
            upstream = requests.get(url, headers={k: v for k, v in self.headers.items() if k.lower() != "host"}, timeout=30)
        except requests.RequestException as exc:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"upstream error: {exc}".encode())
            return
        self.send_response(upstream.status_code)
        for k, v in upstream.headers.items():
            if k.lower() in {"transfer-encoding", "content-encoding", "content-length"}:
                continue
            self.send_header(k, v)
        body = upstream.content
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=8024)
    p.add_argument("--upstream-base-url", default="http://127.0.0.1:9950")
    p.add_argument("--out", required=True, help="Path to write captured request_json")
    p.add_argument("--exit-after-capture", action="store_true", help="Exit once one body has been captured")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Handler.upstream_base_url = args.upstream_base_url.rstrip("/")
    Handler.capture_path = out
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    print(f"[recorder] listening on {args.listen_host}:{args.listen_port} -> {args.upstream_base_url}", file=sys.stderr)
    print(f"[recorder] capture path: {out}", file=sys.stderr)
    if args.exit_after_capture:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        CAPTURED["event"].wait()
        print(f"[recorder] capture complete; shutting down", file=sys.stderr)
        # Give in-flight forwarding a moment to finish before shutdown
        import time as _t
        _t.sleep(2)
        server.shutdown()
        return 0
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
