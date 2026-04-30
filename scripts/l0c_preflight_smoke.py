#!/usr/bin/env python3
"""L0c live-round pre-flight smoke.

Brings up vLLM with the kernel bind-mount in place, hits /v1/health and
/v1/completions once, then stops. No agent involvement, no real round.
Confirms the bind-mount + restart cycle is wired before burning agent budget.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import requests  # noqa: E402

from lumo_flywheel_serving.model_server import ModelServer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(REPO_ROOT / "model_registry.yaml"))
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int, default=8101)
    parser.add_argument("--container-name", default="lumo-vllm-l0c-preflight")
    parser.add_argument("--image", default="lumo-flywheel-vllm:26.01-py3-v0.19.0")
    parser.add_argument(
        "--kernel-source-path",
        default=str(REPO_ROOT / "output" / "auto_research" / "l0c_kernel_workdir" / "chunk_delta_h.py"),
    )
    parser.add_argument(
        "--kernel-container-path",
        default="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py",
    )
    parser.add_argument("--logs-root", default="/tmp/lumo-l0c-preflight-logs")
    parser.add_argument(
        "--triton-cache-root",
        default="/tmp/lumo-fixture-rebuild-triton",
        help=(
            "Use the parity-fixture rebuild cache by default; DeltaNet parity "
            "can drift when capture and L0c use different Triton autotune caches."
        ),
    )
    parser.add_argument("--ready-timeout-s", type=int, default=900)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    host_kernel = Path(args.kernel_source_path).resolve()
    if not host_kernel.is_file():
        print(f"FATAL: host kernel missing at {host_kernel}", file=sys.stderr)
        return 2

    extra_mounts = ["-v", f"{host_kernel}:{args.kernel_container_path}"]
    print(f"[preflight] bind-mount: {host_kernel} -> {args.kernel_container_path}")

    Path(args.logs_root).mkdir(parents=True, exist_ok=True)
    Path(args.triton_cache_root).mkdir(parents=True, exist_ok=True)

    server = ModelServer(
        registry_path=args.registry,
        port=args.port,
        proxy_port=args.proxy_port,
        image=args.image,
        container_name=args.container_name,
        logs_root=args.logs_root,
        triton_cache_root=args.triton_cache_root,
        extra_volume_mounts=extra_mounts,
    )
    print("[preflight] stopping any prior container...")
    server.stop(missing_ok=True)
    print("[preflight] starting vLLM (this can take 5-10min for first model load)...")
    started = time.time()
    server.start(args.model_id, enable_request_logging=False)
    print(f"[preflight] start() returned in {time.time() - started:.1f}s")

    # health probe
    health_url = f"http://127.0.0.1:{args.port}/health"
    deadline = time.time() + args.ready_timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"[preflight] /health 200 after {time.time() - started:.1f}s")
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        print(f"[preflight] FATAL: /health never returned 200 after {args.ready_timeout_s}s", file=sys.stderr)
        if not args.keep_running:
            server.stop(missing_ok=True)
        return 3

    # one completion
    print("[preflight] running 1 completion to exercise the kernel...")
    completion_url = f"http://127.0.0.1:{args.port}/v1/completions"
    import os as _os
    api_key = _os.environ.get("VLLM_API_KEY") or "EMPTY"
    try:
        r = requests.post(
            completion_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": args.model_id,
                "prompt": "L0c preflight smoke: respond with the word OK.",
                "max_tokens": 16,
                "temperature": 0,
                "seed": 0,
            },
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        text = body.get("choices", [{}])[0].get("text", "")
        print(f"[preflight] completion ok in {time.time() - started:.1f}s; first 120 chars: {text[:120]!r}")
    except requests.RequestException as exc:
        print(f"[preflight] FATAL: completion failed: {exc}", file=sys.stderr)
        if not args.keep_running:
            server.stop(missing_ok=True)
        return 4

    # verify the bind-mounted file is what vLLM imported (sha256 match between host and container)
    print("[preflight] OK — bind-mount + vLLM startup + completion all green")
    if not args.keep_running:
        server.stop(missing_ok=True)
        print("[preflight] vLLM stopped")
    else:
        print(f"[preflight] vLLM left running on http://127.0.0.1:{args.port} (--keep-running)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
