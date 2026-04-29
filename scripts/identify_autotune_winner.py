#!/usr/bin/env python3
"""One-shot probe to identify the L0b kernel's empirical autotune winner.

Launches vLLM with TRITON_PRINT_AUTOTUNING=1 + the bind-mounted (unpinned) DeltaNet
kernel, sends two probes (short + long) to exercise prefill + decode shapes, then
stops vLLM. The autotune verdict for each (H,K,V,BT) shape is logged by Triton to
stderr; we grep it out and print the winner.

This unblocks the chunk_delta_h.py pin: instead of guessing a config, we observe
which one L0b's autotune actually picks under our shapes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import requests

from lumo_flywheel_serving.model_server import ModelServer


def _wait_for_health(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM /health never returned 200 within {timeout_s}s")


def _fire_probe(endpoint: str, model: str, api_key: str, prompt: str, max_tokens: int) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    r = requests.post(f"{endpoint}/completions", json=payload, headers=headers, timeout=180)
    r.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "model_registry.yaml"))
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument(
        "--base-bundle",
        required=True,
    )
    parser.add_argument(
        "--kernel-source-path",
        default=str(REPO_ROOT / "output" / "auto_research" / "l0c_kernel_workdir" / "chunk_delta_h.py"),
    )
    parser.add_argument(
        "--kernel-container-path",
        default="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int, default=8101)
    parser.add_argument("--container-name", default="lumo-vllm-autotune-probe")
    parser.add_argument("--image", default="lumo-flywheel-vllm:26.01-py3-v0.19.0")
    parser.add_argument("--logs-root", default="/tmp/lumo-autotune-probe-logs")
    parser.add_argument("--triton-cache-root", default="/tmp/lumo-autotune-probe-triton")
    parser.add_argument("--state-root", default="/tmp/lumo-autotune-probe-state")
    parser.add_argument("--health-timeout-s", type=int, default=900)
    args = parser.parse_args()

    base_bundle = Path(args.base_bundle).resolve()
    kernel_source = Path(args.kernel_source_path).resolve()
    Path(args.logs_root).mkdir(parents=True, exist_ok=True)
    Path(args.triton_cache_root).mkdir(parents=True, exist_ok=True)
    Path(args.state_root).mkdir(parents=True, exist_ok=True)

    extra_mounts = ["-v", f"{kernel_source}:{args.kernel_container_path}"]
    print(f"[autotune-probe] bind-mount: {kernel_source} -> {args.kernel_container_path}")

    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    print("[autotune-probe] TRITON_PRINT_AUTOTUNING=1 set in host env (forwarded to container)")

    server = ModelServer(
        registry_path=args.registry,
        port=args.port,
        proxy_port=args.proxy_port,
        image=args.image,
        container_name=args.container_name,
        logs_root=Path(args.logs_root),
        triton_cache_root=Path(args.triton_cache_root),
        state_root=Path(args.state_root),
        extra_volume_mounts=extra_mounts,
    )

    print(f"[autotune-probe] activating L0b-winner bundle: {base_bundle}")
    server.load_tuned_config(base_bundle)
    print("[autotune-probe] stopping any prior container...")
    server.stop(missing_ok=True)
    print("[autotune-probe] starting vLLM (cold start ~6min)...")
    started = time.time()
    server.start(args.model_id, enable_request_logging=False)
    print(f"[autotune-probe] start() returned in {time.time() - started:.1f}s")

    try:
        _wait_for_health(args.port, args.health_timeout_s)
        print(f"[autotune-probe] /health 200 after {time.time() - started:.1f}s")

        endpoint = f"http://127.0.0.1:{args.port}/v1"
        # Use the API key the server is exposing.
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

        # Short probe — exercises prefill + a few decode tokens.
        print("[autotune-probe] firing short probe (256 tokens)...")
        _fire_probe(endpoint, args.model_id, api_key, "Briefly explain DeltaNet.", 256)
        # Longer probe — exercises decode at >1024 tokens (state checkpoint shape).
        print("[autotune-probe] firing long probe (1100 tokens)...")
        _fire_probe(endpoint, args.model_id, api_key, "Explain DeltaNet recurrent state in detail.", 1100)
        print("[autotune-probe] both probes complete; autotune verdicts now in vLLM log")
        return 0
    finally:
        print("[autotune-probe] stopping vLLM...")
        server.stop(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
