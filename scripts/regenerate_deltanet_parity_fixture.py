#!/usr/bin/env python3
"""Regenerate the deltanet parity fixture against an L0b-winner runtime.

Brings up vLLM with the bind-mounted DeltaNet kernel and a chosen base
bundle's tuned config (kernel_selection / vllm_config / etc.) applied,
then runs build_parity_fixture.py against the live engine, stamping the
bundle's kernel_selection into the new fixture's reference_baseline.

This unblocks L0c rounds whose base bundle's kernel_selection (e.g.
attention_backend=vllm-default) doesn't match the historical fixture
baseline (flash-attn-4) — without that match the parity probe sees
~28x logit drift dominated by config skew, not by kernel mutations.
"""
from __future__ import annotations

import argparse
import os
import subprocess
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "model_registry.yaml"))
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument(
        "--base-bundle",
        required=True,
        help="L0b-winner tuned-config bundle YAML — its kernel_selection becomes the "
        "fixture's reference_baseline, and its vllm_config drives the runtime.",
    )
    parser.add_argument(
        "--kernel-source-path",
        default=str(REPO_ROOT / "output" / "auto_research" / "l0c_kernel_workdir" / "chunk_delta_h.py"),
    )
    parser.add_argument(
        "--kernel-container-path",
        default="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py",
    )
    parser.add_argument("--family-id", default="responses-sdk-adapter-cutover")
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--reproducibility-runs", type=int, default=3)
    parser.add_argument("--kernel-target", choices=["deltanet", "gatedattn", "both"], default="deltanet")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int, default=8101)
    parser.add_argument("--container-name", default="lumo-vllm-fixture-rebuild")
    parser.add_argument("--image", default="lumo-flywheel-vllm:26.01-py3-v0.19.0")
    parser.add_argument("--logs-root", default="/tmp/lumo-fixture-rebuild-logs")
    parser.add_argument("--triton-cache-root", default="/tmp/lumo-fixture-rebuild-triton")
    parser.add_argument("--state-root", default="/tmp/lumo-fixture-rebuild-state")
    parser.add_argument("--health-timeout-s", type=int, default=900)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    base_bundle = Path(args.base_bundle).resolve()
    if not base_bundle.is_file():
        print(f"FATAL: base bundle missing: {base_bundle}", file=sys.stderr)
        return 2
    kernel_source = Path(args.kernel_source_path).resolve()
    if not kernel_source.is_file():
        print(f"FATAL: kernel source missing: {kernel_source}", file=sys.stderr)
        return 2

    Path(args.logs_root).mkdir(parents=True, exist_ok=True)
    Path(args.triton_cache_root).mkdir(parents=True, exist_ok=True)
    Path(args.state_root).mkdir(parents=True, exist_ok=True)

    extra_mounts = ["-v", f"{kernel_source}:{args.kernel_container_path}"]
    print(f"[fixture-rebuild] bind-mount: {kernel_source} -> {args.kernel_container_path}")

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

    print(f"[fixture-rebuild] activating L0b-winner bundle: {base_bundle}")
    server.load_tuned_config(base_bundle)

    print("[fixture-rebuild] stopping any prior container...")
    server.stop(missing_ok=True)
    print(f"[fixture-rebuild] starting vLLM (cold start ~6min)...")
    started = time.time()
    server.start(args.model_id, enable_request_logging=False)
    print(f"[fixture-rebuild] start() returned in {time.time() - started:.1f}s")

    try:
        _wait_for_health(args.port, args.health_timeout_s)
        print(f"[fixture-rebuild] /health 200 after {time.time() - started:.1f}s")

        endpoint = f"http://127.0.0.1:{args.port}/v1"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_parity_fixture.py"),
            "--repo-root", str(REPO_ROOT),
            "--family-id", args.family_id,
            "--probe-count", str(args.probe_count),
            "--reproducibility-runs", str(args.reproducibility_runs),
            "--kernel-target", args.kernel_target,
            "--endpoint", endpoint,
            "--model", args.model_id,
            "--reference-baseline-bundle", str(base_bundle),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        print(f"[fixture-rebuild] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"[fixture-rebuild] build_parity_fixture exited {result.returncode}", file=sys.stderr)
            return result.returncode
        print(f"[fixture-rebuild] OK — fixture regenerated in {time.time() - started:.1f}s")
        return 0
    finally:
        if not args.keep_running:
            print("[fixture-rebuild] stopping vLLM...")
            server.stop(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
