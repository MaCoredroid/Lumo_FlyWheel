#!/usr/bin/env python3
"""Verify a DeltaNet parity fixture by running it against an unmutated kernel.

Brings up vLLM with the bind-mounted kernel and a chosen base bundle's
tuned config applied (no patch), then runs run_parity_probe() against
the fixture. The expected outcome is `pass: true` with `overshoot` near
zero — the same kernel that captured the fixture is being asked to
re-derive the same logits, so any divergence reveals either (a) the
fixture is misaligned with the runtime (kernel_selection mismatch,
weight-version drift, prefix caching contamination), or (b) vLLM's
debug-export hooks are non-deterministic for the L0c probe path.

This is the "no-mutation control" for L0c: if it fails here, no kernel
mutation can ever pass parity, so L0c rounds against this fixture would
spend agent budget chasing ghost divergences.
"""
from __future__ import annotations

import argparse
import json
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
from lumo_flywheel_serving.parity_probe import run_parity_probe


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
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument(
        "--fixture-dir",
        default=str(REPO_ROOT / "benchmark_blueprints" / "families" / "responses-sdk-adapter-cutover" / "parity_fixture"),
    )
    parser.add_argument("--kernel-target", choices=["deltanet", "gatedattn"], default="deltanet")
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
    parser.add_argument("--container-name", default="lumo-vllm-fixture-verify")
    parser.add_argument("--image", default="lumo-flywheel-vllm:26.01-py3-v0.19.0")
    parser.add_argument("--logs-root", default="/tmp/lumo-fixture-verify-logs")
    parser.add_argument(
        "--triton-cache-root",
        default="/tmp/lumo-fixture-rebuild-triton",
        help=(
            "Triton autotune cache directory. MUST be the same path used by "
            "regenerate_deltanet_parity_fixture.py at fixture-build time — "
            "Triton picks marginally different kernel configs per cold start, "
            "and that drift surfaces as ~bf16-LSB cross-session divergence in "
            "the parity probes (max abs diff ≈ 0.32, mean ≈ 0.05). Sharing "
            "the cache pins the compiled kernels so capture and verify "
            "produce byte-identical logits."
        ),
    )
    parser.add_argument("--state-root", default="/tmp/lumo-fixture-verify-state")
    parser.add_argument(
        "--debug-export-dir",
        default=str(REPO_ROOT / "output" / "p2b_fixture_verify"),
    )
    parser.add_argument("--health-timeout-s", type=int, default=900)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    base_bundle = Path(args.base_bundle).resolve()
    fixture_dir = Path(args.fixture_dir).resolve()
    kernel_source = Path(args.kernel_source_path).resolve()
    debug_export_dir = Path(args.debug_export_dir).resolve()
    for path, label in [(base_bundle, "base bundle"), (kernel_source, "kernel source"), (fixture_dir, "fixture dir")]:
        if not path.exists():
            print(f"FATAL: {label} missing: {path}", file=sys.stderr)
            return 2

    Path(args.logs_root).mkdir(parents=True, exist_ok=True)
    Path(args.triton_cache_root).mkdir(parents=True, exist_ok=True)
    Path(args.state_root).mkdir(parents=True, exist_ok=True)
    debug_export_dir.mkdir(parents=True, exist_ok=True)

    extra_mounts = [
        "-v", f"{kernel_source}:{args.kernel_container_path}",
        "-v", f"{debug_export_dir}:{debug_export_dir}",
    ]
    print(f"[verify] bind-mount: {kernel_source} -> {args.kernel_container_path}")
    print(f"[verify] bind-mount: {debug_export_dir} (host=container)")

    # The L0c parity probe path needs LUMO_P2B_* env vars to make vLLM emit .pt
    # exports for each /v1/completions request the probe submits. The staging
    # directory must match run_parity_probe()'s view: it derives staging from
    # debug_export_dir (passed below as `<root>/probe`) as `<that>/staging`,
    # so vLLM must write to the same `<root>/probe/staging` path. The host-side
    # path is bind-mounted at the same location inside the container above.
    probe_export_dir = debug_export_dir / "probe"
    staging_dir = probe_export_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LUMO_P2B_VLLM_DEBUG_EXPORT"] = "1"
    os.environ["LUMO_P2B_DEBUG_EXPORT_DIR"] = str(staging_dir)
    os.environ["LUMO_P2B_DEBUG_PROBE_REQUEST_IDS"] = "*"
    os.environ.setdefault("LUMO_P2B_DEBUG_STATE_TOKENS", "1,1024")
    os.environ.setdefault("LUMO_P2B_DEBUG_STRICT", "1")
    print(f"[verify] LUMO_P2B env: {[k for k in os.environ if k.startswith('LUMO_P2B_')]}")

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

    print(f"[verify] activating bundle: {base_bundle}")
    server.load_tuned_config(base_bundle)

    print("[verify] stopping any prior container...")
    server.stop(missing_ok=True)
    print("[verify] starting vLLM...")
    started = time.time()
    server.start(args.model_id, enable_request_logging=False)
    print(f"[verify] start() returned in {time.time() - started:.1f}s")

    try:
        _wait_for_health(args.port, args.health_timeout_s)
        print(f"[verify] /health 200 after {time.time() - started:.1f}s")

        endpoint = f"http://127.0.0.1:{args.port}/v1"
        print(f"[verify] running parity probe against fixture: {fixture_dir}")
        result = run_parity_probe(
            repo_root=REPO_ROOT,
            fixture_dir=fixture_dir,
            kernel_target=args.kernel_target,
            endpoint=endpoint,
            model=args.model_id,
            api_key="EMPTY",
            debug_export_dir=probe_export_dir,
            request_timeout_s=1800.0,
        )

        print("[verify] result:")
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        if not result.pass_:
            print("[verify] FAIL — fixture is misaligned with the runtime under test", file=sys.stderr)
            return 1
        print(f"[verify] OK — fixture is aligned ({result.probes_passed}/{result.probes_total} probes passed)")
        return 0
    finally:
        if not args.keep_running:
            print("[verify] stopping vLLM...")
            server.stop(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
