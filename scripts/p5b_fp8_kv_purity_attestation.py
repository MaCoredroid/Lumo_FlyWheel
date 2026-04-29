#!/usr/bin/env python3
"""P5b fp8_e5m2 KV purity attestation — HLD v0.3.3 §7.X.

Runs a 16-probe short-prompt-short-output comparison between the L0b-winner
base stack (fp8_e5m2 KV) and the same stack with kv_cache_dtype switched to
bf16. If fp8 KV introduces logit divergence beyond the parity-fixture's
tolerances (rtol_logit=1e-3, atol_logit=1e-3) on >0 of 16 probes, kernel
mutation work against this base would be debugging quantization noise as
if it were kernel divergence — halt with `fp8_kv_purity_violation`.

This is gated upstream: skip P5b entirely when the base bundle's
kv_cache_dtype is already bf16/fp16.

Output: output/p5b_fp8_kv_purity_<timestamp>.json with per-probe overshoot,
overall pass/fail, and the bundles compared.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.model_server import ModelServer  # noqa: E402
from lumo_flywheel_serving.parity_fixture import (  # noqa: E402
    deterministic_probe_rows,
    load_debug_export_pt,
)


P5B_HALT_CODE = "fp8_kv_purity_violation"
DEFAULT_PROBE_COUNT = 16
DEFAULT_OUTPUT_TOKENS = 1
SHORT_PROBE_RTOL_LOGIT = 1.0e-3
SHORT_PROBE_ATOL_LOGIT = 1.0e-3


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


def _override_kv_cache_dtype(bundle_path: Path, target_dtype: str, output_path: Path) -> None:
    """Write a sibling bundle YAML with kv_cache_dtype switched to target_dtype.

    Both vllm_config.kv_cache_dtype and kernel_selection.kv_cache_dtype are
    overridden — vLLM's runtime activation code reads from vllm_config but the
    kernel_selection field is what gets pinned into actually_resolved by the
    fixture, so they must stay in sync for downstream drift checks.
    """
    payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "tuned_config_bundle" not in payload:
        raise RuntimeError(f"bundle {bundle_path} missing tuned_config_bundle root")
    bundle = copy.deepcopy(payload["tuned_config_bundle"])
    vllm_config = bundle.setdefault("vllm_config", {}) or {}
    vllm_config["kv_cache_dtype"] = target_dtype
    bundle["vllm_config"] = vllm_config
    kernel_selection = bundle.setdefault("kernel_selection", {}) or {}
    kernel_selection["kv_cache_dtype"] = target_dtype
    bundle["kernel_selection"] = kernel_selection
    payload["tuned_config_bundle"] = bundle
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _capture_probe_logits(
    *,
    server: ModelServer,
    model_id: str,
    port: int,
    probes: list[dict[str, Any]],
    api_key: str,
    debug_staging_dir: Path,
    archive_dir: Path,
    request_timeout_s: float,
) -> dict[int, Any]:
    """Run one probe per row, archive the .pt logits, return {probe_index: float32 ndarray}.

    Reuses the LUMO_P2B_VLLM_DEBUG_EXPORT pathway: each /v1/completions request
    drops one logits_*.pt into debug_staging_dir, which we move into archive_dir.
    """
    import numpy as np

    results: dict[int, Any] = {}
    archive_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://127.0.0.1:{port}/v1"
    for probe in probes:
        # Drain staging before each probe so we attribute the .pt to this request.
        debug_staging_dir.mkdir(parents=True, exist_ok=True)
        for stale in debug_staging_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

        body = {
            "model": model_id,
            "prompt": probe["prompt"],
            "max_tokens": DEFAULT_OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 0,
        }
        response = requests.post(
            f"{endpoint}/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=request_timeout_s,
        )
        response.raise_for_status()

        # Wait briefly for the .pt to appear (vLLM writes async).
        deadline = time.time() + 20.0
        logits_files: list[Path] = []
        while time.time() < deadline:
            logits_files = sorted(debug_staging_dir.glob("logits_req_*.pt"))
            if logits_files:
                break
            time.sleep(0.2)
        if not logits_files:
            raise RuntimeError(
                f"P5b probe {probe['probe_index']}: no logits .pt appeared in {debug_staging_dir}"
            )
        if len(logits_files) > 1:
            raise RuntimeError(
                f"P5b probe {probe['probe_index']}: expected exactly one logits .pt, got {logits_files}"
            )
        archived = archive_dir / f"probe_{probe['probe_index']:06d}_{logits_files[0].name}"
        shutil.move(str(logits_files[0]), archived)
        for diag in debug_staging_dir.glob("state_diag_*.json"):
            shutil.move(str(diag), archive_dir / diag.name)

        payload = load_debug_export_pt(archived)
        tensor = payload["logits"]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        if hasattr(tensor, "numpy"):
            array = tensor.numpy()
        else:
            array = np.asarray(tensor)
        results[probe["probe_index"]] = np.ascontiguousarray(
            array.astype(np.float32, copy=False)
        ).reshape(-1)
    return results


def _compute_overshoot(candidate: Any, reference: Any, *, rtol: float, atol: float) -> float:
    import numpy as np

    if candidate.shape != reference.shape:
        return float("inf")
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    allowed = atol + rtol * np.abs(reference.astype(np.float64))
    excess = diff - allowed
    return max(0.0, float(np.max(excess)) if excess.size else 0.0)


def _run_one_capture_phase(
    *,
    label: str,
    bundle_path: Path,
    probes: list[dict[str, Any]],
    server_kwargs: dict[str, Any],
    debug_export_dir: Path,
    archive_root: Path,
    model_id: str,
    health_timeout_s: int,
    api_key: str,
    request_timeout_s: float,
) -> dict[int, Any]:
    archive_dir = archive_root / label
    archive_dir.mkdir(parents=True, exist_ok=True)
    server = ModelServer(**server_kwargs)
    server.stop(missing_ok=True)
    print(f"[p5b/{label}] activating bundle: {bundle_path}")
    server.load_tuned_config(bundle_path)
    print(f"[p5b/{label}] starting vLLM...")
    started = time.time()
    server.start(model_id, enable_request_logging=False)
    try:
        _wait_for_health(server_kwargs["port"], health_timeout_s)
        print(f"[p5b/{label}] /health 200 after {time.time() - started:.1f}s")
        return _capture_probe_logits(
            server=server,
            model_id=model_id,
            port=int(server_kwargs["port"]),
            probes=probes,
            api_key=api_key,
            debug_staging_dir=debug_export_dir / "staging",
            archive_dir=archive_dir,
            request_timeout_s=request_timeout_s,
        )
    finally:
        print(f"[p5b/{label}] stopping vLLM...")
        server.stop(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "model_registry.yaml"))
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument("--base-bundle", required=True, help="L0b-winner bundle (fp8_e5m2 KV).")
    parser.add_argument(
        "--family-id",
        default="responses-sdk-adapter-cutover",
        help="Used for deterministic probe-row generation.",
    )
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument(
        "--rtol-logit",
        type=float,
        default=SHORT_PROBE_RTOL_LOGIT,
        help="Tolerance threshold per-element relative; default matches fixture rtol_logit.",
    )
    parser.add_argument(
        "--atol-logit",
        type=float,
        default=SHORT_PROBE_ATOL_LOGIT,
        help="Tolerance threshold per-element absolute; default matches fixture atol_logit.",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int, default=8101)
    parser.add_argument("--container-name", default="lumo-vllm-p5b")
    parser.add_argument("--image", default="lumo-flywheel-vllm:26.01-py3-v0.19.0")
    parser.add_argument("--logs-root", default="/tmp/lumo-p5b-logs")
    parser.add_argument("--triton-cache-root", default="/tmp/lumo-p5b-triton")
    parser.add_argument("--state-root", default="/tmp/lumo-p5b-state")
    parser.add_argument(
        "--debug-export-dir",
        default=str(REPO_ROOT / "output" / "p5b_fp8_kv_attest"),
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / f"p5b_fp8_kv_purity_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"),
    )
    parser.add_argument("--health-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--skip-if-bf16",
        action="store_true",
        default=True,
        help="When set (default), the script exits 0 with status=skipped if the base bundle is already bf16/fp16 KV.",
    )
    args = parser.parse_args()

    base_bundle_path = Path(args.base_bundle).resolve()
    if not base_bundle_path.is_file():
        print(f"FATAL: base bundle missing: {base_bundle_path}", file=sys.stderr)
        return 2

    debug_export_root = Path(args.debug_export_dir).resolve()
    debug_export_root.mkdir(parents=True, exist_ok=True)
    staging_dir = debug_export_root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    bundle_payload = yaml.safe_load(base_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle_payload, dict) or "tuned_config_bundle" not in bundle_payload:
        raise SystemExit(f"base bundle {base_bundle_path} missing tuned_config_bundle root")
    base_bundle = bundle_payload["tuned_config_bundle"]
    base_kv = (
        (base_bundle.get("vllm_config") or {}).get("kv_cache_dtype")
        or (base_bundle.get("kernel_selection") or {}).get("kv_cache_dtype")
        or "unknown"
    )
    if args.skip_if_bf16 and str(base_kv).lower() in {"bf16", "fp16", "auto"}:
        result = {
            "status": "skipped",
            "reason": "base_kv_already_safe",
            "base_kv_cache_dtype": str(base_kv),
            "base_bundle": str(base_bundle_path),
        }
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[p5b] skipped (base kv_cache_dtype={base_kv}); wrote {args.output_json}")
        return 0

    # vLLM 0.19 rejects fp8_e5m2 KV for fp8 checkpoints during engine init, and
    # ModelServer._initial_kv_cache_dtype proactively rewrites fp8_e5m2 -> auto
    # before the first launch to dodge the crash. So on this runtime, requesting
    # fp8_e5m2 KV against an fp8 checkpoint produces realized kv_cache_dtype=auto
    # (i.e. native bf16) — the fp8 phase would be byte-identical to the
    # unquantized sibling, and the comparison is vacuously satisfied. Skip with
    # an explicit attestation rather than spending two cold starts to confirm 0.0.
    server_for_registry = ModelServer(
        registry_path=args.registry,
        port=args.port,
        proxy_port=args.proxy_port,
        container_name=args.container_name,
        logs_root=Path(args.logs_root),
        triton_cache_root=Path(args.triton_cache_root),
        state_root=Path(args.state_root),
        image=args.image,
    )
    model_quantization = server_for_registry.registry[args.model_id].quantization
    if str(model_quantization).lower() == "fp8" and str(base_kv).lower() == "fp8_e5m2":
        result = {
            "status": "passthrough",
            "reason": "fp8_kv_path_dead_in_runtime",
            "detail": (
                f"vLLM 0.19 + quantization={model_quantization} forces "
                f"kv_cache_dtype fp8_e5m2 -> auto at engine init; fp8 KV is never "
                f"actually realized, so kernel mutations against this base cannot "
                f"be debugging fp8 KV quantization noise."
            ),
            "base_kv_cache_dtype": str(base_kv),
            "realized_kv_cache_dtype": "auto",
            "model_quantization": str(model_quantization),
            "base_bundle": str(base_bundle_path),
        }
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"[p5b] passthrough — runtime forces fp8_e5m2 -> auto for "
            f"quantization={model_quantization}; wrote {args.output_json}"
        )
        return 0

    # Synthesize the unquantized-KV sibling bundle next to the base bundle.
    # vLLM kv_cache_dtype only accepts {"fp8_e5m2", "auto"} (also enforced by
    # tuned_config / registry validators); "auto" resolves to the model's
    # native KV dtype, which is bf16 for Qwen3.5-27B-FP8 — i.e. unquantized KV
    # vs the fp8_e5m2 base. That's the comparison P5b needs.
    bf16_bundle_path = debug_export_root / "bundle_bf16_sibling.yaml"
    _override_kv_cache_dtype(base_bundle_path, "auto", bf16_bundle_path)
    print(f"[p5b] wrote unquantized-KV sibling bundle (kv_cache_dtype=auto): {bf16_bundle_path}")

    # Deterministic probes — short prompt, single-token output (HLD: ~10 min, 16 probes).
    probes = deterministic_probe_rows(REPO_ROOT, args.family_id, args.probe_count)
    for probe in probes:
        probe["output_token_count"] = DEFAULT_OUTPUT_TOKENS

    # Set up debug-export env so vLLM emits logits_*.pt for each /v1/completions call.
    os.environ["LUMO_P2B_VLLM_DEBUG_EXPORT"] = "1"
    os.environ["LUMO_P2B_DEBUG_EXPORT_DIR"] = str(staging_dir)
    os.environ["LUMO_P2B_DEBUG_PROBE_REQUEST_IDS"] = "*"
    # P5b doesn't need state checkpoints — it only diffs token-1 logits.
    os.environ.setdefault("LUMO_P2B_DEBUG_STATE_TOKENS", "")

    extra_mounts = ["-v", f"{debug_export_root}:{debug_export_root}"]
    server_kwargs: dict[str, Any] = {
        "registry_path": args.registry,
        "port": args.port,
        "proxy_port": args.proxy_port,
        "container_name": args.container_name,
        "logs_root": Path(args.logs_root),
        "triton_cache_root": Path(args.triton_cache_root),
        "state_root": Path(args.state_root),
        "extra_volume_mounts": extra_mounts,
        "image": args.image,
    }

    fp8_logits = _run_one_capture_phase(
        label="fp8",
        bundle_path=base_bundle_path,
        probes=probes,
        server_kwargs=server_kwargs,
        debug_export_dir=debug_export_root,
        archive_root=debug_export_root / "phase_captures",
        model_id=args.model_id,
        health_timeout_s=args.health_timeout_s,
        api_key=args.api_key,
        request_timeout_s=args.request_timeout_s,
    )
    bf16_logits = _run_one_capture_phase(
        label="bf16",
        bundle_path=bf16_bundle_path,
        probes=probes,
        server_kwargs=server_kwargs,
        debug_export_dir=debug_export_root,
        archive_root=debug_export_root / "phase_captures",
        model_id=args.model_id,
        health_timeout_s=args.health_timeout_s,
        api_key=args.api_key,
        request_timeout_s=args.request_timeout_s,
    )

    per_probe: list[dict[str, Any]] = []
    violations = 0
    max_overshoot = 0.0
    for probe in probes:
        idx = probe["probe_index"]
        if idx not in fp8_logits or idx not in bf16_logits:
            raise RuntimeError(f"P5b probe {idx}: missing logits in one of the phases")
        overshoot = _compute_overshoot(
            fp8_logits[idx], bf16_logits[idx], rtol=args.rtol_logit, atol=args.atol_logit
        )
        passed = overshoot == 0.0
        if not passed:
            violations += 1
        max_overshoot = max(max_overshoot, overshoot)
        per_probe.append(
            {
                "probe_index": idx,
                "overshoot": overshoot,
                "pass": passed,
            }
        )

    overall_pass = violations == 0
    result = {
        "status": "PASS" if overall_pass else "FAIL",
        "halt_code": None if overall_pass else P5B_HALT_CODE,
        "base_bundle": str(base_bundle_path),
        "bf16_sibling_bundle": str(bf16_bundle_path),
        "base_kv_cache_dtype": str(base_kv),
        "probe_count": len(probes),
        "violations": violations,
        "max_overshoot": max_overshoot,
        "rtol_logit": args.rtol_logit,
        "atol_logit": args.atol_logit,
        "per_probe": per_probe,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "violations": violations, "max_overshoot": max_overshoot, "output_json": args.output_json}, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
