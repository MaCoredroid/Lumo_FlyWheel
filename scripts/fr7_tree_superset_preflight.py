#!/usr/bin/env python3
"""Preflight the non-destructive FR7 tree superset B=4 gate.

This does not launch vLLM or the SWE workload. It checks whether a separate
tree vLLM can be started without touching the protected E5 baseline, then prints
the command block for the real correctness run.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path
from typing import Any


REPO = Path("/home/mark/shared/lumoFlyWheel")
BASELINE_CONTAINER = "lumo-vllm-track-b-suffix"
TARGET_CONTAINER = "lumo-vllm-fr7-tree"
SUBSET = "docs/reports/auto_research/swe-bench-agentic-b4-four-verified-20260530.json"


def _run(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _docker_ps() -> dict[str, str]:
    result = _run([
        "docker",
        "ps",
        "--format",
        "{{.Names}}\t{{.Image}}\t{{.Status}}",
    ])
    containers = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            containers[parts[0]] = f"{parts[1]} {parts[2]}"
    return containers


def _gpu_compute_pids() -> list[str]:
    result = _run([
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader",
    ])
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line.strip() != "[Not Supported]"
    ]


def _remote_subset_exists() -> bool:
    result = _run([
        "ssh",
        "-o",
        "ConnectTimeout=8",
        "alienware",
        f"test -f ~/swe_conc_probe/{SUBSET}",
    ])
    return result.returncode == 0


def _command_block() -> str:
    logs = "/tmp/lumo-fr7-tree-logs"
    return f"""source .lumo.local.env
export LUMO_VLLM_CONTAINER_NAME={TARGET_CONTAINER}
export LUMO_VLLM_PORT=9951
export LUMO_VLLM_PROXY_PORT=8089
export LUMO_VLLM_LOGS_ROOT={logs}
export LUMO_VLLM_TRITON_CACHE_ROOT=/tmp/lumo-fr7-tree-triton
export LUMO_VLLM_STATE_ROOT=/tmp/lumo-fr7-tree-state
export LUMO_GPU_MEMORY_UTILIZATION=0.84
export LUMO_TREE_SPINES=2
export LUMO_CODEX_PROXY_MODELS_URL=http://127.0.0.1:8089/v1/models
export LUMO_VLLM_METRICS_URL=http://127.0.0.1:9951/metrics
export LUMO_SWE_DGX_STEPTRACE=/tmp/lumo-fr7-tree-steptrace.jsonl
export LUMO_TRACK_B_REQUEST_METRICS_OUT=/tmp/lumo-fr7-tree-proxy/request_metrics.jsonl
export LUMO_PER_REQ_SPEC_TRACE={logs}/per_req_spec_trace.jsonl
export LUMO_TREE_ACCEPT_PATH_LOG={logs}/tree_accept_path.jsonl
export LUMO_TREE_PATH_LCP_LOG={logs}/tree_path_lcp_max.jsonl
export LUMO_TUNNEL_LOCAL_PROXY_PORT=8089
export LUMO_TUNNEL_LOCAL_METRICS_PORT=9951
bash scripts/swe_x86_helpers/setup_tmux_infra.sh
.venv/bin/python scripts/run_codex_experiment.py \\
  --exp-tag fr7_tree_lcp_b4_$(date -u +%Y%m%dT%H%M%SZ) \\
  --suite swe --config F --mtp 5 --apply-config \\
  --subset {SUBSET} --limit 4 --concurrency 4 --no-commit --nsight off
scripts/verify_tree_path_lcp_superset.py output/<exp-tag>/tree_path_lcp_max.jsonl \\
  --min-rows 1 --json"""


def preflight() -> dict[str, Any]:
    containers = _docker_ps()
    gpu_pids = _gpu_compute_pids()
    ports = {port: _port_open(port) for port in (8089, 9951)}
    issues: list[str] = []
    if TARGET_CONTAINER == BASELINE_CONTAINER:
        issues.append("target container equals protected baseline container")
    if TARGET_CONTAINER in containers:
        issues.append(f"target container already exists: {TARGET_CONTAINER}")
    if BASELINE_CONTAINER in containers and gpu_pids:
        issues.append(
            "active GPU compute process exists while protected baseline is live; "
            "refusing to start a second 27B vLLM"
        )
    for port, open_ in ports.items():
        if open_:
            issues.append(f"required isolated port is already listening: {port}")
    if not _remote_subset_exists():
        issues.append(f"remote SWE subset missing: ~/swe_conc_probe/{SUBSET}")
    return {
        "ok": not issues,
        "issues": issues,
        "baseline_container": containers.get(BASELINE_CONTAINER),
        "target_container": containers.get(TARGET_CONTAINER),
        "gpu_compute_pids": gpu_pids,
        "ports_open": ports,
        "command": _command_block(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = preflight()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "OK" if result["ok"] else "NOT SAFE"
        print(f"FR7 tree superset preflight: {status}")
        for issue in result["issues"]:
            print(f"- {issue}")
        print("\nCommand when safe:\n")
        print(result["command"])
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
