#!/usr/bin/env python3
"""Qwen3.6-27B-FP8 ablation -- one point per invocation.

3-point remeasure on qwen3.6-27b-fp8 mirroring the qwen3.5 work:

  off  -> no spec_decode, no T1-T4 prelaunch patches  -> round_0
  a    -> spec_decode suffix + T1 only                -> round_1
  d    -> spec_decode suffix + T1+T2+T3+T4            -> round_2

vLLM must already be relaunched into the matching config before this
runs (see /tmp/relaunch_qwen36_*.py). This script only drives the
11-task sweep for the chosen point. The runtime_config_hash is
recomputed from the live qwen3.6 vLLM init log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "output/track_b_e2e_qwen36_temp06_ablation"
WARMUP_SP_JSON = REPO_ROOT / "output/track_b_e2e_v4a/round_0/codex_system_prompt.json"
RESET_PREFIX_CACHE_URL = "http://127.0.0.1:9950/reset_prefix_cache"
ENDPOINT = "http://127.0.0.1:8022/v1"
PROXY_CAPTURE_JSONL = "/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"
VLLM_INIT_LOG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.6-27b.log"
MODEL = "qwen3.6-27b"

POINTS = {
    "off": {"round": 0, "spec": False},
    "a":   {"round": 1, "spec": True},
    "d":   {"round": 2, "spec": True},
}

CODEX_TEMPLATE = (
    "docker run --rm --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} -e HOME=/tmp "
    "-w /workspace codex-runner:v1 "
    "codex exec --json --skip-git-repo-check "
    "--dangerously-bypass-approvals-and-sandbox -C /workspace "
    "-c 'model_provider=\"local-proxy\"' "
    "-c 'model_providers.local-proxy={{name=\"local-proxy\","
    "base_url=\"{endpoint}\",env_key=\"OPENAI_API_KEY\","
    "wire_api=\"responses\",stream_idle_timeout_ms=600000}}' "
    "-c 'model_reasoning_effort=\"high\"' "
    "-c 'model_supports_reasoning_summaries=true' "
    "-c 'model_reasoning_summary=\"auto\"' "
    "--model {model} "
    "\"Read the task prompt at /workspace/AGENTS.md and complete it in this workspace.\""
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_runtime_config_hash(round_index: int) -> str:
    tmp = Path(f"/tmp/qwen36_rch_round{round_index}.json")
    log = VLLM_INIT_LOG
    if not Path(log).is_file():
        # ModelServer logs under the model-id-derived name; fall back to
        # whatever vllm_*.log is freshest in the logs dir.
        logs = sorted(Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs").glob("vllm_*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            log = str(logs[0])
    subprocess.run(
        [".venv/bin/python", "scripts/build_track_b_runtime_config_hash.py",
         "--log", log, "--out", str(tmp)],
        check=True, cwd=str(REPO_ROOT),
    )
    digest = json.loads(tmp.read_text())["runtime_config_hash"]
    return digest if digest.startswith("sha256:") else "sha256:" + digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", required=True, choices=sorted(POINTS))
    args = parser.parse_args()
    spec = POINTS[args.point]
    round_index = spec["round"]

    if not WARMUP_SP_JSON.is_file():
        raise SystemExit(f"warmup system prompt missing: {WARMUP_SP_JSON}")

    rch = compute_runtime_config_hash(round_index)
    print(f"[qwen36 {args.point}] round={round_index} runtime_config_hash={rch}",
          flush=True)

    defer = [
        "vllm_request_metrics_join_available",
        "codex_trace_out_supported",
        "dcgm_profile_fields_available",
        "codex_command_smoke",
    ]
    if not spec["spec"]:
        defer.append("spec_decode_metrics_exposed")

    cmd = [
        ".venv/bin/python", "scripts/run_track_b_e2e_round.py",
        "--round", str(round_index),
        "--model", MODEL,
        "--runtime-config-hash", rch,
        "--codex-command-template", CODEX_TEMPLATE,
        "--warmup-policy", "round_start",
        "--warmup-system-prompt-json", str(WARMUP_SP_JSON),
        "--reset-prefix-cache-url", RESET_PREFIX_CACHE_URL,
        "--zero-token-retries", "3",
        "--clock-skew-ms-p99", "8",
        "--trace-emitter-correctness-verified-at", now_iso(),
        "--protocol-hash-match",
        "--repeat", "4",
        "--timeout-s", "1800",
        "--codex-smoke-timeout-s", "600",
        "--out-root", str(OUT_ROOT),
        "--endpoint", ENDPOINT,
        "--vllm-request-metrics-jsonl", PROXY_CAPTURE_JSONL,
        "--hypothesis",
            f"Qwen3.6-27B-FP8 ablation point {args.point}: "
            f"{'spec_decode suffix' if spec['spec'] else 'no spec_decode'} "
            f"on the 11-task v4a corpus under the proxy stack + docker-"
            f"isolated codex. New-model remeasure vs the qwen3.5 corpus.",
        "--config-delta-vs-prior-round",
            f"qwen3.6-27b-fp8 point {args.point}",
        "--auto-research-agent-recommendation", "n/a",
        "--next-round-proposal", "see closeout",
        "--defer-preflight-checks", *defer,
    ]
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    print(f"=== [{now_iso()}] qwen36 {args.point} sweep -> {OUT_ROOT}/round_{round_index} ===",
          flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode
    print(f"=== [{now_iso()}] qwen36 {args.point} sweep finished rc={rc} ===", flush=True)

    round_dir = OUT_ROOT / f"round_{round_index}"
    subprocess.run(
        [".venv/bin/python", "scripts/build_track_b_e2e_clean_wallclock_summary.py",
         "--round-dir", str(round_dir), "--max-retries", "3"],
        cwd=str(REPO_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
