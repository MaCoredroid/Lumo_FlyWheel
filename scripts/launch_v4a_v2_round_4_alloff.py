#!/usr/bin/env python3
"""Round 4 -- all-off vanilla baseline floor for the v4a_v2 ablation.

The 5-point ablation picture is: all-off -> a (T1) -> b (T1+T2) ->
c (T1+T2+T3) -> D (T1+T2+T3+T4). Points a/b/c/D all run on a vLLM
with the suffix spec-decode method active and the T1 session-scoping
source patch applied; they differ only in runtime flags. This round
is the floor below a: vLLM relaunched with NO `spec_decode` (plain
decode, no SuffixDecodingProposer) and a prelaunch shell truncated of
every T1/T2/T3/T4 source patch -- see /tmp/relaunch_track_b_alloff.py.

Mechanically identical to one `run_sweep` iteration of
run_track_b_v4a_e2e_ablation.py, with three differences:

* round index is fixed at 4;
* no runtime-flags file is written (the T2/T3/T4 flags are inert with
  no spec decoder) and the `_lumo_track_b_disabled` patch precheck is
  dropped (the all-off prelaunch never patches suffix_decoding.py);
* the runtime_config_hash is recomputed from the all-off vLLM init log
  rather than reused from the D-point -- the served config genuinely
  differs (spec_decode removed), so the hash must too.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "output/track_b_e2e_v4a_v2_ablation"
ROUND_INDEX = 4
WARMUP_SP_JSON = REPO_ROOT / "output/track_b_e2e_v4a/round_0/codex_system_prompt.json"
RESET_PREFIX_CACHE_URL = "http://127.0.0.1:9950/reset_prefix_cache"
ENDPOINT = "http://127.0.0.1:8022/v1"
PROXY_CAPTURE_JSONL = "/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"
VLLM_INIT_LOG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.5-27b.log"
# Same docker-isolated codex template as the ablation driver.
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


# Written to /tmp first -- the round driver refuses to start if the
# round dir contains any pre-existing file. Copied into round_4/ after
# the sweep completes (see main()).
RCH_MANIFEST_TMP = Path("/tmp/round4_alloff_runtime_config_hash.json")


def compute_runtime_config_hash() -> str:
    subprocess.run(
        [".venv/bin/python", "scripts/build_track_b_runtime_config_hash.py",
         "--log", VLLM_INIT_LOG, "--out", str(RCH_MANIFEST_TMP)],
        check=True, cwd=str(REPO_ROOT),
    )
    digest = json.loads(RCH_MANIFEST_TMP.read_text())["runtime_config_hash"]
    if not digest.startswith("sha256:"):
        digest = "sha256:" + digest
    return digest


def main() -> int:
    if not WARMUP_SP_JSON.is_file():
        raise SystemExit(f"warmup system prompt missing: {WARMUP_SP_JSON}")

    rch = compute_runtime_config_hash()
    print(f"[round_4 all-off] runtime_config_hash = {rch}", flush=True)

    cmd = [
        ".venv/bin/python", "scripts/run_track_b_e2e_round.py",
        "--round", str(ROUND_INDEX),
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
            "Round 4 all-off vanilla baseline (v4a_v2 era): vLLM relaunched "
            "with no spec_decode and a prelaunch shell truncated of every "
            "T1/T2/T3/T4 source patch. Decode floor below ablation point a "
            "on the 11-task v4a corpus under the proxy stack + docker-"
            "isolated codex.",
        "--config-delta-vs-prior-round",
            "spec_decode removed (no SuffixDecodingProposer); prelaunch "
            "shell truncated after the forced tool_choice parser block -- "
            "T1/T2/T3/T4 patches absent. runtime_config_hash recomputed.",
        "--auto-research-agent-recommendation", "n/a",
        "--next-round-proposal", "see closeout",
        "--defer-preflight-checks",
            "vllm_request_metrics_join_available",
            "codex_trace_out_supported",
            "dcgm_profile_fields_available",
            "codex_command_smoke",
            "spec_decode_metrics_exposed",
    ]
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    print(f"=== [{now_iso()}] Round 4 all-off sweep -> {OUT_ROOT}/round_4 ===",
          flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode
    print(f"=== [{now_iso()}] Round 4 all-off sweep finished rc={rc} ===",
          flush=True)

    round_dir = OUT_ROOT / "round_4"
    subprocess.run(
        [".venv/bin/python", "scripts/build_track_b_e2e_clean_wallclock_summary.py",
         "--round-dir", str(round_dir), "--max-retries", "3"],
        cwd=str(REPO_ROOT),
    )
    if RCH_MANIFEST_TMP.is_file() and round_dir.is_dir():
        (round_dir / "runtime_config_hash.json").write_text(
            RCH_MANIFEST_TMP.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
