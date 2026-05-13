#!/usr/bin/env python3
"""Round 4b — e2e ablation against the v4a baseline.

Runs 3 e2e ablation points (T1 only, T1+T2, T1+T2+T3) using the
canonical v4a measurement protocol (round-start warmup, --repeat 4,
--zero-token-retries 3). Point D ("all on") is the existing v4a
baseline at ``output/track_b_e2e_v4a/round_0/`` — no re-measurement.

For each point:

1. Write file-based ablation flags to the live vLLM container's
   ``/tmp/lumo_track_b_runtime_flags.json``. Flag semantics: ``True``
   = technique disabled, ``False`` = enabled.
2. Record the proxy-capture timestamp window for this point.
3. Invoke ``scripts/run_track_b_e2e_round.py`` with the same args
   v4a used (warmup-policy=round_start, --repeat 4, --zero-token-
   retries 3, --protocol-hash-match, deferred checks).
4. Re-compute clean wallclock via
   ``scripts/build_track_b_e2e_clean_wallclock_summary.py``.
5. Slice ``/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl``
   to this point's timestamp window and run
   ``scripts/build_track_b_per_regime_acceptance.py``.

After the loop completes, flags are restored to all-on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "lumo-vllm-track-b-suffix"
FLAGS_PATH = "/tmp/lumo_track_b_runtime_flags.json"
PROXY_CAPTURE_JSONL = "/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"
WARMUP_SP_JSON = REPO_ROOT / "output/track_b_e2e_v4a/round_0/codex_system_prompt.json"
# v4a_v2 era runtime_config_hash (computed from current vLLM init log;
# includes PR #39055 reasoning-parser patches).
RUNTIME_CONFIG_HASH = "sha256:5ae88ac4e10201f83a617e2bda3f1c07da4c7217c80db5482d317a79dd93b43a"
RESET_PREFIX_CACHE_URL = "http://127.0.0.1:9950/reset_prefix_cache"
ENDPOINT = "http://127.0.0.1:8022/v1"
# Docker-isolated codex per attempt (same template as launch_v4a_v2_round_0_baseline.sh).
# `--user $(id -u):$(id -g)` written here as a fixed 1000:1000 since the
# template is rendered at runtime; if uid:gid differ on a future host
# the launcher script should be adapted.
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

# Cumulative ablation: each row adds one technique relative to the previous.
# Flag value semantics (matches suffix_decoding.py _lumo_track_b_disabled):
#   True  → technique disabled
#   False → technique enabled
POINTS = [
    ("a_t1_only",      {"T2": True,  "T3": True,  "T4": True}),
    ("b_t1_t2",        {"T2": False, "T3": True,  "T4": True}),
    ("c_t1_t2_t3",     {"T2": False, "T3": False, "T4": True}),
]


def write_flags(payload: dict) -> None:
    body = json.dumps(payload)
    subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc",
         f"printf '%s' {body!r} > {FLAGS_PATH}"],
        check=True,
        capture_output=True,
    )
    # Read back to confirm.
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", f"cat {FLAGS_PATH}"],
        check=True, capture_output=True, text=True,
    )
    if json.loads(result.stdout) != payload:
        raise RuntimeError(f"flag write-back mismatch: wrote {payload!r} read {result.stdout!r}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_sweep(label: str, out_root: Path, round_index: int, repeat: int) -> tuple[str, str, int]:
    start_ts = now_iso()
    cmd = [
        ".venv/bin/python", "scripts/run_track_b_e2e_round.py",
        "--round", str(round_index),
        "--runtime-config-hash", RUNTIME_CONFIG_HASH,
        "--codex-command-template", CODEX_TEMPLATE,
        "--warmup-policy", "round_start",
        "--warmup-system-prompt-json", str(WARMUP_SP_JSON),
        "--reset-prefix-cache-url", RESET_PREFIX_CACHE_URL,
        "--zero-token-retries", "3",
        "--clock-skew-ms-p99", "8",
        "--trace-emitter-correctness-verified-at", now_iso(),
        "--protocol-hash-match",
        "--repeat", str(repeat),
        "--timeout-s", "1800",
        "--codex-smoke-timeout-s", "600",
        "--out-root", str(out_root),
        "--endpoint", ENDPOINT,
        "--vllm-request-metrics-jsonl", PROXY_CAPTURE_JSONL,
        "--hypothesis",
            f"Round 4b ablation point {label} (v4a_v2 era): cumulative "
            f"technique enablement on the 11-task v4a corpus under the "
            f"§13-§17 proxy stack + docker-isolated codex. T-flag state "
            f"via runtime flags file at {FLAGS_PATH}.",
        "--config-delta-vs-prior-round",
            "ablation point — runtime flags differ from prior round; "
            "container/sample/runtime_config_hash unchanged from v4a_v2 D-point",
        "--auto-research-agent-recommendation", "n/a",
        "--next-round-proposal", "see closeout",
        "--defer-preflight-checks",
            "vllm_request_metrics_join_available",
            "codex_trace_out_supported",
            "dcgm_profile_fields_available",
            "codex_command_smoke",
    ]
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    print(f"\n=== [{now_iso()}] Sweep {label} -> {out_root} (round={round_index}, repeat={repeat}) ===",
          flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode
    end_ts = now_iso()
    print(f"=== [{end_ts}] Sweep {label} finished rc={rc} ===", flush=True)
    return start_ts, end_ts, rc


def recompute_clean(round_dir: Path) -> int:
    cmd = [
        ".venv/bin/python", "scripts/build_track_b_e2e_clean_wallclock_summary.py",
        "--round-dir", str(round_dir),
        "--max-retries", "3",
    ]
    print(f"\n--- Clean wallclock recompute: {round_dir} ---", flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def slice_and_aggregate_regime(
    label: str, start_ts: str, end_ts: str, out_dir: Path
) -> int:
    sliced = out_dir / f"request_metrics.{label}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(PROXY_CAPTURE_JSONL, "r", encoding="utf-8") as fi, \
         open(sliced, "w", encoding="utf-8") as fo:
        for line in fi:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts_request_received")
            if ts and start_ts <= ts <= end_ts:
                fo.write(s + "\n")
                kept += 1
    print(f"\n--- Sliced proxy capture: {sliced} ({kept} rows) ---", flush=True)
    cmd = [
        ".venv/bin/python", "scripts/build_track_b_per_regime_acceptance.py",
        str(sliced),
        "--out", str(out_dir / "per_regime_acceptance.json"),
        "--text",
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Round 4b e2e ablation sweeps.")
    parser.add_argument("--out-root", default="output/track_b_e2e_v4a_v2_ablation",
                        help="Where each point's round_<N>/ dir lands.")
    parser.add_argument("--repeat", type=int, default=4,
                        help="--repeat passed to run_track_b_e2e_round.py (>= 4).")
    parser.add_argument("--only", default=None,
                        help="Run only this point label (a_t1_only/b_t1_t2/c_t1_t2_t3).")
    args = parser.parse_args()

    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if not WARMUP_SP_JSON.is_file():
        raise SystemExit(f"warmup system prompt missing: {WARMUP_SP_JSON}")

    # Confirm container is live + ablation patch present.
    rc = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc",
         "grep -c '_lumo_track_b_disabled' "
         "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0 or int(rc.stdout.strip() or 0) == 0:
        raise SystemExit("FATAL: _lumo_track_b_disabled patch not present in container suffix_decoding.py")

    results: list[dict] = []
    selected = [p for p in POINTS if args.only is None or p[0] == args.only]
    for idx, (label, flags) in enumerate(selected, start=1):
        write_flags(flags)
        time.sleep(1)
        round_dir = out_root / f"round_{idx}"
        start_ts, end_ts, rc = run_sweep(label, out_root, idx, args.repeat)
        if rc != 0:
            print(f"FATAL: sweep {label} returned rc={rc}; aborting.", flush=True)
            results.append({"label": label, "flags": flags, "rc": rc,
                            "start_ts": start_ts, "end_ts": end_ts})
            break
        recompute_clean(round_dir)
        slice_and_aggregate_regime(label, start_ts, end_ts, round_dir)
        results.append({
            "label": label, "flags": flags, "rc": rc,
            "start_ts": start_ts, "end_ts": end_ts,
            "round_dir": str(round_dir),
        })

    # Restore flags to all-on.
    write_flags({"T2": False, "T3": False, "T4": False})

    summary_path = out_root / "ablation_run_log.json"
    summary_path.write_text(json.dumps({
        "schema": "lumo.track_b.round4b_ablation_run_log.v1",
        "ts": now_iso(),
        "container": CONTAINER,
        "out_root": str(out_root),
        "repeat": args.repeat,
        "points": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nAblation run log: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
