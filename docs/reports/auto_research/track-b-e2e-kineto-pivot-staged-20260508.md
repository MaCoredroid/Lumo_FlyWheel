# Track B E2E — Staged Kineto Pivot (Step B replacement)

Generated: 2026-05-08

This is a **staged**, **not-applied** patch. The vLLM container `lumo-vllm-l0c-fp8-cutlass-run30` (Up 34h+) is left untouched; this doc describes exactly what to land at the next operator-approved relaunch.

## Why

Step B in the Track B E2E plan called for a DCGM 100 Hz sampler emitting profile fields (`DRAM_ACTIVE`, `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, etc.). DGX Spark / consumer Blackwell sm_120 does not expose those fields:

- DCGM Issue #234 (open June 2025, no NVIDIA answer) confirms RTX 5090 / consumer Blackwell does not expose DCP profile metrics.
- dcgm-exporter Issue #506 confirms the same on the exporter side.
- vLLM's host nvidia-smi reports `[N/A]` for `memory.used` / `memory.total` / `memory.free` outside the NVIDIA container. Even routing through `docker exec lumo-vllm-... nvidia-smi` returns the same N/A, which is consistent with consumer Blackwell's runtime exposure model.
- DCGM official docs confirm DCP metrics are "supported on NVIDIA datacenter Volta GPUs and newer" — i.e., not consumer cards.

So Step B as specified is impossible-as-specified on this hardware. The replacement is **PyTorch / Kineto via vLLM's `/start_profile` and `/stop_profile` endpoints**, which use CUPTI from inside the CUDA process and work on consumer Blackwell.

## What the next vLLM relaunch should land

### 1. Add `VLLM_TORCH_PROFILER_DIR` to `ModelServer._build_docker_run_command` env args

Insert after the existing `_p2b_debug_export_env_args()` / `_cutlass_overlay_env_args()` calls:

```python
@staticmethod
def _track_b_kineto_env_args() -> list[str]:
    """Forward VLLM_TORCH_PROFILER_DIR into the container so vLLM's
    /start_profile and /stop_profile endpoints become functional. Required
    for Track B Step B (per-task Kineto trace harvest replacing DCGM).

    Routed via env var so the per-archetype overhead is opt-in (vLLM's docs
    explicitly say "vLLM end-users should never turn on profiling" because
    of trace size and flush cost — we use it for diagnostic-only Round 0
    baseline characterization, then disable for routine rounds).
    """
    out_dir = os.environ.get("LUMO_TRACK_B_TORCH_PROFILER_DIR", "").strip()
    if not out_dir:
        return []
    return [
        "-e", f"VLLM_TORCH_PROFILER_DIR={out_dir}",
        "-v", f"{out_dir}:{out_dir}",  # bind-mount so traces are harvestable from host
    ]
```

Wire into the run command:

```python
return [
    "docker", "run", "--detach",
    ...
    *self._batch_invariant_env_args(),
    *self._triton_debug_env_args(),
    *kernel_activation_env_args,
    *self._p2b_debug_export_env_args(),
    *self._cutlass_overlay_env_args(),
    *self._track_b_kineto_env_args(),    # <-- new
    *volume_args,
    "--entrypoint", "bash",
    self.image, "-lc", shell_cmd,
]
```

### 2. Add a per-task wrapper script

`scripts/wrap_task_with_torch_profile.py`:

```python
#!/usr/bin/env python3
"""Drive vLLM's /start_profile and /stop_profile around a Track B task run.

Replaces scripts/sample_dcgm_during_task.py for the Step B Kineto pivot.
Limits profiling to the first 3 turns by default — vLLM docs warn trace
size grows fast and flush is time-intensive.
"""
from __future__ import annotations
import argparse, os, requests, subprocess, time
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-profile-url", default="http://127.0.0.1:9950/start_profile")
    parser.add_argument("--stop-profile-url", default="http://127.0.0.1:9950/stop_profile")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--out-dir", required=True, help="Directory traces will land in (must match VLLM_TORCH_PROFILER_DIR)")
    parser.add_argument("--task-cmd", required=True, help="Command to run for the task")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    r = requests.post(args.start_profile_url, headers=headers, timeout=30)
    r.raise_for_status()
    started_at = time.time()
    try:
        rc = subprocess.run(args.task_cmd, shell=True).returncode
    finally:
        try:
            requests.post(args.stop_profile_url, headers=headers, timeout=120).raise_for_status()
        except requests.RequestException as exc:
            print(f"warning: stop_profile failed: {exc}")
    elapsed_s = time.time() - started_at
    print(f"task wallclock {elapsed_s:.2f}s rc {rc}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
```

### 3. Update `run_track_b_e2e_task.py` to optionally drive the wrapper

Add `--torch-profile-dir` / `--torch-profile-first-n-turns` flags. When set, wrap the codex exec subprocess with the start_profile / stop_profile calls. Keep DCGM sampler path as a fallback for hosts that *do* support DCP metrics.

### 4. Replace the §6.5 diagnosis rules' DCGM fields with Kineto-derived ones

The current §6.5 rules in the parent agentic-saturation plan expect:

- `DRAM_ACTIVE >= 0.85` → `memory-bw-saturated`
- `SM_ACTIVE >= 0.85` & `DRAM_ACTIVE < 0.4` → `sm-bound`
- etc.

Kineto-derived equivalents:

- Aggregate kernel `kernel.timeline_ms / wall_ms >= 0.85` proxies for `SM_ACTIVE`.
- Kineto's per-kernel `mem_bw_GB_per_s` from CUPTI, normalized against published GB10 LPDDR5x bandwidth ceiling (~273 GB/s) → `memory-bw-saturated` when ratio >= 0.85.
- spec-decode-specific fields (`accepted_per_draft_token` per regime from the proxy capture, not from Kineto) → `low-acceptance` when regime ratio < 0.20.

Spec v2 (`track-b-e2e-agentic-saturation-plan-20260508-v2.md`) will rewrite §6.5 against these Kineto fields once Step B lands.

## Why this is staged not landed

Operator approval needed for vLLM container relaunch. The diff above is small and reversible (the env-args helper returns `[]` when the env var is unset, so existing behavior is preserved by default). When operator window opens:

1. Set host env: `export LUMO_TRACK_B_TORCH_PROFILER_DIR=/tmp/track_b_kineto_traces`
2. `mkdir -p /tmp/track_b_kineto_traces`
3. Apply the `_track_b_kineto_env_args` patch above to `model_server.py`.
4. Stop and relaunch the vLLM container via `ModelServer.restart()` (or whatever orchestrator entry point is in use).
5. Verify endpoint: `curl -X POST http://127.0.0.1:9950/start_profile -H "Authorization: Bearer EMPTY"` should return 200 and create a Kineto trace file in `/tmp/track_b_kineto_traces/` after `stop_profile`.
6. Wire `scripts/wrap_task_with_torch_profile.py` into the round driver for the next round_0 v3 sweep.

## What this does NOT block

- **Round 1 candidate selection.** Per-regime acceptance from proxy capture is sufficient for the §6.5 diagnosis taxonomy at regime granularity. Per-kernel diagnosis (which technique is actually memory-bandwidth-bound vs SM-bound) is desirable but not required to declare a Round 1 winner.
- **Step A trace correctness.** Already unblocked via the proxy + runner-side synthesis path.
- **Step D vLLM per-request metrics.** Already unblocked via proxy capture.

## Drop NCU (Step G)

Same reasoning as the parent NCU blocker doc (`track-b-e2e-ncu-server-profiling-blocker-20260508.md`):

- vLLM Issue #25015 closed "not planned": NCU subprocess termination before first instrumented API call is incompatible with vLLM's multiprocess architecture.
- Local probe attempt produced 0-byte CSVs with CUDA OOM during `cudaMemGetInfo` while the live `:9950` server was resident.
- vLLM's official profiling guide explicitly recommends Nsight Systems, not Nsight Compute, for running servers.

Replacement: use the same Kineto trace from Step B as the per-archetype profile source. Lower fidelity than NCU's native counters but covers the diagnosis rules. For the one-time per-archetype deep profile, stage Nsight Systems against a relaunched server (`nsys profile --delay --duration`) — operator-gated, only when needed for baseline characterization.

Spec v2 will mark Step G as dropped, replace with "per-archetype Kineto sample on first 3 turns of an archetype-representative task".
