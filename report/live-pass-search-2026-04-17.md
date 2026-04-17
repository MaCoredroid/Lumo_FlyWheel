# Live Pass Search Report - 2026-04-17

## Outcome

No passing real task was found because no new live family/variant run could be started under the current machine state.

The local vLLM serve path was blocked before the first family/variant attempt because CUDA reported only about `10.68 / 117.51 GiB` free, which is below the launcher floor and far below what the configured `qwen3.5-27b` FP8 runtime can realistically use.

## Repo Change Made

Patched `src/lumo_flywheel_serving/model_server.py` so the initial prelaunch VRAM wait uses the minimum viable floor instead of the full configured `gpu_memory_utilization`. This lets the existing low-memory retry ladder engage when the host is temporarily below the target utilization.

Updated the matching expectations in `tests/test_model_server.py`.

## Validation

Required regression command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py tests/test_smoke_codex_long_variant.py
```

Outcome:

- `134 passed in 0.34s`

Targeted model-server checks for the patched behavior:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_model_server.py -k 'start_waits_for_vram_release_before_first_launch or start_waits_against_lowered_gpu_memory_target_before_retry or switch_model_restarts_when_sleep_mode_state_is_stale'
```

Outcome:

- `3 passed`

Note:

- `tests/test_model_server.py` as a whole currently has six unrelated `StopIteration` failures in `_wait_ready` tests. Those failures were present outside the patched launch-path assertions and were not modified here.

## Live Attempt Record

Ordered attempts made during this session:

1. Local vLLM bring-up for live-task execution: `env LUMO_HOST_MEMORY_RECOVERY=0 ./.venv/bin/python -m lumo_flywheel_serving.cli serve qwen3.5-27b`
   Result: blocked. The process remained inside `_wait_vram_free(...)` because the runtime probe returned `(10.68, 117.51)` GiB free/total and `/health` on `127.0.0.1:8000` never came up.

Family/variant live-task attempts started in this session:

- None. `scripts/run_live_codex_long_task.py` could not be run meaningfully against any pack entry because the required local vLLM endpoint was unavailable.

## Evidence

- `torch.cuda.mem_get_info()` via the repo's serving image returned `(10.68, 117.51)`
- `curl http://127.0.0.1:8000/health` failed while serve was waiting
- `docker ps` showed no active vLLM container
- `nvidia-smi` showed no running compute processes, so the missing free memory appears to be held outside an obvious kill-safe repo-owned process

## Successful Run Data

- First passing family/variant: none
- Exact successful live command: none
- LLD-04 telemetry clean on successful run: not applicable
- Measured one-task elapsed time for first successful run: not applicable

## Remaining Gaps

- Free enough GPU memory for `qwen3.5-27b` so the local vLLM endpoint can start
- Re-run live pack attempts once `/health` on `127.0.0.1:8000` is stable
- Find the first family/variant with `pass=true`, `telemetry_summary.n_tasks=1`, and `anomalies=[]`
