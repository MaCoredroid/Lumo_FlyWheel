# Lumo FlyWheel

Local vLLM serving layer implementation for the Lumo FlyWheel specs.

## Quick start

1. `make venv`
2. `make bootstrap-runtime`
3. `make download-qwen35-27b`
4. `make smoke` for the direct vLLM API smoke check
5. `make gate1` for the signed-off Codex CLI Gate 1 run

The project-local Codex provider config lives in `.codex/config.toml` and points Codex CLI at `http://127.0.0.1:8000/v1` with `wire_api = "responses"`.

The serving stack now builds a repo-owned image tagged `lumo-flywheel-vllm:26.01-py3-v0.19.0` from [docker/Dockerfile.nvidia-vllm](/home/mark/shared/lumoFlyWheel/docker/Dockerfile.nvidia-vllm), which is derived from the required NVIDIA base image `nvcr.io/nvidia/pytorch:26.01-py3`.

## Codex-Long authored pack

The repo now includes an initial authored Codex-Long scenario pack under [scenario_families](/home/mark/shared/lumoFlyWheel/scenario_families), [verifiers](/home/mark/shared/lumoFlyWheel/verifiers), and [verifier_data](/home/mark/shared/lumoFlyWheel/verifier_data). It is intentionally an initial real pack, not a frozen benchmark release: the repo does not yet meet the signed-off 35-family freeze floor from LLD-13, so `split_assignment.yaml` and `benchmark_manifest.lock` are intentionally absent.

Use `.venv/bin/python scripts/validate_codex_long_assets.py` for structural validation and `.venv/bin/python scripts/smoke_codex_long_variant.py --family <family_id> --variant <variant_id> --expect fail` to run a variant through the build plus Phase 2/Phase 3 grading path on its broken state. The smoke script also accepts `--repo-override <dir>` so later red-team rounds can run candidate fixes through the same path without editing the committed assets.

## GB10 host cleanup

`ModelServer.start()` and `ModelServer.stop()` now run the proven GB10 host-memory recovery sequence automatically before startup and after teardown:

`sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true`

By default the launcher tries `sudo -n` for that cleanup. If you want it to authenticate non-interactively in a fresh shell, export `LUMO_SUDO_PASSWORD` for the session before running `lumoserve` or `make smoke`.

Set `LUMO_HOST_MEMORY_RECOVERY=0` to disable the automatic cleanup hook.

## Current machine status

The launcher now auto-retries the Qwen 27B startup path with these machine-specific fallbacks when the GB10 host rejects the spec defaults:

- `--kv-cache-dtype auto` when vLLM rejects `fp8_e5m2` with the FP8 checkpoint
- lower `--gpu-memory-utilization` values (`0.85`, then `0.65`) when free VRAM is below the requested budget
- `--enforce-eager` when startup dies inside the FP8 CUTLASS path

On this machine, the launcher still falls back through `--kv-cache-dtype auto`, lower `--gpu-memory-utilization`, and `--enforce-eager` when the GB10 host rejects the spec defaults. The remaining live-runtime outcome depends on the NVIDIA-based container build rather than the old generic `vllm/vllm-openai` image path.
