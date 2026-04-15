# Lumo FlyWheel

Local vLLM serving layer implementation for the Lumo FlyWheel specs.

## Quick start

1. `make venv`
2. `make bootstrap-runtime`
3. `make download-qwen35-27b`
4. `make smoke`

The project-local Codex provider config lives in `.codex/config.toml` and points Codex CLI at `http://127.0.0.1:8000/v1` with `wire_api = "responses"`.

The serving stack now builds a repo-owned image tagged `lumo-flywheel-vllm:26.01-py3-v0.19.0` from [docker/Dockerfile.nvidia-vllm](/home/mark/shared/lumoFlyWheel/docker/Dockerfile.nvidia-vllm), which is derived from the required NVIDIA base image `nvcr.io/nvidia/pytorch:26.01-py3`.

## Current machine status

The launcher now auto-retries the Qwen 27B startup path with these machine-specific fallbacks when the GB10 host rejects the spec defaults:

- `--kv-cache-dtype auto` when vLLM rejects `fp8_e5m2` with the FP8 checkpoint
- lower `--gpu-memory-utilization` values (`0.85`, then `0.65`) when free VRAM is below the requested budget
- `--enforce-eager` when startup dies inside the FP8 CUTLASS path

On this machine, the launcher still falls back through `--kv-cache-dtype auto`, lower `--gpu-memory-utilization`, and `--enforce-eager` when the GB10 host rejects the spec defaults. The remaining live-runtime outcome depends on the NVIDIA-based container build rather than the old generic `vllm/vllm-openai` image path.
