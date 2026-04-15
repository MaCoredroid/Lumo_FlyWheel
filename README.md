# Lumo FlyWheel

Local vLLM serving layer implementation for the Lumo FlyWheel specs.

## Quick start

1. `make venv`
2. `make bootstrap-runtime`
3. `make download-qwen35-27b`
4. `make smoke`

The project-local Codex provider config lives in `.codex/config.toml` and points Codex CLI at `http://127.0.0.1:8000/v1` with `wire_api = "responses"`.

The serving stack is pinned to `vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`, which reports vLLM `0.19.0` on this host.

## Current machine status

The launcher now auto-retries the Qwen 27B startup path with these machine-specific fallbacks when the GB10 host rejects the spec defaults:

- `--kv-cache-dtype auto` when vLLM rejects `fp8_e5m2` with the FP8 checkpoint
- lower `--gpu-memory-utilization` values (`0.85`, then `0.65`) when free VRAM is below the requested budget
- `--enforce-eager` when startup dies inside the FP8 CUTLASS path

On this machine, those fallbacks get the model through weight load, but the pinned vLLM `0.19.0` image still fails during the post-load profile run with `torch.ops._C.cutlass_scaled_mm(...): RuntimeError: Error Internal`. That is an upstream runtime blocker on this GB10 host, not a missing repo scaffold.
