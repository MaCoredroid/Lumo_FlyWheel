# Qwen 3.6 Availability Audit

Generated: 2026-05-06T01:32:41Z

## Decision

- preferred_target: `Qwen/Qwen3.6-27B-FP8`
- fallback_target: `Qwen/Qwen3.5-27B-FP8`
- status: `preferred_available_pending_local_download`

## Evidence Captured For This Launch

- Hugging Face has a `Qwen/Qwen3.6-27B-FP8` repository with FP8 safetensor shards, Apache-2.0 license metadata, and Qwen-owned namespace.
- The matching dense `Qwen/Qwen3.6-27B` model card lists 27B parameters, hidden size 5120, 64 layers, and the same 16 x (3 DeltaNet + 1 Gated Attention) hybrid layout family used by the Qwen3.5 target.
- The Qwen3.6 model card recommends vLLM >= 0.19.0 and documents a vLLM MTP speculative config path.

## Local Follow-Up

- Download the preferred FP8 checkpoint before live runs.
- Verify tokenizer compatibility against `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl`.
- Re-run the P3a in-process timing pass if local config inspection shows layer/head/layout drift from the model card.
