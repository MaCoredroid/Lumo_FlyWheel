# FR10 P0 cu130 Batch-Invariant Baseline

Date: 2026-06-03

## Status

P0 baseline is booted on `fr10-gdn-tree-kernel` using the digest-pinned cu130-nightly stack. This is the canonical Gate B target for later tree-kernel greedy decode comparison.

## Image And Stack

- Image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- Local image ID: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`
- vLLM: `0.19.2rc1.dev134+gfe9c3d6c5`
- Torch/CUDA/Triton/FlashInfer from stack probe: torch `2.11.0+cu130`, CUDA `13.0`, Triton `3.6.0`, FlashInfer `0.6.8.post1`
- Device: `NVIDIA GB10`

## Launch

Container: `fr10-cu130-p0-s1`

```bash
docker run -d --name fr10-cu130-p0-s1 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 9950:8000 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -v /models:/models \
  -e VLLM_BATCH_INVARIANT=1 -e VLLM_SERVER_DEV_MODE=1 \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 8000 --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --attention-backend FLASH_ATTN --gdn-prefill-backend triton \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
```

Notes:

- `kv_cache_dtype` is intentionally omitted, so the engine reports `kv_cache_dtype=auto`. This is the E3/E5-effective KV behavior; forcing `fp8_e5m2` now hard-errors on fp8 checkpoints.
- `VLLM_BATCH_INVARIANT=1` required an explicit supported attention backend. `--attention-backend FLASH_ATTN` boots and keeps GDN pinned to Triton/FLA.
- `VLLM_SERVER_DEV_MODE=1` is required for `POST /reset_prefix_cache`; without it, the route exists in source but is not mounted and returns 404.

## Boot Evidence

- Health reached `200` at 17:34:14 UTC.
- GDN backend: `Using Triton/FLA GDN prefill kernel`.
- Attention backend: `Using AttentionBackendEnum.FLASH_ATTN backend`.
- CUDA graphs captured:
  - mixed prefill-decode `PIECEWISE=7`
  - decode `FULL=4`
  - graph pool memory `0.54 GiB`
- GPU KV cache size: `228,480 tokens`.
- Reset route: `POST /reset_prefix_cache?reset_running_requests=false&reset_external=false` returned `200`.

## Canonical Streams

Artifacts are under `output/fr10_p0_cu130_boot_batchinv/` and are not committed by repo convention. Preserve these hashes and counts in git:

| Stream | Records | SHA256 |
| --- | ---: | --- |
| `fr10_cu130_p0_s1_batchinv_greedy_tokens.json` | 16 | `b8b1ec327f60e34073fcedf54c8dad402bee47264f650888f3e982176c2e9794` |
| `fr10_cu130_p0_s1_batchinv_greedy_b1_b4_compare.json` | 8 matched prompts | `ebc6a1599ef7f27cf62db5243b00ee66ebfc0d9eeb233b4bdfd1dd8c6ec495c8` |
| `fr10_cu130_p0_s1_batchinv_temp06_b4_samples.json` | 64 | `7d5f0ab0f53b6fa7adab7bf650264d717b16bbb0ef2db39a8059a80fd521f113` |
| `fr10_cu130_p0_s1_batchinv_temp06_logprobs.json` | 16 | `06d80a8fe814154de0bd13c128cabeee363cc404ca0d1ab016049a9f33b73324` |
| `fr10_cu130_p0_s1_batchinv_steptrace_window.jsonl` | 30 | `bb2d6a7dd6ab3aff663fae2df940772fe4a3bd80a7b2ed1cc3b484edfeaa89c8` |

Greedy B1 vs B4 exact-match result: `true` over 8 prompts, with no missing records and no mismatches.

First greedy token-id prefix for prompt 0:

```text
[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 13, 248044]
```

All canonical stream captures had `reset_prefix_cache_error=null`.

## Metrics Snapshot

Post-stream aggregate spec-decode metrics:

- `spec_decode_num_drafts_total = 398`
- `spec_decode_num_draft_tokens_total = 1990`
- `spec_decode_num_accepted_tokens_total = 1206`
- accepted per position: `{0: 273, 1: 266, 2: 253, 3: 238, 4: 176}`

Bounded live steptrace window around a B4 temp=0.6 load:

- rows: `30`
- window: `44.53515648841858 s`
- deltas: `gen=306`, `prompt=192`, `iter_sum=498`, `iter_cnt=37`, `acc=217`, `draft=375`, `drafts=75`, `dec_sum=16.833757460815832`, `pre_sum=5.048701603198424`
- mean step wall time: `1.203652878065367 s`
- tokens per step: `13.45945945945946`
- accepted per draft token: `0.5786666666666667`

The unmodified cu130 OpenAI server exposes aggregate spec counters via `/metrics`. The older per-request accept side-channel file under `/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl` was stale during this run and is not a live P0 source for this stock container.
