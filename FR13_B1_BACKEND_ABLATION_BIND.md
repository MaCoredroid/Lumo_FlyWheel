# FR13 B=1 Backend Ablation Bind

Date: 2026-06-11 UTC

## Verdict

The B=1 chain-width backend ablation is bound. `tree_mtp` with the 5-node
chain tree boots and engages under `FLASH_ATTN`, but its per-forward speed is
still in the prior slow tree-chain class:

- Native MTP-5: `0.218160 s/fwd`
- Tree chain5 / `TREE_ATTN`: `0.304121 s/fwd`, `1.3940x` native
- Tree chain5 / `FLASH_ATTN`: `0.307007 s/fwd`, `1.4073x` native

This separates the cost away from the `TREE_ATTN` full-attention backend as the
dominant tax. In this chain-width regime, the remaining tree
GDN/replay/state path is the dominant speed cost. No lossless pass is claimed.

The B=1 pass bar before returning to B=4 is explicit: B=1 must satisfy the
historical lossless bar, near-native per-forward speed, and superset
accept/event versus native MTP-5. Current cat9 already fails that superset bar
at `2.1515` accept/event versus native `3.1613`; this backend arm also does
not clear it (`2.8712` versus `3.1613`).

## Run

Run dir: `output/fr13_b1_backend_ablation/` (gitignored local artifact).

Substrate at live run start: `19b21bb2` on `main`.

Shared request window:

- Prompts: `output/fr13_acceptance_ladder/prompts_swe4.json`
- `max_tokens=128`, `temperature=0.0`, `top_p=1.0`, `seed=1313`
- `samples_per_prompt=1`, client `batch_size=1`
- `MAX_NUM_SEQS=1`, `GPU_UTIL=0.82`
- Raw speed basis: vLLM `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`

New backend arm:

- `FLASH_ATTN/tree_mtp`
- 5-node chain:
  `[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0)]`
- `NUM_SPECULATIVE_TOKENS=5`
- `FR13_REPLAY_ROUTE=1`, `FR10_ENABLE_TREE_GDN=1`
- `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`, `FR13_BI_TREE_ATTN=0`
- Heavy diagnostics/captures unset.

The arm was not repeated because the first measured result landed directly in
the prior slow tree-chain speed class and did not leave a close/noisy
pass/fail call.

CUDA graph and backend proof:

- Launch args include `attention_backend: FLASH_ATTN`.
- Engine log says `Using AttentionBackendEnum.FLASH_ATTN backend`.
- Engine log says `Using FlashAttention version 2`.
- Mixed prefill-decode capture reached `100%`.
- Decode FULL capture reached `100%`.
- `Graph capturing finished in 6 secs`.

Tree engagement proof:

- `gpu_tree_metadata_rows=142`, all `reason=ok`.
- All logged active `num_draft_tokens` values were `5`.
- `tree_path_lcp_rows=142`.
- Winner path was always `(0, 1, 2, 3, 4)`.
- `superset_violations_count=0`.

`docker ps` was empty after teardown.

## Decision Table

Measured per-forward uses only `/metrics` counters:
`decode_seconds / spec_drafts`.

| arm | decode seconds | spec drafts | draft tokens | drafts/event | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| native MTP-5 / FLASH_ATTN | 27.051817 | 124 | 620 | 5.0 | 0.218160 | 3.161290 | 18.926640 |
| tree chain5 / TREE_ATTN | 36.798581 | 121 | 605 | 5.0 | 0.304121 | 3.256198 | 13.913580 |
| tree chain5 / FLASH_ATTN | 40.524908 | 132 | 660 | 5.0 | 0.307007 | 2.871212 | 12.634205 |

Ratios:

- Chain5 `TREE_ATTN` / native: `1.394026x`
- Chain5 `FLASH_ATTN` / native: `1.407257x`
- Chain5 `FLASH_ATTN` / chain5 `TREE_ATTN`: `1.009491x`

Interpretation: `FLASH_ATTN` does not move chain5 toward native speed. It is
only `0.9491%` slower than the prior chain5 `TREE_ATTN` arm and remains
`~1.41x` native. Therefore, `TREE_ATTN` full-attention backend cost is not the
dominant B=1 chain-width speed tax; the tree GDN/replay/state path is.

The new arm's lower accept/event is not used to infer speed. Per-forward speed
is computed only from the `/metrics` decode seconds divided by speculative
draft events. It is still part of the B=1 pass bar: chain5 `FLASH_ATTN`
accept/event is `2.871212`, below native `3.161290`, so this arm also fails
superset acceptance.

## Coarse Lossless Evidence

Prompt-token identity versus the reused native probe was true for all four
requests. Served streams still fork from native, so this is not a lossless
classification.

| prompt | first fork position | tree FLASH token | native token |
|---:|---:|---:|---:|
| 0 | 82 | 3772 | 4466 |
| 1 | 11 | 26622 | 12182 |
| 2 | 25 | 13766 | 44675 |
| 3 | 117 | 364 | 413 |

This run did not capture final logits or trigger context at forks and does not
replace the historical S1/S2 margin/floor workflow.

## Artifacts

- Reducer summary:
  `output/fr13_b1_backend_ablation/b1_backend_ablation_reduce.json`
- New backend probe:
  `output/fr13_b1_backend_ablation/chain_flash/chain_flash_greedy_probe.json`
- New backend request metrics:
  `output/fr13_b1_backend_ablation/chain_flash/chain_flash_request_metrics.jsonl`
- Raw metrics snapshots:
  `output/fr13_b1_backend_ablation/chain_flash/metrics_before.txt`,
  `output/fr13_b1_backend_ablation/chain_flash/metrics_after.txt`
- Backend/capture proof:
  `output/fr13_b1_backend_ablation/chain_flash/container_env.txt`,
  `output/fr13_b1_backend_ablation/chain_flash/docker_full.log`
- Reused prior rows:
  `output/fr13_b1_chain_speed_discriminator/b1_chain_speed_discriminator_reduce.json`,
  `output/fr13_b1_current_gate/b1_current_gate_reduce.json`

## Scope Notes

No B=4 work was run or analyzed. The optional `naive_mtp + TREE_ATTN`
diagnostic was not run because the primary arm cleanly answered the backend
attribution question.
