# FR13 B=1 Chain-Only Speed Discriminator Bind

Date: 2026-06-11 UTC

## Verdict

The clean B=1 chain-only discriminator is bound. It explains the current
`~1.41x` tree/native speed tax as mostly **tree backend/replay/TREE_ATTN/GDN
path cost**, not branch-row verifier width.

- Chain5 tree is still `1.3940x` native MTP-5 per forward on the first window
  and `1.3890x` on the repeat.
- Caterpillar9 is only `1.0117x` slower than chain5 on the first window and
  `1.0123x` on the repeat.
- Chain5 accept/event is close to native and slightly higher:
  `3.2562` versus native `3.1613`; caterpillar9 remains far below at `2.1515`.
- Chain5 repeats deterministically, but the served stream still forks from
  native on all four prompts. No lossless pass is claimed.

Interpretation: branch rows are not the primary speed tax in this B=1 clean
regime. Speed work should focus on the tree backend/replay/TREE_ATTN/GDN path
even at native-like width. Separately, served-stream forks remain a lossless
blocker until classified by the historical S1/S2 margin/floor workflow.

## Run

Run dir: `output/fr13_b1_chain_speed_discriminator/` (gitignored local
artifact).

Substrate at live run start: `32b91caf` on `main`. The reused native and
caterpillar9 rows were run at substrate `fea42f8e`; `fea42f8e..32b91caf` is
docs-only (`FR13_B1_CURRENT_GATE_BIND.md`, `FR13_TRAIL.md`), so no serving code
changed.

Shared request window:

- Prompts: `output/fr13_acceptance_ladder/prompts_swe4.json`
- `max_tokens=128`, `temperature=0.0`, `top_p=1.0`, `seed=1313`
- `samples_per_prompt=1`, client `batch_size=1`
- `MAX_NUM_SEQS=1`
- Raw speed basis: vLLM `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`

New chain-only tree arm:

- `TREE_ATTN/tree_mtp`
- 5-node spine:
  `[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0)]`
- `FR13_REPLAY_ROUTE=1`
- `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`, `FR13_BI_TREE_ATTN=0`
- Heavy diagnostics/captures unset.
- Final bound run used `gpu_memory_utilization=0.82`, matching the current
  gate's native and caterpillar9 boots.

CUDA graph capture was proven live for the chain arm:

- Mixed prefill-decode capture reached `100%`.
- Decode FULL capture reached `100%`.
- `Graph capturing finished in 7 secs`.

`docker ps` was empty after teardown.

## Decision Table

Measured per-forward uses only `/metrics` counters:
`decode_seconds / spec_drafts`.

| arm | decode seconds | spec drafts | draft tokens | drafts/event | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| native MTP-5 | 27.051817 | 124 | 620 | 5.0 | 0.218160 | 3.161290 | 18.926640 |
| tree chain5 | 36.798581 | 121 | 605 | 5.0 | 0.304121 | 3.256198 | 13.913580 |
| tree caterpillar9 | 50.765643 | 165 | 1485 | 9.0 | 0.307671 | 2.151515 | 10.085561 |

Ratios:

- Chain/native: `0.304121 / 0.218160 = 1.394026x`
- Cat9/native: `0.307671 / 0.218160 = 1.410299x`
- Cat9/chain: `0.307671 / 0.304121 = 1.011673x`

Repeat:

| arm | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|
| native MTP-5 rep2 | 0.218764 | 3.161290 | 18.874349 |
| tree chain5 rep2 | 0.303864 | 3.256198 | 13.925343 |
| tree caterpillar9 rep2 | 0.307603 | 2.151515 | 10.087768 |

Repeat ratios:

- Chain/native: `1.389000x`
- Cat9/native: `1.406095x`
- Cat9/chain: `1.012307x`

Draft-width assertions:

- Chain drafts/event: `605 / 121 = 5.0`
- Native drafts/event: `620 / 124 = 5.0`
- Caterpillar9 drafts/event: `1485 / 165 = 9.0`

## Coarse Lossless Evidence

Same-seed repeat determinism:

- Chain main vs rep2: `4/4` prompts byte-identical, 128 tokens each.
- Native and caterpillar9 determinism are reused from
  `FR13_B1_CURRENT_GATE_BIND.md`: both `4/4` byte-identical on repeat.

Served-stream chain vs native first forks:

| prompt | first fork position | chain token | native token |
|---:|---:|---:|---:|
| 0 | 82 | 3772 | 4466 |
| 1 | 11 | 26622 | 12182 |
| 2 | 58 | 4577 | 4466 |
| 3 | 71 | 464 | 2 |

Compared with caterpillar9, chain5 gets closer to native on accept/event and
pushes three first forks later, but it still forks on all four prompts. This is
not a lossless classification and does not replace the historical S1/S2
margin/floor workflow.

## Artifacts

- Reducer summary:
  `output/fr13_b1_chain_speed_discriminator/b1_chain_speed_discriminator_reduce.json`
- Chain probes:
  `output/fr13_b1_chain_speed_discriminator/chain/chain_greedy_probe.json`,
  `output/fr13_b1_chain_speed_discriminator/chain/chain_greedy_rep2_probe.json`
- Raw chain metrics snapshots:
  `output/fr13_b1_chain_speed_discriminator/chain/metrics_before*.txt`,
  `output/fr13_b1_chain_speed_discriminator/chain/metrics_after*.txt`
- Chain flag/capture proof:
  `output/fr13_b1_chain_speed_discriminator/chain/container_env.txt`,
  `output/fr13_b1_chain_speed_discriminator/chain/docker_full.log`
- Reused current gate rows:
  `output/fr13_b1_current_gate/b1_current_gate_reduce.json`

## Scope Notes

No B=4 work was run or analyzed. No code fix was attempted. An earlier
`GPU_UTIL=0.85` chain run was superseded after verifying the current gate's
native/caterpillar9 boots used `0.82`; only the exact-pairing `0.82` chain row
is bound above.
