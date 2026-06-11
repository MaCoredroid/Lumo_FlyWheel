# FR13 B=1 Current Speed + Lossless Gate Bind

Date: 2026-06-11 UTC

## Verdict

The clean current B=1 gate was run and is bound here. It does **not** pass the
B=1 bar.

- Speed fails decisively: tree/native measured per-forward ratio is
  `1.4103x` on the first window and `1.4061x` on the repeat.
- Superset/accept fails decisively: tree accept/event is `2.1515` versus native
  MTP-5 `3.1613`.
- S1 is still healed: no bonus-row violations, including `[0,2]` winners.
- Greedy served streams still fork from native on all four prompts. Because
  this bounded speed gate did not run final-logit margin captures, those forks
  are not claimed as an S2 pass or classified as inside the historical floor.

No temp-0.6/top_p-0.95 run was launched because the greedy path did not earn a
clean speed/lossless pass.

## Run

Run dir: `output/fr13_b1_current_gate/` (gitignored local artifact).

Substrate at live run start: `fea42f8e` on `main`.

Shared request window:

- Prompts: `output/fr13_acceptance_ladder/prompts_swe4.json`
- `max_tokens=128`, `temperature=0.0`, `top_p=1.0`, `seed=1313`
- `samples_per_prompt=1`, client `batch_size=1`
- `MAX_NUM_SEQS=1`
- Raw speed basis: vLLM `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`

Tree arm:

- `TREE_ATTN/tree_mtp`
- 9-node caterpillar:
  `[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,1), (0,0,1), (0,0,0,1), (0,0,0,0,1)]`
- `FR13_REPLAY_ROUTE=1`
- `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`, `FR13_BI_TREE_ATTN=0`
- Heavy captures/final-logit diagnostics unset. The launcher still emitted the
  normal sampler/LCP trace logs used for engagement/S1 checks.

Native arm:

- `FLASH_ATTN/naive_mtp`
- `NUM_SPECULATIVE_TOKENS=5`
- `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`

CUDA graph capture was proven live in both arms:

- Tree: mixed prefill-decode capture reached `100%`, decode FULL capture
  reached `100%`, `Graph capturing finished in 7 secs`.
- Native: mixed prefill-decode capture reached `100%`, decode FULL capture
  reached `100%`, `Graph capturing finished in 6 secs`.

`docker ps` was empty after teardown.

## Speed

Measured per-forward uses only `/metrics` counters:
`decode_seconds / spec_drafts`.

| arm | decode seconds | spec drafts | draft tokens | drafts/event | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree | 50.765643 | 165 | 1485 | 9.0 | 0.307671 | 2.151515 | 10.085561 |
| native | 27.051817 | 124 | 620 | 5.0 | 0.218160 | 3.161290 | 18.926640 |
| tree rep2 | 50.754538 | 165 | 1485 | 9.0 | 0.307603 | 2.151515 | 10.087768 |
| native rep2 | 27.126764 | 124 | 620 | 5.0 | 0.218764 | 3.161290 | 18.874349 |

Ratios:

- First window: `0.307671 / 0.218160 = 1.410299x`
- Repeat: `0.307603 / 0.218764 = 1.406095x`

This is not near-native parity and is worse than the already-rejected
`1.1x` class. No break-even argument is available because tree is slower per
forward and accepts fewer tokens per event.

## Lossless Classification

Same-seed repeat determinism:

- Tree main vs rep2: `4/4` prompts byte-identical, 128 tokens each.
- Native main vs rep2: `4/4` prompts byte-identical, 128 tokens each.

S1 bonus-row bar:

- `tree_path_lcp_rows=339` across warmup plus the two measured tree windows.
- `bonus_sources={reject_parent_target: 200, tree_self_target: 139}`.
- `[0,2]` full-accept rows: `33`; all served `st[2]`.
- `bonus_violations_count=0`.
- `superset_violations_count=0` with true `spine_path_idx=3`.

Served-stream tree vs native first forks:

| prompt | first fork position | tree token | native token |
|---:|---:|---:|---:|
| 0 | 54 | 47506 | 1671 |
| 1 | 11 | 26622 | 12182 |
| 2 | 21 | 1970 | 3425 |
| 3 | 61 | 20049 | 1901 |

Classification:

- These forks mean the greedy served stream is not a literal native match.
- The historical B=1 bar is not literal equality, but it requires classifying
  forks by margin/trigger against the native/cross-boot floor.
- This bounded gate did not capture final logits or trigger context at the
  first forks, so S2 cannot be declared clean from this run.
- S3 remains failed separately: tree accept/event is `2.1515`, native is
  `3.1613`, delta `-1.0098` accept/event.

Therefore the current B=1 gate is **not a lossless pass**. It is S1-clean,
deterministic, and speed/superset-failing, with served-stream forks requiring a
future S2 margin/trigger reducer only if the B=1 speed/superset work is resumed.

## Artifacts

- Reducer summary: `output/fr13_b1_current_gate/b1_current_gate_reduce.json`
- Tree probes:
  `output/fr13_b1_current_gate/tree/tree_greedy_probe.json`,
  `output/fr13_b1_current_gate/tree/tree_greedy_rep2_probe.json`
- Native probes:
  `output/fr13_b1_current_gate/native/native_greedy_probe.json`,
  `output/fr13_b1_current_gate/native/native_greedy_rep2_probe.json`
- Raw metrics snapshots:
  `output/fr13_b1_current_gate/{tree,native}/metrics_before*.txt`,
  `output/fr13_b1_current_gate/{tree,native}/metrics_after*.txt`
- Flag/capture proof:
  `output/fr13_b1_current_gate/{tree,native}/container_env.txt`,
  `output/fr13_b1_current_gate/{tree,native}/docker_full.log`

## Scope Notes

No code fix was attempted. The gate already fails the pass bar on clean speed
and superset acceptance, and the served-stream forks would need a separate
final-logit/margin discriminator before any lossless pass claim.
