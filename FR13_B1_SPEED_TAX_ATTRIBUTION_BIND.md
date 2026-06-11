# FR13 B=1 Speed-Tax Attribution Bind

Date: 2026-06-11 UTC

## Verdict

The B=1 chain-width speed tax is now narrowed further. Fresh B=1 runs still
show valid chain5 replay-on at `1.392923x` native per speculation forward, but
the new discriminators rule out two more suspected dominant surfaces:

- Replay route staging/replay is not dominant: replay-off is only `2.699%`
  slower than replay-on (`0.311789` vs `0.303595 s/fwd`).
- The tree GDN kernel alone is not dominant: the diagnostic-only
  `FR10_ENABLE_TREE_GDN=0` / `FR10_ALLOW_LINEAR_FALLBACK=1` fallback still
  runs at `1.347138x` native (`0.293616 s/fwd`).

So the remaining removable speed surface is broader than replay or the tree
GDN scan kernel alone: the chain5 `tree_mtp` graph/row-shape/scheduler path
shared by tree mode and fallback. Next speed work should collapse or bypass
that tree-mode path for the pure spine case, then re-check whether the native
MTP-5 row/graph path can be reused before returning to B=4.

No lossless or deployment speed pass is claimed.

## Run

Run dir: `output/fr13_b1_speed_tax_attribution/` (gitignored local artifact).

Substrate at live run start: `f1c7a41a` on `main`.

Shared request window:

- Prompts: `output/fr13_acceptance_ladder/prompts_swe4.json`
- `max_tokens=128`, `temperature=0.0`, `top_p=1.0`, `seed=1313`
- `samples_per_prompt=1`, client `batch_size=1`
- `MAX_NUM_SEQS=1`, `GPU_UTIL=0.82`
- `BATCH_INVARIANT=0`, `FR10_METRICS=0`
- Speed basis: `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`

Valid arms:

- Native: `scripts/fr10_launch_speed_server.sh`,
  `ATTENTION_BACKEND=FLASH_ATTN`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`,
  `FR10_ENABLE_TREE_GDN=0`, `SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}`.
- Primary tree: `scripts/fr13_launch_forked_fa2_tree_server.sh`,
  `ATTENTION_BACKEND=TREE_ATTN`, `FR10_DECODE_MODE_DEFAULT=tree_mtp`,
  chain5 tree, `FR13_REPLAY_ROUTE=1`, `FR10_ENABLE_TREE_GDN=1`.

Diagnostics:

- Replay-off legacy diagnostic: same tree arm with `FR13_REPLAY_ROUTE=0`.
- No-tree-GDN fallback diagnostic: `scripts/fr10_launch_speed_server.sh`,
  `ATTENTION_BACKEND=FLASH_ATTN`, chain5 `tree_mtp`,
  `FR10_ENABLE_TREE_GDN=0`, `FR10_ALLOW_LINEAR_FALLBACK=1`.
  This is diagnostic-only and not gate-valid.

Tree engagement was asserted by the probe:

- Replay-on: `144/144` `gpu_tree_metadata` rows `reason=ok`, draft count `5`,
  `144` tree accept rows.
- Replay-off: `144/144` rows `reason=ok`, draft count `5`, `144` accept rows.
- Fallback: `139/139` rows `reason=ok`, draft count `5`, `139` accept rows.

CUDA graph evidence from logs:

- Native: `PIECEWISE=1 (largest=6), FULL=1 (largest=6)`,
  graph capture finished in `7s`.
- Replay-on: `PIECEWISE=1 (largest=6), FULL=1 (largest=6)`,
  graph capture finished in `9s`.
- Replay-off: `PIECEWISE=1 (largest=6), FULL=1 (largest=6)`,
  graph capture finished in `7s`.
- Fallback diagnostic: `PIECEWISE=1 (largest=6), FULL=1 (largest=6)`,
  graph capture finished in `11s`.

`docker ps` was empty after teardown.

## Decision Table

Measured per-forward uses only `/metrics` counters:
`decode_seconds / spec_drafts`.

| arm | validity | backend | route / GDN | decode seconds | spec drafts | draft tokens | s/forward | ratio vs native | accept/event | warm decode TPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| native MTP-5 | valid reference | FLASH_ATTN | native | 29.205967 | 134 | 670 | 0.217955 | 1.000000x | 2.850746 | 17.530664 |
| chain5 replay-on | valid primary | TREE_ATTN | replay, tree GDN on | 40.985272 | 135 | 675 | 0.303595 | 1.392923x | 2.807407 | 12.492292 |
| chain5 replay-off | diagnostic legacy | TREE_ATTN | replay off, tree GDN on | 42.091471 | 135 | 675 | 0.311789 | 1.430519x | 2.807407 | 12.163985 |
| chain5 no-tree-GDN fallback | diagnostic only | FLASH_ATTN | fallback, tree GDN off | 38.170019 | 130 | 650 | 0.293616 | 1.347138x | 2.946154 | 13.413669 |

Draft-count proof:

- Native: `670 / 134 = 5.0`
- Replay-on: `675 / 135 = 5.0`
- Replay-off: `675 / 135 = 5.0`
- Fallback diagnostic: `650 / 130 = 5.0`

Attribution from the fresh native baseline:

- Valid replay-on tax: `0.303594607 - 0.217954979 = 0.085639628 s/fwd`.
- Replay-off minus replay-on: `0.008194063 s/fwd`, only `9.568%` of the
  valid replay-on tax. Replay-on is slightly faster, so replay route is not
  the cause of the residual slowdown.
- Replay-on minus no-tree-GDN fallback: `0.009979074 s/fwd`, only `11.652%`
  of the valid replay-on tax. Because fallback is invalid and uses
  `FLASH_ATTN`, this is only a diagnostic bound, but it is enough to reject
  "tree GDN kernel alone dominates the `~1.39x` tax."

## Profiler Status

No valid kernel-time table is bound in this doc.

Profiler feasibility artifact:
`output/fr13_b1_speed_tax_attribution/profiler_feasibility.log`.

Findings:

- Host tools are present: `nsys 2025.3.2` and `ncu 2025.3.1`.
- The vLLM image does not include `nsys` or `ncu` by default.
- Mounting `/opt/nvidia` and `/usr/local/cuda-13.0` makes host Nsight tools
  visible inside the image.
- The canonical FR10/FR13 launchers used for these gates detach `docker run`
  and do not wrap the in-container `vllm serve` process. The only existing
  `LUMO_NSYS_WRAP_VLLM` hook is in `model_server.py`, not these launchers.

Therefore a real profiler run would require adding an off-by-default wrapper
to the canonical launchers or routing this exact FR13 launch through
`model_server.py`. I did not mutate launcher code in this attribution bind.

## Decision

The next removable speed surface is the `tree_mtp` chain-mode execution path
that remains even when branch width, `TREE_ATTN`, replay route, and tree GDN
are separated away:

- CUDA graph row shape is still the tree verifier shape (`root + 5` rows)
  rather than an ordinary native MTP-5 spine path.
- The fallback diagnostic still pays most of the tax with tree GDN off.
- Replay-off makes the run slower, not faster.

Next concrete fix/discriminator: create an off-by-default kernel table or graph
node table for native MTP-5 versus chain5 tree mode, then target the first
shared residual component: tree-mode scheduler/metadata/committer graph nodes,
tree verifier row materialization, or redundant per-row MTP/tree bookkeeping.
The pass bar remains near-native `/metrics` s/fwd; do not use TPS divided by
acceptance as a proxy.

## Artifacts

- Reducer:
  `output/fr13_b1_speed_tax_attribution/b1_speed_tax_attribution_reduce.json`
- Native:
  `output/fr13_b1_speed_tax_attribution/native/native_probe.json`,
  `output/fr13_b1_speed_tax_attribution/native/docker_full.log`,
  `output/fr13_b1_speed_tax_attribution/native/container_env.txt`
- Replay-on:
  `output/fr13_b1_speed_tax_attribution/chain_replay_on/chain_replay_on_probe.json`,
  `output/fr13_b1_speed_tax_attribution/chain_replay_on/docker_full.log`,
  `output/fr13_b1_speed_tax_attribution/chain_replay_on/container_env.txt`
- Replay-off diagnostic:
  `output/fr13_b1_speed_tax_attribution/chain_replay_off_diag/chain_replay_off_probe.json`,
  `output/fr13_b1_speed_tax_attribution/chain_replay_off_diag/docker_full.log`,
  `output/fr13_b1_speed_tax_attribution/chain_replay_off_diag/container_env.txt`
- Fallback diagnostic:
  `output/fr13_b1_speed_tax_attribution/chain_no_tree_gdn_fallback_diag/chain_no_tree_gdn_fallback_probe.json`,
  `output/fr13_b1_speed_tax_attribution/chain_no_tree_gdn_fallback_diag/docker_full.log`,
  `output/fr13_b1_speed_tax_attribution/chain_no_tree_gdn_fallback_diag/container_env.txt`
- Profiler feasibility:
  `output/fr13_b1_speed_tax_attribution/profiler_feasibility.log`

## Scope Notes

No B=4 work was run. No accept/event optimization was attempted. The fallback
arm is not a valid gate result because it used `FR10_ALLOW_LINEAR_FALLBACK=1`.
