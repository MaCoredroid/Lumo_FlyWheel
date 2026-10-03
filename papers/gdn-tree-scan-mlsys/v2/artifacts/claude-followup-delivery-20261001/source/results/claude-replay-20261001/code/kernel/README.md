# GDN tree-verification kernel benchmark (timing + memory)

Ready-to-run, **not yet run**. Measures, for the evaluated LumoTree geometry, the GPU time and memory of GDN tree
verification and accepted-path commit for LumoTree, the TreeWY and Weaver author implementations (through the frozen M1
adapters), and two naive baselines that materialize a full recurrent state per tree node. Each method's numerical error
against a float64 reference is reported next to its time (M1 comparator; informational, not a gate).

## Run

```bash
# on GB10, GPU idle (the launcher refuses otherwise)
bash /home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/kernel/run_kernel_bench.sh            # full run
bash .../run_kernel_bench.sh --dry-run          # host checks + print the docker command, launch nothing
bash .../run_kernel_bench.sh --smoke            # 5 repeats, wiring check, results labelled SMOKE
bash .../run_kernel_bench.sh --methods "lumo_fixed32 treewy_author_default" --no-sweep --repeats 200
bash .../run_kernel_bench.sh --extra "--stratum mixed-stress"   # M1 near-zero-key / extreme-gate stress operands
```

Output: `/home/mark/shared/lumotree-v2exp-runs/kernel/<UTC>/` with `methods/<method>.json`, `sweep/sweep_<family>.json`,
`summary.json`, `summary.md`, `logs/` (one log per process, `rc.txt`, versions, nvidia-smi before/after), `launch.json`,
`docker_run.txt`, `container.log`.

Launcher refusals (exit 3): any running container, any v2exp arm driver or queue driver (`queue.sh`/`queue_v2.sh`; the queue
starts its next step 20 s after it sees an idle GPU), any GPU compute process, GPU utilization > 5 %, host `MemAvailable`
< 24 GiB, pinned image missing. The container is named `lumo-kbench-<UTC>` (deliberately not `v2exp*`, which the queue
deletes) and only that container is ever removed. Timeout `KBENCH_TIMEOUT_S` (default 3 h).

Image: `sha256:ffa30d66...61cdc` = `vllm/vllm-openai@sha256:3dbe092e...e776` (`vllm/vllm-openai:cu130-nightly`), the image
M1 used and the deployed LumoTree serving image (torch 2.11.0+cu130, triton 3.6.0 are recorded and compared). Mounts (all
read-only except `/out`): the v2exp worktree at `/workspace`, the M1 review repo `/home/mark/lumotree-review-20260927` at
`/review` (M1 adapters/loaders, author code, Q1 Backend, C2 oracle), the served model's `config.json` at
`/model_config.json`, the run directory at `/out`. `--network none`. Environment = the deployed fixed32 GDN flags (below).

## Geometry (asserted; any mismatch exits 2 with `refused_geometry`)

| item | value | asserted against |
|---|---|---|
| GDN layers | 48 (+16 full-attention, not benchmarked) | `/models/qwen3.8-27b-nvfp4/config.json` `layer_types` |
| heads / dims | H_k = 16 key heads, H_v = 48 value heads (GQA 3), d_k = d_v = 128 | config `linear_*`; M1/Q1 constants |
| tree | Hydra27: 27 drafts + root padded to 32 physical rows (inactive rows 18, 23, 25, 27), max depth 11, 28 accepted paths | `scripts/fr13_fixed32_topology.py` (sha c02703b4) |
| batch | B = 1 (the deployed B1 route) | |
| dtypes | q, k, v, a, b bf16; A_log, dt_bias fp32; recurrent state fp32 [48,128,128] = 3 MiB/layer | M1 raw boundary |
| Lumo scan | BV = 8 (`FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`), 2 launches/layer: level 0 = spine 0-1-4-9-14 (grid 48x16x1), level 1 = 11 paths (grid 48x16x11) | kernel subtree cache |
| sources | `fr10_gdn_tree_kernel.py` d9dd0c69 in **both** trees, Q1 runner ab27fc9c, native op 000ab899 | sha256 at start |

Operands: M1 shared synthetic operands (`m1_cycle_driver.build_shared_operands` -> `q1_2b_fixtures.make_instance`, seed
20260928, stratum ordinary-random, one independent instance per layer, warm S0). Every method sees identical bytes.

## Methods and how each is invoked

| method | verify | commit / replay | source |
|---|---|---|---|
| `lumo_fixed32` | 48 x `launch_tree_gdn_prepared` (two-level path scan, transient cut states, in-kernel replay-ring export) | D2D copy of the accepted path from a device table + `launch_tree_gdn_replay_all_layers` (device asserts + replay of the preseeded fixed16 committer CUDA graph: staging gather + 48 native `fused_sigmoid_gating_delta_rule_update`) | Q1.2b `Backend` storage + production boot, kwargs identical to `Backend.scan/publish` |
| `weaver_author_default` | per layer: `fused_gdn_gating` -> g/beta stash, `l2norm_fwd_strided` q, k -> stash, v -> stash, `tree_gdn_triton_verify(precision="tf32", bf16_mode="none")` | `advance_ssm_states_along_accept_paths` (1 launch, all layers) | M1 `m1_adapters_v3` plans |
| `weaver_aligned_local` | same author kernels, M1 LOCAL fp32 rsqrt(+1e-6) normalization and unrounded fp32 g/beta | same | M1 plans (labelled LOCAL) |
| `treewy_author_default` | per layer: physical->DFS gather, ONE fused `tree_wy_tree_commit_capture_triton` (commits the previous accepted leaf, then verifies), DFS->physical remap | inside the next fused call; final deferred flush = one extra call/layer | M1 plans (`dot_bf16=True`) |
| `naive_native_paths` | Lumo's own 2-level path cover through the vLLM native kernel with per-token state stores (every node's fp32 state written) | leaf-state copy into the durable slot | this directory |
| `naive_torch_node` | pure PyTorch, sequential per-node update storing all 32 node states (the literal baseline) | same | this directory |

Optional: `treewy_dotbf16_false` (author `dot_bf16=False` setting).

Deployment mode: the LumoTree serving stack uses `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`, so **verification runs inside the
captured decode graph** (the `graph` numbers are the deployment numbers) and **commit is an eager wrapper that replays its
own preseeded committer graph** (identical in both modes). Every method is timed in `graph` mode (verify and commit captured
separately; per-step metadata copy stays eager) and `eager` mode; a failed capture is recorded under
`graph_capture`/`_capture_failures` and never replaced by an eager number.

Deployed fixed32 flags (identical to M1 `LUMO_REQUIRED_ENV` and to the v2exp tree arm `container_env.txt`):
`FR13_FIXED32_MODE=hydra27_fixed32 FR13_SUBTREE_PARALLEL=1 FR13_SCAN_ALIGN=0 FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_RING_EXPORT=1
FR13_TREE_RUNROW_INIT=1 FR13_FLAGS_INKERNEL=1 FR13_FIXED32_COMMIT_DEVICE_FILL=1 FR13_FIXED32_KV_REMAP16=1` and `=0` for
`COMMITTER_LAYER_BATCH, GDN_SINGLE_LAUNCH_PRODUCTION, GDN_GQA_GROUP3_PRODUCTION, TAW_NATIVE_PRECOMPUTE, CONV_COMMIT_ZERO_TAIL`.

## What is measured

Timing (`timing.<mode>.<experiment>`): CUDA-event pairs, `--warmup 20`, `--repeats 100` (>= 50 enforced outside
`--smoke`), median / p10 / p90 / mean / min / max per region and per iteration, raw milliseconds kept. Before each timed
iteration (untimed): 256 MiB L2-flush write and a device synchronize. State is reset to S0 before each experiment; the
28 accepted paths are cycled per iteration from a device table.

| experiment | regions |
|---|---|
| `step` (all) | `verify` + `commit` (Lumo, Weaver, naive); `fused_commit_prev_and_verify` (TreeWY) |
| `verify_layer0` / `layer0_step` | one layer (per-layer cost measured directly; `derived.*_per_layer_*` = per-step / 48) |
| `verify_only_sentinel` (TreeWY) | same fused step with the positive gcum sentinel set, so the in-kernel commit is skipped; `derived.commit_estimate` = step - verify-only |
| `final_flush` (TreeWY) | one extra fused call per layer (finite-session flush) |
| `topology_rebuild` (Weaver) | `build_tree_structure_into_fast` + `build_tree_ancestor_masks` (production rebuilds per step; NOT included in `step`) |

Memory (`memory`, and `timing.*.*.memory`): `torch.cuda` caching-allocator `memory_allocated`, `max_memory_allocated`,
`max_memory_reserved`, `mem_get_info` at process start, after setup, after numerics, around graph capture and around every
experiment (peaks reset per experiment; transient peak = max allocated - allocated before). `memory.accounting` lists each
method's scratch by tensor: Lumo cut-state export buffer (allocated 32 x 3 MiB = 96 MiB, shared by all layers; logical
N_c = 5 rows {0,1,4,9,14} = 15 MiB written per layer), replay rings, committer fixed16 staging; Weaver stashes + verifier
workspace cache + topology; TreeWY vt/kk/gc stashes (pool 2, one slot used) + DFS staging; naive node-state bank (32 x 3 MiB
per layer, all 48 layers resident = 4.5 GiB). The paper's logical formula is reported beside it:
`M_cut = 4 * N_c * B_v * d_k` = 4 x 5 x 8 x 128 = 20,480 B per (head, value tile) x 768 tiles = 15,728,640 B per layer.

Numerics (`numerics`): one verification from S0 (outputs of the 28 active nodes, all 48 layers) and the committed durable
state after accepting `root-only`, `n14`, `n15`, `n31` (depth 11), each compared with the float64 C2 reference
(`q1_2b_fixtures.c2_node_refs` + `q1_oracle.per_head_errors`, the M1 comparator): max-abs, RMS, relative max-abs,
nonfinite counts, per-layer maxima.

Tree-size sweep (`kernel_sweep.py`, verify only, 48 layers, 50 repeats): shapes `chain12` (depth-only spine, root + 11),
`tree8`, `tree16` (root + first 8 / 16 active drafts in level order), `tree27` (all 27 active drafts, no padding).
Families: `lumo_generic` (the same path-scan kernel through the GENERIC subtree route, `FR13_FIXED32_MODE` unset; records
the schedule, critical path and N_c per shape), `treewy_verify_only`, `weaver_verifier` (pre-normalized q/k, preparation
excluded), `naive_native_depth` (one native launch per depth level), `naive_torch_node`. Layer-0 outputs are checked against C2.

## Expected cost

Roughly 20-30 min wall clock for the full run (first run includes Triton JIT for every kernel), GPU exclusively held the
whole time; device peak about 6 GiB (naive node-state bank 4.8 GiB + graph pools), Lumo about 2.5 GiB (1.5 GiB deployed
page storage), host RAM about 2 GiB per process (float64 references are built one layer at a time).

## Caveats

1. **Lumo eager numbers are not deployment numbers**: `launch_tree_gdn_prepared` carries substantial Python validation per
   call; the deployed verify is graph-captured. Its commit wrapper (D2D copy, device asserts, graph replay) is eager in both.
2. Only the 48 GDN scans are captured, not the whole decode forward: in serving, attention/MLP weights stream through L2
   between GDN layers; here an L2 flush happens only between iterations.
3. Convolution-state commit (tree conv) is excluded for every method (TreeWY/Weaver do not handle it); so are MTP/drafting,
   attention/KV, sampling and pending-token work. This is GDN verification + accepted-state publication only.
4. TreeWY has no commit-only entry point: its commit is fused into the next verify; the commit cost is a derived difference
   with the sentinel-skipped run. The final deferred flush is reported separately as finite-session overhead.
5. Weaver's static topology is reused across steps (M1 policy); the per-step production rebuild is timed separately.
6. `weaver_aligned_local` is a labelled LOCAL normalization/precision-aligned port, not the author default. TreeWY uses its
   author normalization `x/max(||x||, 1e-12)`; Lumo/native use `x*rsqrt(sum x^2 + 1e-6)`: identical operands are not
   identical mathematical inputs for near-zero keys (use `--stratum mixed-stress` to see it).
7. All methods compute the 4 inactive padded rows (32 slots). The deployed fixed32 Lumo kernel cannot shrink with the active
   node count, so its tree-size sensitivity is flat by construction; `lumo_generic` in the sweep is NOT the deployed route.
8. B1 only. The B2-B4 batch kernel (`launch_tree_gdn_prepared_fixed32_batch`) needs credentialed selectors and is not
   exercised.
9. Lumo is bound through the M1/Q1 Backend (deployment-shaped storage and the public boot sequence) but timed by calling the
   same kernel entry points with the same kwargs as `Backend.scan/publish`, without their per-call `.to()`,
   `torch.cuda.synchronize()` and host `torch.tensor()` of the accepted path (acceptance is device-resident in deployment).
   Author methods use the frozen M1 plan builders and dispatch (`resolve_refs` -> callable -> `bind_result`) without M1's
   per-call sync, digests and logging.
10. Naive commit copies the leaf state into the durable slot (two passes over 48 x 3 MiB). An index-swap commit (point the
    next step's initial-state index at the leaf slot, vLLM MTP style) would make commit nearly free while keeping all 32
    slots live per request; it is not timed.
11. Memory figures are PyTorch caching-allocator statistics on GB10 unified memory (reserved includes cached blocks; graph
    private pools count as allocated). Accounting bytes are tensor sizes, not measured traffic.
12. Synthetic operands (M1 ordinary-random); timing is data-independent for these kernels except through numerics.

## Files

`run_kernel_bench.sh` (host launcher), `in_container.sh` (per-method processes, sweep, aggregation), `kernel_bench.py`
(per-method runner), `kbench_lumo.py`, `kbench_author.py`, `kbench_naive.py`, `kbench_base.py`, `kbench_common.py`
(geometry checks, timing, memory, M1 comparator), `kbench_trees.py`, `kernel_sweep.py`, `aggregate.py`,
`tests/test_kbench_cpu.py` (CPU-only: topology, formulas, timing plumbing, naive baselines vs C2 on CPU, native-kernel slot
layout via a torch transcription of the pinned kernel, M1-plan plumbing with fake author kernels, aggregation).
Run the tests with
`KBENCH_REVIEW_ROOT=/home/mark/lumotree-review-20260927 KBENCH_WORKSPACE=/home/mark/shared/lumotree-v2exp-20260930 CUDA_VISIBLE_DEVICES= python3 -m pytest -q tests/test_kbench_cpu.py`
(needs Python >= 3.10 with torch and pytest).
