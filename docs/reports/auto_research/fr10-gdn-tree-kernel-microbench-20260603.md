# FR10 GDN Tree Kernel Microbench

Date: 2026-06-03

Status: initial standalone Phase 2 Triton microbench passed CPU-oracle parity
and CUDA graph replay for static padded tree families. This is not yet the vLLM
FLA op fork/integration.

## Stack

Artifact: `output/fr10_phase2_gpu_stack_probe_20260603.json`

| Field | Value |
|---|---|
| container | `lumo-vllm-audit:v0.22.0-cu129-min` |
| GPU | NVIDIA GB10 |
| driver | 590.48.01 |
| Python | 3.12.3 |
| PyTorch | `2.10.0a0+a36e1d39eb.nv26.01.42222806` |
| torch CUDA | 13.1 |
| CUDA toolkit | 13.1.115 |
| Triton | 3.6.0 |
| vLLM | 0.22.0 |
| FlashAttention | 2.7.4.post1 |

## Kernel

Artifact: `scripts/fr10_phase2_triton_tree_gdn_microbench.py`

The microbench implements the P1 tree-ancestry chunk algebra in one Triton kernel
over Qwen3.6 GDN dimensions:

- heads: 48
- key/value dim: 128
- state per node/head: `128 x 128`
- public node families: `{2,3,6,8,14}`
- padded compile families: `{2,4,8,16}`

The public `--nodes` path fail-closes on unwarmed public node counts. The
branch-depth table uses one fixed base tree and appends one row at varying
branch depths inside the same padded compile bucket to isolate marginal leaf
cost.

The kernel has no Triton autotune decorator (`rg '@triton.autotune'` finds none
in the microbench). This is intentional: deterministic kernel config is a
losslessness prerequisite for Phase 2/3, because autotune config flips can break
parity and CUDA graph capture.

## Correctness

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -v /tmp/vllm-0.22-src:/tmp/vllm-0.22-src:ro \
  -w /workspace --entrypoint python -e PYTHONWARNINGS=ignore \
  lumo-vllm-audit:v0.22.0-cu129-min \
  scripts/fr10_phase2_triton_tree_gdn_microbench.py --capture --cpu-oracle
```

Artifact: `output/fr10_phase2_triton_tree_gdn_microbench_cpu_oracle_20260603.json`

| nodes | padded | graph bit-exact | graph us | eager us | max out vs CPU | max state vs CPU |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 2 | yes | 14.252 | 13.395 | 1.779e-07 | 4.649e-06 |
| 3 | 4 | yes | 42.120 | 41.906 | 3.654e-07 | 1.269e-05 |
| 6 | 8 | yes | 172.970 | 169.107 | 4.410e-07 | 1.607e-05 |
| 8 | 8 | yes | 209.351 | 191.731 | 9.462e-07 | 1.616e-05 |
| 14 | 16 | yes | 741.648 | 748.151 | 1.383e-06 | 2.168e-05 |

All rows compare the GPU Triton tree output and per-node recurrent state against
serial per-path replay through vLLM's CPU `recurrent_gated_delta_rule()`.
`graph_ok` is set only after zeroing output/state buffers, replaying the captured
graph, and verifying graph replay output/state are bit-exact equal to eager
output/state.

## Native GPU Gate D

The installed `vllm._C` extension in this CUDA-13 audit image imports against
`libcudart.so.12`, which is absent. To avoid that package-level import blocker,
the native-GPU comparison loads the vLLM 0.22 FLA op source files under a private
module namespace and shims only `vllm.triton_utils` / `vllm.platforms`. The kernel
source exercised is still the vLLM 0.22 `chunk_gated_delta_rule()` path.

Active backend detection:

- Direct `ChunkGatedDeltaRule()` construction in the audit container is blocked
  before the constructor by the `libcudart.so.12` import failure above.
- Source-level resolver inputs for the sampled E3/E5-style bundles point to
  `triton` / native FLA: no `additional_config.gdn_prefill_backend`, empty
  `kernel_selection`, GB10/CUDA13/head_k_dim=128, and no installed `flashinfer`
  or `nvidia-cutlass-dsl-libs-cu13` packages in the audit image.
- Artifacts: `output/fr10_phase2_active_backend_probe_20260603.json`,
  `output/fr10_phase2_backend_package_probe_20260603.json`,
  `output/fr10_phase2_active_backend_deduction_20260603.json`.

The single-spine comparison is a drift diagnostic, not a final losslessness
pass: for a spine, the tree-ancestry mask is the linear causal mask, so the tree
kernel and native FLA chunk are the same algebra, but their bf16-input reduction
order differs. This table uses bf16 inputs and fp32 recurrent state.

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -v /tmp/vllm-0.22-src:/tmp/vllm-0.22-src:ro \
  -w /workspace --entrypoint python -e PYTHONWARNINGS=ignore \
  lumo-vllm-audit:v0.22.0-cu129-min \
  scripts/fr10_phase2_triton_tree_gdn_microbench.py --capture --native-gpu --input-dtype bf16 --single-spine-table
```

Artifact: `output/fr10_phase2_native_gpu_single_spine_20260603.json`

| nodes | padded | graph bit-exact | native full-spine max out | native full-spine final state | note |
|---:|---:|---|---:|---:|---|
| 2 | 2 | yes | 5.651e-05 | 7.074e-04 | reduction-order drift |
| 3 | 4 | yes | 6.537e-05 | 8.780e-04 | reduction-order drift |
| 6 | 8 | yes | 5.994e-05 | 6.694e-04 | reduction-order drift |
| 8 | 8 | yes | 5.824e-05 | 9.432e-04 | reduction-order drift |
| 14 | 16 | yes | 9.496e-05 | 7.378e-04 | reduction-order drift |

Branched trees have no native single-pass equivalent; the whole FR10 kernel
exists to provide that. For branched shapes, the production reference remains
serial native per path plus the CPU oracle.

State-commit consequence: the tree kernel's recurrent state is not automatically
byte-identical to native decode state. The robust FR10 plan is to use tree-kernel
logits for accept/reject verification, then re-run the accepted short path
through the canonical native decode update (`fused_recurrent_gated_delta_rule` /
production decode path) and commit that state. The decisive production gate is
end-to-end greedy token stream equality against native decode on real prompts.

## Exact Production Gate D On cu130 GB10

Follow-up run after P0 stack resolution used the digest-pinned cu130-nightly image:

- Image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- Local image ID: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`
- GB10 production GDN prefill backend: Triton/FLA `forward_native`
- Precision/convention: bf16 q/k/v/g/beta inputs, fp32 initial state, raw `g`, `use_qk_l2norm_in_kernel=True`, production scale `1/sqrt(128)`

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  --entrypoint python3 -e PYTHONWARNINGS=ignore \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  scripts/fr10_phase2_triton_tree_gdn_microbench.py \
  --capture --production-gdn --production-scale --input-dtype bf16 --single-spine-table
```

Artifact: `output/fr10_cu130_gate_d_production_gdn_single_spine_batchinv_followup_clean_20260603.json`

| nodes | padded | production method | graph bit-exact | tree graph us | production max out | production final state |
|---:|---:|---|---|---:|---:|---:|
| 2 | 2 | `forward_native` | yes | 12.556 | 6.104e-05 | 8.978e-04 |
| 3 | 4 | `forward_native` | yes | 43.164 | 6.104e-05 | 8.745e-04 |
| 6 | 8 | `forward_native` | yes | 308.687 | 6.104e-05 | 6.552e-04 |
| 8 | 8 | `forward_native` | yes | 352.262 | 6.104e-05 | 9.473e-04 |
| 14 | 16 | `forward_native` | yes | 1041.471 | 6.104e-05 | 7.828e-04 |

Gate D interpretation:

- Single-spine tree mask equals the linear causal mask, so this is the apples-to-apples production algebra check.
- Output agreement is one bf16 quantum for all rows: max `6.103515625e-05`.
- Final-state difference remains reduction-order drift below `9.473264217376709e-04`.
- CUDA graph replay equals eager bit-exact for all rows.
- Therefore verifier logits are usable for accept/reject, but tree-kernel recurrent state must not be committed.

The cu130 production decode state commit primitive is:

- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- import: `fused_recurrent_gated_delta_rule_packed_decode`
- call site around the decode path passes `mixed_qkv`, `a`, `b`, `A_log`, `dt_bias`, `scale=self.head_k_dim**-0.5`, `initial_state=ssm_state`, `out=core_attn_out[:num_actual_tokens].unsqueeze(1)`, `ssm_state_indices`, and `use_qk_l2norm_in_kernel=True`.

FR10 state-commit rule: tree kernel output/logits drive accept/reject only. After an accepted path is selected, commit the canonical recurrent state by replaying the accepted short path through the native decode update (`fused_recurrent_gated_delta_rule_packed_decode` / the production decode path) and discard tree verifier state.

## Canonical State Commit Probe

Implementation artifact: `scripts/fr10_canonical_state_commit_probe.py`

This probe exercises the exact cu130 production decode state-update primitive:

`vllm.model_executor.layers.fla.ops.fused_recurrent_gated_delta_rule_packed_decode`

It constructs production-shaped synthetic decode inputs:

- packed bf16 `mixed_qkv` with `[q, k, v]` layout matching `gdn_linear_attn.py`
- bf16 `a` and `b`
- fp32 `A_log`, `dt_bias`, and recurrent state bank
- valid `ssm_state_indices` slot `1`; slot `0` remains the invalid/null state
- `scale = 1/sqrt(128)`
- `use_qk_l2norm_in_kernel=True`

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  --entrypoint python3 -e PYTHONWARNINGS=ignore \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  scripts/fr10_canonical_state_commit_probe.py --tokens 14 --accepted-tokens 5 --capture
```

Artifact: `output/fr10_canonical_state_commit_probe_20260603.json`

Results:

- packed-decode replay output bit-exact: `true`
- packed-decode replay state bit-exact: `true`
- packed replay max output delta: `0.0`
- packed replay max state delta: `0.0`
- one-token CUDA graph replay bit-exact: `true`
- packed decode vs sequence recurrent diagnostic max output delta: `0.0`
- packed decode vs sequence recurrent diagnostic max state delta: `2.9802322387695312e-08`
- accepted path length: `5`
- state read bytes per token: `3,145,728`
- state write bytes per token: `3,145,728`
- accepted-path state read bytes: `15,728,640`
- accepted-path state write bytes: `15,728,640`

This is the implementation rule for lossless FR10 commit:

1. Tree verifier kernel computes candidate logits and accept/reject decisions.
2. Tree verifier recurrent state is treated as scratch and never committed.
3. The accepted token path is replayed through `fused_recurrent_gated_delta_rule_packed_decode`.
4. The native packed decode output/state is committed to the request state.

The probe demonstrates that this canonical commit path is deterministic,
CUDA-graph-compatible for warmed one-token decode, and numerically aligned with
the sequence recurrent diagnostic to fp32 roundoff.

## Marginal Branch Cost

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -v /tmp/vllm-0.22-src:/tmp/vllm-0.22-src:ro \
  -w /workspace --entrypoint python -e PYTHONWARNINGS=ignore \
  lumo-vllm-audit:v0.22.0-cu129-min \
  scripts/fr10_phase2_triton_tree_gdn_microbench.py --capture --cpu-oracle --branch-depth-table
```

Artifact: `output/fr10_phase2_branch_depth_cost_20260603.json`

| tree shape | branch depth | nodes | shared trunk tokens | new leaf tokens | kernel us | marginal leaf us | memory bytes | state reads | state writes | equivalent serial us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `5->6 padded 8->8` | 0 | 6 | 1 | 1 | 163.026 | 11.900 | 18,874,368 | 6 | 6 | 302.697 |
| `5->6 padded 8->8` | 1 | 6 | 2 | 1 | 162.141 | 6.289 | 18,874,368 | 6 | 6 | 309.838 |
| `5->6 padded 8->8` | 2 | 6 | 3 | 1 | 163.161 | 9.226 | 18,874,368 | 6 | 6 | 303.223 |

These are initial standalone kernel numbers, not optimized FLA-fork numbers. The
important Phase 2 signal is that one extra branch row can be measured inside a
fixed padded graph-captured shape and validated against the CPU oracle.

## Real-Dimension Tree Cost Versus cu130 FLA

Implementation artifact: `scripts/fr10_real_dims_tree_vs_fla_cost.py`

This follow-up runs in the digest-pinned cu130 production stack and uses real
Qwen3.6 GDN dimensions:

- key heads: `16`
- value heads: `48`
- key/value head dim: `128`
- bf16 q/k/v/g/beta inputs
- fp32 recurrent state
- raw q/k are normalized inside both kernels
- raw `g` is passed to the FLA chunk path

Command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  --entrypoint python3 -e PYTHONWARNINGS=ignore \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  scripts/fr10_real_dims_tree_vs_fla_cost.py --capture --iters 300 --repeats 5
```

Artifact: `output/fr10_real_dims_tree_vs_fla_cost_20260603.json`

Timing is median microseconds over `5` repeats of `300` iterations. The FLA
column is cu130 `fla_chunk_gated_delta_rule` with
`use_qk_l2norm_in_kernel=True`.

| nodes | padded | tree graph bit-exact | tree graph us | FLA chunk us | tree minus FLA us | state read bytes | state write bytes |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 2 | yes | 12.325 | 135.023 | -122.698 | 6,291,456 | 6,291,456 |
| 3 | 4 | yes | 45.339 | 134.661 | -89.322 | 9,437,184 | 9,437,184 |
| 6 | 8 | yes | 306.008 | 135.876 | 170.132 | 18,874,368 | 18,874,368 |
| 8 | 8 | yes | 340.857 | 134.458 | 206.398 | 25,165,824 | 25,165,824 |
| 14 | 16 | yes | 996.084 | 135.971 | 860.114 | 44,040,192 | 44,040,192 |

Fixed-base marginal rows used `5->6 padded 8->8` at branch depths `0/1/2`.
Measured medians were `-13.939 us`, `-13.005 us`, and `-11.282 us`. These are
not interpreted as a real negative marginal cost. In this standalone monolithic
kernel the same padded block does most of the work regardless of active node
count, and the extra active row is below compile/codegen/timing noise at this
measurement granularity. The bandwidth fact remains fixed and useful for the
production design: one additional Qwen3.6 GDN verifier node carries
`3,145,728` bytes of fp32 state read traffic and `3,145,728` bytes of fp32 state
write traffic per value-head state slot (`48*128*128*4`).

Speed interpretation:

1. Current standalone tree kernel is graph-capturable and faster than cu130 FLA
   for tiny `2/3` node verifier shapes.
2. It is slower than cu130 FLA for `6/8/14` node shapes and is not yet an
   optimized FLA-fork deliverable.
3. The real optimization target is reducing redundant padded-block state work
   and state memory traffic while preserving the no-autotune, graph-capturable
   contract.

Cost-gate interpretation:

- This is a red flag for the large-tree path, not a success result. cu130 FLA is
  essentially flat around `135 us` for `2..14` tokens, while the dense masked
  tree kernel climbs to `996 us` at `14` nodes.
- The negative fixed-base marginal rows are a padding-bucket artifact. Base and
  extended trees both run in the same padded-8 graph shape, so the measured
  negative delta is timing/codegen noise and must not be reported as a free
  leaf.
- Next decision point is profiling the `14`-node case: if dense KKT/triangular
  solve dominates and tree sparsity can cut the work from padded `N^2` toward
  `N*depth`, prototype the sparse path. If that cannot plausibly beat the
  `~135 us` FLA flat cost for `6..14` nodes, narrow the speed case to the
  `<=4` node niche.

## Cost-Gate Stage Profile

Implementation artifact: `scripts/fr10_tree_kernel_stage_profile.py`

Artifacts:

- `output/fr10_tree_kernel_stage_profile_6n_20260603.json`
- `output/fr10_tree_kernel_stage_profile_8n_20260603.json`
- `output/fr10_tree_kernel_stage_profile_14n_20260603.json`

Command template:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  --entrypoint python3 -e PYTHONWARNINGS=ignore \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  scripts/fr10_tree_kernel_stage_profile.py --nodes 14 --capture --iters 300 --repeats 5
```

The stage profile uses graph-replay medians over `5x300` iterations. It is a
variant profile, not an exact Nsight additive trace, but it isolates whether the
dense padded solve or the state/output traversal is the main cost center.

| nodes | padded | full us | dense solve variant us | solve/full | state-output-only us | strict ancestor pairs / dense lower | visible pairs / dense square |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 303.243 | 160.921 | 0.531 | 221.282 | 11/28 = 0.393 | 17/64 = 0.266 |
| 8 | 8 | 334.785 | 160.861 | 0.480 | 285.975 | 19/28 = 0.679 | 27/64 = 0.422 |
| 14 | 16 | 995.777 | 636.507 | 0.639 | 352.529 | 36/120 = 0.300 | 50/256 = 0.195 |

Profile conclusion:

- Dense triangular solve is the largest single cost center for the 14-node
  public tree: about `636 us` of a `996 us` full path.
- Tree sparsity is real, but simple pair-count sparsity does not plausibly
  rescue the large public trees. Even an optimistic pair-scaled estimate leaves
  14 nodes well above the `~135 us` FLA flat cost, and 8 nodes has too little
  sparsity inside the padded-8 bucket.
- The 6-node case is borderline only under optimistic scaling, but FR10 needs a
  broad `6..14` tree speed case. Do not invest in large-tree optimization unless
  a more radical STree-style accumulated-state recurrence removes the padded
  dense solve/state traversal rather than merely skipping masked pairs.
- Next speed path is the explicitly narrowed `<=4` node niche.

## Next

Move the standalone algebra into the vLLM 0.22 FLA op fork:

- `solve_tril.py`: tree-ancestry mask in the triangular solve.
- `chunk_scaled_dot_kkt.py`: ancestry-masked KKT interaction.
- `cumsum.py`: ancestor-path gate accumulation.
- `chunk_delta_h.py` and `chunk.py`: per-node tree state and fixed descriptors.

Before integrating, pin or remove autotune for the tree path and control the
`chunk_delta_h.py` `use_cuda_graph` autotune behavior so capture and parity are
not dependent on a selected autotune config.
