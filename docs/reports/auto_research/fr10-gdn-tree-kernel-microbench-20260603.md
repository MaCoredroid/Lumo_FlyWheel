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

## Next

Move the standalone algebra into the vLLM 0.22 FLA op fork:

- `solve_tril.py`: tree-ancestry mask in the triangular solve.
- `chunk_scaled_dot_kkt.py`: ancestry-masked KKT interaction.
- `cumsum.py`: ancestor-path gate accumulation.
- `chunk_delta_h.py` and `chunk.py`: per-node tree state and fixed descriptors.

Before integrating, pin or remove autotune for the tree path and control the
`chunk_delta_h.py` `use_cuda_graph` autotune behavior so capture and parity are
not dependent on a selected autotune config.
