# FR12 Results

## WY Tree Recurrence Gate

Command:

```bash
python3 scripts/fr12_wy_tree_recurrence_check.py --json-out output/fr12_wy_tree_recurrence_check.json
```

Scope:
- Uses FR10 tree descriptors from `scripts/fr10_gdn_tree_algebra_reference.py`.
- Runs the gated delta recurrence in float64 to avoid the vLLM CPU oracle's fp32 floor.
- Validates parent-inherit plus one-reflector append T against rebuilding WY on each path.
- Validates full per-node state/output against serial per-path GDN semantics.

Result:
- Verdict: `PASS` at threshold `1e-8`.
- Max append-vs-rebuild T/basis error: `0.0`.
- Max append-vs-rebuild operator error: `0.0`.
- Max homogeneous S0 map error: `3.3306690738754696e-16`.
- Max full state vs serial error: `3.0531133177191805e-16`.
- Max output vs serial error: `8.673617379884035e-18`.

Interpretation:

`TREE_ANCESTRY_T_RECURRENCE_CONFIRMED`

The FR12 WY tree recurrence is algebraically exact at float64 floor for the tested FR10 tree shapes. This validates the parent T inheritance plus append rule before building the Triton kernel.

## Existing WY Fused Probe Baseline

Command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  /workspace/output/gdn_novel_research/wy_tree_fused_probe.py
```

Preflight:
- `docker ps` showed no running containers.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed no active compute processes.

Result:
- Device: `NVIDIA GB10`.
- Tree nodes: `14`, padded nodes: `16`.
- Existing WY fused skeleton: `563.1969451904297 us`.
- Dense FR10-shaped fused kernel: `990.0962829589844 us`.
- Flat FLA chunk baseline: `138.1873607635498 us`.
- WY skeleton vs dense output maxabs: `0.013397216796875`.
- WY skeleton vs dense state maxabs: `0.1730390340089798`.

Interpretation:

`EXISTING_WY_FUSED_PROBE_IS_NOT_CORRECT`

The existing probe is a useful launch-shape and timing skeleton, but its reconstruction shortcut is not an acceptable FR12 implementation. The next kernel step must replace the reconstruction math and re-check against the serial/WY oracle; speed alone is not sufficient.

## Corrected WY Tree Solve Probe

Command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  /workspace/scripts/fr12_wy_tree_kernel_probe.py
```

Preflight:
- `docker ps` showed no running containers.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed no active compute processes.

Result:
- Device: `NVIDIA GB10`.
- Tree nodes: `14`, padded nodes: `16`.
- Corrected WY tree solve: `557.0748901367188 us`.
- Dense FR10-shaped fused kernel: `1038.9421081542969 us`.
- WY vs dense output maxabs: `2.7939677238464355e-09`.
- WY vs dense state maxabs: `2.9802322387695312e-08`.
- WY vs serial output maxabs: `8.702627383172512e-07`.
- WY vs serial state maxabs: `1.8596649169921875e-05`.
- Dense vs serial output maxabs: `8.703209459781647e-07`.
- Dense vs serial state maxabs: `1.8611550331115723e-05`.

Interpretation:

`CORRECTED_WY_TREE_SOLVE_MATCHES_DENSE_TREE_KERNEL`

The corrected kernel builds the tree-ancestry triangular WY factor `T = inv(I + A)` with `A[i,j] = beta_i <k_i,k_j> exp(G_i-G_j)` on strict ancestors, then applies `T @ (beta*v)` and `T @ (beta*exp(G)*k)`. It matches the dense tree kernel to fp32 roundoff while cutting the one-launch probe from `1038.94 us` to `557.07 us`. The shared `~1.86e-05` state gap versus the Python serial oracle is also present in the dense kernel, so it is the existing Triton-vs-Python arithmetic-order floor for this probe rather than a WY-specific mismatch.

## Serving Kernel WY Swap: Real Tensor Gate

Code change:
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` now uses the FR12 tree-ancestry WY triangular factor inside the existing `launch_tree_gdn_prepared` ABI.
- The FR10 vLLM patcher already imports `launch_tree_gdn_prepared`, so the no-copy tree verify path picks up the WY solve without changing the h0 bank loader, output buffer, or state commit/remap plumbing.

CPU checks:
- `python3 -m py_compile src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`
- `pytest -q tests/test_fr10_phase4_sampled_committer_wiring.py tests/test_fr10_decode_modes.py`
- Result: `20 passed in 0.10s`.

GPU command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  /workspace/scripts/fr10_phase4_real_tensor_validation.py \
  --payload /workspace/output/fr10_phase4_tree_capture_probe_20260603/tensors/language_model_model_layers_0_linear_attn_spec_gdn.pt \
  --out /workspace/output/fr12_wy_tree_real_tensor_validation_layer0.json
```

Preflight:
- `docker ps` showed no running containers.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed no active compute processes.

Result:
- `gate_d_real_tensor_decode_update_pass`: `true`.
- `tree_kernel_vs_native_decode_update_path_out_abs`: `7.450580596923828e-09`.
- `tree_kernel_vs_native_decode_update_path_state_abs`: `7.62939453125e-06`.
- `tree_kernel_vs_native_update_path_out_abs`: `7.450580596923828e-09`.
- `tree_kernel_vs_native_update_path_state_abs`: `7.62939453125e-06`.
- `tree_kernel_vs_gqa_ref_state_abs`: `5.7220458984375e-06`.
- `native_linear_vs_tree_ref_non_linear_nodes_abs`: `0.5618356466293335` negative-control contamination remains powered.

Interpretation:

`WY_SERVING_KERNEL_REAL_TENSOR_GATE_PASS`

The integrated serving kernel preserves the previously banked real-tensor decode-update parity while replacing the dense value-wise solve with the FR12 WY triangular factor. This validates the in-repo kernel swap before patching/running the full vLLM server path.
