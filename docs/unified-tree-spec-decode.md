# Unified Tree Spec Decode

`scripts/swe_x86_helpers/relaunch_qwen36_round.py --config Fb` is the maintained MTP path. It always uses the per-path drafter, the expanded parent-state target GDN verifier, and the engine `tree_path_lcp_max` sampler. `--spines=1` is the E5-equivalent chain through this same code path; `--spines>=2` is a regular multi-spine tree.

Surviving CLI options:

- `--config D`: legacy suffix-stack launch, kept outside the tree path.
- `--config Fb`: unified MTP tree launch.
- `--mtp`: MTP depth for `Fb`.
- `--spines`: number of regular tree root spines for `Fb`; `1` is the chain baseline.
- `--tree`: explicit regular `speculative_token_tree` literal for `Fb`.
- `--tree-debug`: enables bounded tree draft/accept debug logging.
- `--kv-cache-dtype`: overrides realized KV cache dtype when validating cache behavior.

Surviving environment options:

- `LUMO_GPU_MEMORY_UTILIZATION`: overrides the bundle GPU memory fraction.
- `LUMO_VLLM_MAX_NUM_SEQS`: overrides the bundle max concurrent sequences.
- `LUMO_VLLM_PORT`: direct vLLM server port.
- `LUMO_VLLM_PROXY_PORT`: proxy port used by `ModelServer`.
- `LUMO_VLLM_CONTAINER_NAME`: container name used by `ModelServer`.
- `LUMO_VLLM_LOGS_ROOT`: log directory mounted into the container.
- `LUMO_VLLM_TRITON_CACHE_ROOT`: Triton cache directory.
- `LUMO_VLLM_STATE_ROOT`: vLLM state directory.
- `LUMO_ENFORCE_EAGER`: requests eager execution in the generated bundle.
- `LUMO_CUDAGRAPH_MODE` / `LUMO_CUDA_GRAPH_CAPTURE`: overrides CUDA graph capture mode.
- `LUMO_CUDAGRAPH_CAPTURE_SIZES`: overrides CUDA graph capture sizes.
- `LUMO_FA_PACKED_CUDAGRAPH_SIZES`: shorthand for packed unique-node capture sizes.
- `LUMO_MTP_DRAFT_TRACE_FILE`: writes native MTP draft rows for diagnostics.
- `LUMO_TREE_PER_PATH_DRAFTER_LOG`: writes per-path drafter rows.
- `LUMO_TREE_PATH_LCP_LOG`: writes engine path0/winner LCP rows.
- `LUMO_TREE_ACCEPT_PATH_LOG`: writes accepted tree path rows.
- `LUMO_TREE_SPINES`: default value for `--spines` when the CLI flag is omitted.
- `LUMO_TREE_DRAFT_DEBUG`: internal export set by `--tree-debug`.
- `LUMO_BATCH_INVARIANT_VLLM`: set automatically for `Fb`; keep enabled for measurements.
- `LUMO_FA_UNIQUE_NODES`: internal export set automatically for `Fb`.
- `LUMO_FA_ACTIVATION_REPLAY_COMMIT`: internal escape hatch for accepted-path GDN replay commit diagnostics.
- `LUMO_FA_TREE_GROUP_SIZE`: internal diagnostic fallback for replay grouping.

Teacher-forced tree measurement must use `scripts/measure_spec_teacher_forced.py measure --mode tree`; its acceptance comes from the engine `tree_path_lcp_max` event for that forced prefix, not from a separate verifier query.
