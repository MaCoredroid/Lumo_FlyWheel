# Round-F CUDA Graph Captureability Spec - 2026-05-30

## Purpose

This spec records the intended CUDA graph boundary for the current Round-F
tree-delta / F-spine work after pulling `main` to `2e48fc0c`.

The important conclusion is that the new approach is not trying to force the
entire GDN/tree verifier into full CUDA graph capture today. The deliberate
shipping boundary is:

- keep the fixed-shape decode scaffolding and static tensor plumbing
  capture-compatible where vLLM can dispatch `FULL`;
- let vLLM downgrade unsupported GDN/tree attention regions to
  `FULL_AND_PIECEWISE` / piecewise execution;
- explicitly tag `gdn_attention_core` as `cudagraph_unsafe` when the
  correctness-preserving path needs to stay out of full capture;
- measure the actual runtime mode rather than trusting the requested config.

That boundary is consistent with the current evidence: correctness is the
primary win, while the speed target remains only partially met.

## Commit Readthrough

Pulled range:

- `232deda3` added record-only paired-run telemetry.
- `bfd5e95a` fixed cudagraph telemetry import scope.
- `9d5e7ed` fixed the B=4 SWE-Verified subset.
- `d7773fed` added `scripts/summarize_round_f_agentic_arm.py`.
- `6d9d3b47` and `5b02da2e` recorded the clean E3 arm.
- `fb6c99fa` made F tree depth-row tensors capture-safe.
- `e92d32bb` recorded the clean F-spine arm.
- `2e48fc0c` merged the branch.

The commit that matters most for captureability is `fb6c99fa`. It changes the
depth-row path from unconditional per-call `torch.tensor(...)` /
`torch.arange(...)` creation to cache-backed reuse. That is directionally
correct, but it should be read as a warmup/cache priming fix, not as a proof
that the path is safe on cache miss during active stream capture.

The older closeout report is more explicit about the intended non-captured
boundary. The best correctness/speed point used:

```bash
export LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1
export LUMO_FA_PACKED_CUDAGRAPH_SIZES=1
export LUMO_CUDAGRAPH_MODE=full
```

and the report says a FULL/FULL attempt without
`LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1` reached READY, but vLLM downgraded to
`FULL_AND_PIECEWISE` because `GDNAttentionBackend` was not full-capture safe in
this stack.

## Deliberate Non-Captured Parts

### 1. GDN core custom op

`scripts/swe_x86_helpers/relaunch_qwen36_round.py` includes
`_FA_GDN_CORE_CUDAGRAPH_UNSAFE_BLOCK`, which changes the vLLM custom op
registration for `gdn_attention_core` to include:

```python
tags=(torch._C.Tag.cudagraph_unsafe,)
```

That is not accidental. It tells PyTorch/vLLM graph machinery that this custom
op should not be captured as part of the graph-safe region. This matches
PyTorch guidance: custom ops are assumed graph-safe by default, so ops that
contain unsupported CPU work, dynamic behavior, or other unsafe behavior must be
explicitly marked unsafe.

### 2. Backend-level downgrade for GDN/tree attention

The clean B=4 F-spine report says `CUDAGraphMode.FULL` was requested, but
`GDNAttentionBackend` only reported `UNIFORM_BATCH` support, so vLLM forced
`FULL_AND_PIECEWISE`.

That is also deliberate. vLLM's CUDA graph dispatcher resolves the runtime mode
from both the requested mode and backend support. Its docs say unsupported full
capture modes are automatically downgraded to the closest supported mode. For
`UNIFORM_BATCH`, that can mean `FULL_AND_PIECEWISE`: full graph for compatible
uniform decode batches and piecewise handling for other work.

### 3. CPU/debug telemetry around captured regions

Telemetry is a measurement surface, not part of the graph contract. The pulled
code adds runtime-mode telemetry in `gpu_model_runner.py` and summarizes it in
`scripts/measure_track_b_real_workload.py`. That telemetry must remain outside
captured GPU work, or be guarded so it does not perform `.cpu()`, `.tolist()`,
`.item()`, file writes, or other host work while the stream is being captured.

The current code already has an example of the right guard:

```python
None if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
else tensor.detach().cpu().tolist()
```

This pattern should be applied consistently.

## Capture-Compatible Parts

These are the surfaces that should stay capture-compatible:

- persistent metadata buffers sized by `decode_cudagraph_max_bs`;
- `copy_`, `fill_`, `index_select`, `index_copy_`, and kernel launches that
  operate on stable addresses;
- fixed tree shapes or pre-bucketed capture shapes;
- packed cudagraph sizes from `LUMO_FA_PACKED_CUDAGRAPH_SIZES=1`;
- the F-spine top-1 chain shape, where depth and node count are stable enough
  to justify capture buckets.

`fb6c99fa` belongs here. It moves depth-row tensor materialization toward
long-lived cache entries so the captured path can reuse addresses.

## Capture Hazards Still Present

### Cache miss during capture

The current depth-row and clip-cache logic still creates tensors on cache miss.
If a miss happens while CUDA graph capture is active, the code is still
capture-risky. Caching is only sufficient if all keys that can appear during
capture are populated during warmup.

The stricter invariant should be:

```text
During active CUDA graph capture, metadata cache lookup must be hit-only.
No tensor allocation, CPU extraction, file I/O, or shape/topology discovery may
occur on the captured path.
```

### Python control flow and host scalar extraction

The Round-F patch surface still contains many `.item()`, `.tolist()`, `.cpu()`,
`torch.tensor`, `torch.arange`, `torch.empty`, and `torch.zeros` sites. Some are
prelaunch patch text, eager diagnostics, or known non-captured paths. The risk
is not their existence; the risk is accidentally moving any of them into a
runtime path that vLLM later captures.

### Dynamic topology

Full capture assumes static graph topology. Branched tree decode changes parent
structure and useful row subsets more often than the F-spine path. That is why
the branched path is currently research-only: even after correctness fixes, it
pays extra tree/GDN overhead and is less naturally compatible with fixed
cudagraph buckets.

## External Research Summary

### NVIDIA CUDA

NVIDIA's CUDA Programming Guide describes stream capture as recording work
issued into a stream, then replaying the instantiated graph later. It also
forbids synchronizing or querying a stream/event under capture, forbids
legacy-stream dependencies that would connect captured and non-captured work,
and invalidates the capture graph after invalid operations.

Relevant source:

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

### PyTorch

PyTorch documents the practical inference contract:

- warm up before capture;
- keep long-lived input/output tensors because replay uses the same addresses;
- no CPU/GPU sync such as `.item()` inside capture;
- no dynamic shapes;
- CPU work inside capture is not replayed;
- unsafe parts can run eagerly while graph-safe parts are captured separately.

Relevant sources:

- https://docs.pytorch.org/docs/2.12/notes/cuda.html
- https://docs.pytorch.org/docs/2.9/torch.compiler_cudagraph_trees.html

### vLLM

vLLM's current CUDA graph design makes full capture orthogonal to compilation
and uses a dispatcher to choose `FULL`, `PIECEWISE`, or `NONE` at runtime.
The attention backend advertises CUDA graph capability via an
`AttentionCGSupport` enum. `UNIFORM_BATCH` supports same-query-length batches
and can be used for speculative decode, but is weaker than `ALWAYS`.

Relevant sources:

- https://docs.vllm.ai/en/latest/design/cuda_graphs.html
- https://docs.vllm.ai/en/stable/api/vllm/v1/cudagraph_dispatcher/

### Upstream GDN Spec-Decode Context

The upstream vLLM issue/PR around Qwen3.5 hybrid GDN speculative decoding
confirms the recurrent-state class of bug: after accepting N speculative tokens,
the next step must read SSM/conv state from the accepted position, not from a
stale base block. The Round-F tree-delta work is solving the same family of
state-rollback correctness before optimizing capture.

Relevant sources:

- https://github.com/vllm-project/vllm/issues/39273
- https://github.com/vllm-project/vllm/pull/40738

## Proposed Engineering Contract

### Capture-safe fast path

For a path to be considered CUDA graph captureable:

1. All tensor buffers used by the captured path are allocated during init or
   warmup.
2. All metadata cache keys expected during replay are populated before capture.
3. The runtime path is hit-only for caches while
   `torch.cuda.is_current_stream_capturing()` is true.
4. The path performs no `.cpu()`, `.tolist()`, `.item()`,
   `torch.cuda.synchronize()`, file I/O, or stream/event query while captured.
5. Tree topology and tensor shapes are fixed for the selected capture bucket.
6. Runtime-mode telemetry proves `CUDAGraphMode.FULL` or the expected
   `FULL_AND_PIECEWISE` split.

### Capture-unsafe boundary

For GDN/tree attention regions that still need dynamic state repair or backend
support below `ALWAYS`:

1. Keep `gdn_attention_core` tagged `cudagraph_unsafe`.
2. Let vLLM dispatch piecewise/eager where required.
3. Treat `FULL_AND_PIECEWISE` as a valid deliberate runtime mode, not a failure,
   when the report's expected backend support is `UNIFORM_BATCH`.
4. Fail only if runtime mode is worse than expected, e.g. unexpected
   `CUDAGraphMode.NONE`, missing runtime telemetry, or an unplanned eager
   fallback spike.

### Fail-closed guard

Any allocation-prone cache lookup on the candidate captured path should follow
this structure:

```python
if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
    if key not in cache:
        raise RuntimeError("capture cache miss: warmup did not prime key")
else:
    cache.setdefault(key, build_tensors_for_key(...))
```

This avoids the subtle failure mode where a run succeeds after an eager cache
miss but fails or silently changes behavior during graph capture.

## Mitigation Plan: Full Capture and Cache Misses

This section is the proposed implementation plan for turning the F-spine path
from "mostly capture-compatible scaffolding plus deliberate piecewise GDN" into
a stricter full-capture candidate, while preventing the `fb6c99fa` cache fix
from hiding cache misses until capture time.

### Track A: eliminate capture-time cache misses

The current cache-backed depth-row fix is necessary but not sufficient. CUDA
graph replay assumes fixed topology, fixed kernel parameters, and stable memory
addresses. PyTorch and NVIDIA both recommend warming up before capture so
allocations and kernel initialization happen outside capture. Therefore, the
cache mitigation should be implemented as an explicit warmup contract:

1. Enumerate every cudagraph bucket before capture:
   - `num_spec_decodes`
   - `num_spec_decode_tokens`
   - `actual_conv_rows`
   - `fa_tree_depth_rows`
   - tree topology id (`spine_d3`, `spine_d4`, `branch_k2_d3`, etc.)
   - batch size / padded token count used by vLLM's `BatchDescriptor`

2. Add a `prime_fa_tree_metadata_cache()` helper in the vLLM patch block. It
   should run during model warmup, before vLLM starts stream capture, and build:
   - `fa_tree_parent_indices_tensor`
   - `fa_tree_depth_row_tensors`
   - `fa_tree_depth_query_start_tensors`
   - `_lumo_fa_tree_depth_cache`
   - `_lumo_fa_conv_depth_clip_cache`

3. Convert all captured-path cache accesses to hit-only lookups:

```python
_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
if _capturing and key not in cache:
    raise RuntimeError(
        "LUMO_FA capture cache miss: "
        f"key={key}; run warmup/prime_fa_tree_metadata_cache first"
    )
if key not in cache:
    cache[key] = build_tensors_for_key(...)
```

4. Count misses even outside capture:

```python
self._lumo_fa_tree_depth_cache_misses += int(key not in cache)
```

The measurement harness should fail any "captureable" claim if the final
artifact reports nonzero cache misses after warmup.

### Track B: replace dynamic caches with fixed buffer tables

For the F-spine path, a stronger approach is to remove runtime cache building
entirely. Use fixed module-owned buffers sized by maximum capture bucket:

```text
parent_indices_buf:       [max_nodes]
depth_row_indices_buf:    [max_depth, max_nodes]
depth_row_counts_buf:     [max_depth]
depth_query_start_buf:    [max_depth, max_nodes + 1]
clip_row_indices_buf:     [max_actual_rows, max_depth, max_nodes]
clip_row_counts_buf:      [max_actual_rows, max_depth]
```

The captured path then consumes fixed-address buffers and count tensors instead
of Python tuples. Empty depth rows become count `0`, not missing Python entries.
This avoids:

- `torch.tensor(_rows, ...)`
- `torch.arange(...)`
- Python tuple reconstruction
- cache-key hashing during capture
- changing the number of row tensors seen by the loop

For full capture, the loop topology must also be fixed. That means iterating
over `range(max_depth)` and passing count `0` for unused rows, rather than
iterating over `zip(_depth_rows, _depth_row_tensors, ...)` whose length can vary
by tree. If the kernel path cannot tolerate empty row dispatches efficiently,
bucket by depth and capture one graph per depth.

### Track C: make GDN core full-capture eligible

The current `gdn_attention_core` unsafe tag should remain until the GDN core
can satisfy the same rules as the scaffolding. To remove the tag safely:

1. Remove host scalar extraction from the full-capture candidate path:
   - no `int(tensor.item())`
   - no `.cpu().tolist()`
   - no Python parent traversal over GPU tensors
   - no file/debug logging inside capture

2. Move parent/depth/topology derivation out of the captured forward path. The
   captured path should receive tensors, not derive them.

3. Convert dynamic accepted-row/state-copy decisions into tensorized kernels:
   - accepted count is a device tensor or fixed scalar bucket value;
   - state copy runs as a fixed-shape kernel over `[batch, max_nodes]`;
   - rejected suffix rows are masked, not skipped by Python control flow.

4. Use one full-capture bucket per stable topology/shape:
   - F-spine depth 3 is the first candidate;
   - F-spine depth 4 is a separate graph;
   - branched trees are separate graphs or remain piecewise until their topology
     is stable enough.

5. Only after those changes should the backend advertise stronger graph support
   than `UNIFORM_BATCH`. In vLLM terms, the target is for this patched GDN path
   to behave like an `ALWAYS` backend for the selected bucket. Until that is
   true, `FULL_AND_PIECEWISE` is the honest mode.

### Track D: runtime enforcement and artifact gates

Add an explicit "capture contract" object to the debug JSONL and final summary:

```json
{
  "capture_contract": {
    "requested_mode": "full",
    "expected_runtime_modes": ["CUDAGraphMode.FULL"],
    "metadata_cache_misses_after_warmup": 0,
    "capture_cache_misses": 0,
    "gdn_core_cudagraph_unsafe": false,
    "fixed_buffer_metadata": true,
    "topology_bucket": "spine_d3",
    "full_capture_claim": true
  }
}
```

For the current deliberate piecewise boundary, the gate should instead expect:

```json
{
  "expected_runtime_modes": [
    "CUDAGraphMode.FULL",
    "CUDAGraphMode.PIECEWISE"
  ],
  "gdn_core_cudagraph_unsafe": true,
  "full_capture_claim": false
}
```

This prevents ambiguous results where `LUMO_CUDAGRAPH_MODE=full` was requested
but vLLM silently downgraded.

### Track E: implementation sequence

1. Add cache-miss counters and fail-closed guards around
   `_lumo_fa_tree_depth_cache` and `_lumo_fa_conv_depth_clip_cache`.
2. Add warmup cache priming for all configured `LUMO_CUDAGRAPH_CAPTURE_SIZES`
   / packed sizes.
3. Rerun the current F-spine configuration and require:
   - no capture cache misses;
   - expected `FULL_AND_PIECEWISE` runtime mode;
   - unchanged acceptance distribution within noise.
4. Replace tuple caches with fixed buffer tables for `spine_d3`.
5. Remove `gdn_attention_core` unsafe tag only for `spine_d3` behind a new env
   flag, for example `LUMO_FA_GDN_FULL_CAPTURE_EXPERIMENT=1`.
6. Rerun with runtime telemetry requiring `CUDAGraphMode.FULL` on the target
   bucket. If vLLM still downgrades, inspect backend support rather than
   treating the run as a full-capture success.
7. Expand from `spine_d3` to additional fixed topology buckets only after the
   first full-capture bucket is invariant-clean.

### Track F: what not to do

Do not use `capture_error_mode="relaxed"` or rely on `cudaMallocAsync` graph
memory nodes as the first mitigation for this path. CUDA supports graph memory
nodes, but this code is built on PyTorch/vLLM and already depends on stable
replay addresses, fixed vLLM batch descriptors, and cacheable metadata. The
lower-risk fix is to preallocate and prime deterministic buffers. Relaxed
capture can mask unsafe side effects and make correctness failures harder to
attribute.

## Measurement Requirements

Every future Round-F captureability claim should include:

- requested `LUMO_CUDAGRAPH_MODE`;
- actual `cudagraph_runtime_summary.runtime_modes`;
- `full_count`, `piecewise_count`, and `eager_fallback_count`;
- acceptance distribution;
- invariant failures;
- whether `LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1` was enabled;
- whether `LUMO_FA_PACKED_CUDAGRAPH_SIZES=1` was enabled;
- a cache-miss counter for depth-row and clip-cache metadata;
- if Nsight is used, whether the sqlite contains CUDA kernel tables, not just
  GPU metrics tables.

## My Recommendation

Do not try to force the current GDN core into full graph capture as the next
step. The current evidence says that path is correctness-sensitive and that vLLM
already downgrades it based on backend capability. The next useful work is:

1. make the capture-compatible scaffolding strictly hit-only during capture;
2. add cache-miss counters and fail-closed guards;
3. preserve `gdn_attention_core` as cudagraph-unsafe until a dedicated backend
   patch can advertise stronger `AttentionCGSupport`;
4. rerun paired E3/F-spine measurements on the same B=4 workload with runtime
   mode telemetry enabled;
5. only then decide whether a deeper backend rewrite to make GDN full-capture
   safe is worth the complexity.

In short: the deliberate non-captured part is a feature, not a bug, for this
iteration. The bug would be claiming full capture while letting cache misses,
CPU telemetry, or GDN dynamic state repair leak into the captured path.
