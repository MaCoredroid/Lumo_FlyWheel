# Fixed32 SFWD conv/post-prep fusion

Status: **source-only candidate complete; not runtime-qualified or served**.

Base revision: `c49c8eb5370e4d4035aceffaa8476aea31f921f5`.

This candidate fuses exactly one fixed32 layer's frontier-5 tree-conv producer
with its immediate post-conv preparation. It directly writes distinct
unnormalized `query_spec`, `key_spec`, and `value_spec` recurrence buffers plus
`value_tree`, FP32 `g`, FP32 `beta`, and the unchanged commit-source stage. The
BF16 full-conv surface is absent on the hot path; a compile-time optional tap
retains it for diagnostics without restoring the consumer read.

The boundary ends before GDN. GDN, residual update, and the next layer remain in
their existing order. Cross-layer fusion is explicitly forbidden.

## Exactness

- 32 physical rows, width 4, state length 34, C=10,240
- K64/root1 qualification profile and exact fixed32 parent vector
- BF16 input/weight product rounded to BF16 for every tap
- ordered FP32 adds: bias, tap0, tap1, tap2, tap3
- SiLU result rounded to BF16 before every recurrence/post-prep store
- independent storage for query, key, value-spec, and value-tree outputs
- exact fused-post-prep softplus threshold and FP32 g/beta algebra
- unchanged `prior[0:3] ++ x[0:32] ++ zero` commit-source stage
- fail-closed shape, dtype, stride, storage-bound, and output-alias checks

## Static result

The no-tap B1 path removes 2,228,224 logical global bytes per layer and
106,954,752 bytes across 48 layers. It changes 2 launches/layer to 1, removing
48 launches across the model. B4 byte counts are exactly four times B1; launch
counts remain one candidate launch per layer.

Host-only SM121a compilation with `CUDA_VISIBLE_DEVICES` explicitly empty
produced 56 registers/thread and zero stack, local, and shared bytes for B1,
B4, B1+tap, and B4+tap. No binaries or compiler caches are checked in.

## Scope

There is no patcher import, runtime selector, byte-gate harness, graph capture,
Docker run, GPU run, service, request, task, response, timing, or acceptance
claim in this branch. A later served route requires a dedicated fail-closed
selector and authenticated real SWE byte A/B qualification.

The package contains only source hashes, deterministic static ledgers, reduced
offline codegen metadata, and host test output.
