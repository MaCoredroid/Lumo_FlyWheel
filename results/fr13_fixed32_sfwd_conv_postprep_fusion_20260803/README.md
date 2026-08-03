# Fixed32 SFWD conv/post-prep fusion

Status: **default-off eager and FULL-capture wiring complete; served arm blocked
pending a real-task byte-gate credential; not GPU measured**.

Base revision: `c49c8eb5370e4d4035aceffaa8476aea31f921f5`.
Guarded source revision: `bd225f7f80f9911d19a731cef109028767ac82d3`.

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
- fail-closed shape, dtype, stride, state-bank value, storage-bound, and
  output-alias checks
- eager SSI values checked directly; FULL capture requires the exact persistent
  pregather self-check lease, clamps the replay-time bank row before address
  formation, and permanently clears the existing sticky-committer scalar on an
  out-of-range row
- capacity-sized output banks allocated before capture, with exact persistent
  object/data-pointer bindings for every B1-B4 view

## Static result

The no-tap B1 path removes 2,228,224 logical global bytes per layer and
106,954,752 bytes across 48 layers. It replaces the incumbent conv launch,
three `rearrange_mixed_qkv` contiguous-copy launches, and post-prep launch with
one launch per layer: 5 to 1, removing 192 launches across the model. B4 byte
counts are exactly four times B1; launch counts remain one candidate launch per
layer.

Host-only SM121a compilation with `CUDA_VISIBLE_DEVICES` explicitly empty
produced 64 registers/thread for guarded no-tap B1/B4 and 56 for B1/B4 with the
diagnostic tap. All four profiles use zero stack, local, and shared bytes. No
binaries or compiler caches are checked in.

## Scope

The patcher and real-task launcher expose a fail-closed selector that remains
off by default. Eager mode retains the direct SSI range check. FULL capture is
accepted only after final-FULL preseed binds all 48 layer outputs, the exact
builder-owned SSI objects, conv-bank views, source stages, pregather lease, and
sticky-committer state. The capture launch path has no `.item()`, CPU copy, or
device synchronization. Valid rows add no sticky store; an invalid row is
clamped before its bank address is formed and atomically makes the later
committer assertion fail closed.

A naked `FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1` is rejected. Selection also
requires a raw-SHA-bound, single-link regular PASS and source manifest whose
schema, candidate, source revision, and current patcher/generator/module/kernel
hashes all match. No checked-in gate emits that PASS and no launcher forwards
those credentials yet, so this checkpoint cannot serve or time the candidate.

No Docker, GPU, service, real SWE task, response, timing, or acceptance run was
performed for this guarded wiring checkpoint. The static byte and launch ledger
is not a speed claim. A real-task byte gate and dedicated timing wrapper must be
completed before real B1 and B4 measurement.

The package contains source hashes, deterministic static ledgers, reduced
offline codegen metadata, and host/generated-patch verification output.
