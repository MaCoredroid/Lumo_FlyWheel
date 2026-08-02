# Fixed32 B4 FA2 qrow32 fused initial K/V page address

Status: **host/source verified; GPU compile, SASS inspection, real-event byte
qualification, and timing pending**.

## Selection

The first paged K and V tiles used two calls to the same per-thread page
resolver. Under the private qrow32 gate, both calls had identical thread,
logical K-block, block-table, partial-block, page-stride, row-stride, and
head-stride coordinates. Only the K/V allocation base pointers differ.

Later K and V resolver sites are intentionally unchanged because the FA2
pipeline advances V at `n_block` and K at `n_block - 1`. The compile-time-dead
append-KV path is also unchanged.

## Change

The qrow32-only `kStaticQueryBatch` branch now resolves the initial relative
page address once and adds it to the separate K and V base pointers. Its active
source resolver-call count changes from two to one. The qrow16 static-page
fallback retains two calls, and the generic paged fallback retains two calls.

The shared address remains exactly:

`physical_page * 1,048,576 + page_offset * 1,024 + column_offset`

The API gate still requires a 1,024-row physical page and identical canonical
K/V page, row, and head strides. Dynamic sequence length, block-table entries,
and final partial-tile clamping remain inputs to the single resolver call.

## Preservation

- Launch geometry remains `6 x 4 x 4 = 96` CTAs per layer.
- K-loop order and the staggered K/V pipeline are unchanged.
- QK/PV accumulation order, softmax, masking, tree bias, and output/LSE paths
  are unchanged.
- Dynamic K length and final partial-tile size remain dynamic.
- qrow16 and stock/generic paged attention retain their original address paths.

The source edit removes one duplicate active resolver expression. It is not a
measured instruction-count or performance claim. GPU compilation and private
SASS inspection must establish whether the compiler had already eliminated the
duplicate and must confirm resource usage before timing.

## Verification

- FA2 candidate, qrow32 gate, and static-gate host tests: `28 passed`.
- Python byte compilation: pass.
- Pinned-header transform: pass.
- Second-pass transform idempotence: pass.
- Exhaustive relative-address equivalence over 64 threads, 64 logical K blocks,
  four nonidentity physical pages, and all partial sizes 1 through 64: pass.
- Fail-closed rejection without the static K/V stride prerequisite: pass.
- `git diff --check`: pass.

No GPU command, synthetic performance probe, or real-task measurement was run.
Acceptance remains restricted to the standing real SWE-Verified task sets.

This directory contains aggregate source-level metadata only. It contains no
task text, prompts, responses, patches, model traffic, raw logs, environment or
process identities, binaries, credentials, or timing samples.
