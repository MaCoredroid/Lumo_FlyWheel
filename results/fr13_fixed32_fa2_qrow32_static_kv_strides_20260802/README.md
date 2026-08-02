# Fixed32 B4 FA2 qrow32 static K/V strides

Status: **host/source verified; GPU compile, SASS inspection, real-event byte
qualification, and timing pending**.

## Selection

The qrow32 translation unit already instantiates `Is_causal=false` and
`Is_local=false`, so causal and window branches are compile-time discarded.
There was no additional safe runtime predicate to remove in that control path.

The next hot address dependency was the paged K/V resolver. Its block/page
coordinates were already static, but page, row, and head strides still arrived
through runtime parameters at every resolver call site.

## Change

The private qrow32 trait now fixes the canonical contiguous BF16 K/V layout in
elements:

| K/V dimension | Static stride |
| --- | ---: |
| Physical page | 1,048,576 |
| Token row | 1,024 |
| KV head | 256 |

The fail-closed API gate verifies both K and V batch/page strides in addition
to the existing row and head stride checks. The private page resolver selects
the trait constants with `if constexpr`; qrow16 and all stock traits retain the
runtime stride formula.

For qrow32 the resolved element address remains exactly:

`physical_page * 1,048,576 + page_offset * 1,024 + column_offset`

The K/V base-head offset similarly remains `kv_head * 256`. Page-table row
selection and physical page identifiers remain dynamic and unchanged.

## Preservation

- Launch geometry remains `6 x 4 x 4 = 96` CTAs per layer.
- Dynamic sequence lengths and page-table contents are unchanged.
- The split-forward function retains 14 dynamic K-length uses, four GEMM call
  sites, three softmax call sites, two mask sites, and two tree-bias sites.
- K-loop order, QK/PV accumulation order, suffix bias, masking, and output/LSE
  layouts are unchanged.
- The qrow16 translation unit does not specialize the stride trait.

These are source and contract facts, not measured instruction counts. GPU
compile and SASS inspection must confirm constant lowering, resource use, and
the absence of regressions before the candidate is timing-eligible.

## Verification

- FA2 candidate, qrow32 gate, and static-gate host tests: `27 passed`.
- Python byte compilation: pass.
- Pinned-header and pinned-resolver migration: pass.
- Second-pass transform idempotence: pass.
- Exhaustive page/row/column address equivalence model: pass.
- Split-forward arithmetic/control census preservation: pass.
- `git diff --check`: pass.

No GPU command, synthetic performance probe, or real-task measurement was run.
Acceptance remains restricted to the standing real SWE-Verified task sets.

This directory contains aggregate source-level metadata only. It contains no
task text, prompts, responses, patches, model traffic, raw logs, environment or
process identities, binaries, credentials, or timing samples.
