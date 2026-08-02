# Fixed32 B4 FA2 qrow32 static batch specialization

Status: **host/source verified; GPU compile, binary-resource inspection,
real-event byte qualification, and timing pending**.

## Change

The private B4 qrow32 path still launches 96 complete-attention CTAs per layer,
but encodes them as `6 GQA lanes x 4 sequences x 4 KV heads` instead of
`1 query tile x 4 sequences x 24 query heads`. The mapping
`query_head = kv_head * 6 + lane` is a bijection over the prior batch/head
domain. For this private trait, `blockIdx.z` is already the KV head, so the
six repeated `query_head / 6` address expressions become one compile-time-dead
dynamic fallback.

The exact four-sequence gate already verifies packed query prefixes
`[0, 32, 64, 96, 128]`. Under that invariant, the four split-forward query and
output prefix lookups are replaced by `batch * 32 * row_stride`. The existing
compile-time paged-KV route is also enabled for qrow32, retaining the fixed
1024-row physical page and division-free 64-row K-block resolver.

## Preserved work

- CTA count remains 96 per layer and 1,536 across 16 target layers.
- Every `(batch, query_head)` pair is visited exactly once.
- Each CTA keeps `m_block = 0`, one complete reverse-ordered K traversal, and
  no split-K or combine kernel.
- Dynamic per-sequence K lengths and the page-table contents are unchanged.
- The pinned split-forward function retains four GEMM call sites, three
  softmax call sites, two mask sites, two tree-bias sites, and the same three
  K-loop forms.
- Tree-bias bounds, suffix masking, QK/PV accumulation order, and output/LSE
  element layouts are unchanged.
- Stock traits remain on the original dynamic coordinate path.

## Static evidence

| Split-forward source census | Before | Candidate |
| --- | ---: | ---: |
| `bidh / h_h_k_ratio` expressions | 6 | 1 dynamic fallback |
| Direct `binfo.q_offset` calls | 4 | 0 |
| Static packed-offset calls | 0 | 4 |
| Dynamic K-length uses | 14 | 14 |
| GEMM call sites | 4 | 4 |
| Softmax call sites | 3 | 3 |
| Mask call sites | 2 | 2 |
| Tree-bias call sites | 2 | 2 |

The remaining division is inside the nonstatic `if constexpr` branch. A GPU
compile and SASS inspection are required before claiming that the private
binary contains no integer divide sequence or redundant query-prefix loads.

## Verification

- FA2 candidate, qrow32 gate, and static-gate host tests: `26 passed`.
- Python byte compilation: pass.
- Pinned-header transform and second-pass idempotence: pass.
- Exhaustive 96-CTA bijection and packed-offset CPU proof: pass.
- `git diff --check`: pass.

No GPU command, synthetic performance probe, or real-task measurement was run.
Acceptance remains restricted to the standing real SWE-Verified task sets.

This directory contains aggregate source-level metadata only. It contains no
task text, prompts, responses, patches, model traffic, raw logs, environment or
process identities, binaries, credentials, or timing samples.
