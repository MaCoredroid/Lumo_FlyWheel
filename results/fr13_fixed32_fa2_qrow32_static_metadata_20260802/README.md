# Fixed32 B4 FA2 qrow32 static paged metadata

Status: **host/source verified; GPU compile, SASS inspection, real-event byte
qualification, and timing pending**.

## Change

The private qrow32 trait now constructs only the metadata required by fixed32
paged verification. Its lightweight object sets the query extent to 32 and
loads the dynamic K extent directly from `seqused_k[batch]`.

This is exact under the fail-closed private contract:

- `seqused_k` is nonnull and supplies the active K length.
- Paged varlen FA2 rejects left padding, and the gate requires
  `leftpad_k == nullptr`.
- The varlen entry point zero-initializes cache remapping, and the gate requires
  `cache_batch_idx == nullptr`.
- The gate requires a nonnull 1024-row paged block table.
- The private template forbids split-K and append-KV.

The same trait forms `block_table + batch * block_table_batch_stride` directly
and uses the already-specialized KV head for K/V base offsets. The complete
generic `BlockInfo`, cache-remap, dense-KV, and nullable-table path remains in
the nonstatic `if constexpr` branch used by stock traits and qrow16.

## Source work model

Under the gated values, generic `BlockInfo` logically consumes two query-prefix
entries, two K-prefix entries, and one `seqused_k` entry per CTA. The private
constructor consumes only the `seqused_k` entry. These are contract-evaluated
source loads, not measured instructions; compiler and SASS confirmation remain
pending.

| Pinned split-forward census | Before | Candidate |
| --- | ---: | ---: |
| Required dynamic metadata sources | Q prefix, K prefix, `seqused_k` | `seqused_k` only |
| Contract-evaluated metadata loads/CTA | 5 | 1 |
| Static-path cache-remap checks | 1 | 0 |
| Static-path block-table null checks | 1 | 0 |
| Dynamic K-length uses | 14 | 14 |
| GEMM call sites | 4 | 4 |
| Softmax call sites | 3 | 3 |
| Mask call sites | 2 | 2 |
| Tree-bias call sites | 2 | 2 |

The launch remains `6 x 4 x 4 = 96` CTAs per layer. Query/KV row mapping,
page resolution, suffix bias, masking, softmax, QK/PV accumulation order, K
loop order, and output/LSE layouts are unchanged.

## Verification

- FA2 candidate, qrow32 gate, and static-gate host tests: `27 passed`.
- Python byte compilation: pass.
- Pinned-header composed transform and second-pass idempotence: pass.
- CPU proofs for dynamic K length and block-table row equivalence: pass.
- Split-forward arithmetic/control census preservation: pass.
- `git diff --check`: pass.

No GPU command, synthetic performance probe, or real-task measurement was run.
Acceptance remains restricted to the standing real SWE-Verified task sets.

This directory contains aggregate source-level metadata only. It contains no
task text, prompts, responses, patches, model traffic, raw logs, environment or
process identities, binaries, credentials, or timing samples.
