# Fixed32 Hydra27 DFWD attention batching audit

Status: **stopped; attention-only fusion is not lossless**.

This is a host-only audit based on repository commit
`c49c8eb5370e4d4035aceffaa8476aea31f921f5`. It used no GPU, Docker,
serving runtime, main campaign worktree, or synthetic timing traffic. The
historical 6.967564 ms/event value is retained only as an attribution ceiling;
it is not a current measurement or candidate speed claim.

## Exact dependency

Hydra27 has four post-root MTP model forwards. At level `i`, the unified
attention output is consumed by the rest of that model forward. The resulting
hidden state is projected through the K64 head, and rank 0 of the exact top-3
result becomes `input_ids` for forward `i+1`. The returned hidden state is also
an input to that next forward. Before its attention runs, sequence length is
incremented and the next paged-KV suffix row is written.

The blocking chain is therefore:

```text
A_i.out -> post-attention model -> K64 logits -> top3 rank0
        -> next input token + prior hidden -> Q_(i+1)
```

The Hydra27 runner-up ranks are packing-only, but that does not remove the
rank-0 spine recurrence. The four attention nodes are nonconsecutive CUDA-graph
nodes separated by model, head, selection, metadata, and KV-write work. A
single same-stream attention kernel would prevent those intervening nodes from
producing its later queries. Fusing them would require fusing the entire MTP
model recurrence, not an attention-only kernel.

The graph also reuses query, output, and sequence-length workspaces across the
four forwards. Deferred attention would read the final values unless every
level added snapshots. The existing BM8 diagnostic snapshots query and length
for precisely this reason; those copies do not make production deferral valid
because each attention output is already needed inside its level.

## Static ledger

Let `S` be the sequence length at the first post-root attention call, under the
explicit no-max-length-rollover guard. The four calls see `S`, `S+1`, `S+2`,
and `S+3`.

| Item | Exact count |
|---|---:|
| Unified-attention launches/event | 4 |
| BM8 CTAs/launch | 4 KV heads |
| BM8 CTAs/event | 16 |
| Query reads/event | 49,152 bytes |
| Output writes/event | 49,152 bytes |
| K+V bytes per sequence row | 4,096 bytes |
| K+V reads/event | `4096 * (4*S + 6)` bytes |
| BM8 physical QK+PV FLOPs/full 32-row tile | 1,048,576 |
| Live-head QK+PV FLOPs/full 32-row tile | 786,432 |

The byte rows count mandatory Q/output/K/V payload and source-level logical
block-table loads; the work rows count QK and PV dot products. They do not
claim compiler-dependent metadata transactions or physical HBM sectors.

The current Triton source forms 32 identical block-table addresses per
32-row tile and KV-head CTA. Replacing that vector lookup with one scalar
lookup is algebraically safe only under the guarded 1024-row page and 32-row
tile layout. Its optimistic logical saving is 496 bytes per full tile across
all four KV heads versus 131,072 K/V payload bytes, or 0.3784%. It does not
prove a physical-memory saving because identical addresses can coalesce and
the small page table is cache-resident. No runtime candidate is integrated for
an unproven sub-0.4% logical-byte change.

Preparing the common K/V prefix once is strictly worse at unchanged precision:
the original four reads cost `4B`; read+write staging followed by four consumer
reads costs `6B`. The block table itself is stable across these sites. Expanding
it once from one entry per 1024-row page to one entry per 32-row tile is exact
when built through `S+3`, but materializes 32 entries per source page and then
adds consumer reads. A direct scalar page lookup needs no expanded map and has
the same unproven physical-memory value described above.

## Decision

No attention kernel, launcher selector, or production credential changed. The
minimum lossless attention-only launch count remains four. `audit.json` binds
the exact topology, algebra, layouts, aliases, recurrence edges, source hashes,
and symbolic byte/work ledger. Focused tests fail closed on K64/root1,
physical32, Hydra27 mask, B1 Q1/BM8, paged-KV layout, metadata, and alias drift.
