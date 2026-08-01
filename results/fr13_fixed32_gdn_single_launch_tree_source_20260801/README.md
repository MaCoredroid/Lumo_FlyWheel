# FR13 fixed32 GDN depth-first single-launch source candidate

Status: **source candidate only, default OFF, no GPU measurement**.

## Exact schedule

The fixed32 logical schedule remains one five-node root path plus eleven
terminal paths. This candidate changes only physical execution. One CTA owns
one request, value head, and `BV=8` value tile. It advances the root chain and
interleaves the side paths whose parent state is currently live:

```text
node  0 -> paths 1,2  -> continue node 1
node  1 -> paths 3,4  -> continue node 4
node  4 -> paths 5,6  -> continue node 9
node  9 -> paths 7,8  -> continue node 14
node 14 -> paths 0,9,10
```

Each side path starts from a branch copy of the current root tile. The root
tile remains unchanged and advances only through the next root-chain node.
Inter-path ordering is unobservable: every node has exactly one output and
K/V/A/B ring writer, and every recurrence edge uses the same parent-state
bytes and unchanged `_gdn_node_step` body as the two-launch path route.

The physical program keeps two fp32 state tiles live: the current root tile
and one branch tile. At `BV=8`, `DIM_K=128`, this is 8,192 aggregate state
bytes per CTA. With the deployed eight warps, ideal even distribution is eight
fp32 state values per thread; compiler temporaries, layout replication, and
actual register allocation remain a required SM121 compile check.

## Work model

The observer deliberately preserves the logical schedule census while adding
a distinct physical recurrence census:

| Metric | Original path | Parent-group | Single launch |
| --- | ---: | ---: | ---: |
| Logical critical path | 12 | 12 | 12 |
| Physical recurrence critical path | 12 | 17 | 32 |
| Launches per layer | 2 | 2 | 1 |
| CTA units per request/layer/VH/V-tile | 12 | 6 | 1 |
| Export-state writes | 5 | 5 | 0 |
| Parent-state reads | 11 | 5 | 0 |

At 48 layers, 48 value heads, `DIM_V=128`, and `BV=8`:

- B1 candidate CTAs/event: 36,864, down from 442,368 original and 221,184
  parent-group.
- B4 candidate CTAs/event: 147,456, down from 1,769,472 original and 884,736
  parent-group.
- B1 eliminated state handoff: 2,415,919,104 bytes versus original, or
  1,509,949,440 bytes versus parent-group.
- B4 eliminated state handoff: 9,663,676,416 bytes versus original, or
  6,039,797,760 bytes versus parent-group.
- Physical CTAs per layer remain 768 at B1 and 3,072 at B4.

These counts are descriptor-derived, not timing or achieved-bandwidth claims.
The central tradeoff is explicit: the candidate removes the launch boundary
and all fp32 state HBM handoff, but serializes 32 recurrence updates in each
CTA instead of the original physical critical path of 12 or parent-group's 17.

A stronger follow-on should retain the two-launch/root-export boundary and run
each same-parent level-1 group with member-SIMD lanes inside one CTA. Its group
maximum path lengths are `(7, 7, 1, 1, 1)`, so the physical recurrence stays
`root 5 + group max 7 = 12` while parent loads and CTA units remain reduced
from 11 to 5. The cost is a padded `[4, 8, 128]` fp32 state tensor, about 128
ideally distributed fp32 values per thread at eight warps before temporaries,
plus masked lane work. That resource tradeoff should be compiled alongside
this physical-32 candidate once the GPU is free.

## Route and reference safety

The route is armed only by `FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=1` or a
worker-visible `.arm` file containing exactly `1`. It requires Tail23 or
Hydra27 fixed32 mode and exact `FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`. It is
mutually exclusive with parent-group and path-BV selectors and fails closed on
topology, schedule, writer, geometry, or descriptor drift.

Both B1 and batched B2-B4 launchers use one grid-z program per request. The
batched byte-reference helper passes `force_reference_structure=True`, which
removes both structural descriptors before launching the incumbent two-level
path kernels. This prevents candidate-vs-candidate qualification.

No production authorization is included. Existing batched byte-gate export
surface semantics still assume a state-exporting candidate; a dedicated live
gate must compare the externally consumed output/ring/flag/counter surfaces
and prove the eliminated export scratch is dead before production use.

## Verification

Focused single-launch, parent-group, and BV8 production-sidecar tests: 24
passed. The full fixed32 test family with `PYTHONPATH=.`: 781 passed, 8 skipped,
3 failed only because this isolated worktree intentionally lacks the private
`.venv` and no-symlink `.cache` runtime prerequisites. Work-census self-test:
PASS, 167 tamper tests. Python compilation and `git diff --check`: PASS.

No GPU command was run while the real B4 SWE-Verified campaign owned the
device. Triton compilation, SM121 register/local/shared-memory resources,
raw-byte equality, CUDA graph capture/replay, and real SWE-Verified B1/B4
full-step timing remain unresolved.
