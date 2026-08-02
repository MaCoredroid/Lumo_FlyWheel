# Fixed32 GDN two-launch batch candidate

Source base: `e45a7e7355b9ba9fea3ee6c30afbf7d8e85b1cdd`

Status: source and CPU/static contract complete. No CUDA build, GPU correctness
gate, or performance measurement has been run. The live real SWE-Verified
campaign was not touched.

## Kernel contract

The existing exact fixed32 caller launches two path levels once per request.
Across 48 GDN layers this is:

| Batch | Before | Candidate | Removed |
| ---: | ---: | ---: | ---: |
| B1 | 96 | 96 | 0 |
| B2 | 192 | 96 | 96 |
| B3 | 288 | 96 | 192 |
| B4 | 384 | 96 | 288 |

The candidate keeps B1 on the existing `_tree_gdn_path_kernel`. For B2-B4,
request is folded into path-grid axis 2:

```text
request = global_path // paths_in_level
path = global_path % paths_in_level
row = request * 32 + node

level 0 grid paths = B * 1
level 1 grid paths = B * 11
```

Each request and path retains the same `_gdn_node_step` order. Arithmetic and
CTA counts still scale with real requests; only independent per-request launch
nodes are coalesced.

No new scratch is allocated. The existing 32-row fp32 handoff buffer is used
as `[B, 5]` compact slots for root-path nodes `(0, 1, 4, 9, 14)`. B4 consumes
20 rows. The level-1 parent slots are `(4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)`.

## Verified without GPU

- The legacy path-kernel AST hash is unchanged from the exact-safe base.
- CPU reference mapping proves B1-B4 path order, output/ring row coverage,
  request isolation, compact scratch isolation, and exactly two level launches.
- Targeted result: `5 passed, 1 skipped` (the skip is the existing CUDA exact-I/O
  test because this source work intentionally did not acquire the GPU).
- Python compile and `git diff --check` pass.

## Required deployment gates

1. Build the production image and confirm the new Triton specialization
   compiles for B2-B4 without changing the B1 binary.
2. Byte-compare legacy versus batched output, ring K/V/A/B, staging flags,
   invocation counter, and untouched scratch/capacity rows for B2-B4.
3. Exercise both running-row and accepted-column h0 selection with distinct
   request state-bank rows, then CUDA graph capture/replay for B2-B4.
4. Measure only the standing real SWE-Verified exact4 set, reporting full wall
   TPS, SFWD GPU time per physical step, actual batch histogram, and kernel
   launch counts. Compare against the current exact-safe arm, not July 23-25
   pre-fixed32 B4 artifacts.
