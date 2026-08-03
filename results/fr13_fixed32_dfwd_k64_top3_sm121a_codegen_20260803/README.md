# Fixed32 DFWD K64 mapped-top3 SM121a codegen

Status: **BUILT_UNQUALIFIED** at source commit
`fef06f1eb2ab17d99849bd28d99b4cfc37649e66`.

The fixed32 K64 B1 drafter has five logit heads per event: one eager root and
four captured MTP heads. The stock path performs a separate BF16 argmax,
`torch.topk(..., 3)`, subset-ID gather, and graph-output copy for each head.
The default-off candidate scans one contiguous `[1,65536]` BF16 row, produces
the descending top three subset indices, maps them through the pinned K64 ID
map, and writes the persistent spine and width-three graph outputs in one
CUDA launch.

## Motivation

The historical real-SWE B1 attribution package
`results/fr13_fixed32_b1_nsys_20260731T013952Z_curated` contains 881 DFWD
events and is attribution-only, not acceptance-valid candidate evidence. It
records 4,405 argmax launches and, for top-k, 8,810 launches each of block
digit counts, digit cumulative sum, scan by key, and within-k counts plus
4,405 gathers. This is 10 reduction launches per head and five heads per
event.

If the new route engages at its exact geometry, its structural reduction
count is five launches per event instead of 50, or 45 fewer. It also removes
the separate argmax scan of 131,072 logit bytes per head, an analytical
minimum of 655,360 redundant input bytes per event. These are source and
shape accounting values, not measured GPU traffic or performance.

## Offline build

- PyTorch: `2.11.0+cu130`
- CUDA release: `13.0`
- Target: `sm_121a`
- Grid/block: `1 x 256`
- Registers/thread: `30`
- Stack/local bytes: `0 / 0`
- Static shared bytes: `1,216`
- Binary SHA-256: `55229a9db7364fc8c0811fe34d3eaf06bc577626a3455fcc25a0fb9990aa480b`
- Binary bytes: `159,288`

The build completed in the pinned container without GPU device passthrough.
The shared object registered with the pinned PyTorch dispatcher, and
`cuobjdump` found exactly one `sm_121a` cubin. Static SASS contains no local
loads, local stores, or calls.

## Qualification boundary

The selector defaults off and fails closed unless the run is B1, fixed32,
root-enabled K64, single-logits, width three at all five depths, and full
CUDA-graph mode. The launcher verifies the candidate SHA-256, mounts it read
only at a fixed container path, and the real gate requires ready, engaged,
and four-call graph-capture markers.

No candidate GPU kernel, service, SWE-Verified task, timing run, acceptance
run, or byte-equivalence run was executed for this artifact. The historical
profiler census only motivated the kernel. The checked-in package omits the
raw shared object, cubin, PTX, SASS, compiler cache, raw logs, task content,
credentials, process IDs, and container IDs. The real one-task diagnostic
must run first; standing acceptance still requires the canonical four-task
set (and the 16-task set where required).
