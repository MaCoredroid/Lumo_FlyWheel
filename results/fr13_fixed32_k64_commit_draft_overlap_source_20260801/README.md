# Fixed32 K64 commit-draft overlap source candidate

Status: `SOURCE_READY_GPU_NOT_RUN`

- GPU used for this work: `false`
- Source base: `d383ec46d03a08b5138b86471aecec1199a14ae3`
- Candidate source commit: `ed157906617c664fa2ea69b4682da3f64ebe58e5`
- Branch: `agent/fixed32-k64-dfwd-cfwd-overlap-20260801`
- Scope: physical-32 Tail23/Hydra27, B1/B4, K=65536, root reduction on
- Default: off; no production bake or performance claim

## Schedule

After rejection sampling publishes the accepted paths, the candidate records a
default-stream ready event and queues these exact incumbent operations on one
persistent side stream:

1. fixed32 48-layer convolution column-0 commit;
2. fixed32 accepted-path input copies, guards, and the one-replay 48-layer GDN
   state/ring commit graph;
3. fixed32 flags clear;
4. the 16-layer target full-attention KV linear remap.

The independent MTP/K64 drafter remains on the default stream. A completion
event is joined immediately after DFWD, before draft copying and connector
finalization. A second pre-connector fence covers proposal-less/max-length
paths. Two reusable event slots prevent hot-path allocation. Exact mode, batch,
step sequence, and request-owner tuples are checked at every lifecycle edge.

The moved target state banks and target KV caches are disjoint from the MTP
drafter state and its single MTP KV cache. Accepted path/lens inputs are
read-only after publication. No math, kernel launch geometry, state update
order within the commit tail, RNG call, or rejection-sampling decision changes.

## Unhidden work

This candidate does not hide SFWD, rejection sampling, bonus/path selection,
accepted-output publication, host-side target-KV validation/payload setup,
committer accounting, CUDA event handoff, connector finalization, or any commit
tail that outlasts DFWD. It also does not reduce mandatory weight traffic.
The candidate CFWD timer covers the remaining default-stream prefix; the new
task-boundary census reports the overlapping tail separately, so those
components must not be summed.

## Ceiling

The current real exact4 B1 K64 Hydra27 reference is:

- wall: 232.779790071 ms/event, 24.718146718 full-wall TPS;
- SFWD: 159.619263244 ms/event;
- DFWD: 36.813368134 ms/event;
- CFWD: 20.677390557 ms/event;
- other: 15.669768137 ms/event;
- floor: 119.658015414 ms; one-sided U95 cap: 137.6067177261 ms.

Even the impossible upper bound that hides all CFWD behind DFWD is
`159.619263244 + max(36.813368134, 20.677390557) + 15.669768137 =
212.102399515 ms/event`: 27.127863792 TPS, 1.772571597x floor, and
74.495681789 ms above the cap. It saves at most 20.677390556 ms/event
(8.882812% wall, 9.748777% TPS). The implemented exact-safe tail is smaller
than all CFWD, so this is a strict unattainable ceiling for this lever alone.

## Contention warning

The earlier real SWE-Verified exact16 B4 overlap campaign at commit `8822ffe5d`
is a no-bake precedent. Its candidate measured 158.588491672 ms/event and
36.589505599 TPS versus 145.206705579 ms/event and 38.011849711 TPS for the
reference, a 3.74% wall-TPS regression despite a lower committer span. K64 may
change the balance, but DFWD head/weight traffic, GDN state traffic, and target
KV copies still share GB10 unified-memory bandwidth. The exact4 wall result,
not isolated tail time, decides this candidate.

## Live qualification

Run only after the active GPU campaign releases the device and after composing
this commit onto the final K64 physical32 stack. For each Tail23/Hydra27 and
B1/B4 cell:

1. run the canonical real SWE-Verified exact4 subset with stock and candidate
   arms, with the only delta `FR13_FIXED32_COMMIT_DRAFT_OVERLAP=0/1`;
2. keep K=65536, root reduction on, physical drafts=31/rows=32, canonical block
   map SHA-256, work census, timers, graph mode, and request isolation pinned;
3. require every immutable overlap task-boundary snapshot to reconcile
   `begun == sealed == fenced+flush_fenced == timed_spans`, pending=false, and
   the expected B1/B4 occupancy histogram;
4. reduce the pair with `scripts/fr13_fixed32_commit_draft_overlap_gate.py` to
   report full wall TPS, SFWD, DFWD, default-stream CFWD, residual wall, and the
   overlapping commit-tail time for the exact task interval;
5. reject immediately on any wall regression, lifecycle/order failure, work
   census drift, output corruption, or request-owner mismatch;
6. if exact4 improves wall time, repeat the byte/state gate and formal exact16
   run. Production still requires one-sided U95 wall <= 137.6067177261 ms.

No synthetic workload or one-task probe may qualify this candidate.

