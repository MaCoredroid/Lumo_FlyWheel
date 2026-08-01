# Fixed32 CFWD state-only layer-batch candidate

Status: **default off; source/static ready; GPU compile, byte gate, and timing
pending**.

## Change

The accepted-path committer invokes the 48-layer fused GDN operator only for
its in-place recurrent-state update. Its returned per-token output is not
assigned or consumed. The existing one-launch layer-batch candidate still
loaded `q`, normalized it, evaluated the state-by-`q` output dot product, and
wrote an output tensor for every layer, request, path slot, and value head.

This candidate keeps the ordered FP32 state recurrence unchanged and removes
only that dead output suffix from the unqualified one-launch candidate:

- no `q` pointer/load or q-normalization work;
- no output dot product, scale, output pointer, or output store;
- no candidate `obuf` allocation;
- the same 48 logical layers, 16 path slots, K/V/gate inputs, state reads,
  state-update order, intermediate/final state stores, and one physical
  recurrence launch;
- the native 48-launch graph remains the reference on every new process, and
  the first real nonzero accepted path must match every authoritative FP32
  running-state byte before the candidate may serve.

The runtime contract now requires `state_only_output_elided=true` whenever the
layer-batch route reports one physical recurrence call. This prevents an older
output-producing layer-batch binary from satisfying the new candidate route.

## Removed work

For the pinned 48-layer geometry (`B` requests, 16 path slots, 48 value heads,
K=V=128), the removed output dot product is:

`48 * B * 16 * 48 * 128 * 128` FP32 FMAs per event.

| Occupancy | Removed FP32 FMAs | Logical q-load bytes | Dead BF16 stores |
| --- | ---: | ---: | ---: |
| B1 | 603,979,776 | 37,748,736 | 9,437,184 |
| B4 | 2,415,919,104 | 150,994,944 | 37,748,736 |

The q-load count reflects the existing program geometry and is not a DRAM
traffic claim; cache reuse can reduce physical memory traffic. No latency or
throughput improvement is claimed before a real SWE-Verified timing pair.
The FMA count covers only the removed output dot product; removed q
normalization and scale operations are additional and are not folded into it.

## Qualification-integrity blocker

The inherited first-real byte gate is correct for state equality but is not a
normal one-launch event. A powered gate executes the 48-call reference graph,
the one-call candidate comparison graph, and then the one-call served candidate
graph: 50 physical recurrence calls plus host synchronization and state
clone/compare/restore work. A zero-accept event before the gate passes serves
the 48-call reference. The inherited static contract currently reports one
candidate call for either case.

Therefore this source must not be timed or accepted yet. The runtime counters
now expose B-indexed byte-gate pass and attempt maps so a timing boundary can
prove the gate was already passed and did not execute in a measured event.
Before GPU qualification, wire that guard into the timing reducer. Any gate or
pre-gate event must be rejected from the production work census.

## Static verification

- `81 passed` across the focused committer, GDN schedule, full-preseed,
  conv-commit wiring, and exact-commit suites.
- Python compile and Ruff passed for the changed kernel/test files; the patcher
  compiled successfully.
- `git diff --check` passed.
- The broader sampled-committer test file retains three failures already
  present at base `080c417ed`; none intersects this candidate.

## Required next gate

1. Compile the candidate on SM121 and require zero stack/local spill plus a
   resource comparison against the 48-launch reference.
2. Require `passed_before[B] == passed_after[B] == 1` and a zero gate-attempt
   delta at every measured timing boundary.
3. Run real SWE-Verified B1 and exact4 B4 Tail23/Hydra27 gates from the exact
   pushed source. Require first-real-nonzero raw-byte equality for all 48
   authoritative running-state rows at every used occupancy.
4. Require the work census to report 48 logical layers, one physical recurrence
   call, one graph replay, and zero fallback/readback on served post-gate events.
5. Time same-source stock versus candidate on the standing real task sets and
   report full-wall TPS plus CFWD ms/event. This candidate cannot independently
   close the current hardware-floor gap.
