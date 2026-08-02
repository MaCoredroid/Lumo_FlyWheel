# Fixed32 CFWD GDN layer-batch source candidate

Status: **source candidate only; default OFF; no GPU or throughput claim**.

## Attribution

The 48 `fused_sigmoid_gating_delta_rule_update_kernel` calls per event are the
accepted-path **state committer**, one call for each of the model's 48 GDN
layers. They are not drafter-forward work. The completed verifier forward has
already exported each layer's `k/v/a/b` ring, so each call reads that layer's
ring and gates and writes that layer's distinct fp32 recurrent-state bank.
`accepted_paths` and `accepted_lens` are shared read-only inputs. There is no
cross-layer data dependency.

The real SWE-Verified Nsight attribution artifact reports:

- 42,288 kernel instances / 881 complete ranges = exactly 48/event.
- 3,596,371,408 ns / 881 = 4.082147 ms/event.
- The same artifact's CFWD projected envelope is 22.755077 ms/event. A newer
  timer cited in the work handoff is about 20.05 ms/event; using that denominator
  makes this kernel group about 20.36% of CFWD.

The current fixed32 committer already captures the 48 calls inside one CUDA
graph and performs one graph replay per event. Thus CPU launch dispatch and the
number of graph replays are already constant across B=1..4. Necessary device
work scales with B: the native grid has `4 * 48 * B = 192B` CTAs per layer.
At B=1 that is already four CTAs per each of 48 SMs, so layer batching should
primarily remove device launch boundaries, not the recurrence work.

## Candidate

`FR13_FIXED32_COMMITTER_LAYER_BATCH=1`, or the equivalent `.arm` sidecar before
boot, selects a one-physical-launch Triton candidate. Its third grid dimension
packs `(layer, request, value_head)`. It preserves the native realization:

- `BK=128`, `BV=32`, four warps, three stages.
- The complete ordered token recurrence remains inside one program.
- Native softplus, sigmoid, q/k normalization, state update, intermediate
  state stores, and output store remain in the same source order.
- Layer base addresses use the already-proven aligned anchor plus int64 offset
  table; raw pointer-table casts are not used.

When armed, boot captures both the incumbent 48-call graph and candidate graph.
Zero-accept boot warmups keep replaying the incumbent. On the first real event
with at least one accepted draft, both graphs execute from identical state and
inputs; every byte of every touched running-state row is compared. A mismatch
restores the incumbent state and raises. Only a raw-byte pass enables the
candidate graph for that B.

## Ceiling and decision

The absolute, impossible best case is deleting all 4.082147 ms. Against the
current Tail exact4 cap overrun of 98.638 ms, that closes at most 4.14% and
leaves 94.556 ms. Since the candidate performs the same recurrence and B=1
already has 192 CTAs/layer, the credible gain is much smaller and must be
measured on real SWE-Verified traffic.

Do not stack or claim this candidate until all of these pass:

1. SM121 compile and capture at B=1 and B=4.
2. `BYTE-GATE PASS` for every used occupancy on real SWE-Verified paths.
3. Nsight proves one physical committer recurrence launch/event and unchanged
   state traffic/work.
4. Real-task B1 diagnostic shows a measurable CFWD/full-wall gain.
5. Formal exact4 then exact16 acceptance uses the standing full-wall protocol.

The candidate is secondary, not a route to the hardware floor by itself.

