# FR13_STEP_GRAPH — one CUDA graph per decode step (design)

**User call 2026-07-26: this is the active structural build, upfront; verify
state-bytes levers (KV fp8, mamba dtype) PARKED.**

## Why

Measured step anatomy at B=4 (composed projection ~292ms): verify forward
~95ms (≈ the weight-read floor — already one FULL graph), drafter ~46-56
(R4 graph), committer ~6-10 (CG graph after bake), sampler ~4, and **~60-90ms
of inter-phase host python/scheduler/sampler glue and phase gaps** that no
kernel lever can touch. Three separate graphs with host hops between them is
the ceiling of the current architecture; one graph per step is the floor of
the next one. Projected step ~150ms => floor_ratio ~1.5; beyond that accept is
the campaign lever (physics asymptote ≈1.2 — drafter bytes + KV/state streams
sit above the 98.6ms pure-weight floor by construction).

## Why it is capture-legal (the three hard parts are already solved)

1. **Committer under varying accept-len**: FR13_COMMITTER_GRAPH already proved
   state-neutral fixed-shape padding (a=-1e4 => decay 1, k=v=b=0 => no write)
   is BYTE-IDENTICAL to the varlen committer across accept-len 1..12 × active
   batch 1..4 (fr13_committer_graph_varying ALL-IDENTICAL). The committer body
   is capture-legal at fixed (MAX_B × MAX_PATH) shapes TODAY.
2. **Sampler**: vLLM's rejection sampler emits fixed-shape output
   [B, max_spec_len+1] with -1 padding — data-dependent VALUES, static SHAPES.
   Its kernels (uniform draws included — pre-generated per-step RNG tensor as
   graph input) are capture-legal. The host-side .tolist() consumers move
   POST-replay (async-output pattern, already shipped for sampled ids).
3. **Drafter**: R4 already captures the full 4-iter spine loop with static
   root/hidden/pos/slen inputs. In S1 the root gather (committed-leaf hidden →
   drafter input) moves INSIDE the graph: it is a tensor-indexed gather off
   the sampler's accepted-len tensor — GPU-data-dependent indexing is
   capture-legal (values flow, shapes fixed).

## Staging

### S1 — sample+commit+draft in ONE graph (the near rung)
Capture boundary: from `rejection_sampler(...)` kernel entry through CG
committer body through R4 drafter loop end (packed draft tensor + static
spine/wide views). Host hops per step: exactly one (verify forward -> S1
replay), plus deferred DtoH of sampled ids (unchanged async pattern).
- Inputs (static buffers): verify logits view, RNG uniforms [B, max_spec+1]
  (filled per step pre-replay), seq-len/slot tensors.
- Outputs: accepted_len [B], committed path [B, MAX_PATH], draft package
  (spine/leaf/wide statics — unchanged from R4).
- The current host code between phases (accepted-path python walk, ssi
  prebuild host arithmetic, drafter root staging) must be re-expressed as
  tensor ops — most already are (SSI_PREBUILD, batched gathers); audit the
  residue with the same offline map used for R4.
- Gate ladder: offline patch-and-parse -> same-boot byte gate (S1-graph vs
  staged path, temp 0.6 fixed seed, in-process) -> probe accept band ->
  4-task live arm vs composed baseline (eps-matched pair, one lever).

### S2 — fold the verify forward in (the far rung)
One graph per step. Additional requirements:
- GPU-resident metadata: seq-lens/slot-mapping increments as tensor ops
  (FA2 fork consumes seqused_k tensors already; GDN spec_state_indices are
  value-static per topology; audit the scheduler-owned host ints).
- APC page-boundary steps allocate blocks on host: hybrid cadence — replay the
  full-step graph between boundaries, fall back to the staged path on the
  boundary step (or pre-allocate pages for an N-step horizon).
- Scheduler stays out of the graph: steady-state decode steps only; any
  prefill/mixed step uses the staged path (UNIFORM_DISPATCH_GUARD pattern).

## Cost-gate arithmetic
S1 eliminates the sampler->committer->drafter host glue (~30-50ms measured
class); S2 the remaining scheduler/prepare hop (~20-40ms). Both are
glue-elimination, not byte reduction — orthogonal to accept and to the parked
byte levers, composable with everything shipped.

## First build steps (S1)
1. Map every host touchpoint between sampler entry and drafter end in the
   live-container source (read-first discipline): .tolist()/.item()/host
   branches; classify each as (a) already-tensor, (b) tensorizable, (c) must
   move post-replay.
2. Pre-generated RNG tensor input for the rejection sampler (seeded per step —
   determinism preserved for byte gates).
3. Single capture wrapping the three existing bodies; statics inventory
   extends R4's dict.
4. Byte gate harness: same-boot staged-vs-graph, 512-step probe, temp 0.6.
