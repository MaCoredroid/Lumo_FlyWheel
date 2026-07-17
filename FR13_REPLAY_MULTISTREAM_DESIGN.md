# FR13_REPLAY_MULTISTREAM — overlap the 48 independent per-layer GDN replays

## Why (overturns the earlier "72ms replay is the floor" conclusion)
The committer's dominant cost is the accepted-path GDN replay: ~48 per-layer `launch_tree_gdn_replay`
calls. FR13_REPLAY_GPU_TIMER (wraps EACH launch, fr10_gdn_tree_kernel.py:1370) measured:
  tail6_rt: 17.95 gpu_s / 12950 spans = 1.386 ms PER LAYER-LAUNCH  => x48 = 66.5 ms/step GPU time.
So the replay is GPU-bound (66.5 of ~72ms wall; only ~5.5ms host). BUT the per-layer kernel is TINY:
state_bank [B~1.3, heads 16, 128, 128] fp32 = 13.6MB, read+write ~27MB => 0.1ms at 273 GB/s; compute
~30 MFLOP => ~0.03ms. 1.386ms is **14x the bandwidth floor** => the kernel is LATENCY/OCCUPANCY-bound
(GPU idle during each launch, waiting on unified-LPDDR5X memory latency), NOT bandwidth- or compute-bound.

If the 48 launches OVERLAP, aggregate traffic 48x27MB=1.3GB / 273 GB/s = ~4.8ms bandwidth floor.
=> potential ~60ms win (66.5 -> ~5-15ms). This is the single largest committer lever.

## Why this is NOT the refuted batched-fused replay
FR13_SAMPLED_REPLAY_BATCHED (launch_tree_gdn_replay_all_layers, ONE fused kernel over all layers via a
bank-POINTER TABLE) was LOSSLESS but SLOWER on GB10 (76.5->111.9ms) — diagnosed as strided cross-bank
pointer-indirection on unified memory. MULTI-STREAM is different: 48 SEPARATE kernels, each doing
CONTIGUOUS access to its OWN bank; only the DISPATCH overlaps. No strided gather. Distinct hypothesis.

## Correctness (byte-identical when ON — independent writes)
- Each layer L's replay writes ONLY its own `_fr13_ssm_bank` (from `_fr13_replay_layers[L]`). Different
  layers => different banks => write-independent => order-independent => concurrent-safe => byte-identical.
- `runrow_commit`/`runrow_init`/`burn_node_bank` are BOOL constexpr (kernel :1186-1188), NOT shared tensors.
- Shared reads only: `_accepted_path_buf`, `_accepted_lens_buf` (read-only). Safe.
- The custom-kernel launch (kernel :1292-1328) is ASYNC: num_spec_decodes is a Python int, grid/strides are
  host metadata, no internal .item()/sync => launches CAN overlap. Deployed path (FR13_COMMITTER_NATIVE is
  diagnostic-only, launcher :411 `:-0`) uses this custom path.

## Implementation (patcher, sampled committer replay loop ~9961-10037), FR13_REPLAY_MULTISTREAM default 0
GATE ON only when: env==1 AND not `_fr13_bnd_on` AND not `_fr13_rdab_on` (diagnostics need per-layer serial
order) AND not FR13_SAMPLED_REPLAY_BATCHED. Else the existing per-layer serial loop (byte-identical floor).
1. Module-level stream pool created once: `_FR13_REPLAY_STREAMS = [torch.cuda.Stream() for _ in range(N)]`
   (N=4 to start; sweep 2/4/8). Guard creation behind the flag.
2. BATCH the flag validation before the loop: gather all layers' `_fr13_replay_flags` -> ONE DtoH, assert
   all[:,0]==1 and all[:,1]==rows. Removes the per-layer `.item()` syncs (2x48) that would serialize streams.
3. Dispatch loop: for i,prefix in enumerate(sorted(layers)): `with torch.cuda.stream(streams[i % N]):`
   run `_fr13_replay_launch(...)`, then that layer's `_fr13_publish_apc_ssm_leaf` (if not runrow_commit) and
   `_fr13_flags[0].fill_(0)` ON THE SAME STREAM (ordered after its replay). NO `.item()` inside.
4. After the loop: make the default stream wait on all pool streams (event per stream + `wait_event`), OR
   `torch.cuda.synchronize()` once. The committer must NOT return before all replays land.

### CRITICAL cross-stream dependency (the easy bug)
The SCAN (runs on the default/forward stream, BEFORE the committer) WRITES the per-layer `_fr13_ssm_bank`
and the k/v/a/b rings that each replay READS. If a pool stream launches its replay without waiting for the
scan, it reads STALE/incomplete state => garble. Required dance:
  - After scan, before dispatch: `_ready = Event(); _ready.record(default_stream)` (once).
  - Each pool stream: `stream.wait_event(_ready)` BEFORE its replay launch.
  - After the loop: for each used pool stream `e=Event(); e.record(stream); default.wait_event(e)` (join).
The per-layer `.item()` flag validations run on the DEFAULT stream (scan already done) => cheap, and do NOT
block pool streams (different streams) => overlap preserved even if kept per-layer. Still prefer batching
them (design step 2) to avoid 96 default-stream syncs/step.

## Measurement (CRITICAL: not the per-launch timer)
FR13_REPLAY_GPU_TIMER does `_e.synchronize()` PER launch => it SERIALIZES the streams => cannot see overlap.
Use FR13_COMMIT_FULL_GPU_TIMER (CF2, brackets whole committer, one sync/step, :10646) instead:
  A) tail6 + CF2, multistream OFF  -> committer gpu_s baseline (~expect replay 66.5ms inside)
  B) tail6 + CF2, multistream=1 N=4 -> committer gpu_s (does replay collapse?)
Then LIVE B4-16 (subset_b4_sixteen) end-to-end: derived_tps_gpu + accept must be byte/accept-IDENTICAL
A vs B (independent writes => must match exactly; any drift = a concurrency bug, STOP).

## Red-team / kill-criteria
- SM occupancy: one replay grid = num_spec(~5) x num_vh(16) x cdiv(128,BV=8)=16 = ~1280 blocks @ num_warps=8.
  If that already saturates GB10 SMs, concurrent kernels have no SMs => multistream serializes => NO win.
  COUNTER: latency-bound kernels leave SM cycles idle even at high block residency => cross-kernel latency
  hiding can still help. ONLY the measurement settles it. If B ~= A (no speedup) => multistream dead on GB10
  (same memory-serialization verdict as batched-fused) => record + revert, replay IS the floor.
- If B < A but accept/bytes drift => concurrency correctness bug (missing sync / shared write missed) => STOP.

## Status: DESIGNED (this turn). Next: implement flag (default off, byte-identical gate) -> CF2 A/B -> live gate.

## MEASUREMENT ATTEMPT 1 — BLOCKED by B=4 cold-boot GPU-OOM (NOT multistream) [2026-07-17]
First A/B (ms_strm + ms_base, B=4) BOTH died "container died before health" with NO python traceback.
dmesg (audit-infra-first) = the real cause: `NVRM GPU0 Out of memory [NV_ERR_NO_MEMORY] @ mem_desc.c:1393`
at 21:17-21:20 (ms_strm capture) AND 21:32:42 (ms_base capture). The gpu_oom_guard (floor 9000MiB) then
docker-kills the container => the no-traceback death. ROOT = B=4 cold-boot capture/autotune memory spike at
GPU_UTIL~0.78 dips unified-avail below the guard floor. CORRECTS my premature "multistream poisons capture"
read: ms_base has multistream OFF and OOM'd identically => OOM is the sole blocker so far. The capture-guard
(6f55d2a19) stays (multistream IS capture-hostile in principle) but was not the bug.
PLAN: get the multistream SPEED signal via a B=1 A/B (less KV => boots under the guard floor; B=1 is also a
CLEANER per-step committer measure, no co-residency confound). Then fix the B=4 boot-OOM (lower GPU_UTIL /
locked launcher) for the final no-drift + lossless gate at B=4.

## VERDICT [2026-07-17]: REFUTED at B=4 — multistream is SLOWER. Both cheap committer attacks dead.
After fixing the worker-env-drop (sidecar), multistream ENGAGED at B=4 and measured (CF2, n=150):
  multistream (N=4):  91.6 ms/step
  serial (custom):    76.6 ms/step  (also native-kernel replay: 77.3 ms/step)
=> multistream is ~15ms SLOWER. The SM-occupancy kill-criterion FIRED: one replay grid = ~1280 blocks
already saturates GB10 SMs, so the 4 streams cannot run concurrently (no spare SMs) — they serialize on
the shared unified-LPDDR5X bandwidth AND pay stream-creation + cross-stream-event overhead => net slower.
Same physics as the refuted batched-fused replay: overlapping the per-layer GDN replays does NOT help on
GB10 (memory-serialized, not latency-hideable-with-spare-bandwidth). Bake OFF (default; no sidecar).

## COMMITTER-OVERHEAD ATTACKS — ALL CHEAP LEVERS EXHAUSTED (measured, B=4)
- Native-kernel replay (FR13_COMMITTER_NATIVE): 77.3ms == custom 76.6ms. Kernel-agnostic (latency-bound).
- Multistream overlap (FR13_REPLAY_MULTISTREAM): 91.6ms > 76.6ms. SLOWER (SM-saturated, can't overlap).
- Batched-fused replay (FR13_SAMPLED_REPLAY_BATCHED): slower (strided). [earlier]
- Copy-not-replay: infeasible (13.7GB per-node state export). [earlier]
=> The ~77ms committer replay is the GB10 HARDWARE FLOOR for the per-layer recompute architecture. The
only remaining lever is ARCHITECTURAL (fuse the accepted-path advance into the forward like native, or the
stateless-tree rewrite) — NOT cheap. Multistream + native-replay + gate-diag code stay flag-gated OFF.
