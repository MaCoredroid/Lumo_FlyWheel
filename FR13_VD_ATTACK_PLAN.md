# FR13 V+D ATTACK PLAN — verify-forward + drafter, accept-agnostic (2026-07-27)

User directive: attack the two accept-agnostic denominators first — the verify
forward and the drafter (= MTP head + suffix path combined). Scope guard: using
the suffix to SKIP MTP iterations is the deleted adaptive-skip path (collapsed
accept 3.6→2.0) — excluded by the accept-agnostic criterion itself. The
suffix side already costs ~0.3ms (host walk); the whole drafter cost is the
5-level MTP head loop.

## Bytes ledger (model config, /models/qwen3.6-27b-fp8)
64 layers = 48 GDN linear-attention + 16 full-attention (interval 4);
hidden 5120, intermediate 17408, vocab 248320, attn 24 heads x 256 (4 KV),
GDN 48 v-heads/16 k-heads x128. MTP head = 1 full-attn-style layer
(~0.39GB fp8: qkvo ~73M + mlp ~267M + fc ~52M params) sharing embeddings.
GB10: 273 GB/s unified; model weight read 27GB fp8 = the 98.6ms verify floor.
- Drafter per-iteration floor: MTP layer 0.39GB + DVK-64k lm_head slice
  (0.33GB fp8, pre-materialized at boot — verified in the shim: index_select
  once, cached) + KV ≈ ~0.8-1.1GB → ~3-4ms/iter → 4-iter loop floor ~12-16ms.
  MEASURED ~54ms/step → ~3.5-4.5x over floor INSIDE the R4 captured graph
  (launch overhead already gone) = real kernel inefficiency.
- Prime drafter suspect (inference, needs the probe): the fp8 GEMM on the 64k
  slice at M=1..4 — DVK bake moved 94.9→56.3 (−38.6ms ≈ −7.7ms/call x5),
  implying ~7.4ms/call remaining for a ~1.2-1.5ms read → ~5x off bandwidth.
- Verify fixed excess (155−98.6 ≈ 56ms): full-attn KV reads at agentic context
  (16 layers x ~30-75k ctx x bf16 = multi-GB/step, scales with CONTEXT not
  eps) + norm-soup remnants + per-layer fixed costs. Marginal 31/event =
  22-row tree work (scan/attn/norms).

## What is already refuted / shipped (do NOT redo)
- Shipped since the 2026-07-22 nsys profile: RING_EXPORT (B1), CONV_WB_FUSED
  (B2a), KV_REMAP_SYNCFREE, batched committer, burn-off, SLOT_REORDER,
  PARENT_GATHER (micro, ~3ms), subtree-parallel, DVK-64k, R4+L5 drafter graph.
- Refuted: GEMM M-tile scaling (rows free), lm_head as tree lever, verify
  layer-fusion (online residual dep), committer-fuse-into-verify (accepted
  path unknown), replay multistream (SM-saturated: 91.6>76.6ms), BV widen
  re-tile (register wall), .contiguous() kills (proven no-ops), HC (retired),
  drafter meta-reuse (dead heat), adaptive-skip (deleted).

## Ranked levers (accept-agnostic; sized honestly)
- D1 (largest, pending probe): drafter lm_head fp8 GEMM efficiency at tiny M —
  if the probe confirms ~7ms/call for a 0.33GB read, candidate fixes: better
  gemv kernel/config for the sliced head (fp8 block-quant path at M=1), or a
  resident bf16 slice (0.67GB, 2.4ms floor) via simple gemv. Target −15-25ms/step.
- V1 (cheap, built): FR13_CONV_WB_BATCHED (B2c) — gate + bake, ~2-5ms/step.
- V2 (probe-ranked): tree-attention 22-row decode config (the +14/draft bucket;
  splitkv/num_splits investigation) — up to ~10-20ms/step at operating eps.
- V3 (medium-high): GDN scan h_cache→shared-memory + O(1) parent load (§10 #2;
  the parent-only contract is now PROVEN by baked PARENT_GATHER). ~6-9ms/draft
  era-sizing; re-rank from the probe.
- V4 (structural, committer-adjacent, after V/D): committer-under-drafter
  overlap — the drafter needs only the committed TOKEN, not the advanced GDN
  state; the state replay + DtoH could hide under the next propose. Bounded by
  the multistream SM-saturation lesson (overlap idle, not bandwidth). Sized by
  the cfwd share that is genuinely idle — needs the probe's committer window.
- Parked by user (state-bytes, behavior-gated, NOT in this pass): KV fp8,
  mamba dtype.

## Instrument (committed this pass)
FR13_TORCH_PROF="<skip>:<active>" — one torch.profiler window driven from the
sfwd timer's per-step tick; the three span timers enter record_function windows
(FR13_W_VERIFY / FR13_W_DRAFTER / FR13_W_COMMITTER) so kernels attribute per
component; dump = per-window cuda ms/step + top-40 kernels (.json/.txt).
Fail-safe: any error → one-shot needle + disabled, never crashes the worker.

## Probe (queued behind tempfix1): output/fr13_msr/run_vdprof1.sh
EAGER-LABELED diagnostic boot (ENFORCE_EAGER=1, FR13_DRAFTER_GRAPH=0, canonical
tail6 shape, temp 0.6): profiler window 150:24, ~40 completion probes. Read
per-kernel GPU times + component ratios ONLY (walls inflated by eager+CUPTI).
Deliverable: named kernels for D1 and re-ranked V2/V3/V4 → then author the
top fix behind a flag with a same-boot byte/band gate.

## PROBE RESULTS (run_20260727T195333Z, 24 active steps, EAGER basis)
DRAFTER window 79.2 ms/step (72 window-calls):
- gemvx **BF16** 27.9 ms/step, n=217 (~9/step, ~3.1ms/call ≈ 216GB/s — the
  gemv itself is near-bandwidth-EFFICIENT; the DVK slice is resident in BF16
  = 0.67GB/read; the lever is TRAFFIC, not kernel choice)
- cutlass fp8 blockwise (MTP layer projections) 8.6 | everything else < 1
=> D1' (fp8 DVK slice, ~−14ms) — **PARKED by user 2026-07-27**: draft-logit
   quantization is distribution-touching; joins KV-fp8 + mamba-dtype in the
   parked state-bytes/quantization family. The campaign stays EXACT-MATH.
   Remaining drafter lever: the call-multiplicity audit (~9 gemv/step vs ~5
   expected — if redundant, exact-math savings ~6-9ms; one code-read decides).
VERIFY window 306.1 ms/step eager (kernel sum ~148; eager launch-gap
inflation ~150 — graphs already reclaim this in prod):
- fp8 GEMMs 115.6 (n~256/step) = only ~17% over the 98.6 weight floor →
  GEMMs are near-efficient, NOT a lever (re-confirms the M-tile refutation)
- tree-specific kernels are SMALL: _tree_gdn_path 12.9 + index_copy 4.1 +
  scatter_gather 3.7 + conv_wb_fused 2.8 + soup ~5 ≈ ~28 ms/step
- attention ~0.9 at the probe's SHORT context — the agentic verify fixed
  excess must be KV reads at long context (not probed here; state-bytes
  levers are the matching attack, currently user-PARKED)
=> V-front honest re-rank: kernel-side verify headroom ≈ 20-28ms (scan+soup),
   NOT ~90; the remainder of graphed-verify-over-floor at agentic contexts =
   KV traffic + in-graph bubbles. B2c/V3 still valid but small.
COMMITTER window 43.3 ms/step eager (kernel sum ~14; dominated by the 248k
softmax 4.1 + hundreds of micro-kernels/step) → launch-count-bound, already
graph-captured in prod; matches the ~40-45ms full-coverage staged spans.
NEXT: regress1 arm (running) delivers the per-step F/m regression; then
author D1' (fp8 DVK slice) behind a flag + accept-gate.

## PER-STEP REGRESSION (regress1 harvest, n=3010, mid-run preliminary)
  step_wall = 210.9(±1.4) + 61.4(±0.6)·drafts   R²=0.933
  sfwd      = 130.8(±1.1) + 43.3(±0.5)·drafts   R²=0.911
- The pooled arm-level fit (235.5+49.2) was WRONG both ways: fixed lower
  (211), marginal higher (61.4) — resolves the high-eps above-line drift
  (tempfix1 +22.5 at eps 2.57 was the pooled m underfit).
- Verify fixed 130.8 vs 98.6 weights => fixed excess only ~32 (KV at this
  workload + fixed kernels + fixed idle).
- **IN-SPAN IDLE (the campaign, user call 2026-07-27 "lets tackle in span
  idle"): verify marginal 43.3/event vs ~11-12/event of measured kernels =>
  ~30ms/EVENT of non-kernel time INSIDE the verify span, scaling with
  events (~75ms/step at eps 2.5); + drafter span ~15 and committer span ~20
  over their kernel content => ~110ms/step total in-span idle.** The earlier
  "host/gaps ~11ms" counted only BETWEEN-span gaps — the in-span idle is
  cuda-event-bracketed stream idle = host work landing between the start
  event and kernel launches (per-request input-prep, metadata/block-table
  python, publishes). PRIME SUSPECTS: per-request prep loops inside
  execute_model (scales with spec requests — matches the per-EVENT scaling
  signature), inter-phase host seams, committer serial waits.
ATTACK PLAN (exact-math only):
 1. Name it: host-phase CPU timers (perf_counter, no syncs) around the prep
    phases inside execute_model + propose + committer dispatch — sampled,
    sidecar-dumped; rides the B2c gate boot.
 2. vLLM --async-scheduling / prep-overlap: investigate whether input-prep
    (N+1) can overlap forward(N) in this build + patch compatibility — the
    structural fix if the suspect confirms.
 3. Vectorize the hottest per-request python loops (cheap, targeted).
 4. Committer-under-drafter overlap for the committer span's serial wait.

## DISPATCH-TAG RESULT (b2c1 live, first 365 samples): HYPOTHESIS 1 REFUTED
**365/365 pure-decode verify forwards ran as FULL graph replays** — zero
piecewise fallback. The ~28ms/event in-span idle is NOT capture-coverage;
it sits between the span's start event and/or around the FULL replay:
per-request host work (attn/GDN metadata assembly, patched seams) or a
residual per-event DtoH sync (invisible to kernel tables — consistent with
the profiler seeing nothing). NEXT DISCRIMINATOR is CPU-side and free:
code-inspect the per-event paths between sfwd-begin and the replay launch
for .item()/.cpu()/synchronize + python request loops (the 07-25
SAMPLER_SYNC_KILL pattern); then host-phase timers if inspection is
ambiguous. B2c riders healthy: CONV_WB_BATCHED preseeded needles firing;
ctrace corpus flowing (mis-pathed to /vllm-workspace/1 — snapshotted;
schema RICHER than .commit: per-node draft_token_id/parent/target_argmax/
target_prob_draft => the branch-tail join can condition on exact rescue
events with accept probabilities).
CONFOUND RECORDED (do NOT quote b2c1 speed): cfwd 106.3 ms/span vs regress1
42.1 — the ctrace debug rider does per-row .cpu() pulls inside the committer
path = a large injected observer effect (the very per-event-sync pathology
under investigation; sfwd samples unaffected — debug lives in the sampler
region). b2c1 deliverables = band + dispatch tags (FULL at n=2727) + corpus.
B2c's REAL speed gate = a clean arm without the debug rider.

## SUBSPAN FIRST SPLIT (subspan1 live, first 54 samples): HYPOTHESIS 2 ALSO
## REFUTED — the idle is NOT pre-launch host python
  host_ms (begin→replay-enqueue): mean 0.0, median 0.0
  exec_ms (enqueue→span-end):     mean 250.9, median 268.2 (early/cold basis)
The span's start event is immediately followed by the replay enqueue. The
in-span excess therefore lives AFTER enqueue, in one of two places the
current mark cannot separate:
 (a) INSIDE the replay's stream time — e.g. captured cross-stream wait nodes
     (subtree-parallel side streams, overlap fences) baked into the graph;
 (b) HOST TAIL — python between model-return and the span's stop event (our
     post-forward capture hooks etc.); a late-recorded stop event completes
     immediately when the stream has drained => elapsed folds host tail in.
NEXT INSTRUMENT (mark #2, authored for the boot after subspan1): an event at
model-return + CPU perf_counter deltas at mark/stop records — separates
in-replay stream time from host tail at zero extra sync cost. Meanwhile
subspan1 accumulates n for the exec-vs-drafts marginal regression (the
~28ms/event should reappear in exec's marginal).

## FINAL DISCRIMINATOR (subspan2, n=435): THE IDLE IS INSIDE THE GRAPH
  cpu_tail (mark→span-end host path): mean 1.1ms, median 0.4ms
  exec (stream time):                 mean 266.9ms
  ratio ~0.001-0.005 => HOST TAIL REFUTED. The host enqueues the replay and
  reaches the stop event in ~1ms; the STREAM genuinely runs the whole span.
=> The ~26ms/event + fixed excess is NON-KERNEL STREAM TIME CAPTURED INSIDE
the FULL graph: captured cross-stream wait/fence nodes are the prime suspect
(subtree-parallel side streams + overlap fences — the multistream lesson
says side streams SERIALIZE on this SM-saturated part; captured joins would
add that serialization to every replay). Secondary: capture-shape kernel
specialization differing from the eager probe's kernel times.
ELIMINATION LADDER COMPLETE: piecewise ✗ (8k FULL) → pre-launch host ✗
(35µs) → post-model host tail ✗ (0.4ms) → IN-REPLAY STREAM TIME ✓.
NEXT: subtree_ab probe — launcher-direct A/B boot (NEEDS-assertion exempt),
FR13_SUBTREE_PARALLEL=1 vs 0, timers + samples, graph mode, probe workload;
if exec collapses at =0, the captured side-stream topology is the cost and
subtree-parallel's +4.7% B=1 claim gets eps-adjusted re-pricing at B=4.
[superseded by the live-SWE subtree0 arm per the kernel-measurements-on-SWE rule]

## SUBTREE0 VERDICT (live SWE, n=569, window-matched control): BOTH EFFECTS
  subtree=1 (control, subspan1 early window): 132.2 + 37.49(±1.15)·drafts
  subtree=0 (this arm):                        99.5 + 59.79(±4.75)·drafts
  (control full-run 134.0 + 37.22 — no window/context confound)
1. **Captured side-stream topology CONVICTED for the fixed tax: +32.7ms/step**
   — at =0 the fixed term sits AT the 98.6 weight floor. The entire fixed
   in-graph excess was captured cross-stream waits.
2. **The subtree DECOMPOSITION is a real marginal win: −22.3ms/event** (serial
   scan 59.8 vs 37.5). Crossover eps ≈ 1.47 → at operating eps 2.2-2.7,
   subtree=1 nets −16..−27ms/step BETTER. Do NOT unbake.
FIX HYPOTHESIS => **FR13_SUBTREE_SINGLESTREAM**: launch the same subtree
kernels on the MAIN stream (the multistream refutation says side streams
cannot overlap here anyway → the side-side topology contributes only the
baked wait nodes). Expected: keep −22.3/event, delete +32.7 fixed →
~−33ms/step at operating eps on top of the current board. Design read next;
flag-gated build; live gate per standard.
**REFUTED BY SOURCE READ (before building — 2026-07-28):** `_launch_paths`
(fr10_gdn_tree_kernel.py:3828) is ALREADY single-stream — one
`_tree_gdn_path_kernel` launch per path LEVEL on the current stream; no side
streams, no wait nodes anywhere in the subtree path (the only Stream() near
it is the committer-graph capture warmup, standard idiom, once at capture).
=> the +32.7ms fixed lives in the PER-LEVEL LAUNCH STRUCTURE: 48 layers ×
level barriers × small-kernel durations (level-1 sibling kernels are
tiny-work but latency/occupancy-bound) and/or inter-level drain bubbles —
unresolvable by reading. MEASUREMENT NEXT BOOT (per the
kernel-measurements-on-SWE rule): subtree_nsys arm — the proven July-22
nsys-on-live-task runner adapted (14598 marathon, 300s window, SIGSTOP
teardown freeze, reduce_tail6_nsys.py reducer) — names level-kernel
durations vs gaps AND delivers the real-context kernel floor (the standing
caveat) in one boot. Fix design follows the named numbers: candidates are
level-count tightening, sibling fold-in with in-register branch emission
(out_j from the parent state without carry), or accepting the level barrier
as the price of path concurrency and re-drawing the floor.

## SUBTREE_NSYS RESULTS (2026-07-28 01:07Z, 300s window, 691 drafts, B=1
## real 14598 context — per-draft ≈ per-step at CONC=1): THE FLOOR MOVES UP
Top kernels (ms/draft): gemm_mlp 142.6+12.4+2.8 ≈ 158 | **attention
flash_splitkv 63.8 + unified 13.4 ≈ 77** | **lm_head 29.5** | tree scan
9.85 | norm-soup ≈ 27 | committer bits ~6 (conv_wb_batched 1.97 = B2c ✓).
Kernel sum ≈ ~310/draft ≈ the B=1 step wall — **at REAL context the verify
span is mostly REAL KERNEL WORK.** The "26ms/event in-span idle" was
substantially a SHORT-CONTEXT artifact of the eager probe's kernel floor
(attention measured ~0.9 there vs ~77 here). Consequences:
1. FLOOR REWRITE: real-ctx attention (~77 at B=1; scales per-event with
   context) + the found verify lm_head (29.5) move INTO the measured
   baseline; the attackable idle pile SHRINKS accordingly. Honest floor
   ratio is closer than the short-ctx accounting suggested.
2. NEW DATA-RANKED LEVERS (exact-math): (a) attention splitkv at real ctx
   runs ~4.5x over its ~14ms analytic KV-read floor → attention
   config/kernel efficiency = now the TOP verify lever candidate;
   (b) verify lm_head 29.5 vs ~5-9ms read floor → efficiency lever
   (quantization variants are PARKED; kernel/config only);
   (c) **cudaMemcpyAsync: 169 calls/draft, 282ms/draft CPU-side** — a
   structural smell needing attribution (which subsystem issues 169
   copies per draft; if pageable-staging serializes the stream it is a
   major hidden cost, if fully overlapped it is host-CPU burn only).
3. Subtree 32.7 fixed tax (real-task A/B) still stands; the level-gap
   query runs offline on the exported sqlite (748MB, on disk).
NEXT (all CPU, no GPU needed): sqlite queries — per-level path-kernel gap
analysis, memcpy source attribution, per-step bucket normalization vs the
subspan fits. Then the fix ranking gets rebuilt from real-context numbers.

## SQLITE ATTRIBUTIONS (2026-07-28, offline): the honest real-context map
- **GPU busy fraction 0.89**: kernels 265.4s + memcpy 1.5s of the 299.9s
  window at B=1 => TOTAL non-GPU idle ≈ 48ms/draft across EVERYTHING
  (committer serialization + host + drafter gaps + sampler). The step is
  work-dominated at real context; overlap territory is bounded at ~48.
- **Memcpy smell = BENIGN**: GPU-side memcpy totals 2.2ms/draft (390
  D2D/draft = in-graph ring/static copies, 288MB at high BW). The 282ms
  CPU-side API time overlaps under an 89%-busy GPU — host burn, not
  critical path. NO lever.
- **Sync events: 7.0ms/draft** (type-3 stream/event) — small.
- **Path kernel: 103.7 instances/draft ≈ 48 layers × ~2.16 levels, mean
  100µs => scan total 10.4ms/draft GPU.** The subtree +32.7ms fixed tax
  CANNOT be path-kernel durations or launch counts (+56 launches ≈ 0.3ms)
  — mechanism UNRESOLVED but bounded; keep=1 stands (net positive at
  operating eps); deprioritized behind the bigger levers below.
- **REBUILT LEVER RANKING (per-draft @ B=1 real ctx, exact-math only):**
  1. attention 77 vs ~14 KV-read floor => ~60ms class (splitkv config /
     kernel efficiency at long ctx) — THE new campaign target
  2. verify GEMM ~150 vs ~99 weight floor => ~50ms class (fp8 blockwise
     dequant/config at M=22; verify against per-shape autotune first)
  3. lm_head 29.5 vs ~5-9 floor => ~20ms class
  4. norm-soup ~27 (fusion continuations)
  5. total idle ~48 (committer-under-drafter overlap; bounded)
  6. scan 10.4 — already efficient, close to done
CAVEAT retained: the marginal split (kernels vs waits inside 37.5/event)
still needs the real-context kernel measurement (nsys-on-live-task) — the
eager probe's kernel floor mixes subtree-eager mode and short context.
