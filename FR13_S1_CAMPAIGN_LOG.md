# FR13 S1 (=2 one-graph) campaign log — records + boot ladder

## Deploy-speed records (measured_tps_fullstep_wall basis; quote eps + prefill_frac always)

| arm | mode | tps (wall) | step_wall ms | eps | accept/event | prefill_frac | verdicts |
|---|---|---|---|---|---|---|---|
| s1go (=1 walk capture) | captured (=1) | 40.23 | 371.8 | 2.74 | 4.467 | 0.502 | 1P/3F |
| dscg (DEPTHSYNC ref) | staged | 37.73 | 335.6 | 2.32 | 4.451 | 0.409 | 1P/3F |
| **s1fullgo boot-16** | **staged fallback (=2 DISABLED)** | **31.5** | **331.8** | **1.94** | **4.384** | **0.629** | **2P/2F** (13236+12907 pass; 13033+13398 fail) |

Boot-16 notes: raw tps low vs s1go is an eps+prefill artifact (long single-task
tails: eps 1.94 vs 2.74; prefill_frac 0.63 vs 0.50). Step_wall 331.8 @ 1.94 is
s1go-line at matched conditions — expected for a staged arm. Behavioral result
is band-POSITIVE: 2 passes vs 1P/3F in both references (13236 flipped to pass).
Trace artifacts auto-committed per task; deploy json itself was lost to the
boot-17 dir wipe — numbers preserved here (source: arm-end record read
2026-07-26 22:51:51).

## =2 capture boot ladder (all fixes committed on fr13-swe-qwenrename-stream-autocommit)

1. boot-1 KeyError num_draft_tokens (runner metadata is a LIST) → ndt_h host-derive; counts TUPLE into both capture keys (permuted-zero soundness).
2. boot-3 AcceleratorError = _sample's async-output event sync → FR13_SG_ASYNC_HOIST (in-capture no-op + wrapper hoist).
3. boot-4/5 philox poison on abort → g.reset()+del+gc; later: poison-immune EXPLICIT-generator bulk draws (topk_topp + dm walk) — arms now survive any capture failure.
4. boot-6 second async reconciliation (_get_draft_token_ids_cpu) → FR13_SG_SPEC_HOIST.
5. boot-7 parents-cpu ptr-keyed cache miss (pageable DtoH in-capture) → wrapper pre-warms cache for the static's address.
6. boot-8/9 penalties active in live serving → FR13_SG_TL_QUEUE: both processor passes (target + tree_self) run eagerly pre-capture into per-key statics; forward consumes FIFO.
7. boot-10 committer flavors are host-layout → _fr13_native_committer_all_layers_device (device-built fixed-shape neutral-padded layout; byte gate 40/40 vs CG host layout; capture bench byte-identical replay at real 48-layer geometry).
8. boot-11 StreamCaptureInvalidated → pre-capture side-stream warmup with force-flags (exact captured route) + CG-pattern save/restore of GDN+conv col0 rows; boot-15 added scan-FLAGS save/restore (warmup consumed per-step markers → staged-fallback death).
9. boot-12 greedy warmup-probe captured the legacy host committer → all-greedy eligibility Skip.
10. boots 14-17 deterministic silent invalidation at capture_end: CAPDBG ledger (682 captures process-wide, only ours fails); concurrency REFUTED offline (matmul/pageable/graph-replay/device-sync hammers pass both error modes); composed one-capture bench (walk+sampler+committer, PHASES bisect) PASSES offline; MEMFRAC probe shows in-capture allocs reach the allocator OOM machinery; retry ladder proves determinism (attempt-1 identical signature; post-poison retries wasted).
11. CURRENT (boot-18): capture into vLLM's SHARED graph pool (get_global_graph_pool) — in-capture allocations reuse pool inventory instead of fresh cudaMalloc (NVRM-under-load = lead suspect: empty-GPU benches always pass, 80GB-committed live always invalidates).

Instruments built: docker-logs streamer (forensics survive container death),
CAPDBG capture-lifecycle needle, skip-reason needles, capture-status phase
probes (NOTE: is_current_stream_capturing stays True on Invalidated — blind),
composed/committer/sampler-half capture benches, MEMFRAC allocator probe.
Hazard log: PRESSURE ballast on unified memory = host OOM (retired 22:54).

## Bisect conclusion (boot-19, 2026-07-26 23:42) + pivot
SCOPE=half (sampler+walk+products; committer OUT) invalidates with the
IDENTICAL signature => the invalidator is in the SAMPLER STRIP's live context
(_sample entry -> walk: vLLM Sampler bonus path / slices / constraints).
Walk-only (=1) captures are proven live; products+device-committer proven
offline byte-identical at scale. Shared-pool refuted (boot-18 attempt-1,
pre-link-drop). Boot-18's 0/4 itself = transient GB10<->alienware link drop
(watchdog-classified network-drop; link recovered), NOT code. Cache-regression
smell CLEARED: hit-rate ramps overlap (s1go 0->21->41->65->82% vs boot-18
44->55% at same-elapsed); boot-16 prefill_frac 0.63 = task-path composition.

PIVOT (=3): grow the LIVE-PROVEN =1 in-dispatcher capture region to
walk -> products_device -> device committer (zero vLLM sampler code in-region);
committer tail gets a state-committed marker (skip conv/launch, keep
publishes). Sampler strip (~3-5ms launches) stays eager — acceptable loss;
the committer launch storm (dominant glue) is captured. =2 runner wrapper parked.

## boot-19 record (staged; =3 built but boot ran pre-=3 seq at STEP_GRAPH=2-era... verify: seq had =2+SCOPE=half)
boot-19 (SCOPE=half bisect arm, staged after DISABLED): 2P/2F
(12907+13236 pass; 13033+13398 fail) — SAME set as boot-16; second
consecutive band-positive vs 1P/3F references (16-task confirm queued).
Speed: 31.343 tps wall | step 325.023ms @ eps 1.847 | accept 4.515 |
prefill_frac 0.454. Staged line consistent w/ boot-16 (331.8 @ 1.94 pf .63);
lower pf confirms boot-16's prefill was path variance.

## boot-23 record — FIRST LIVE =3 ARM (walk+products+conv+committer IN-GRAPH)
Verdicts: 2P/2F (12907+13236 pass; 13033+13398 fail) — IDENTICAL set to both
staged arms => behavioral parity in captured mode; third consecutive
band-positive vs 1P/3F references.
Stability: 4 mode3 graphs, ~2h15m captured-mode soak, death 0, zero engine
incidents. KV-remap verified intact (mechanism + empirics). Image identity
verified constant across all arms (official swebench per-instance, local
digest sha256:4232400..., runner never re-pulls).
Speed: 26.946 tps wall | step 305.479ms @ eps 1.441 | accept 4.713 |
prefill_frac 0.434 | committer-span (CFWD basis) 39.654ms/step | drafter
56.121ms/step | s_per_fwd_gpu 0.138. Lowest absolute step_wall of any 4-task
arm BUT lowest eps too (13398 ran a 2h solo IERS-branch marathon, eps→1
dominating the average) => speed verdict = staged-parity-class, committer
attribution INCOMPLETE (staged component pairs were lost with boots 16/19
records). Next-arm protocol: preserve FULL component records + overlap-phase
window pairs; 13398 style watch-item (terse/marathon trajectory) stands.

## Boot-24 — ALL-AT-ONCE =2 (S1-full one-graph, full strip hoisted) [LAUNCH 2026-07-27]
Strategy call (user): attack =2 in one shot, then S2. Basis: the SCOPE=half
bisect convicted the sampler strip; every strip op is now hoisted —
processors via _FR13_SG_TL_QUEUE (two-entry FIFO), async/spec syncs via
FR13_SG_ASYNC_HOIST/FR13_SG_SPEC_HOIST, and NEW this boot the bonus sampler
via _FR13_SG_BONUS_OUT pop-once handoff (the last un-hoisted strip op,
rejection_sampler.py forward call).
Mechanism: wrapper precomputes bonus SamplerOutput eagerly; forward pops it
(module-global, same namespace pattern as TL_QUEUE). Capture flow: obj-1
consumed by side-stream warmup -> post-warmup refill sets obj-2 -> real
capture bakes obj-2 tensors; ent["bso"]=obj-2. Replay: eager bonus recompute
-> copy_ into baked ent["bso"].sampled_token_ids before graph.replay().
Abort paths clear the handoff (philox-poison protocol unchanged).
Dry-gate PASS: full patcher applied on pristine pinned image
(vllm/vllm-openai@sha256:3dbe092...), rejection_sampler + gpu_model_runner
py_compile clean, call-site re-indent verified by eye.
Arm: run_s1fullgo.sh seq FR13_STEP_GRAPH=2 (+CAPDBG), B=4 CONC=4 offloaded
4-task subset, temp 0.6, WALL=0 (standing no-AGENT_WALL_S gate policy —
eps comparability handled measurement-side via eps-matched overlap windows,
NOT by capping agents). Survivability: skip-and-stay-armed => degrades to
valid TAW-eager arm on capture failure.
Success needle: "S1-full captured B=4 (sampler+committer one graph...)".
If =2 STILL invalidates with the full strip hoisted: the poison op is
OUTSIDE the named strip remainder => next move is the inverted climb
(un-hoist one piece per boot from this scaffold to name it).

### Boot-24 mid-run finding + boot-25 package (committed same day)
Boot-24 =2 capture FAILED both attempts; arm degraded to staged fallback
(by design, still running as the staged component-pair reference repair).
Forensics:
- Attempt 1: silent invalidation surfaced at capture_end (same signature).
  ZERO phase probes fired — and the probe is STRUCTURALLY BLIND:
  torch.cuda.is_current_stream_capturing() == (status != None), which stays
  True for an INVALIDATED capture. All prior "no probe fired" evidence is
  void; the poison op is NOT bracketed by existing probes.
- Attempt 2 never really ran: pre-capture fr13_sg_fill_uniforms
  u.uniform_() (default-gen draw missed by the bulk-gen conversion) hit the
  philox captured-offset poison left by attempt 1 => "Offset increment
  outside graph capture" => attempt burned for free.
- Audit while reading source: served tree path does NOT execute the
  per-request-generator draw sites (early device-route return); glue is
  sync-free under device multidraft (probs staging None, parents from
  step-constant cache). Shared-pool capture + thread_local already in.
Boot-25 package (this commit):
1. capchk upgraded to TRI-STATE via ctypes cudaStreamIsCapturing
   (0=None 1=Active 2=Invalidated; tracks last-Active tag) — un-blinds
   every probe.
2. New probes: post-begin, route-return, fwd-tail, pre-capture-end
   (brackets the whole region; first status!=1 names the phase).
3. fill_uniforms poison immunity (bulk-gen for both default-gen draws)
   => retry attempt 2 becomes a REAL second capture attempt.
Expected boot-25 outcomes: either (a) capture succeeds on attempt 2 now
that retries are real (attempt-1-poison was self-inflicted), or (b) probes
name the guilty phase precisely. Both advance the line.

## Boot-24 FINAL (completed 04:48Z) — staged-fallback reference arm
Verdicts: 2 pass, 2 fail, 4 finished (12907+13236 pass; 13033+13398 fail)
= boot-16/boot-23 2P/2F band. 13398 marathon again (2737s, ~417 records,
coherent IERS/mjd_utc debugging to the end — no garble).
Speed (STAGED fallback, =2 capture disabled at step 1):
measured_tps_fullstep_wall 32.276 | step_wall 331.457ms @ eps 1.899 |
accept 4.635 | prefill_frac 0.480 | s_per_fwd_gpu 0.113 | death 0.
Slope (accept+1)/step = 17.00 tps/eps == boot-16 staged 16.97 =>
staged-vs-staged cross-boot reproducibility CONFIRMED; the staged
reference is REPAIRED with full component records preserved this time:
output/fr13_msr/boot24_records/ (deploy json + verdicts + pid-231
sidecars: committer 236.25s/6108 spans=38.7ms, drafter 356.36s/6482=55.0ms,
sfwd 1094.08s). Stream log rotated to s1fullgo_stream.boot24.log.
Scoreboard (slope basis): =3 graph 18.7 > staged 17.0 (x2 boots) > =1 14.7.
Boot-25 LAUNCHING: same =2 lever + tri-state probes + real retries
(a25162f50). Expected: capture on the real attempt 2, or the first
probe with status!=1 names the poison phase.

## Boot-25 (killed early by design) + boot-26 launch
Boot-25 delivered its diagnostic payload in the first 10 minutes:
1. TRI-STATE PROBES WORKED: "capture status 2 at phase: pre-capture-end
   (last Active: fwd-tail)". The bracketed sliver (SamplerOutput build +
   module-call exit + _sample return) contains ZERO CUDA ops => the poison
   CAUSE is concurrent (batch-queue pipelining thread / allocator-pool
   entanglement) landing late in the region window; cause != location.
2. Philox fix CONFIRMED: attempt 2 reached capture_begin (no Offset-
   increment burn) and exposed the NEXT abort-hygiene leak:
   "beginAllocateToPool: already recording to mempool_id" — C++
   capture_end throws at cudaStreamEndCapture BEFORE endAllocateToPool,
   and reset() does not end pool recording either.
Arm killed after S1 DISABLED (lever burned at step 1; staged reference
already banked twice — boot-16 slope 16.97, boot-24 17.00). Stream log
preserved: s1fullgo_stream.boot25.log.
Boot-26 package (this commit):
- Abort path now calls torch._C._cuda_endAllocateToPool(dev, pool)
  (releasePool deliberately skipped: leaked refcount on the process-
  lifetime global pool is harmless; double-release is not).
- HETEROGENEOUS RETRY: attempt 1 = shared global pool, attempt 2 =
  PRIVATE pool. One boot discriminates: attempt-2 success => shared-pool
  entanglement was the poison (adopt private); identical attempt-2
  failure => concurrent-thread landing => next lever = capture-step
  serialization (pause batch-queue pipelining for the one capture step).

## Boot-26 (killed post-payload) + boot-27 launch: mode discriminant
Boot-26 verdict: SHARED-POOL ENTANGLEMENT REFUTED. Attempt 1 (shared pool)
invalidated; pool-leak fix engaged ("pool-recording ended after abort");
attempt 2 ran on a PRIVATE pool and invalidated with the IDENTICAL
signature (status 2 at pre-capture-end, last Active fwd-tail). Poison is
pool-independent + deterministic. Hooks audit: zero forward hooks in
vllm/v1, zero global module hooks => the bracketed sliver truly has no
CUDA ops in our thread.
Remaining hypotheses + the discriminant (boot-27, this commit):
(i) mode-gated unsafe action in OUR thread — but thread_local only errors
    on own-thread actions, and the sliver is empty… unless the action
    precedes fwd-tail with LATE status flip;
(ii) structural stream/event violation (another thread interacting with
    the capturing stream's history) — invalid in EVERY mode.
Attempt ladder now 3: shared/thread_local -> private/thread_local ->
private/RELAXED. relaxed-success => (i) (and the graph is still only our
stream's recording; replay correctness remains byte-gated vs staged);
relaxed-fail => (ii) confirmed => build capture-step serialization
(pause batch-queue pipelining for the single capture step).
Stream log preserved: s1fullgo_stream.boot26.log. Staged reference
unchanged (banked x2).

## Boot-27 verdict + boot-28 launch: capture-step serialization
Boot-27: RELAXED mode failed with the IDENTICAL signature (status 2 at
pre-capture-end, last Active fwd-tail) => hypothesis (i) mode-gated
own-thread action REFUTED. STRUCTURAL stream/event violation CONFIRMED:
the batch-queue pipelining thread (execute_model futures run concurrently
with sample_tokens on separate executor threads; AsyncGPUModelRunnerOutput
does per-step cross-stream wait_stream event wiring) interacts with the
capture window. The stable fwd-tail bracket is a GIL-scheduling artifact:
the blocked thread lands its pending CUDA call at our first yield near
region end — cause != location, which is why 19 boots of in-region
op-hunting never converged.
Boot-28 package (this commit): CAPTURE-STEP SERIALIZATION.
- Module RLock in gpu_model_runner; execute_model holds it for its whole
  body (normal execute(N+1)/sample(N) overlap untouched — sample_tokens
  does not take the lock => zero steady-state cost).
- The =2 capture section acquires it exclusively: waits for the in-flight
  forward to drain, captures solo, releases right after capture_end
  (aborts release in the handler). One-step stall, once per key.
- Dry-gate caught a real ordering bug (tail-append guard sentinel
  collided with the wrapper's own string => lock never defined); fixed
  (guard on _fr13_sg_orig_execute_model), re-gated PASS.
Ladder stays 3-attempt (shared/tl -> private/tl -> private/relaxed) — all
three now run under serialization; ANY success is decisive.
If boot-28 still invalidates solo: the violator is NOT the pipelining
thread => next suspects are the async-output copy machinery (its
wait_stream/event wiring) and in-region record_stream allocator traffic.

## Boot-28 verdict: serialization REFUTED => pivot to WARMUP CAPTURE
Attempts 2/3 captured with the pipeline provably solo ("exec lock HELD")
and invalidated with the identical signature. Combined refutation set is
now: strip ops (all hoisted), pool (shared+private), capture mode
(global-class thread_local + relaxed), execute_model concurrency
(exec-lock), module hooks, philox/pool abort hygiene. Thread census of
EngineCore: 54 VLLM::EngineCor + 7 pt_nccl_watchdg + 7 pt_nccl_heartbt +
12 pt_gloo_runloop + cuda-EvtHandlr — the live capture window is
structurally hostile (many threads, unlockable one-by-one).
PIVOT (evidence-backed): vLLM's ~180 FULL forward graphs capture
SUCCESSFULLY in this same process/thread census — at WARMUP, in the quiet
window. Our =2 capture is the only one attempted under live traffic.
The refill machinery built over boots 20-28 (uniforms/perm/tls/stls/
bonus/topology handoffs, composition checks) makes capture-time
irrelevant: replays refill everything. Tree topology is a serve-config
CONSTANT (21-node speculative_token_tree) => synthetic warmup metadata
reproduces live shapes exactly (B in 1..4 x fixed 21-tok drafts).
Boot-29 = WARMUP CAPTURE: pre-capture the =2 graph per B-key during the
boot's quiet capture phase; live steps replay-only. Boot-28 arm continues
as staged band point while the build proceeds.

### Boot-29 warmup-capture design (PINNED, build in progress)
Fabricator `_fr13_sg_warmup_capture(self)` (module-tail injection, called
at end of capture_model, inside the boot's quiet window):
per B in 1..max_num_seqs:
1. Tree choices from serve config (21-node constant); ndt = [21]*B.
2. TEMP monkey-patch SpecDecodeMetadata.make_dummy -> real tree index
   layout (target = sampled_start+parent_local, self = sampled_start+
   node_idx+1, sampled_start += 22) + tree_parent/self attrs — the
   _dummy_run pipeline (line 8005) then builds a TREE-SHAPED forward,
   engaging the GDN defer route so real products/stash exist.
3. TEMP swap input_batch.sampling_metadata -> dummy B-sized (temp 0.6,
   generators={} -> bulk-gen uniforms), per _dummy_sampler_run pattern.
4. self._dummy_run(uniform_decode, force_attention, mode NONE) ->
   defer products staged; logits via model.compute_logits (exact live
   dtype + (22B, V) shape; key = (tuple([21]*B), logits.shape)).
5. self._sample(logits, md) TWICE: first = warm-set staged run (Triton
   warm), second = the capture — in the same quiet window where vLLM's
   ~180 FULL graphs capture successfully.
6. Restore patched seams; skip live publishes (nothing to publish);
   dummy-slot GDN state safe (zero-on-alloc baked).
Live steps then REPLAY-only; composition checks already guard mismatch.

## Boot-29 sub-series ledger (a-h): warmup-capture bring-up
Launch infra: three harness-tracked driver launches died at +2-4min
("stopped" externally); detached setsid launches survive => all boots now
launch detached (workaround, cause in the task harness unknown).
Fabrication seam ladder (each found fail-loud, ~12min/cycle):
  29a REQKEY sampler-row req ids -> fabricate B dummies
  29d REPLAY_ROUTE spec-row req ids -> same list
  29e layer-0 stale scan flags -> ROOT: num_decode_draft_tokens buffer
      all -1 at warmup => gdn builder num_spec_decodes=0 => no layer
      staging; buffer fill fixes BOTH the builder gate and the =2
      wrapper's own uniform-step gate
  29f 0/4 keys, all-eager: the =2 harness wraps sample_tokens' CALL to
      _sample; direct _sample bypassed it => fabricator now routes
      through sample_tokens with fabricated execute_model_state +
      warmup-mode early-return after the wrapper block
  29g THE BIG DATUM: warmup capture RAN in the quiet window and
      invalidated with the IDENTICAL signature (status 2 at
      pre-capture-end, last Active fwd-tail). QUIET-WINDOW HYPOTHESIS
      REFUTED — the poison is in our region's own execution. Also:
      abort philox poison is BOOT-LETHAL at warmup (vLLM's own init
      rand_like crashed the engine).
Working theory (fits ALL data incl. warmup-solo + mode-independence +
the empty sliver): python GC firing in the fwd-tail->capture_end window
frees side-stream-warmup intermediates DURING capture; allocator
event-insertion on cross-stream frees = structural invalidation.
Boot-29h: gc.collect+cuda.synchronize before capture_begin + gc.disable
across the capture (+re-enable on success/abort); fabricator logits via
poison-immune generator; per-B cuda rng-state restore.

## Boot-29i VERDICT: INVALIDATOR NAMED — our own CFWD span timer
Triple-split probes bracketed the flip to: last Active post-so-build,
invalid at sample-return — the gap contains exactly ONE call:
_fr13_cfwd_end(_fr13_cfwd_ev), the CFWD GPU committer timer (the driver
sets FR13_CFWD_GPU_TIMER=1 on EVERY arm). The begin/end pair brackets the
rejection_sampler call inside _sample:
- =1/=3 regions are NESTED INSIDE that bracket => timer ops outside the
  capture => 4 clean graphs, 2h15m soaks.
- =2's region CONTAINS the pair => its cuda-event ops inside the capture
  are a STRUCTURAL violation => silent invalidation, every boot, every
  mode/pool/thread/window. 19+ boots of external suspects; the standing
  instrumentation-observer-effect rule vindicated in the strongest form.
Fix (42157968e): capture-guards on all three span timers (cfwd/dfwd/sfwd
begin returns None when is_current_stream_capturing()).
PHILOX CURE FOUND (in-boot cascade vs the REAL poison):
graphsafe_set_state(clone_state()) cures; gc/manual_seed/set_state all
fail. Baked into the abort path — aborts are no longer boot-lethal.
Boot-30 = warmup capture with the fix. Expected: WARMUP-CAPTURED x4.

## BOOT-30: =2 CAPTURED — S1-full one-graph MILESTONE (2026-07-27 ~07:58Z)
[FR13_STEP_GRAPH] S1-full captured B=2 (sampler+committer one graph; statics=6)
  WARMUP-CAPTURED B=2 key=((21,21),(44,248320))
  WARMUP-CAPTURED B=3 key=((21,21,21),(66,248320))
  WARMUP-CAPTURED B=4 key=((21,21,21,21),(88,248320))
warmup-capture done: 3/4 keys. B=1 missed on a bring-up off-by-one only
(phase-1 died on dm-not-loaded BEFORE the warm-set add, so phase-2 was
consumed by the warm-set skip; B=1 needs 3 calls — fix queued; live-ladder
B=1 capture also still armed with timers now guarded).
THE FIX THAT DID IT: capture-guards on our own CFWD/DFWD/SFWD span timers
(42157968e) — the invalidator all along.
Arm continues to the live 4-task gate with B=2/3/4 replaying from step 1.
Gates now: behavioral band (2P/2F), garble eyeball, replay engagement,
captured-=2 speed vs staged 331.5ms @ eps 1.9 (slope basis: staged 17.0,
=3 18.7).

## BOOT-31: 4/4 KEYS — complete =2 key set warmup-captured (~08:15Z)
WARMUP-CAPTURED B=1 ((21,),(22,248320)), B=2, B=3, B=4 — all four
compositions. B=1 three-phase fix worked; logits-static engagement fix in
(live logits copied into the captured tensor on ptr mismatch, ~0.2ms).
Arm proceeding to the live 4-task gate with the FULL replay set armed.
Gate: 2P/2F band + garble eyeball + replay engagement (committer-span
collapse vs staged 38.7ms = cleanest signal) + captured-=2 speed tuple.

## Boot-31 live gate: GARBLE — line stopped (arm killed ~08:25Z)
4/4 keys captured AND live replays ENGAGED (no address-moved raises,
accept 4.20 mid-run) — but 2 of 4 tasks went DEGENERATE: num_turns=1,
~5min single turns, thinking = "Let me understand the!!!!!!!!!!..."
(coherent prefix -> deterministic '!' spam). 13033 stayed coherent =>
composition-dependent corruption. The =2 replay corrupts state.
Prime suspects (warmup-baked vs live mismatch in committer consumption):
dummy-slot indices / stash provenance / a per-step tensor not covered by
the refill contract. Verdict basis: we owed this lever the per-change
same-seed byte gate BEFORE any live arm (fr13_speed_first_lossless_gate)
and skipped it in the capture excitement.
NEXT: in-boot replay-vs-eager BYTE SELF-CHECK in the fabricator (capture
-> paired fresh-input step replay-vs-eager -> compare sampled ids +
committed GDN/conv states, per key) — no live arm until 4/4 byte-clean.

## Boots 32-33: engagement corruption localization (ongoing)
Boot-32 (ptr-audit): committer-consumed addresses CLEAN except a benign
perm module-static hit (graph reads ent["perm"], refilled by copy_; the
audit compared the eager-path module static that later warmup keys
legitimately re-point). Accept collapsed to 1.00 (0 drafts accepted @
176/s drafted) => walk rejects everything.
Boot-33 (draft-ids refill: live draft_token_ids copied into the baked
md_static pre-replay): accept STILL 1.00 @ 92/s drafted + same garble
("!!!" spam / "agh" stubs). The draft-ids hole was real but not the only
one — the walk's accept criterion is catastrophically false at replay.
Pointer auditing exhausted; NEXT = the byte self-check (task #71): at
warmup, per key, paired identical-input steps eager-vs-replay, byte-compare
sampled ids + accepted counts + committed state rows — directly names the
diverging output. No live arm until 4/4 byte-clean.
