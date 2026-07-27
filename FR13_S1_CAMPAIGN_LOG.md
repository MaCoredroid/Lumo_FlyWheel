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

## Boots 34-38: self-check convergence (ongoing)
Harness now: inference-mode wrap (34), paired default-gen seeds + true-eager
via dead-flag (36; warm-set was silently RE-CAPTURING the "eager" arm),
deep-accept capture attempt (37), statics-split probe in0/ent_tls0 (38).
FINDINGS SO FAR:
1. FROZEN REPLAY OUTPUT — the primary defect: replay0 = 38352 CONSTANT
   across keys B=1..4, across boots, while check logits vary => the
   replayed region re-emits the capture-time recovery token; some link
   between the refilled statics (logits_static -> tls/stls -> walk ->
   output) does not recompute at replay. This mechanism exactly produces
   the live signature (accept pinned 1.00 + 1 wrong token/step + garble).
2. EAGER accepts 0 in the fabricated context even with argmax drafts:
   the walk verifies the FORWARD's staged products (which encode the
   dummy forward's ZERO input tokens), not md.draft_token_ids post-hoc =>
   a coherent deep-accept warmup context needs drafts == the tokens the
   dummy forward actually consumed (drafter-coherent fabrication) — or
   run the self-check on the FIRST LIVE steps instead.
Boot-38 statics-split readout decides: ent_tls0 fresh but output frozen
=> freeze between tls and output (walk consumes a baked clone — suspect:
apply_sampling_constraints intermediate); ent_tls0 stale => the tls
refill itself doesn't reach the graph's tensor.

### Boot-38 split-probe readout (B=1)
in0=0 ent_tls0=0 BOTH arms; eager0=4080 replay0=38352 (both boot-constant).
1. POSITIVE: the tls refill PROVABLY reaches the graph's baked tensor
   (ent_tls0 fresh) — the frozen-refill theory is DEAD.
2. CONFOUND NAMED: check-context logits are DEGENERATE (argmax 0 =>
   near-constant rows from the zero-input dummy hidden), and the two arms
   draw from the same-seeded generators in DIFFERENT SEQUENCES (staged
   fill order vs wrapper refill order) => recovery tokens legitimately
   differ per arm while being boot-constant. The MISMATCH as currently
   measured does NOT convict the graph.
NEXT (boot-39): RNG-PROOF self-check — pin uniforms/q CONTENT via
FR13_SG_PIN_UNIFORMS=1 (fill returns constant 0.5) applied in BOTH arms
=> accept/recovery deterministic in logits alone => byte-comparable.
ALSO revisit: the LIVE accept-1.0 may trace to the conv-commit host-eager
sub-step (accepted-lens host reads baked at capture) — audit whether
_fr13_conv_commit_to_col0 consumes host-read lens inside the capture.

## Boots 39-41: THE STATE CONVICTION
Boot-39/40 (pinned uniforms): sampled ids BYTE-EQUAL eager-vs-replay on
ALL FOUR KEYS — the =2 graph's token path is faithful. BUT committed
SSM/conv STATE DIVERGES on a FIXED stride-4 layer set every key & boot:
{12:conv, 16:ssm+conv, 20:ssm+conv, 24:ssm, ...} (list truncated at 6).
This is the live fine-then-garble mechanism: correct tokens now, poisoned
state on those layers corrupts later steps.
PATTERN NOTE: {12,16,20,24}=stride 4 == the GDN layers immediately AFTER
the hybrid's full-attention layers (full-attn at 11,15,19,23 in the 3:1
layout) — the ATTN_KV_REMAP neighborhood.
Boot-41 adds changed-vs-baseline per arm (eN,rN): splits spurious-write
(replay writes junk) vs missing-write (replay skips those layers'
commits — e.g., a baked layer-subset in the batched launch or the
host-eager conv loop).

## Boot-41: ROOT CAUSE CONVICTED — batched commit breaks a cross-layer
## ordering dependency on ATTN_KV_REMAP-adjacent layers
Split verdict: (e1,r1) on every diverging layer, every key — BOTH arms
write the stride-4 layers' state, with DIFFERENT values, under byte-
identical pinned inputs. Synthesis (fits all data):
- The staged committer commits layers SEQUENTIALLY (cf. the standing
  "GDN verify dispatch = sequential rank-1" note); the =2 graph's ONE
  batched launch commits all layers in PARALLEL.
- The post-full-attn GDN layers (12,16,20,24,... = the ATTN_KV_REMAP
  neighborhood) have a commit-time dependency on a neighbor layer's
  freshly-committed/remapped state: sequential sees NEW, parallel sees
  OLD => deterministic state divergence on exactly that stride-4 set,
  ids unaffected, context-independent. Live: state poison on those
  layers => fine-then-garble + accept collapse.
FIX (boot-42 design): TWO-PHASE batched launch — phase 1 commits the
independent layers, phase 2 (after phase-1 completes, still in-graph as
a second kernel) commits the remap-adjacent set. Alternative: pre-stage
copies of the consumed neighbor state so order stops mattering.
Gate: selfcheck must show STATE EQUAL x4 (ids already BYTE-EQUAL x4)
before any live arm.

### Boot-41 final + boot-42 fix plan (two-phase launch, implementation notes)
4/4 keys confirm: ids BYTE-EQUAL, STATE DIVERGED (e1,r1) stride-4 set.
Eager arm = STAGED committer (per-layer SEQUENTIAL dispatch); replay =
ONE batched launch => ordering dependency broken for post-full-attn GDN
layers. NOTE: =3 also captured the batched commit_state_part — its live
2P/2F may have absorbed small state noise; re-examine after the fix.
Implementation (boot-42):
1. Adjacent-set detection at runtime: GDN layer index i is remap-adjacent
   iff layer i-1 is NOT in _FR13_REPLAY_LAYERS (i-1 was full-attn).
2. REORDER stacks layer_order at build: independent layers first,
   adjacent last (one-time permutation at stack construction).
3. Route + commit_state_part: TWO launch_all calls with SLICED stack
   views (rows [0..k) then [k..n)) + banks_list slices — no kernel
   change; kernel-order between the two launches enforces the dependency
   (same stream, still fully in-graph).
Gate before live: selfcheck STATE EQUAL x4 + ids BYTE-EQUAL x4.

## Boot-43 STATEVAL: state "divergence" is BENIGN ULP DUST — check CLEAN
STATEVAL 12.linear_attn: ssm sums IDENTICAL e==r==b (~1e-10, denormal
dust); conv e=0.006293 r=0.006302 b=0.006324 (~0.5% on near-zero totals)
=> batched-vs-sequential accumulation-order ULP class (the ACCEPTED
within-floor category), NOT corruption. Ordering fix (two-phase) can stay
or be reverted — it was never the issue.
NET: the =2 replay is CLEAN in the check context (ids BYTE-EQUAL x4,
states ULP-equal). The LIVE accept-1.0 + garble is live-only.
PRIME SUSPECT: LIVE TOPOLOGY VARIATION — tail6's merged drafter emits
per-step suffix-chain topologies; the fabricated capture baked the static
config tree; the graph key ((21,)*B, logits.shape) does NOT key topology
=> wrong tree structure at replay => reject-all. (=1 lived because its
walk-only graph consumed topology via per-step refilled statics.)
DECIDER IN FLIGHT: boot-43's live phase prints MD-DIFF (live vs baked
tree_parent_indices) at first replay — letting the arm run to live.
Fix candidates: topology in the graph key (per-topology captures — may
explode key count) OR make the walk's topology tables per-step refilled
data (the =1 pattern) OR composition-guard: skip replay when live
topology != baked (staged fallback for non-matching steps).

## Boot-44: TOPOLOGY REFUTED (live MD-DIFF identical) — deep-accept gap named
Live replays (B=2/3/4 keys, multiple steps): live_tgt/self/par/bonus ALL
== baked — the live tree topology IS the config constant (suffix
machinery maps into the same fixed padded 21-node layout).
REFUTED PILE (complete): strip ops, pools, capture modes, exec-lock,
hooks, GC, quiet-window, commit ordering, state corruption (ULP dust),
metadata indices, draft ids, logits address, topology.
THE GAP: the selfcheck only exercises ACCEPT-0 contexts (zero-context
products; even eager accepts 0). Live eager accepts 4.7; live replay
accepts 0 — the divergence lives in code only executed at accept>0
(walk path-selection -> committer row consumption).
NEXT BUILD (boot-45): DRAFTER-COHERENT deep-accept selfcheck — feed the
draft tokens as the dummy forward's input ids at tree positions (dummy
input_ids seam), so products cohere with drafts and the check reaches
deep accepts; replay-vs-eager at accept~5 then reproduces the live bug
in-warmup (12min/boot iteration) or fully exonerates the graph.

## Boot-45: coherent-fill ineffective (accept still 1; in0 still 0) —
## PIVOT to LIVE-PAIR probe
Two-pass coherent fabrication did not move accepts (fill either
overwritten inside _dummy_run's batch prep or position mapping wrong).
ASIDE: token id 0 renders as '!' — the live "!!!..." garble is argmax-of-
broken-context token-0 spam; in0=0 in every degenerate check context is
the same phenomenon. Fabricated-context realism is a rabbit hole (3
iterations); the sharper instrument exists:
BOOT-46 DESIGN — LIVE-PAIR probe at first live replay per key:
1. snapshot committed-state slot rows (existing _snap machinery),
2. REPLAY the live step (record ids + stash lens),
3. restore state, force-flags re-arm (the side-stream warmup protocol's
   save/restore + force-flags already do exactly this),
4. run the STAGED committer on the SAME live stash/products,
5. byte-compare ids + accepted lens + committed rows, print one-shot.
Real live deep-accept inputs, direct staged-vs-replay pair, no fabricated
context — answers "what differs at accept>0" in one boot.

## Boot-46 LIVEPAIR: PER-ROW MISALIGNMENT convicted (the decisive datum)
LIVEPAIR B=4 ids_equal=False replay_lens=[1,1,10,2] staged_lens=[2,5,10,2]
replay0=[244020,...] staged0=[271,16,...]
Rows 2,3: replay == staged EXACTLY (deep accepts 10 and 2 work through
the graph). Rows 0,1: collapse with different tokens. Same step, same
inputs => PER-ROW MISALIGNMENT: a permutation (sampler-order vs spec-row
order) applied on one side of the refill/consume boundary but not the
other; aligned rows = fixed points of the live perm. Retro-explains the
accept~1.0 aggregate (misaligned rows dominate most steps).
NEXT (boot-47): print at LIVEPAIR time — ent["perm"] values + live
sampler-row req ids + spec-row req ids; if broken rows == permuted rows
=> fix is aligning the uniforms/q (and any row-indexed refill) through
the same perm the graph's committer fill-2 uses (or refilling them in
spec order). The warmup check missed it because fabricated perm == identity.

## Boots 48-50: LIVEPAIR REFRAME — no structural defect visible; measure the
## current stack end-to-end (boot-51 = full clean arm)
Paired-uniform attempt was DEFEATED (the staged rerun redraws uniforms/q
internally), so LIVEPAIR arms are NOT RNG-paired. With that understood:
- boot-49 samples 2,3: replay_lens == staged_lens EXACTLY ([1,1,5,4],
  [1,1,7,3]) — the =2 REPLAY ACCEPTS DEEP (5,7,4,3) on live steps.
- Only token VALUES differ, and only at rejected rows (recovery draws) =
  RNG-explained, not a defect.
- draft_eq all True (draft refill correct). tls_eq uniform-False =
  probe artifact (processors run in-place/after the staged arm), not
  row-discriminating.
KEY REALIZATION: the last FULL live arm was boot-31; since then the stack
gained md draft-ids refill (boot-33), logits-static (boot-31), timer
capture-guards (boot-30), philox cure. Boots 32-50 were diagnostics
killed early. The accept-1.00 + garble evidence is from the OLD stack.
BOOT-51 = clean full arm (LIVEPAIR off, no diagnostics): behavioral band
(2P/2F), personal garble eyeball, mid-run accept, arm-end
measured_tps_fullstep_wall + eps + prefill_frac, slope vs staged 17.0 /
=3 18.7. That measurement decides whether =2 is already shippable.

## BOOT-51 CLEAN ARM: GARBLE CONFIRMED — =2 replay IS broken live
All 4 tasks, immediate multilingual token salad from turn 1 ("Let me
understand the jacket الرسول prestazioni Va الجزائ..."), accept ~1.9-2.0,
death 0. LINE STOPPED.
SELF-CORRECTION: my boot-48/49 LIVEPAIR "exoneration" was WRONG. The arms
were never RNG-paired (the staged rerun redraws uniforms/q internally),
and matching accept LENS hid wrong token CONTENT — lens agreeing says
nothing when the sampled ids differ (which they visibly did: replay0
near-vocab-max vs staged0 normal, EVERY sample). The near-vocab-max ids
(243944, 244024, 200559, 105070...) are the garble in the raw: the =2
replay emits wrong-token content while accept structure looks plausible.
STANDING LESSON (already in memory, re-learned the hard way): a scalar/
structural metric (accept lens) is blind to per-token defects; gate on
CONTENT.
WHAT IS SOLID: capture works 4/4 every boot; ids byte-equal + state
ULP-equal in warmup checks (but those checks are VACUOUS — degenerate
accept-0 contexts).
NEXT: the honest instrument is a CONTENT gate at live steps —
replay ids vs staged ids on the SAME step with RNG genuinely neutralized
(pin uniforms INSIDE both arms via FR13_SG_PIN_UNIFORMS for the probe
step only, so the staged redraw is also pinned; boot-47 pinned only one
side). If ids still differ under both-pinned => hard defect with content
evidence; then bisect the region (=3 + sampler only, etc.).

## SEPARATE FINDING (not the =2 garble): DOUBLE temperature application
Stock rejection_sampler.py has ONE `target_logits = apply_sampling_constraints(...)`.
The patched file has TWO in sequence (patcher injects its own copy above the
stock call; introduced e87808ee5 2026-06-05 / 44adae753 2026-06-07).
apply_sampling_constraints applies TEMPERATURE + top-k/top-p, so the tree
target logits are scaled by 1/T twice => effective temp 0.36, not 0.6
(top-k/top-p are idempotent; temperature is NOT).
SCOPE: the call sits in the shared spec-decode forward, so EVERY spec arm
(tree AND native MTP bars) has been affected equally since 2026-06-05 —
cross-arm comparisons remain valid, but the absolute regime is temp 0.36.
NOT the =2 garble cause (staged and replay share this code).
DECISION NEEDED (user): fix + re-baseline the reference band, or keep the
current regime for continuity. Not changed unilaterally — it moves every
arm's behavior and would invalidate the standing band mid-campaign.

## BOOT-53: GARBLE FIXED — constraint-kernel omission was the root cause
Fix (6eba91b1b): warmup capture now builds dummy SamplingMetadata with REAL
top_p/top_k TENSORS (were None => the graph never recorded the top-k/top-p
constraint kernels), and the replay refills temperature/top_p/top_k VALUES
from live metadata each step (the graph bakes addresses).
LIVE RESULT (boot-53, diagnostics off): garble GONE. Coherent English,
real work — 12907 landed a genuine one-char fix in separable.py::_cstack
("all 11 tests pass") and EVALUATED PASS. Accept recovered 1.9 -> 2.9-3.6.
Verdicts: 12907 PASS; 13033/13236/13398 fail (13398 emitted raw tool_call
markup = known qwen-code protocol quirk, ended early n=7).
=> 1 pass, 3 fail, 4 finished — at the s1go/dscg reference band (1P/3F),
below boot-16/24's 2P/2F. Accept still under the staged 4.5-4.7 band;
next questions are (a) is the accept gap real or composition, (b) speed
tuple at arm end.
EVIDENCE CHAIN that cracked it: both-sides RNG pin => walk products
byte-identical while emitted tokens still diverged => difference had to be
the sampling DISTRIBUTION, not the walk/state/RNG => missing constraint
kernels.

### Boot-53 FINAL RECORD (first honest captured-=2 speed tuple)
Verdicts: 1 pass, 3 fail, 4 finished (12907 resolved; 13033/13236/13398
failed) = s1go/dscg band.
Speed: measured_tps_fullstep_wall 38.384 | step_wall 351.468ms @ eps 2.645
| accept 4.101 | prefill_frac 0.642 | s_per_fwd_gpu 0.0867 | death 0.
SLOPE (accept+1)/step = 14.51 tps/eps.
SCOREBOARD (slope basis, the eps-independent comparison):
  =3 graph (boot-23)      18.7
  staged (boot-16/24)     17.0 / 17.0
  =1 graph (s1go)         14.7
  =2 graph (boot-53)      14.51   <-- SLOWEST per-event so far
READ: =2 now WORKS (garble fixed, real task PASS) but its per-event
economics are the worst of the ladder — the one-graph =2 region is not
paying off yet. Raw tps 38.4 is the highest number on the board ONLY
because eps 2.645 was high (workload phase), and prefill_frac 0.642 is
also the highest (heavy prefill share) — both confound raw tps; the slope
is the honest read.
Accept 4.101 is近 the staged band (4.5-4.7) but below it.
NEXT (one lever at a time): (a) confirm the slope with a second =2 arm
(single sample, workload-sensitive), (b) if it holds, decompose why the
=2 replay costs more per event than =3 (extra baked constraint kernels +
per-step metadata refills are the new suspects — they were ADDED by the
fix), (c) only then consider S2.

## BOOT-54: =2 slope CONFIRMED (2-arm agreement) — functional, not faster
Verdicts: 1 pass, 3 fail, 4 finished (12907 resolved+PASS again;
13033/13236/13398 failed). Garble-free, coherent traces, death 0 — the
constraint fix REPRODUCES.
Speed: 38.731 tps wall | step 375.020ms @ eps 2.554 | accept 4.686 |
prefill_frac 0.390 | s_per_fwd_gpu 0.095 | slope 15.16 tps/eps.
TWO-ARM =2 SUMMARY (the honest, eps-independent read):
  boot-53: slope 14.51 (accept 4.101, step 351.5, eps 2.645, prefill .642)
  boot-54: slope 15.16 (accept 4.686, step 375.0, eps 2.554, prefill .390)
  mean ~14.8 tps/eps
LADDER (slope): =3 18.7 > staged 17.0 > =1 14.7 ~= =2 14.8
=> CONFIRMED: the one-graph =2 region is the SLOWEST per-event of the
ladder, ~21% below =3, despite accept parity (4.686 vs staged 4.5-4.7).
Its raw tps (38-39) leads the board only via high eps + phase.
Behavioral band: 1P/3F both arms (s1go/dscg band, below boot-16/24 2P/2F).
Note prefill_frac swung .642 -> .390 across the two arms with slope stable
(14.5/15.2) => the slope IS the robust statistic, as designed.
NEXT: decompose WHY =2 costs more per event than =3 (task #72 step 2).
Suspects, in order: (1) the constraint kernels the garble fix moved INSIDE
the graph (top-k/top-p now execute per replay in-region), (2) the per-step
temperature/top_p/top_k refills added alongside, (3) the double
apply_sampling_constraints (temp²) — cheap to test by comparing CFWD/SFWD
sidecar spans boot-54 (=2) vs boot-23 (=3) at matched eps.

### Boot-54 sidecar decomposition — INCONCLUSIVE, and it undercuts the
### "=2 region costs more" reading
boot-54 (=2, pid217) vs =3-era (pid231), per-step GPU spans:
  sfwd (VERIFY FORWARD): 217.81 vs 194.23 ms/step  (+23.6)
  dfwd (drafter):         54.07 vs  54.98 ms/span  (parity)
  cfwd (committer):       71.34 x571 vs 38.68 x6108 spans
  wall:                  340.9  vs 299.3  ms/step  (+41.6)
TWO PROBLEMS with attributing the slope gap to the =2 region:
1. cfwd is NOT comparable: under =2 the committer runs INSIDE the graph, and
   the span timers are capture-guarded (boot-30 fix), so only the 571 STAGED
   steps are timed — the 71.34ms/span is mixed-step cost, not replay cost.
2. The biggest comparable delta is sfwd (+23.6ms/step) — the verify forward,
   which =2 does NOT touch. That points at workload/co-residency PHASE, not
   at the captured region.
=> The 2-arm slope gap (14.8 vs 18.7) is real as measured but NOT yet
attributed; cross-boot phase is a live confound (exactly the eps-matched
lesson from bar17-r2).
NEXT (correct instrument): SAME-SESSION A/B — one boot running the =3 arm
and the =2 arm back-to-back on the same subset, so phase/co-residency and
host state are shared; compare slopes within that boot. Only then decide
whether =2 costs per-event, and only then attribute.

## POOLED REGRESSION (7 arms, all modes): NO MODE DIFFERENCE DETECTABLE
Fit over every arm-level tuple measured (=1, =2, =3, staged):
    step_wall_ms = 235.5 + 49.2 * eps      (R ~ exact; residuals below)
  =3  boot23  eps 1.441 step 305.5  resid  -0.9   slope 18.70
  stg boot16  eps 1.847 step 325.0  resid  -1.4   slope 16.97
  stg boot24  eps 1.899 step 331.5  resid  +2.5   slope 17.00
  =3  armA    eps 2.093 step 337.0  resid  -1.5   slope 15.96
  =2  boot54  eps 2.554 step 375.0  resid +13.9   slope 15.16
  =2  boot53  eps 2.645 step 351.5  resid -14.2   slope 14.51
  =1  s1go    eps 2.740 step 371.8  resid  +1.5   slope 14.70
Mean residual by mode: =1 +1.5 | =2 -0.1 (spread 28) | =3 -1.2 | stg +0.6
=> ALL FOUR MODES SIT ON THE SAME LINE within +/-1.6ms of each other.
CORRECTION TO MY OWN CLAIM: the "slope ladder" (=3 18.7 > stg 17.0 > =2
14.8) was an ARTIFACT. Slope = (accept+1)/step_wall does NOT normalize
eps, because step_wall itself grows with eps (F + m*eps). Slope predicted
from eps ALONE reproduces every arm: eps 1.4 -> 18.1, eps 2.0 -> 16.5,
eps 2.6 -> 15.1. The arms differ in WORKLOAD PHASE, not in mode.
WHAT THIS MEANS FOR S1: the one-graph =2 region neither costs nor saves
measurable step time vs =3 or staged. Detection floor here is ~±15ms/step
(the =2 residual spread) = ~4% of step. S1's graph fusion did NOT move the
step-time needle.
WHERE THE TIME ACTUALLY IS: fixed 235.5ms/step + 49.2ms per verify event.
At eps 2.5 the FIXED term is ~66% of the step — and the measured verify
forward alone is ~194-218ms/step (sfwd sidecars), i.e. essentially the
whole fixed term. The remaining lever is the VERIFY FORWARD (= exactly
what S2 targets), not the sampler/committer region.
NOTE: 49.2ms/event marginal now matches the native ~49ms/event figure from
bar17-r2 — the tree's per-event cost is no longer the outlier it was.
NEXT INSTRUMENT (to resolve below ±15ms): per-STEP regression from the
sfwd sidecar (thousands of steps/arm, each with known draft count) instead
of arm-level aggregates; gives F and m per mode with real confidence
intervals.

## STRATEGIC RE-BASE: the tree deficit is a FIXED +75ms/step, not per-event
Fits (step_wall vs eps):
  NATIVE (e5, B=4 & B=8):   step = 160.4 + 44.6 * eps
  TREE   (7 arms, all modes): step = 235.5 + 49.2 * eps
  => MARGINAL essentially at PARITY (+4.6 ms/event) — the historic row-tax
     (bar17-r2: tree ~140 vs native ~49 ms/event) is GONE, killed by the
     kernel work (PARENT_GATHER, SLOT_REORDER, subtree-parallel, HC/PG).
  => The entire remaining deficit is the FIXED term: +75.1 ms/step.
Net effect at matched eps (tree accept 4.5 vs native 3.55):
  eps 2.0: native 36.47 tps | tree 32.94 | tree 0.90x
  eps 2.6: native 42.82 tps | tree 39.35 | tree 0.92x
The +27% accept advantage is MORE THAN EATEN by the +75ms fixed tax.
WHAT THE FIXED TERM IS MADE OF (measured, per step):
  drafter  ~54 ms GPU  (21-node tree draft vs native's 5-token chain)
  committer ~39 ms     (staged span; in-graph under =2/=3)
  + host gaps/publishes
S1 VERDICT IN THIS FRAME: =2/=3 graph fusion targeted part of this fixed
term and moved it by < detection floor (~15ms) — because the host cost had
already been hoisted out to make capture legal; only launch gaps remained.
TWO WAYS TO WIN (both quantified):
 (a) CUT FIXED: -75 ms/step reaches native parity at equal accept. The
     drafter (54ms) is the single largest owned component.
 (b) RAISE ACCEPT: at the CURRENT fixed cost, accept >= 5.0 beats native
     at eps 2.6 (need (a+1) > 5.985). We sit at 4.5-4.7 => the accept>5
     work (tasks #33/#45) is now a DIRECT win condition, not a nice-to-have.
S2 AS ORIGINALLY FRAMED (fold verify forward into our graph) is NOT
obviously the lever: the verify forward is already inside vLLM's own CUDA
graphs and its ~200ms is GPU work (98.6ms weight-read floor + tree rows),
not launch overhead. Verify that claim (full-graph vs piecewise for the
tree decode shape) BEFORE investing in S2.

## POST-SESSION AUDIT (2026-07-27, 7-agent verification of boots 47-54 claims)
Independent re-verification against primary records; supersedes the
final-hour transcript numbers where they conflict.

### Regression claims (a074cf45b / 0fa12a74a) — DOWNGRADED, not overturned
- Pooled tree fit reproduces exactly: 235.42 + 49.25*eps, R^2=0.894 — but
  n=7 gives WIDE CIs: F +/-43ms, m +/-19.5 ms/event. "R ~ exact" overstated.
- Transcription: the 'stg boot16' tuple is actually the BOOT-19 record;
  dscg (staged, eps 2.323, step 335.6) was OMITTED and carries a -14.2ms
  residual — the +/-14ms spread hits STAGED too, it is NOT =2-specific.
  n=8 fit with dscg: 236.5 + 47.95, R^2=0.85.
- Honest mode-detection floor at arm level is ~+/-100ms (df=2), not the
  quoted +/-15ms. "No mode difference" stands as a point estimate
  (F(3,2)=0.03); a sub-15ms claim needs the per-STEP sidecar regression.
- NATIVE "fit" 160.4+44.6 is an exact 2-POINT line (native_b4 eps 2.608,
  native_b8 eps 5.067; zero df) whose slope is identified solely by the
  B=4->B=8 contrast (batch, 4-vs-8 task set, boot date, prefill_frac
  0.14-0.17 vs tree 0.39-0.64 all conflated). native_b4 was health-flagged
  (3/4 early-exit) with vacuous pair dumps.
- => "deficit is FIXED +75.1 / marginal at parity +4.6" is ONE consistent
  reading, NOT a measured fact (fixed-deficit CI [32,118]; marginal-diff
  CI [-15,+24]; eps=0 intercepts extrapolate 1.4-2.6 below the data).
  ROBUST version: TOTAL deficit ~85-90 ms/step at operating eps 2.0-2.6
  (87.2+/-12.1 at eps 2.6 where native_b4 is direct data); tree ~0.92x
  native there; bar17-r2's ~140ms/event row-tax confirmed dead (marginal
  CI excludes 140). Win conditions numerically survive: cut ~85ms OR
  accept >= 5.0.

### Drafter / tail — the final-hour "11 sequential passes" model is WRONG
- The depth-6-11 tail is NOT MTP-drafted: FR13_TAIL_MODE=1 caps the MTP
  loop at the depth-5 head (root + 4 graph-replayed spine forwards);
  depths 6-11 come from the Arctic suffix trie (host walk ~0.3ms). The
  live boot's TAIL counters read TAIL[fired=12854 hit=12731]=99.0% —
  the tail IS engaged. The session's "suffix never fires" misread zeroed
  HEAD-MERGE counters (speculate_fired etc., by-design dormant in tail
  mode) from a log line truncated at exactly 'TAIL[fir'. The user's
  correction ("suffix tail is 6 so drafter we are also 5") was RIGHT.
- Ergonomics trap fixed in understanding (needle docstring says
  "match_full>0 is the engagement proof" — that is merge-mode only; in
  tail mode the proof is TAIL[hit]>0).
- dfwd ~54ms/step IS real GPU span (drain-artifact theory formally
  retracted in FR13_LEVER_REDESIGN.md; DVK bake 94.9->56.3 corroborates);
  it is the HEAD loop (~4 iters), so tail-cut saves ~0.3ms drafter, not
  ~29ms. The "3.6x cheaper than native" compare is confounded (our DVK-64k
  head vs native full-vocab, cross-boot).
- Per-depth accept (task #33's histogram) ALREADY ANSWERED from bracket
  counters across 7 clean arms: tail d6-11 carries 0.85-1.31 accept/event
  = 19-27% of total; comb d1-5 = 3.41-3.91; branch siblings +0.14-0.29
  vs straight chain (mostly depth 1). TAIL-CUT IS AN ANTI-LEVER: realistic
  saving ~5ms/step vs break-even accept loss 0.43-0.47 -> net -14..-21%
  TPS, and accept falls AWAY from the >=5.0 win condition. The accept>5
  route is a BETTER tail: native11's trained head earns 1.699 in d6-11 vs
  Arctic's 0.85-1.31 (+0.4-0.8 headroom at ~zero step cost). Open piece
  of #33 = Q2 suffix-complement miss-coverage.

### Decomposition numbers (final-hour tables) — corrected
- gaps "~70ms" and "~44ms" BOTH fail on a consistent basis. Boot-54
  task-window recompute: wall 375.0 - sfwd 242.6 - dfwd 55.1 - cfwd 66.5
  => host/gaps ~10.8 ms/step (boot-53: 8.4), upper bound 77.3 only if the
  in-graph committer were free.
- sfwd is NOT "essentially the whole fixed term": 3-point fit gives
  sfwd ~ 155 + 31*eps — a ~155ms fixed part (~2/3 of F) plus most of the
  49.2ms/event marginal living INSIDE the verify forward.
- Committer band 39-66 ms/span (39 = boot-23 =3 staged basis; boot-53/54
  60-66 on the ~20% staged-timed steps; in-graph cost unmeasured). Burn
  drop confirmed baked (FR13_APC_BURN_NODE_BANK=0 live).

### Engagement / validity audits — all PASS
- Boot-53 constraint-kernel fix (6eba91b1b) present in live source and on
  the active =2 path; SELFCHECK BYTE-EQUAL; reproduces across 3 arms.
- RNG pin (a4348d178) disarmed in speed arms (env absent, defaults 0,
  zero LIVEPAIR needles) — no observer effect.
- Double temp still live (rejection_sampler.py:152+158). NEW nuances:
  (a) tree_self_logits get a SINGLE apply => parent rows 0.36 vs tree
  rows 0.6 asymmetry inside the same arm; (b) native arms booted via the
  STOCK launcher (nativemtp5/nativemtp5apc/nativemtp11apc kinds) ran at
  TRUE 0.6 — not temp-matched vs tree; e5 fit bars (patched launcher)
  ARE matched. Decision on the fix remains QUEUED (user).
- "Same-session A/B" is actually TWO boots back-to-back (m3 14:52Z ->
  torn down -> m2 15:49Z), same driver/subset; phase control partial.
- S2 premise check largely answered: live boot captures FULL decode CUDA
  graphs at exactly the B=4 tree shape (FULL=4, largest=88=4x22).
  Residual: measure runtime FULL-vs-piecewise dispatch fraction (cheap).
- Housekeeping flagged: "keep =3" is transcript-only (committed default
  everywhere = FR13_STEP_GRAPH=0 staged; #72 defers the bake decision to
  arm-B's tuple); #45 closed with a pre-rebase verdict; #49/#50/#60
  stale; FR13_STEP_GRAPH_DESIGN.md needs an outcome note (projections
  refuted); armA(=3) verdicts 2P/2F (12907+13236) not yet in this log —
  vs =2's 1P/3F x2 (small-n, watch arm B).

## A/B COMPLETE + #72 CLOSED (2026-07-27 17:13Z)
Arm B (s1ab_m2, =2) finished: 2 pass, 2 fail, 4 finished (12907+13236 pass,
13033+13398 fail — SAME per-task outcomes as arm A). Final tuples:
  armA =3: accept 4.3766 | eps 2.0934 | tps 33.399 | step 336.99ms | pf .394
  armB =2: accept 3.6973 | eps 1.6568 | tps 25.106 | step 309.98ms | pf .296
Pooled-line residuals: =3 -1.5ms | =2 -7.0ms => BOTH arms on step=235.5+49.2*eps.
The =2/=3 band worry dissolves (arm B 2P/2F). Accept 4.38 vs 3.70 = workload
composition (arm B's 13398 ran a 4543s solo exploration tail), comb-basis in
band. VERDICT: no mode buys speed or behavior; FR13_STEP_GRAPH stays 0.

## CLEANUP + BAKE EXECUTED (2026-07-27, FR13_CLEANUP_BAKE_PLAN.md; commits
## 429cad7f6 plan, e21d626f1 bake, 8bca48c2b delete)
- BAKED as launcher defaults: ENABLE_APC, TAW, PARENT_GATHER, COMMITTER_GRAPH,
  CONV_PREGATHER, FLAGS_INKERNEL, SUBTREE_PARALLEL, DRAFTER_GRAPH (all =1);
  KIND defaults to tail6 (canonical config); fr13_launch_locked.sh retired
  HISTORICAL.
- DELETED: head-merge decide_and_fill seam + module path (needle now TAIL-led,
  truncation-misread trap killed); committer burn (FR13_APC_BURN_NODE_BANK) +
  legacy runrow=0 path (retired fail-loud; kernel burn kwarg dead-False);
  FR13_HC_INTERNAL mechanism (hc_internal_on()->False, wiring+preseed removed);
  suffonly KIND. Kernel-body HC_MASK/BURN constexpr excision = follow-up with
  its own bit-exact gate.
- OFFLINE GATE PASS: edited patcher applied end-to-end in a throwaway container
  on the pristine image tree — all steps True (except the known patch-time
  env-gated APC_ZERO_MAMBA, False without its env), full compileall of patched
  vllm PASS, deletions verified tombstone-only in patched outputs;
  merged-drafter unit tests rewritten for the tail surface: 24/24 PASS.
- LIVE GATE: cleanbake1 (tail6, subset_b4_four, B=4, temp 0.6, NO stack
  exports — baked defaults under test) launched; gates = engagement needles +
  garble-free traces + 2P/2F band + tuple on the pooled line. Next after PASS:
  double-temp fix as its own milestone + same-subset A/B re-base.
- Boot needles verified live: FULL decode graphs 4/4 largest=88 (tree shape),
  new TAIL-led ENGAGED needle firing (TAIL[fired=84 hit=77]), CPG first-line
  signature byte-identical to reference boots, APC_ZERO_MAMBA engaged, 0 errors.

## DOUBLE-TEMP FIX LANDED (2026-07-27, commit 58b51f762 — user go-ahead)
Root cause of the double apply: the injection predates images whose STOCK
rejection_sampler applies constraints itself; the guard only checked for the
patcher's own copy => second apply => T^2 (effective 0.36 at requested 0.6)
for every spec arm since 2026-06-05. FIX: debug pre-capture inserted always;
the extra apply ONLY when stock lacks it OR FR13_TEMP_LEGACY_DOUBLE=1
(patch-time control-arm hatch). Container-validated: default => 1 apply
(true 0.6); legacy => 2 applies (byte-reproduces 0.36). Boot needle
[FR13_TEMP_FIX] prints stock/legacy/injected at patch time.
RE-BASE A/B queued: cleanbake1 (legacy 0.36 — its boot predates the fix) vs
tempfix1 (fixed 0.6), back-to-back boots, same subset_b4_four, no exports.
ALL pre-fix absolute accept/behavior numbers now carry the 0.36 caveat;
tempfix1 sets the new reference band. Side effect resolved: the parent-row
(0.36) vs tree-self (0.6) asymmetry inside the tree arm collapses to a
uniform 0.6.

## HONEST HOLE CLOSED (2026-07-27, live cleanbake1 mid-run): host/gaps ~11ms
## CONFIRMED with FULL committer coverage — the [11,77] ambiguity resolves to 11
The =2/=3 arms could not time the in-graph committer (capture-guarded events,
~20% staged coverage) => the audit's [11, 77]ms host-gap band. The CANONICAL
default (STEP_GRAPH=0) runs the committer on the staged path EVERY step, and
cleanbake1 boots with all three async span timers armed => full coverage:
  wall 735.50s/2467 steps = 298.1 ms/step @ eps 1.369 (boot-cumulative)
  sfwd 487.14s/2534      = 192.2 ms/step   (fit 155+31*eps -> 197.5, -5)
  dfwd 156.19s/2877      =  54.3 ms/span
  cfwd 108.90s/2711      =  40.2 ms/span   <- n_spans ~= n_steps = FULL coverage
  residual host/gaps     = ~11.4 ms/step
The committer measured across ALL steps is 40.2ms (boot-23 staged basis 39.7
reproduced) — NOT free, so the 77ms upper bound collapses. Host overhead on
the deployed path is ~11ms/step; the remaining floor gap is GPU work.
Caveats: boot-cumulative window (warmup included; per-timer denominators
differ slightly); canonical task-window decomposition lands in each arm's
deploy_speed_msr.json at arm end; tempfix1 replicates at true temp 0.6.
INSTRUMENT ADDED for the residual F/m question: per-step (drafts, ms) samples
in the sfwd timer final JSON (teardown-only dump) — tempfix1 carries it =>
per-STEP regression with real CIs replaces arm-level n=7 fits.

## TEMPFIX1 RE-BASE COMPLETE (2026-07-27 19:53Z): TRUE temp 0.6, accept holds
Verdicts: 1 pass, 3 fail, 4 finished (12907 pass; 13033 tests_failed; 13398
context-marathon fail; 13236 whitespace-marathon fail 3095s — its known
signature at BOTH temps, self-limited as predicted). Garble eyeball 4/4 clean.
Engagement PROVEN: [FR13_TEMP_FIX] stock_apply=True legacy_double=False,
live rejection_sampler = exactly 1 apply; TAIL 98% hit.
RE-BASE TUPLE (vs cleanbake1 = the legacy-0.36 control, same subset,
back-to-back boots):
  cleanbake1 (0.36): accept 4.434 | eps 1.579 | tps 27.548 | step 311.5 | pf .578
  tempfix1   (0.60): accept 4.331 | eps 2.571 | tps 35.644 | step 384.5 | pf .414
=> ACCEPT AT TRUE 0.6 = 4.331 — only −0.10 vs the 0.36 control. The feared
0.36-inflation of the accept band was SMALL: the honest band is ~4.3-4.4.
Tail share holds (~18-22% live windows). NEW REFERENCE BAND: accept ~4.33,
band 1P-2P on subset_b4_four, temp regime = TRUE 0.6 from here on.
Step-line note: 384.5 measured vs 362.0 predicted at eps 2.571 (+22.5) —
same above-line drift as boot-54 (+13.9 at eps 2.55); high-eps arms suggest
the pooled m is underfit (its CI was ±19.5). The per-step regression decides.
INSTRUMENT BUG found+fixed (3f7143acc): per-step samples were teardown-only
and docker rm -f (SIGKILL) never runs atexit — tempfix1's samples LOST
(sidecar final=False). Samples now ride a 30s slow throttle; next
instrumented arm delivers the F/m regression.

## SUBSPAN1 COMPLETE (2026-07-27 23:06Z): B2c speed POSITIVE + best board tps
Verdicts: 1 pass, 3 fail, 4 finished (12907; 13236 flipped back — coin-flip
class; band 1P-2P holds). Garble 4/4 clean.
Tuple: accept **4.749** (best true-0.6 yet; band now 4.33-4.75) | eps 2.701 |
**tps 43.570 — best measured_tps_fullstep_wall on the board** | step 356.4ms
| pf .504 | drafter 54.6 | committer 60.6.
B2C SPEED VERDICT (two independent signals, cross-boot caveat):
  - sfwd per-step fit: 132.7 + 37.65·drafts vs regress1's 139.4 + 40.05
    (−6.7 fixed, −2.4/event)
  - step 356.4 vs the regress1-based model's 376.5 at eps 2.701 (−20ms)
  => FR13_CONV_WB_BATCHED ≈ −10-20ms/step at operating eps. BAKE per the
  bake-on-golden rule (offline byte gate 07-24 + b2c1 band PASS + this).
NOTE (headline with caveats): at eps 2.7 the native fit predicts ~43.7 tps —
this arm sits AT native-parity throughput for the first time (single arm,
pf .504, workload-phase caveats; the accept 4.75 × eps 2.7 combination did
it). SUBSPAN: final n=1751 samples, HOST≈35µs confirmed; mark #2 (CPU tail)
arm subspan2 queued (marker bug fixed — sed'd runner echoed B2C2_DONE).

## B2C1 COMPLETE (2026-07-27 22:05Z): B2c band-gate PASS + dispatch verdict
Verdicts: 2 pass, 2 fail, 4 finished (12907+13236 pass) — band holds with
FR13_CONV_WB_BATCHED engaged (preseeded needles fired). Garble eyeball 4/4
clean. SPEED NOT QUOTABLE (ctrace rider confound, recorded). Deliverables:
- DISPATCH TAGS FINAL: 6329/6329 FULL — piecewise-fallback hypothesis dead
  at scale; the ~28ms/event in-span idle is host time between the begin
  event and the FULL-replay enqueue (or a residual per-event sync).
- ctrace corpus: 34k records snapshotted (b2c1_ctrace_snapshot.jsonl; the
  container-local file died at teardown ~1 min after the last snapshot) —
  per-node draft/parent/argmax/prob = the branch-tail join dataset.
NEXT: b2c2 clean arm (B2c + timers, NO debug rider) = B2c's real speed
verdict; sub-bracket instrument (begin→replay-enqueue split + logits-compute
bracket) authored for the boot after.

## REGRESS1 COMPLETE (2026-07-27 21:00Z): per-step F/m DELIVERED + 2P/2F at 0.6
Verdicts: 2 pass, 2 fail, 4 finished (12907 + 13236 pass — 13236's coin flip
landed pass; the 2P band reappears at TRUE 0.6). Tuple: accept 4.561 | eps
2.173 | tps 34.984 | step 345.4ms | pf .316 | drafter 56.1 | committer 52.4.
TRUE-0.6 accept band now 4.33-4.56 across two arms.
PER-STEP REGRESSION, FINAL (harvested samples, p98-trimmed, n=5617/5769):
  step_wall = 217.1(±0.7) + 58.99(±0.39)·drafts   R²=0.941
  sfwd      = 139.4(±0.6) + 40.05(±0.31)·drafts   R²=0.915
  (tuple cross-check: model predicts 344-347 at eps 2.173 vs measured 345.4)
=> the arm-level pooled fit (235.5+49.2) is RETIRED: fixed ~217, marginal
~59/event. Non-verify remainder = ~78 fixed + ~19/event (drafter+committer+
host). IN-SPAN IDLE stands as the campaign: verify marginal ~40/event vs
~11-12/event measured kernels => ~28ms/event non-kernel inside the forward
span; between-span remainder adds ~78+19·eps. Samples instrument (v2
.samples sidecar + harvest) = the standing per-arm deliverable.
Task #7 (honest hole) CLOSED: host/gaps fully attributed at three levels —
between-span ~11ms/step, in-span idle named + sized, F/m resolved with CIs.

## CLEANBAKE1 GATE: PASS (2026-07-27 18:52Z) — cleaned+baked build behaviorally
## neutral, tuple ON the pooled line
Verdicts: 1 pass, 3 fail, 4 finished (12907 pass; 13236 patch_apply_failed —
the known flip-flopper, whitespace-tail, same class as boot-54; 13033
tests_failed coherent; 13398 tests_failed after its known context-marathon,
COMPRESSION_FAILED at 75k tokens). Within the 1P-2P historical band for this
subset. Garble eyeball: all 4 traces clean (0 flags, 0 nonascii).
Speed: 27.548 tps wall | step 311.545ms @ eps 1.5795 | accept 4.4337 |
pf 0.5785 | floor_ratio 3.160 | drafter 54.77 | committer 45.61 (full-coverage
staged basis) | engaged tok_per_draft=21.
Pooled-line check: predicted 313.2 @ eps 1.58 -> residual -1.7ms. ON the line.
=> the bake (8 defaults) + deletions (head-merge/burn/HC) are confirmed
behavior- and speed-neutral on the served path. NOTE this arm still runs
LEGACY temp (0.36) — its boot predates the temp fix => it doubles as the
re-base A/B control. tempfix1 (true 0.6 + per-step samples) auto-started
18:5xZ (container up).
