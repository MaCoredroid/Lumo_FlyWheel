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
