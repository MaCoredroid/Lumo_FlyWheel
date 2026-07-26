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
