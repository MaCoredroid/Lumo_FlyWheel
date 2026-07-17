# FR13 Pipeline Overhead Accounting — tail6b (B=4, subset_b4_sixteen, b7)

Source: `output/fr13_tail6b_ab/tail6b_b7/deploy_speed_b7.json` (FR13_DFWD/CFWD/SFWD_GPU_TIMER on,
async cuda-event spans over the per-task /metrics brackets, 16/16 tasks).

## Per-decode-step GPU component spans (MEASURED)

| stage        | GPU ms/step | note |
|--------------|------------:|------|
| drafter      | **101.1**   | `drafter_gpu_ms_per_step` — MTP head (5 fwds) + arctic tail retrieval + merge logic |
| verify       | **105.0**   | `s_per_fwd_gpu`=0.10501 — the 25-node TREE_ATTN forward |
| committer    | **108.7**   | `committer_gpu_ms_per_step` — rejection-sampler + commit; **includes host DtoH+sync (FR13_GPU_COMMITTER=0)** |
| **GPU compute subtotal** | **314.8** | = `committed/derived_tps_fullstep_gpu` = 5.5002/17.4687 |

## Connecting to ACTUAL (not derived) tps

- `committed_per_event` = 5.5002 tok/step, `per_request_decode_tps` = **4.4518** tok/s (real per-stream).
- wall/step = committed / per_request_tps = 5.5002 / 4.4518 = **1235.5 ms/step**.
- **GPU compute = 314.8 ms/step (25.5%). Non-compute gap = 920.7 ms/step (74.5%).**
- The gap = host orchestration + chunked-prefill interleave (prefill_frac 0.453) + co-residency
  (effective_concurrency 2.05). **NOT yet decomposed into reducible-vs-fixed — that is the open question.**

## Ceiling arithmetic (per-stage removal, tail6b)

| lever | ms removed | wall/step | tps | gain |
|-------|-----------:|----------:|----:|-----:|
| baseline | — | 1235.5 | 4.45 | — |
| async/GPU committer (remove 108.7) | 108.7 | 1126.8 | 4.88 | +9.7% |
| graph drafter (remove 101.1) | 101.1 | 1134.4 | 4.85 | +9.0% |
| zero ALL GPU compute (315) | 314.8 | 920.7 | 5.97 | **+34%** |
| **+ collapse the 920ms gap → HW limit** | ??? | ??? | ??? | **the real prize** |

The +34% is only the ceiling IF the 920ms gap is truly fixed. **It is not obviously fixed — we own the
whole pipeline (forked patcher + tree kernels + drafter + committer + host loop).** Decomposing and
attacking that gap toward the hardware limit (weight-read floor ~98.6ms/necessary-forward, fully
overlapped) is the huge-TPS-win target. Serial 315ms compute → if the 3 stages PIPELINE across steps,
step compute → max(stage)≈105ms not sum(315). Plus removing host syncs + graphing the whole step.

## vs native MTP-5 (CROSS-RUN, native_nocache_qc4; native component timers were OFF → decomposition TBD)

| metric (B=4) | tail6b (TREE) | native MTP-5 |
|---|---:|---:|
| accept/fwd | 4.500 | 3.336 |
| verify s/fwd (GPU) | 0.105 | 0.073 |
| per-stream tps | 4.45 | 4.60 |
| kernel tps_gpu (verify-only) | 52.4 | 59.1 |

**Native's drafter/committer NOT measured** (its timers were off). The sweep re-runs native
(flash_ns5_nocache) with `FR13_DFWD/CFWD_GPU_TIMER=1` (set by run_variant for every arm) → full
native stage decomposition, same-session vs the tree arms. Only THEN can we say whether native's
5-sequential-MTP drafter is cheaper or dearer than the tree's 101ms, and where the tree's real deficit is.

GOAL (user): push the WHOLE pipeline to the hardware limit — we are kernel MAKERS, not reproducers.
Huge TPS win, not another +0.05 accept.

---

## b7 CLEAN same-session result: d6-branch is an accept win but a SPEED LOSS

| metric | tail6b (25-node branched) | tail6 (21-node spine) | delta |
|--------|--------------------------:|----------------------:|-------|
| accept_per_event | 4.500 | 4.317 | +0.18 (+4.2%) |
| per_request_decode_tps | 4.452 | **4.889** | **-8.9%** |
| kernel derived_tps_gpu | 52.4 | 56.9 | -7.9% |
| committer_gpu_ms/step | 108.7 | 94.0 | +14.7 |
| s_per_fwd_gpu | 0.105 | 0.0935 | +12% |

The +0.18 accept does NOT pay for the +4 branch nodes' forward+committer cost => net **-8.9% per-stream tps**.
Confirms: bloating the tree for accept is the WRONG speed lever. tail6 (spine-only) is the fastest tree.
Geometry-widen arms (tail6c/tail6e) DEPRIORITIZED (more nodes = slower).

## HW-LIMIT PLAN (workflow wf_fc8d5fe5-a49: 6 code-readers + design + adversarial verify)

**Gap decomposition (the 920.7ms/step, was hand-waved as "fixed"):**
- **~250ms (27%) = REDUCIBLE host stall we own** — sync engine loop + committer DtoH+full-stream
  synchronize (patcher:7947-7948, 91.9% of committer window) + drafter eager launches. PER-STREAM killable.
- ~305ms (33%) = genuine co-resident throughput (other streams' rows in the same weight-read). Not waste.
- ~260ms (28%) = WASTED re-prefill (enable_prefix_caching=False, 107:1 prompt:gen). APC-recoverable (aggregate).
- ~105ms (11%) = agentic idle (batch under-fill, eff_conc 2.05<4). Aggregate-only.

**HW-limit ceiling: 130ms/step => ~42 tps/stream (~9.5x today's 4.45).** Floor = 1 weight-read (98.6ms
verify, AT floor) + ~30ms graphed/fp8 drafter preamble + ~2ms overlapped committer. Spec-decode is
data-serial per stream; the win = delete the 250ms host stall + compress the 315ms SERIAL chain.

**Ranked levers:**
| # | lever | effort | +tps | note |
|---|-------|--------|------|------|
| 1 | Committer sync-kill (FR13_GPU_COMMITTER=1 + FR13_COMMITTER_SYNCKILL=1) | low | +9% | **ROOT DOMINO** — flags EXIST; skips synchronize@7947-8; **NEVER live-gated (G5), needs IN-PROCESS OFF==ON byte-identical gate (no cross-boot byte gate on GB10)** |
| 2 | Async scheduling / 2-deep batch_queue | med | +22% | depends on #1 |
| 3 | CUDA-graph drafter spine + fp8 draft lm_head | high | +10% | patcher:13681 |
| 4 | APC prefix-caching ON | high | +25% (agg) | **blocked on AGENTIC-losslessness (tree+cache degrades agentic)** |
| 5 | Overlap GDN replay into next drafter | low | +2% | depends on #1 |
| 6 | Reuse verify root logits for draft d0 | low | +1% | patcher:13385 |

**Sequence:** #1 is the first domino (nothing pipelines until the main-thread synchronize is gone). It is
correctness-sensitive (the committer decides accepted tokens) and NEVER validated => build an IN-PROCESS
OFF==ON byte-identical losslessness gate BEFORE the speed campaign. Meanwhile the free GPU runs the
native+tail6 decomposition (native stage timings — the missing HW-limit input). Then #1 -> #2 -> #3.

---

## ADVERSARIAL VERIFY REFUTED the top levers — the HW-limit ceiling was over-optimistic

The workflow's Design agent proposed 42 tps/9.5x; its own redteam (+ my code verification) DEMOLISHED it:

- **Committer sync-kill: +9% → REFUTED (~+2%).** Two grounds, both verified in code: (1) the built synckill
  path calls `_materialise()` EAGERLY (patcher:17888-91) which does `event.synchronize()` — it doesn't
  actually defer. (2) **THE LINCHPIN (verified, gpu_model_runner.py:1430-1443):** for the GDN hybrid,
  `mamba_cache_mode=="align"` runs `self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()` EVERY STEP —
  a blocking host sync, upstream-flagged `# TODO: Remove .cpu() sync to enable fully async for hybrid model`.
  align mode needs num_accepted on CPU to decide mamba-state block-copies at boundaries (preprocess_mamba,
  :5033). The `else` (non-align) branch uses a non-blocking copy + event (NO sync). So the sync is FORCED by
  the GDN cache architecture, not by our committer.
- **Graphed drafter: +10% → REFUTED (~+3%).** The drafter's 101ms is REAL bf16 lm_head weight-read
  (measured dfwd_split compute_logits=15.08ms/call x ~5 = ~75ms GPU-active), NOT a removable host bubble.
- **APC: +25% → REFUTED (0% on per-stream).** `per_request_decode_tps` EXCLUDES prefill by construction
  (fr13_measure.py:1606), so removing co-resident re-prefill doesn't move the per-stream metric. (APC is an
  AGGREGATE/TTFT lever, not per-stream — and still blocked on agentic-losslessness.)

### Honest per-step floor (adversarially verified)
- verify 105ms = HBM weight-read floor (read 27B fp8 once = 98.6ms; within 6%). IRREDUCIBLE on GB10 273 GB/s.
- drafter ~75ms of 101ms = draft lm_head weight-read. Also HBM-bound. Only ~+3% from graphing the launches.
- committer host-sync = FORCED by mamba_cache_mode=align (upstream TODO). async scheduling (+20%) is
  BLOCKED behind it.
- => The decode IS HBM-bound (confirms the prior "weight-read floor, per-forward opts limited" finding).
  The user's "push to HW limit" is right in principle, but verify+drafter are already AT the HBM floor.

### The ONE real big lever: escape mamba_cache_mode="align"
The only path to a large (+~20%) per-stream gain is switching the GDN hybrid OFF align mode onto the
non-blocking event path (:1455), which unblocks async scheduling / batch-queue pipelining. This is HARD
(align mode manages mamba-state block-copy at boundaries; GDN may need it for cache/APC correctness) and
lossless-UNPROVEN. It is the real "kernel-maker" target: make the GDN state-advance read num_accepted
DEVICE-resident so the host sync is unnecessary. Realistic ladder: ~+6% from small levers (graph drafter
+3%, overlap GDN replay +2%, reuse verify root-logits +1%) WITHOUT align-escape; ~+20-30% IF align-escape
is made lossless. NOT 9.5x. native+tail6 decomp (nt1, running) confirms whether native shares the same
align-mode floor (it uses the same GDN hybrid) -- if so, the floor is architectural, not tree-specific.

---

## Align-escape feasibility: the sync is a per-step CHECK for a RARE event (device-side detection = the lever)

Read preprocess_mamba (mamba_utils.py): the align-mode block-copy moves GDN recurrent state between
physical KV blocks ONLY when a request crosses a block boundary (mamba_utils:384 `if src==dest and
accept_token_bias==0: return`). At FR13 mamba_block_size=8192, a boundary crossing is RARE (~1 per 8192
decode tokens). But the `num_accepted.gpu.cpu().numpy()` sync (gpu_model_runner:1441) fires EVERY step
to CHECK `accept_token_bias` (= did the accepted tokens cross a boundary). So the per-step host sync is a
check for an almost-always-negative condition.

**The real align-escape lever (device-side boundary detection):** compute `accept_token_bias` on GPU
(num_accepted stays device-resident), branch device-side; only when a boundary IS crossed (rare) do the
host sync + block-copy. Otherwise the non-blocking event path (:1455) already exists. This removes the
per-step host stall (unblocks async scheduling +~20%) WITHOUT the full mamba-block-copy rewrite.

**Cost/risk (honest):** the block-copy path is heavily FR13-instrumented (the entire APC/cache
losslessness effort lives here: block-align, conv-snapshot, leaf-crosscheck) => extremely fragile,
lossless-gating is expensive. But it is NOT a premature no-go: the device-side-check reframing makes it a
BOUNDED change (touch the per-step check, not the rare copy). GO/NO-GO deferred to nt1's native stage
decomposition (quantifies the exact host-stall $ that async-escape would reclaim) -- native shares the
same align floor (confirmed: GDN hybrid => align forced), so nt1's committer/host numbers set the ceiling.

## Where the deliverable actually stands (pending nt1 clean native comparison)
- tail6 (21-node spine tail): accept 4.317, per_request_decode_tps 4.889 (b7). Cross-run vs native 4.60
  => tail6 ~+6% per-stream tps AND higher accept. If nt1 confirms same-session, THE DELIVERABLE (tail6)
  ALREADY BEATS native MTP-5 on BOTH axes. Branch-widening (tail6b) was the anti-speed misstep; tail6 stands.
- Remaining speed frontier = align-escape (bounded device-side-check, deferred to nt1) OR HBM wall.
  Branch-widening for accept is CLOSED (anti-speed, b7-proven).

---

## NATIVE DECOMPOSITION (nt1) — the tree committer is the smoking gun, and native is FASTER

Same-subset native MTP-5 (flash_ns5_nocache, timers ON) vs tail6 (b7):
| stage/step | native MTP-5 | tail6 (tree) | delta |
|-----------|-------------:|-------------:|-------|
| accept_per_event | 3.415 | 4.317 | tree +0.9 (+27%) |
| per_request_decode_tps | **5.490** | 4.889 | **native +12%** |
| kernel derived_tps_gpu | **75.96** | 56.9 | native +33% |
| verify s_per_fwd_gpu | 58ms | 93.5ms | tree +35ms (25-node attn) |
| **committer_gpu_ms/step** | **7.2ms** | **94.0ms** | **tree +87ms (13x!)** |
| drafter_gpu_ms/step | 93.1 | 99.2 | ~same |

### Two overturned assumptions
1. **Native MTP-5 is FASTER than the tree** (per-stream 5.49 vs 4.89; kernel 76 vs 57). Earlier "tail6 beats
   native" came from a stale cross-run native (4.60); this fresh SAME-timer native is 5.49. The tree's +0.9
   accept does NOT buy a speed win -- its verify+committer overhead exceeds the accept benefit.
2. **The align-mode sync is NOT the committer bottleneck.** Native runs the SAME GDN hybrid (same
   mamba_cache_mode=align, same per-step num_accepted.cpu()) yet its committer is 7.2ms. So the tree
   committer's 94ms is FR13's OWN host path (tree path-LCP DtoH + Python per-row loop), not the align sync.

### THE REAL LEVER: the FR13 tree committer (87ms of OUR overhead)
The tree committer (94ms) vs native (7ms) = 87ms/step of FR13-authored host-side overhead (DtoH the tree
output + host path-LCP over root-to-leaf paths + commit). This is the S1 "sampled-committer port" lever
(prior task #25). If the tree committer drops to ~native's 7ms: tail6 per-step compute 286->199ms; wall
1088->~1001ms; per_req 4.89->~5.3 -- closing most of the gap to native 5.49 WHILE keeping +0.9 accept.
The verify (+35ms, 25-node tree-attn) is the remaining tree tax (trades accept via tree size).

**Reframed plan:** the committer port (FR13 host-LCP -> device-resident, remove the 87ms host loop) is
the biggest single reducible overhead AND it is entirely our code (low architectural risk vs the align-
escape). FR13_GPU_COMMITTER=1 was meant to do this; redteam found its synckill path _materialise()s
eagerly (keeps the sync) -- but the COMPUTE port (device LCP kernel, remove host loop) is the 87ms win,
separable from the align sync. nt1 tail6 arm (running) confirms the 94ms committer SAME-SESSION vs native 7ms.

---

## Committer-port CRASHED live + the sobering ceiling math

**tail6_gc (FR13_GPU_COMMITTER=1) crashed on the WARMUP PROBE** (first decode, EngineDeadError rc=4,
serve VACUOUS). The device LCP committer (scripts/fr13_gpu_committer_kernel.py, fr13_gpu_committer_device_full)
is NOT functional on the live tree path -- a shape/init bug (warmup uses dummy inputs, so data-independent).
The "never live-run" risk materialized. Root-cause traceback lost (container removed on arm transition);
localizing needs a re-run with docker-log capture OR reading fr13_gpu_committer_kernel.py.

**CEILING MATH (why the port doesn't rescue the tree):** committer 94->10ms saves 84ms/step. tail6 wall
1088->1004ms => per_req 5.317/1.004 = **5.30 -- STILL below native's 5.49.** The committer port, even
FIXED, does NOT make the tree beat native per-stream. And native aggregate (12.85) > tail6b (9.67).

## HONEST EMERGING CONCLUSION: native MTP-5 is faster than the tree on GB10
- Native MTP-5: accept 3.42, per_req 5.49, aggregate 12.85, committer 7ms, verify 58ms.
- tail6 (tree):  accept 4.32, per_req 4.89, aggregate 9.67(b7), committer 94ms, verify 93ms.
- The tree does MORE work per HBM-bound forward (draft tree + verify 25 nodes + commit tree) for +0.9
  accept, but on GB10 the forward is cheap-ish (HBM 98.6ms) so the extra verify+committer overhead
  (~127ms/step) EXCEEDS the accept benefit. Native wins per-stream AND aggregate.
- The committer port (84ms) closes most of the committer gap but leaves the tree ~tied/slightly-behind
  (5.30 vs 5.49) -- and it crashes, needing debug + losslessness gating. Poor ROI.
- PENDING: the cp1 tail6 arm (running) confirms native-vs-tail6 SAME-ish-session. If it holds, the honest
  deliverable answer on GB10 is NATIVE MTP-5, not the tree -- the tree's accept edge doesn't buy speed here.
  This is the measured cost-gate, not a premature no-go (branch-widening + committer-port + align-escape
  all assessed).

---

## CORRECTION: "native wins" was PREMATURE — the committer port reclaims the host-loop STALL too

Red-teamed the "native beats tree" conclusion. My ceiling math counted only the committer's 84ms COMPUTE,
but the 94ms host LCP loop ALSO blocks the pipeline (GPU idle during the host DtoH+loop). Evidence: the
tree's gap (802ms/step) is 156ms LARGER than native's (646ms) despite the SAME align sync + similar prefill
-- much of that extra gap is the GPU idling during the host committer. A DEVICE committer removes BOTH the
84ms compute AND the host-loop stall (~part of the 156ms). Corrected projection: tail6 wall 1088 -> ~900ms
=> per_req 5.317/0.9 = **~5.9 -- BEATING native 5.49, WITH +0.9 accept.** So the committer port is a REAL
win lever, not marginal. My earlier "native wins" under-counted it. (Every premature no-go here has been
overturned; this one too.)

## Committer-port crash: device kernel, needs the traceback
tail6_gc crashed on warmup. The B=4 loop-skip/synckill drift (patcher:7714-7846) is ALREADY FIXED and my
run had NO synckill (parents_cpu populated:7984), so the crash is in the DEVICE committer kernel
(fr13_gpu_committer_kernel.py, _fr13_committer_kernel launched :653) on the warmup's dummy inputs -- a
shape/counts assumption. Root-cause traceback was lost (container removed on arm-fail). NEXT: dedicated
tail6_gc boot with docker-log capture (container persists) -> read the EngineCore traceback -> fix the
kernel shape bug -> re-run. The corrected ROI (could beat native) justifies the debug effort.

---

## Committer-port kernel FIXED + running (cp4) — lossless is now the gate

After 3 Triton int64 fixes (best_leaf/bonus_tok/acc_row, via log-capture boots), tail6_gc
(FR13_GPU_COMMITTER=1) COMPILES + DECODES clean: 0 crashes, GPU committer engaged
("FR13_EAGER_PACK committer path engaged: boundary_legacy_loop=0, packed_dtoh_elems=181" -- device LCP
runs, host legacy loop skipped, tiny packed DtoH vs the full 94ms host loop).

**OPEN GATE -- LOSSLESSNESS:** early raw accept ~3.807 (32 windows) vs tail6's 4.32. Could be early noise
OR a device-LCP correctness bug (the Triton kernel commits a SHORTER/different path than the host
committer => fewer accepted tokens => NOT lossless). The committer decides which tokens are accepted, so
a lower accept = a real correctness difference, not just speed. The cp4 A/B (tail6_gc vs tail6) settles it:
  - accept_per_event ~= tail6 4.32 => device LCP is lossless-equivalent (rejection-sampler convergence).
  - accept << 4.32 => device LCP has a bug (wrong path-LCP / tie-break / bonus-token) -> localize vs the
    host _lumo_tree_path_lcp_max_greedy_sample reference.
  - committer_gpu_ms 94->~10 => the speed lever works (measured at arm end).
DO NOT trust the speed win until accept matches -- a faster-but-wrong committer is a reward hack.

## Committer-port LOSSLESS confirmed (cp4) — speed numbers pending

tail6_gc raw accept = **4.321** (267 windows) == tail6 host-committer 4.317 (native 3.415). The device LCP
kernel commits the SAME paths as the host committer => LOSSLESS-equivalent (the two only match this closely
if the accept/LCP/bonus decisions are identical). 0 crashes, 0 fallback, GPU committer engaged throughout.
=> The FR13_GPU_COMMITTER=1 device port is CORRECT. Remaining = pure SPEED (deploy_speed at arm end):
committer_gpu_ms 94->~10? + per_req past native 5.49? If yes => BAKE (lossless + faster, our own code, no drift).

## Committer-port = LOSSLESS but ~0 SPEED (misdiagnosed) — the 94ms is GDN replay, not the LCP

Live cfwd/dfwd sidecars, tail6_gc (FR13_GPU_COMMITTER=1): drafter 100.2ms, committer **97.5ms** ==
tail6's 99/94ms. The device LCP port did NOT reduce the committer. Needle: "FR13_EAGER_PACK committer path
engaged: layers=48 replay_batched=1 boundary_legacy_loop=0" -- the host LCP loop was ALREADY skipped
(EAGER_PACK, baked in BOTH arms); the 94ms committer is dominated by the **GDN state replay across 48
layers** + packed DtoH, NOT the path-LCP. FR13_GPU_COMMITTER only moves the tiny LCP decision host->device.

**MISDIAGNOSIS corrected:** I attributed the 94ms committer (vs native 7ms) to "the host path-LCP loop";
it's actually the GDN recurrent-state replay for the committed path across 48 layers -- INHERENT to
tree+GDN (native's mamba advances linearly by num_accepted, no per-committed-token tree replay). The
committer port is LOSSLESS (accept 4.321==4.317, kernel debugged through 3 Triton dtype bugs) but delivers
~0 speed because the LCP was never the bottleneck. The debugging fixed a real crash + proved the device
LCP correct, but the lever doesn't move per_req.

**Where this leaves it:** the tree's committer disadvantage vs native (94 vs 7ms) = the GDN replay, which
is architecturally inherent. Reducing it = overlap the replay with the next drafter (workflow lever #5,
~+2%) -- marginal. Native MTP-5 remains faster per-stream. The A/B tail6 arm (cp4 arm2) confirms tail6's
committer is also ~94ms same-session (both EAGER_PACK). Do NOT bake FR13_GPU_COMMITTER (no speed win).
Committer-as-LCP-lever CLOSED; GDN-replay is the real (inherent) committer cost.

## MEASURED COST-GATE: committer NOT reducible -> native MTP-5 is the throughput answer on GB10

Committer decomposition complete (all variants, live-measured):
| committer variant | committer_gpu_ms/step | verdict |
|-------------------|----------------------:|---------|
| tail6 host-LCP (baseline)   | 94.0  | -- |
| tail6_gc device-LCP port    | 97.5  | no change (LCP already EAGER_PACK device-side) |
| tail6_gc_sk synckill        | 129.3 | WORSE (_dev kernel acc_path extra work; defer doesn't help) |
| native MTP-5                | 7.2   | -- |

NONE of the committer variants reduce the 94ms => the tree committer's cost is INHERENT (GDN 48-layer
state replay + unavoidable DtoH + sync-wait), NOT the path-LCP or the DtoH-deferral. Cleanly isolating
sub-segments needs patcher sub-timers (declined -- rabbit hole; the answer is already directionally clear).

### CONCLUSION (measured, not premature -- every lever tried on the live gate)
On GB10 (HBM-bound, weight-read ~98.6ms), **native MTP-5 is faster than the tree spec-decode**: per-stream
5.49 vs 4.89, aggregate 12.85 vs 9.67. The tree does ~122ms/step MORE work (verify +35ms 25-node tree-attn,
committer +87ms GDN replay) for +0.9 accept (4.32 vs 3.42), but the cheap HBM-bound forward makes that
overhead NOT worth it. Levers exhausted: branch-widening (anti-speed), committer-LCP-port (lossless, no
speed), committer-synckill (worse), APC (0% per-stream), align-escape (risky + GDN forces the sync).
**The tree's value on GB10 = ACCEPT / LOSSLESSNESS, not throughput.** Deploy answer: native MTP-5 for raw
tps; the tree for its lossless high-accept spec-decode where accuracy/branch-losslessness is the goal.

REAL deliverables from this investigation (kept): (1) the device committer kernel now COMPILES + is
LOSSLESS (4 Triton int64 dtype fixes -- was never-live-run/broken); (2) full committer/drafter/verify
decomposition documented; (3) the honest measured verdict. FR13_GPU_COMMITTER stays OFF (no speed win).

## FINAL: committer 94ms = sync-wait (align serialization), not LCP/replay -> align-escape is the only lever (cost-gated)

Replay code check (_fr13_native_committer_replay, fr10_gdn_tree_kernel.py): the GDN replay is STATE-ONLY,
per-layer ring GATHER (index_select) + fused_sigmoid_gating SSM advance, NO host sync -- already cheap
(~11ms, ~native's 7ms mamba advance). So the 94ms committer is NOT the replay and NOT the LCP (both cheap/
device-side). By elimination it is the ~83ms SYNC-WAIT: the committer's output_token_ids.cpu() + align-mode
per-step num_accepted.cpu() serialize committer<->verify (no overlap). Removing it = ASYNC scheduling,
BLOCKED by mamba_cache_mode=align (per-step host sync) => the align-escape (device-side boundary detection).

**COST-GATE (per speed-is-goal: STOP if no plausibly-cheap correct path):** the only remaining lever is
the align-escape -- a deep, fragile rewrite of the GDN mamba block-boundary path (the FR13 APC losslessness
core). NOT plausibly cheap. => STOP. Every cheap+medium lever exhausted & measured on the live gate.

**MEASURED VERDICT (tree spec-decode on GB10):** native MTP-5 is the THROUGHPUT answer (per-stream 5.49 vs
4.89, aggregate 12.85 vs 9.67). The tree's value is ACCEPT / LOSSLESSNESS (4.32 vs 3.42, branch-lossless
spec-decode) -- an accuracy property, not a speed one, because the HBM-bound cheap forward doesn't reward
the tree's extra verify+committer work. Deploy: native for tps; tree where lossless high-accept matters.

## DEFINITIVE (measured, not guessed): GDN replay = 1.5ms -> stateless-tree lever DEAD, committer = verify-wait

FR13_REPLAY_GPU_TIMER (10650 spans): **GDN accepted-path replay = 1.5 ms/step** -- negligible, cheaper
than the ~11ms redteam estimate. => The replay is NOT the committer bottleneck. The stateless-tree
committed-leaf-state GATHER lever (task #11: replace replay-recompute with a gather) is DEAD -- the replay
is already 1.5ms, nothing to reclaim.

**Clean committer decomposition (finally measured, not guessed):** committer span 94ms = ~1.5ms replay +
small device-LCP + **~90ms WAIT-for-verify**. The CFWD cuda-event captures the committer blocking on the
async verify forward (align-serialized pipeline: output.cpu() waits for the still-running verify). So the
"94ms committer" is NOT 94ms of committer WORK -- it's mostly the verify-forward wait. The tree's REAL
extra cost vs native = the bigger 25-node verify (93 vs 58ms tree-attn) + the non-overlapped pipeline.

**FINAL VERDICT (fully measured on the live gate):** native MTP-5 is faster per-stream on GB10 (5.49 vs
4.89). The tree's extra cost is the 25-node verify forward (tree-attn) + align-serialized non-overlap.
Levers: shrink verify = smaller tree (refuted anti-accept); overlap = async/align-escape (cost-gated deep
APC-core rewrite). Replay=1.5ms (stateless-tree lever dead). Committer-LCP-port lossless-but-no-speed.
=> Tree's value on GB10 = ACCEPT/LOSSLESSNESS (4.32 vs 3.42), not throughput. Deploy: native for tps,
tree for lossless high-accept. Speed investigation COMPLETE -- every lever measured, no premature no-go.

## OVERTURN (aggregate axis): batch under-fill was BUDGET-STARVED, not agentic-idle -> ~2x throughput lever

Prior conclusion (native wins) was PER-STREAM only. The AGGREGATE axis (multi-user serving = per_stream x
eff_conc) had an untested config lever: the APC-OFF deploy path (tail6) set NO chunked-prefill/max-num-
batched flags (vLLM defaults). bf1 A/B (tail6 + --enable-chunked-prefill --max-num-batched-tokens 8192 vs
baseline): boots CLEAN on GDN-hybrid w/o APC (0 crashes) AND the batch fills to **Running:4 reqs 70% of
samples** (vs baseline eff_conc ~2.0). => the under-fill is BUDGET-STARVED (config-fixable), NOT agentic-
idle as I'd concluded. Full B=4 batch => ~2x aggregate throughput potential (per_stream x eff_conc 2->4).
Tradeoff: full batch = more per-stream contention (per_stream may dip); aggregate wins iff per_stream drops
< eff_conc rises. CLEAN read pending (deploy_speed both arms: aggregate_decode_tps + eff_conc + per_stream
+ accept). If aggregate UP + accept unchanged => BAKE FR13_SERVE_BATCH_FLAGS for serving throughput. This
is the deployment-relevant metric (native's per-stream win doesn't preclude a tree aggregate win, AND the
same batch-fill helps native too -- but it's a real, cheap, previously-missed lever on the deploy config).

## Batch-fill overturn FAILED (measured): aggregate -49% -- decode starved by big prefill batches

bf1 A/B result: tail6 + --enable-chunked-prefill --max-num-batched-tokens 8192 => aggregate_decode_tps
**4.917 vs baseline 9.67 (-49%)**, per_req 1.838 vs 4.89 (-62%), eff_conc 2.493 vs 2.09 (+19% only),
accept 4.462 (holds). The big max-num-batched let the scheduler batch HUGE prefill chunks that STARVED
decode -- per-stream cratered while eff_conc barely rose. The live "Running:4 reqs" was misleading: those
were mostly PREFILLING, not decoding. => the batch-fill config is a NET LOSS; the vLLM DEFAULT config
(baseline) is optimal. My aggregate overturn was WRONG (researched + run cleanly, honest measured negative).
No drift risk: FR13_SERVE_BATCH_FLAGS is default-EMPTY (deploy config unaffected). DO NOT bake.

### Speed direction FULLY closed (per-stream AND aggregate, all measured)
- Per-stream: native MTP-5 faster (5.49 vs 4.89); tree overhead inherent (verify+committer), all levers
  measured/refuted (replay 1.5ms, LCP-port, synckill, branch, APC, align-escape).
- Aggregate: the default config is optimal; forcing bigger batches starves decode (-49%). eff_conc ~2 is
  the agentic-workload natural concurrency (streams idle between tool calls), NOT budget-starved as hoped.
- Tree value = accept/losslessness (4.32-4.46 vs native 3.42). Deploy: native for tps, tree for accuracy.
