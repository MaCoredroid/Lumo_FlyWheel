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
