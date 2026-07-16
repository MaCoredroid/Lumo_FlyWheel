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
