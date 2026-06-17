# FR13 Stage D: cat6-vs-E5 +28ms speed-gap LOCALIZATION (workflow wottfrivv, 2026-06-17)

User reframe: cat6 commits +17% (4.82 vs 4.11) but deploy TPS only +4% (18.51 vs 17.8). User insisted the
slow part is OUR code / what differs from E5, NOT s/fwd. Workflow read cat6-vs-E5 per-committed-token diffs.

## VERDICT: the +28ms is CONSUMER-PACED IDLE, NOT our-code compute.
- OUR-CODE compute is sub-1% (MEASURED py-spy device profile): device committer 0.8%, GDN-replay 0.08%,
  detok 0.0018ms/tok. The greedy-LCP + device committer are both per-step-FIXED (no per-accepted-token
  Python/sync bubble in the engaged path). GDN-replay = static_range unroll, acc-invariant grid. None scale
  per-committed-token. Combined per-step-fixed delta sub-ms (~3% of the 28.5ms).
- The +28ms = the NON-forward per-step IDLE absorbed into request_decode_time (last_token_ts - first_token_ts,
  fr13_measure.py L1612, idle-inclusive). Cross-check: committed/s_fwd = forward-basis TPS 30.0(E5)/34.9(cat6),
  but realized per_request_decode_tps = 17.8/18.51 = ~1.8x LOWER -> ~93-122ms/step is non-forward idle (the
  B=1 agent-loop inter-step gap / consumer pacing), dwarfing the ~138ms forward AND the sub-ms our-code.
- cat6 produces +0.71 tok/step into that idle-dominated span -> lands +4% not the +17% its tokens would give
  if the span were forward-bound. The GPU IS ~+17% faster (more committed per equal forward); the DEPLOY just
  can't realize it (idle/consumer-bound).

## CORRECTION to the workflow (it erred, conclusion survives):
Workflow assumed deploy temp=0.0 (greedy LCP committer) from serve_variant L280's LUMO_PROXY_FORCE_TEMPERATURE=0.0.
WRONG: that's the LOCAL proxy; the OFFLOAD proxy (codex's path) forced 0.6 (offload_proxy_env.txt). DECISIVE
deploy log: "FR13_DEVICE_MULTIDRAFT engaged: device-side temp>0 multidraft committer (NO per-node Python loop),
n_req=1". So deploy = temp-0.6, DEVICE committer (my bake aa39fe07 APPLIES), no per-node .item() walk -> the
device committer is the FAST path (~0.8%), strengthening "our-code is small". (12.9% flip-rate also = sampled
not greedy.) The workflow's analysis-5 "12ms/tok per-node walk" = the host-ref path, which NEVER ran.

## LEVER: NOT kernel/our-code (~0 gain, sub-ms of a 230-260ms step). The idle is the deployment lever:
- B>1 concurrency (overlap one stream's idle with another's forward) — lossless risk NONE (scheduling).
- accept (cat6 already +17% committed; the win is real on-GPU, masked by idle).
- (Future temp>0 with a SLOW committer: FR13_GPU_COMMITTER+SYNCKILL — but the current device path already has
  no per-node loop, so N/A.)

## CONFIRM (running): stream=FALSE realized TPS (no consumer backpressure) + s_per_fwd_gpu (idle-independent
GPU forward) for cat6 vs E5. PREDICTION: stream=false realized widens toward +17% (cat6's true token advantage),
+ s_per_fwd_gpu ~equal -> proves the deploy +4% is the consumer/idle span. scripts/fr13_synth_realized_tps.sh.
The prior stream=TRUE probe was INVALID (slow python SSE client backpressured + inflated rdt to 0.226).

## FINAL VERDICT (cat6 vs E5 stream=false synthetic + diff, 2026-06-17)
Measured (scripts/fr13_synth_realized_tps.sh, stream=false, generic CPU-explanation prompt, B=1):
| metric | cat6 | E5 |
|---|---|---|
| s_per_fwd_gpu (pure verify forward, idle-excl) | 0.1196 | 0.1177 |
| realized_decode_tps | 14.02 | 15.12 |
| accept/event | 2.208 | 2.333 |
| committed/step | 3.200 | 3.333 |
| non-forward gap (wall/step - s_per_fwd_gpu) | 0.1086 | 0.1027 |

1. VERIFY FORWARD EQUAL: s_per_fwd_gpu cat6 0.1196 ~= E5 0.1177 (~1.6% tree tax; deploy was 0.7%). Not the diff.
2. OUR-CODE per-step delta ~6ms (cat6 non-forward gap +0.0059 over E5 = tree drafter top-k + device committer
   vs E5 chain). Tiny, NOT the +28ms.
3. DEPLOY +28ms is CONSUMER-DOMINATED: synthetic (no consumer) reproduces only ~6ms; the other ~22ms is the
   deploy consumer/agent-loop (codex side, outside vLLM rdt). Kernel/our-code is NOT the deploy bottleneck.
4. ACCEPT IS WORKLOAD-DEPENDENT (new): generic synthetic text -> E5 accepts MORE (2.333 vs 2.208) -> E5 FASTER
   (15.12 vs 14.02). codex SWE deploy -> cat6 accepts more (3.82 vs 3.11) -> cat6 faster (18.51 vs 17.8). cat6's
   win is SPECIFIC to the structured-code (codex SWE) workload its tree was tuned for, NOT universal.

VERDICT: cat6 WINS on the DEPLOYED workload (codex SWE: +4% TPS + lossless, the relevant result). The kernel is
sound (forward equal, our-code ~6ms). The deploy +28ms is consumer/agent-loop-bound -> kernel/committer tuning
gives ~0 deploy-TPS. Further speed = B>1 concurrency (overlap the consumer idle) or a better-accept shape; NOT a
kernel lever. The user's "more accept should be way faster" is right ON-GPU but the deploy is consumer-paced.
