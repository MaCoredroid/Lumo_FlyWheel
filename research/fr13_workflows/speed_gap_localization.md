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
