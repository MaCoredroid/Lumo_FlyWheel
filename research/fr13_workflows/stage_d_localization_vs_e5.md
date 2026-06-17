# FR13 Stage D: localize the +28ms cat6-vs-E5 (workflow wwtr26lo9, 2026-06-17)

Supersedes the REFUTED `stage_d_overhead_is_stock_depthbound.md`. Research+adversarial-verify pass
(7 agents: read OUR patched path + E5 baseline + online SOTA) per the user's "measure which part differs
before tuning / don't call it not-improvable without research" directive.

## CONFIRMED (code-read or directly measured) — the NEGATIVES
- **DRAFTER ≈ E5** (code, not inferred): cat6root -> `FR10_CATERPILLAR_NATIVE_SPINE_TOP2` branch
  (fr10_phase4_patch L11402-11410), NOT `self.propose_tree`. Spine loop L11774 `for token_index in range(4)`
  = 4 SINGLE-ROW MTP forwards (== E5 stock chain propose, container eagle.py L626-639). cat6 has
  `_fr10_leaf_steps=frozenset()` -> ZERO interior-leaf topk. Net extra vs E5 = ONE `topk(k=2, V=151936)` at
  root + a `torch.stack` of 6 ints, all on GPU, <0.1ms/step. **The user's model was right.**
- **VERIFY forward EQUAL** (measured): s/fwd_gpu cat6 0.13798 vs native 0.13704 (+0.94ms, ~3%). The forked-FA2
  tree-bias reads a STATIC 7x7 tree_attn_bias (built once in __init__) inside the kernel -> already in this number.
- **COMMITTER COMPUTE ≈ E5** (measured): device multidraft commit 0.34-0.35ms isolated, ~= E5 stock
  rejection_sample 0.35ms. **But isolated/GPU-only — does NOT capture per-step .item()/DtoH sync stalls.**

## THE +28ms IS NOT MEASURED (critical caveat)
Derived from wall=committed/deploy-TPS (overhead_profiling_future_work.raw.json): basis-dependent (native
0.082 wall-span vs 0.094 deploy-TPS), MID-ARM (cat6 2/4), LUMPED. CONFOUND: cat6 ran at prefill_frac 0.98 vs
native 0.65, and cat6 is FASTER per-stream (18.51 vs 17.80 TPS). The longer step is largely the cost of
committing MORE tokens (4.82 vs 4.11), not pure waste. => MUST be diffed on a MATCHED-prefill clean B=1
raw-prompt boot with a DIRECTLY-measured step wall, or part of it is a confound (H4).

## RANKED HYPOTHESES (mechanism-grounded, UNMEASURED — the boot-profile resolves them)
- **H1 (most likely, ~10-20ms): lost pipeline overlap from committer + GDN-replay per-step .item()/DtoH syncs.**
  Device committer per-node `multinomial().item()` (kernel L237/245/253/268/350/375) + per-req seed .item() +
  per-node `torch.tensor` HtoD (L386) each BLOCK the host on the GPU queue, draining run-ahead so the next
  step's drafter forward launch is delayed. At B=1 single-stream there is NO second request to fill the bubble
  -> every sync becomes wall. The 0.35ms isolated microbench CANNOT see this. Only mechanism consistent with
  ALL measured facts (verify=, committer-compute=, drafter=, yet wall +28ms).
- **H2 (likely, ~5-10ms): GDN durable-state REPLAY-COMMIT host work (FR13_REPLAY_ROUTE, baked `if True:`
  L7739/L7770).** cat6-EXCLUSIVE postprocess: builds accepted_gdn_node_paths host lists (L7633-7660), publishes
  per-GDN-layer device buffers (L7693-7717), builds _LUMO_FA_TREE_ACCEPT_BY_REQ dict (L7765-7769), replays
  accepted paths on EVERY registered GDN layer (~48) every step. E5 = one in-place no-copy recurrent commit.
  **NEW suspect — no prior analysis mapped it. Largest un-bracketed cat6-exclusive host span.**
- **H3 (possible, 3-8ms): host-path committer Python walk** — ONLY if a boot ran FR13_DEVICE_MULTIDRAFT=0;
  the b1 verdict says DEVICE path was engaged -> likely N/A. Confirm the boot's flag state.
- **H4 (the +28ms is partly ARTIFACT)**: basis/prefill confound (see above). Could shrink the real target by up to ~10ms.
- **H5 (REFUTED): propose_tree growing-row forwards** — cat6 doesn't take that path. The old doc is wrong.

## INSTRUMENTATION PLAN (FR13_STEP_PHASE_TIMER, default-OFF, clone the FR13_SFWD_GPU_TIMER machinery)
Run cat6 AND E5 at MATCHED prefill_frac / same B=1 raw-prompt seed; RECORD FR13_DEVICE_MULTIDRAFT state.
- **STEP-WALL** (the canonical boundary; without it the +28ms stays derived): container gpu_model_runner.py
  execute_model ENTRY (~L3771) -> sample_tokens RETURN (~L4330), host perf_counter, NO per-step sync, gated on
  pure-decode predicate. BOTH. Must reproduce ~0.260/0.231.
- **t_forward (verify)**: reuse FR13_SFWD_GPU_TIMER (async cuda-events). BOTH. (already 0.138/0.137)
- **t_verify_metadata_build**: gpu_model_runner.py L2260 `builder.build(...)`, perf_counter, key by
  TreeAttentionMetadataBuilder vs FlashAttentionMetadataBuilder. BOTH. (tests the weakest "equal" claim)
- **t_committer**: bracket `_lumo_tree_canonical_multidraft_sample(...)` (patch L9306) with
  `cuda.synchronize();t0; call; cuda.synchronize();t1` (syncs REQUIRED — we WANT to charge the stalls). E5:
  identical bracket on `rejection_sample(...)` (container rejection_sampler.py forward L141). BOTH. **H1 test.**
- **t_gdn_replay_commit**: bracket FR13_REPLAY_ROUTE (patch L7629 -> past L7790) with cuda.sync. OURS ONLY. **H2 test.**
- **t_postprocess**: around the `record_function('postprocess')` block. BOTH.
- **RESIDUAL** = step_wall - sum(phases). Should be small + cancel in the diff.

## TUNABLE LEVERS (CONTINGENT on the measurement confirming the localization)
- **BATCH the FR13_REPLAY_ROUTE GDN publish** (the H2 lever): replace per-layer host list-comp + per-layer small
  HtoDs (L7633-7717) + the dict build with ONE stacked HtoD consumed by all GDN layers (EAGER_PACK already does
  this for the b1 publish L7662-7702 -> extend to the whole replay). EFFORT medium, **LOSSLESS RISK LOW (pure
  host-transport, no numeric effect)**, payoff = H2 share. CLEANEST lever.
- **KILL committer per-node .item()/DtoH syncs** (the H1 lever): rewrite fr13_device_multidraft_commit as ONE
  fused ancestry-aware tree-commit Triton kernel (collapse per-node multinomial+seed .item() + HtoD to a single
  end-of-step DtoH). EFFORT high, LOSSLESS RISK MEDIUM (distribution-lossless not byte-identical -> MUST stay
  behind the per-token argmax gate fr13_gold_margin_probe). Highest payoff IF H1 confirmed.
- static-shape metadata cache (low payoff, tree build already <= E5), cudagraph drafter+verify (low-med, B=1
  launch latency), fuse host-path softmax (N/A, device path engaged).

CONFIDENCE: medium-high on the NEGATIVES (drafter/verify/committer-compute all = E5); LOW on positive
attribution (derived+lumped+confounded). NEXT ACTION = MEASURE (FR13_STEP_PHASE_TIMER boot-profile), not tune.
