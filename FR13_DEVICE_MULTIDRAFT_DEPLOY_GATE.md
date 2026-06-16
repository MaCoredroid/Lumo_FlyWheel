# FR13 DEVICE-MULTIDRAFT DEPLOYMENT SPEED GATE (SpeedPair)

Canonical PAIRED deployment-speed artifact for the temp-0.6 device committer
`FR13_DEVICE_MULTIDRAFT` (scripts/fr13_device_multidraft_kernel.py), flag-gated
default-OFF. The device path computes the IDENTICAL SpecInfer/multi-draft
residual-mix accept distribution ON-DEVICE (no [nodes x vocab] host softmax DtoH,
no Python per-node loop), offline-proven within 1e-9 (22/22,
fr13_device_multidraft_offline_gate.py). This is the REAL deploy-temp (0.6) speed
lever (OPT-1 was greedy-only).

## Regime (both arms IDENTICAL except the flag)
- Vehicle: `scripts/fr13_bigdenom_swe_serve_variant.sh <arm> cat9 subset_b4_four.json`
- Subset: output/fr13_b1_gold_swe/subset_b4_four.json (4 astropy SWE-Verified tasks)
- DEPLOY_FORCE_TEMP=0.6  OFFLOAD_CODEX=1  MAX_NUM_SEQS_OVR=4  SWE_CONCURRENCY=4 (B=4 co-residency)
- cat9 locked pipeline, num_spec=9, tok/draft=9.0 engaged both arms (class-9 gate PASS)
- HEAD c6b0d92f (fr13-speedfix); reducer = scripts/fr13_measure.py deploy-speed (s/fwd OFF-mode)
- Real codex agent loop on alienware, GB10 vLLM-only (uncontended deploy-speed); per-task /metrics brackets

## Engagement (the discriminator)
- DEVICE (FR13_DEVICE_MULTIDRAFT=1): needle FIRED — `rejection_sampler.py:3066
  FR13_DEVICE_MULTIDRAFT engaged ...` from (EngineCore pid=176). 0 failures, no
  host fallback. tok/draft=9.0 all 4 tasks. drafts=30819 (firm).
- HOSTREF (FR13_DEVICE_MULTIDRAFT=0): device needle ABSENT (host reference
  `_lumo_tree_canonical_multidraft_sample` ran, byte-identical to HEAD default).
  tok/draft=9.0 all 4 tasks. drafts=23338.
- Both: GRAPH_CAPTURE_OK, worker-environ needle OK, teardown hygiene clean (0 swap).

## s/fwd (deploy-speed, OFF-mode, basis = d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total))

| task                    | device s/fwd | host s/fwd | ratio dev/host | device faster |
|-------------------------|-------------|------------|----------------|---------------|
| astropy__astropy-12907  | 0.6005      | 0.7786     | 0.7713         | 22.9%         |
| astropy__astropy-13033  | 0.5233      | 0.7525     | 0.6955         | 30.5%         |
| astropy__astropy-13236  | 0.6004      | 0.7525     | 0.7979         | 20.2%         |
| astropy__astropy-13398  | 0.6004      | 0.7525     | 0.7979         | 20.2%         |

- AGGREGATE: device s/fwd = **0.5875**, host s/fwd = **0.7562** -> ratio 0.7769
  = **device 22.3% LOWER (faster) per forward**.
- per-task ratio mean 0.766, stdev 0.042, min 0.696, max 0.798 (tight, all 4 same direction).
- Raw counters: decode-seconds nearly equal (device 18106s vs host 17648s, +2.6%)
  but device did **+32% drafts** (30819 vs 23338) and **+39% gen tokens** (139472 vs
  100526) in that decode-time = signature of a cheaper per-event committer, NOT a
  co-residency artifact (eff_concurrency device 2.51 vs host 2.46 = matched).

## Secondary (reporting-only)
- accept/event: device 3.530 vs host 3.313 (+0.217; distribution-lossless, NOT byte —
  device RNG draws differ; accept noise, not an s/fwd driver since s/fwd is per-EVENT).
- per_request_decode_tps: device 6.71 vs host 5.65 (+18.7%).
- prefill_frac: device 0.313 vs host 0.359.

## Toward the spine target (temp-0.6 cat9 -> chain5)
- The host tree committer's t0.6 tax (MEMORY: host ~1.4x vs chain5 spine ~1.11x) is
  the host softmax DtoH + per-node Python loop. The device committer removes it:
  cat9 t0.6 s/fwd drops 0.7562 -> 0.5875, landing at native-E5-GREEDY s/fwd level
  (banked nativeE5_b4 greedy s/fwd=0.596, tok/draft=5) WITH a 9-node tree. That is
  the chain5-spine direction (per-event committer cost no longer the tree tax).

## FIRM vs noise
FIRM. The earlier 2-prompt fr10_quick_decode_tps_probe (~1.4%/prompt noise) showed
only device +6.8% TPS. This canonical run is ~30K drafts/arm, 4 like-for-like tasks,
all 4 same direction at 20-30%, aggregate 22.3% lower s/fwd — an order of magnitude
beyond the GB10 cross-boot autotune/co-residency noise band, and the raw-counter
decomposition (same decode-seconds, +32% drafts) confirms the mechanism.

## Artifacts (output/ gitignored; numbers above are the record)
- output/fr13_device_multidraft/deploy_speed_device.json
- output/fr13_device_multidraft/deploy_speed_hostref.json
- output/fr13_bigdenom_swe/dm_device/ , output/fr13_bigdenom_swe/dm_hostref/ (boots)
