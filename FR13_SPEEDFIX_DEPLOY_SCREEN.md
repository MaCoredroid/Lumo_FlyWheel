# FR13 speed-campaign deployment screen (2026-06-15)

GOAL: push cat9 clearly > native B=1 decode-TPS via the cat-shape swap (R4 cat6root /
cat10) + the two levers (OPT-A GB10 fp8 GEMV cfg, OPT-1 committer sync-kill), LOSSLESS
held. Measured on the DEPLOYMENT regime (real SWE-Verified + codex, astropy-12907, B=1
MAX_NUM_SEQS=1), via the canonical infra (fr13_measure deploy-speed, instrument OFF;
fr13_bigdenom_swe_serve_variant.sh extends the canonical serve to the candidate arms).

All NEW arms ran at AGENT_WALL_S=600 (DEV-iteration bounded wall) + a fresh cat9 baseline
at the SAME wall for an apples-to-apples lever/shape delta. The full-wall banked cat9 /
native_e5 are the deployment reference (cat9 18.88 / native 18.28 TPS).

## DEPLOYMENT SPEED SCREEN (s/fwd = d(decode_seconds)/d(drafts), instrument OFF)

| arm          | drafts | gen_tok | s/fwd  | accept | committed | derived TPS | trajectory |
|--------------|-------:|--------:|-------:|-------:|----------:|------------:|------------|
| cat9_w600    |   1319 |    5930 | 0.2436 | 3.500  | 4.500     | 18.47       | resolved, patch 504 (baseline) |
| OPT-A on     |   1322 |    5771 | 0.2440 | 3.369  | 4.369     | 17.91       | resolved, patch 504 (comparable) |
| OPT-1 on     |    --- |     --- |   ---  |  ---   |   ---     |   CRASH     | EngineCore dead (build defect) |
| cat10        |   1153 |    5255 | 0.2489 | 3.558  | 4.558     | 18.31       | resolved, patch 504 (comparable) |
| cat6root r2  |   1097 |    4625 | 0.2394 | 3.216  | 4.216     | 17.61       | resolved, patch 504 (comparable) |
| cat6root r1  |   4619 |   18968 | 0.1318 | 3.106  | 4.106     | 31.15       | FAILED, CONFOUNDED (3.5x tokens, degenerate fork) |

reference (full-wall banked): native_e5 s/fwd 0.2334 / accept 3.267 / TPS 18.28;
                               cat9      s/fwd 0.2481 / accept 3.685 / TPS 18.88.

## VERDICTS

- **OPT-A (FR13_GB10_FP8_GEMV_CFG=1)**: ENGAGED (confirmed live in EngineCore worker
  environ as =1; the GB10/sm_121 skinny-decode fp8 GEMV config override fires at forward
  time). s/fwd 0.2440 vs cat9 0.2436 = SPEED-NEUTRAL (+0.2%, within noise). The decode-M
  6-10 tree-verify GEMM is NOT the per-forward bottleneck on the deployment trajectory.
  Lossless: structurally bit-identical (BLOCK_SIZE_N/K pinned, default-OFF byte-identical).
  NOT A SPEED WIN -> does not ship.

- **OPT-1 (FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1)**: BUILD DEFECT, crashes the
  EngineCore. Root cause (precisely located): the synckill fork in the EAGER_PACK preamble
  of `_lumo_tree_path_lcp_max_greedy_sample` (rejection_sampler.py ~L1419) NULLS the host
  committer-input lists (parents_cpu/drafts_cpu/parent_targets_cpu/self_targets_cpu/
  bonus_targets_cpu = None) on the ASSUMPTION that the device arm fully owns the committer
  decision -- but the SAME function then falls through to its LEGACY per-node loop at
  L1637 `parents = parents_cpu[start:start+node_count]` which subscripts the nulled list ->
  `TypeError: 'NoneType' object is not subscriptable` -> EngineCoreDead. The G2.a
  input-nulling and the device-arm replacement live in two code regions that do NOT
  compose: nulling the inputs without making the legacy loop skip is a dispatch bug. The
  CPU byte-A/B gate passed because it tested the device-kernel emulation in ISOLATION, not
  the live in-process `_lumo_tree_path_lcp_max_greedy_sample` dispatch. NEEDS A REAL FIX
  (guard/skip the legacy loop under synckill + source committer outputs from the device
  arm before return) before it can even be screened. NOT SHIPPABLE as built.

- **cat6root (R4)**: ENGAGED (exact-match tree, tok/draft=6). At a COMPARABLE trajectory
  (rep2, resolved/patch 504): s/fwd 0.2394 vs cat9 0.2436 = -1.7% per-forward (modest, as
  expected from 6 vs 9 verify rows / pad8 vs pad16), but accept 3.216 vs 3.500 = LOWER
  (fewer interior leaves -> fewer accepted tokens/event), netting TPS 17.61 = -4.7% vs
  cat9. NOT A SPEED WIN. NOTE: rep1's 31.15 TPS / 0.1318 s/fwd was a TRAJECTORY ARTIFACT
  (the codex agent forked to a high-throughput degenerate-ish loop: 4619 drafts / 18968
  tokens, verdict FAILED, 3.5x the comparable arms) -- bug-class #12 (non-like-for-like
  trajectories). s/fwd is regime-robust ONLY at comparable context-length distributions;
  rep2 is the apples-to-apples number.

- **cat10 (cat9 + (1,) root sibling)**: ENGAGED (tok/draft=10). COMPARABLE trajectory
  (resolved/patch 504, 27 pair dumps ~ cat9's 24). s/fwd 0.2489 vs cat9 0.2436 = +2.2%
  per-forward (the 10th verify row costs more in the lm-head GEMV + tree-attn), accept
  3.558 vs 3.500 = +1.7% (the root sibling rescues a few argmax-flip events), netting TPS
  18.31 = -0.9% vs cat9 = TPS-NEUTRAL. The accept gain is exactly offset by the extra
  per-forward cost. NOT A CLEAR WIN. The prior 2.932-accept revive concern was correct to
  re-measure: on the deployment regime cat10 accept is 3.558 (per-event, depth-matched d5),
  NOT the class-12-denom-artifact 2.932 -- but the accept edge does not buy net TPS.

## BOTTOM LINE

NO candidate clears the bar (cat9 STRICTLY > native at B=1 deployment TPS held by the
candidate). cat9 stays the deployed shape: 18.47 TPS at the 600s wall (18.88 full-wall),
already TPS-competitive-to-faster than native (18.28). The two levers do not move
deployment per-forward time (OPT-A neutral, OPT-1 broken); the reshapes trade s/fwd for
accept with no net gain (cat6root -4.7%, cat10 -0.9%). Lossless gate is moot -- a candidate
that does not improve TPS does not ship regardless of losslessness.

The s/fwd ranking (OPT-A neutral, cat6root -1.7%, cat10 +2.2%) confirms the verify-forward
cost scales ~linearly with tree-node count but is NOT the TPS lever on this HBM-bound B=1
regime -- accept/event (committed numerator) and the codex trajectory dominate. Pushing
cat9 clearly past native needs a per-forward win that the fp8-GEMV-cfg / node-count knobs
do not deliver, OR an accept/event gain that does not cost proportional per-forward time
(cat10's root sibling is the closest but breaks even).

All artifacts: output/fr13_measure/deployment/{cat9_w600,opta,cat10,cat6root_r2,cat6root}
_deploy_speed_b1.json. Boots: output/fr13_bigdenom_swe/{cat9_w600,opta_w600,opt1_w600,
cat10_w600,cat6root_w600,cat6root_w600r2}/. Infra: scripts/fr13_bigdenom_swe_serve_variant.sh
(commit 67678ac4). Builds screened: reshape 14f6a528, OPT-1 68e44f22 (defect), OPT-A
e90de7ef.
