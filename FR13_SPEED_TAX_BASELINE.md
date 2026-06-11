# FR13 Speed-Tax Baseline — first table (CPU backfill from existing artifacts, 2026-06-10)

**Producer**: `scripts/fr13_speed_tax_gate.py` @ aa261e16 (`reduce` + `fit`, CPU-only; NO GPU run for this doc).
**Outputs** (gitignored, reproducible): `output/fr13_speed_tax_gate/backfill_reduce.json`, `backfill_fit.json`
(re-run verified byte-identical modulo timestamp/arm-order: `backfill_reduce_verify.json`, `backfill_fit_verify.json`).

## Measurement rules (bestiary class 12, FR13_BUG_CLASS_PLAYBOOK.md — binding)

- **NEVER present TPS/accept division as a measured per-forward number** (that hand-roll was retracted twice).
  The gate script raises `HandRollGuardError` on any such derivation and emits UNAVAILABLE when the /metrics basis is missing.
- The MEASURED per-forward basis = `request_decode_time_seconds_sum / spec_decode_num_drafts_total` from /metrics
  scraped before/after a window (the wacoxe6i2 method). Validity check per arm:
  `spec_draft_tokens / spec_drafts == num_spec_tokens` proves drafts == forwards (column `drafts==fwds`).
- `decode_seconds` is a per-request SUM under concurrency — valid as a **RATIO at matched shape**
  (same prompts/B/max_tokens/samples), **NOT** an absolute latency. All ratios below passed the pairing gate
  (prompts sha256 byte-identical, same seed, same temp/top_p/B/mt/spp, BI on BOTH arms — class 9).

## Pinned pairing state (all arms)

- Prompts: `output/fr13_acceptance_ladder/prompts_swe4.json` (sha256 `5df7fa46b0977ebaf7cb4a629893678409455073c4ef23c7ddedd23a1f0dc4b3`, 4 prompts)
- B=1, samples_per_prompt=1, max_tokens=128, greedy (temp 0.0 / top_p 1.0), seed 1313, BATCH_INVARIANT=1 both arms
- Tree arms: forked-FA2 TREE_ATTN + tree GDN (`scripts/fr13_launch_forked_fa2_tree_server.sh`), FR13_BI_TREE_ATTN=1, GPU_UTIL=0.82
- Native arm: FLASH_ATTN naive MTP-5 (`scripts/fr10_launch_speed_server.sh`), GPU_UTIL=0.86
- Replay-b2 arm BI: run_header boot2 records only `other_env: same as boot1`; BI=1 supplied via documented `--flags`
  override and recorded in the arm label (class-9 refusal worked as intended; future run_headers must write the full flag set per boot)

## THE TABLE — speed tax vs native MTP-5 (cell labels: **[M]**=measured /metrics basis, **[R]**=measured ratio of paired [M] cells, **[P]**=traffic-model prediction NOT a measurement, **[F]**=regression fit, **UNAVAILABLE**)

| arm | route | topology (N draft, depth, pad) | accept/event | warm TPS | per-req TPS mean | s/fwd (sum basis) | drafts==fwds | tax ratio vs native | pred GB/fwd | pred floor ms | caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native_mtp5_bi1 | native | MTP-5 chain (5) | 3.047 [M] | 18.95 [M] | 12.40 [M] | **0.2127** [M] | Y (635/127=5) | baseline | 28.06 [P] | 102.8 [P] | — |
| legacy_chain5 | legacy | chain5 (5, d5, pad8) | 2.277 [M] | 9.16 [M] | 7.96 [M] | **0.3517** [M] | Y (795/159=5) | **1.653x** [R] | 30.10 [P] | 110.3 [P] | — |
| legacy_cat9_preconvfix | legacy | caterpillar9 (9, d5, pad16) | 1.819 [M] | 7.15 [M] | 6.43 [M] | **0.3936** [M] | Y (1638/182=9) | **1.850x** [R] | 31.78 [P] | 116.4 [P] | pre-conv-fix boot |
| legacy_cat9_b1 | legacy | caterpillar9 (9, d5, pad16) | 2.024 [M] | 7.65 [M] | 6.83 [M] | **0.3936** [M] | Y (1530/170=9) | **1.850x** [R] | 31.84 [P] | 116.6 [P] | replay-campaign boot1, flag OFF |
| legacy_cat9_convfix | legacy | caterpillar9 (9, d5, pad16) | 2.215 [M] | 7.99 [M] | 6.81 [M] | **0.3976** [M] | Y (1170/130=9) | **1.869x** [R] | 31.90 [P] | 116.8 [P] | conv-fix ON (c0b53f5d), bootB |
| replay_cat9_b2 | replay | caterpillar9 (9, d5, pad16) | 1.583 [M] **accept-bug CONFOUNDED** | 7.87 [M] | 7.07 [M] | **0.3270** [M] | Y (1791/199=9) | **1.537x** [R] | 27.54 [P] | 100.9 [P] | **replay-ON accept-bug confounded** (accept 2.02→1.58, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND); per-forward ratio at matched shape still informative; BI=1 via documented --flags |
| b1_naive_greedy / b1c_nonmtp_greedy / b2_nonmtp_greedy | native refs on tree boots | — | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | — | UNAVAILABLE | — | — | window dirs EMPTY: naive_mtp/non_mtp on tree boots kill the engine (EagleProposer.positions, eagle.py:1430, flag-independent, tracked separately); matched baseline substituted = native_bi1_greedy (s1s2s3 campaign, same pinned prompts/seed/BI) |

Availability caveats:
- The task-listed `output/fr13_replay_gpu_gates/{b1_naive_greedy, b1c_nonmtp_greedy}` (+`b2_nonmtp_greedy`) contain no probe
  json / metrics — they are UNAVAILABLE rows, not zeros. The native baseline above is the matched
  `output/fr13_s1s2s3_discriminate/native_bi1_greedy` arm (separate FLASH_ATTN boot, same pinned battery).
- Per-forward seconds are sum-basis: comparable **within this table only** (matched shape); not absolute latency.
- accept/event spread across legacy cat9 arms (1.82–2.22) is trajectory-confounded across boots (see ladder log);
  it barely moves the per-forward basis (s/fwd 0.3936–0.3976, ±0.5%) — per-forward is robust to accept differences
  by construction (divides by forwards, not tokens). Do NOT multiply these columns into a "speed" number (class 12).

## n=2 scaling signal — chain-5 vs caterpillar-9 at legacy route (direction only)

What 2 distinct N values can and cannot say (legacy route, both d=5, so the added 4 nodes are **width**):

- **Direction: the per-forward tax GROWS with N.** N=5 → N=9: 0.3517 → 0.3936–0.3976 s/fwd
  (+0.042–0.046 s/fwd; ratio 1.653x → 1.850–1.869x). All three N=9 arms agree within 1%, across three different
  boots/campaigns — the step is real, not boot noise.
- **[F] OLS over the 4 legacy points** (backfill_fit.json): slope **0.0108 s/fwd per draft node**
  (95% normal-approx CI [0.0095, 0.0121], small-n indicative only), intercept 0.298 s; preferred over the
  N-invariant constant by dof-adjusted MSE (SSE 1.1e-5 vs 1.4e-3). This is a FIT label, not a measurement,
  and with only 2 distinct N it is **direction, not shape**: any monotone curve fits 2 x-values.
- **Magnitude vs the traffic model:** the w78aq6xum row model (legacy rows = 3N+2a+1) predicts a floor step of only
  +6.3 ms/fwd from N=5→9 (110.3→116.6); measured step is +42–46 ms/fwd ≈ **7x the row-traffic floor slope**.
  So the marginal cost per node is NOT explained by GDN state-row traffic alone — candidates: the n_pad step
  (8→16 exactly at this comparison — a built-in confound of these 2 points), verifier mask/sampler work, drafter
  rollout width. Cannot be decomposed with 2 N values.
- **Replay single point (CONFOUNDED label carried):** replay@N=9 = 0.3270 s/fwd (1.537x) — cheaper per forward
  than legacy at N=5 (0.3517). Directionally consistent with the replay N-invariance claim (rows = 2+a), but it is
  ONE point at ONE N from an accept-bug-confounded boot: no invariance test is possible
  (fit correctly reports preferred_model=None, INSUFFICIENT_POINTS for linear).

## Ladder rows (FR13-TAX, emitted by the gate; flag-state+seed headers inline)

```
FR13-TAX | arm=native_mtp5_bi1 | route=native | topo=native_mtp5 | accept/event=3.0472 | warm_tps=18.952 | per_fwd=0.2127s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=baseline | pred=28.06GB/fwd floor=102.8ms | BI=1 | seed=1313 | temp=0.0
FR13-TAX | arm=legacy_chain5 | route=legacy | topo=chain5(N=5,d=5,pad=8) | accept/event=2.2767 | warm_tps=9.157 | per_fwd=0.3517s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=1.653x | pred=30.10GB/fwd floor=110.3ms | BI=1 | seed=1313 | temp=0.0
FR13-TAX | arm=legacy_cat9_preconvfix | route=legacy | topo=caterpillar9(N=9,d=5,pad=16) | accept/event=1.8187 | warm_tps=7.147 | per_fwd=0.3936s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=1.850x | pred=31.78GB/fwd floor=116.4ms | BI=1 | seed=1313 | temp=0.0 | label=pre-conv-fix caterpillar
FR13-TAX | arm=legacy_cat9_b1 | route=legacy | topo=caterpillar9(N=9,d=5,pad=16) | accept/event=2.0235 | warm_tps=7.652 | per_fwd=0.3936s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=1.850x | pred=31.84GB/fwd floor=116.6ms | BI=1 | seed=1313 | temp=0.0
FR13-TAX | arm=legacy_cat9_convfix | route=legacy | topo=caterpillar9(N=9,d=5,pad=16) | accept/event=2.2154 | warm_tps=7.990 | per_fwd=0.3976s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=1.869x | pred=31.90GB/fwd floor=116.8ms | BI=1 | seed=1313 | temp=0.0 | label=caterpillar+convfix (bootB)
FR13-TAX | arm=replay_cat9_b2 | route=replay | topo=caterpillar9(N=9,d=5,pad=16) | accept/event=1.5829 | warm_tps=7.869 | per_fwd=0.3270s/fwd(sum-basis,probe_metric_delta,drafts==fwds=Y) | ratio_vs_native_mtp5_bi1=1.537x | pred=27.54GB/fwd floor=100.9ms | BI=1 | seed=1313 | temp=0.0 | label=replay-ON accept-bug confounded (accept 2.02->1.58, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND); BI=1 per run_header boot2 other_env=same-as-boot1
```

## What the GPU sweep adds (matrix already generated: `output/fr13_speed_tax_gate/sweep/sweep_commands.sh`)

Shapes under the N_PAD=16 cap (n_pad = next pow2 of N+1 ⇒ max 15 draft nodes; the requested 16-node shape is
REJECTED, n=17→pad32, documented as a no-arm catalog entry):
chain5 (N=5,pad8) · caterpillar9 (N=9,pad16, deployed default) · caterpillar12_w3_d5 (N=12, wider at fixed depth-5)
· caterpillar13_d6 (N=13, depth-6) · caterpillar15_w3_d6 (N=15, max under cap). Each shape × {legacy, replay}
+ one native MTP-5 FLASH_ATTN reference boot; /metrics before/after snapshots; `--require-tree-engagement`
+ `--expected-draft-count N` (class 9); serial-GPU teardown between boots.

It will discriminate what n=2 cannot:
1. **Legacy linear-in-N**: 4 distinct N at pad16 (9,12,13,15) + the pad8 point → within-pad slope, separating the
   n_pad-step from the per-node cost, and testing the 7x-over-floor excess against the 3N+2a+1 row model.
2. **Replay N-invariance** (THE scaling claim, rows = 2+a): flat s/fwd across N=5..15 vs legacy's slope.
3. **Width vs depth**: caterpillar12_w3_d5 (width at d5) vs caterpillar13_d6 (depth 6) at adjacent N.
4. A same-boot-pair native reference, removing the cross-campaign baseline substitution noted above.

**When to run: AFTER the replay live accept-bug fix.** Replay arms require the unmerged branch
`fr13-replay-route@9d4d22e3` (FR13_REPLAY_ROUTE launcher passthrough is not on main), and replay accept is live-
confounded (2.02→1.58, seams R1/R6g/R8 per FR13_REPLAY_GPU_GATES_BIND.md) — per-forward ratios would be informative
but the accept-dependent columns (and any superset claim) would inherit the confound; the legacy+native arms of the
sweep could run earlier if a GPU window opens, but a single post-fix campaign is cheaper (one native boot amortized).
Depth>5 shapes carry a drafter-engagement caveat: confirm `--expected-draft-count` on first boot of any new shape.

## Reproduce (CPU)

```bash
python3 scripts/fr13_speed_tax_gate.py reduce \
  --arm native_mtp5_bi1=output/fr13_s1s2s3_discriminate/native_bi1_greedy \
  --arm legacy_chain5=output/fr13_s1s2s3_discriminate/chain_greedy \
  --arm legacy_cat9_preconvfix=output/fr13_s1s2s3_discriminate/tree_greedy \
  --arm legacy_cat9_b1=output/fr13_replay_gpu_gates/b1_tree_greedy \
  --arm legacy_cat9_convfix=output/fr13_convfix_ab/b_greedy \
  --arm replay_cat9_b2=output/fr13_replay_gpu_gates/b2_tree_greedy \
  --baseline native_mtp5_bi1 \
  --route native_mtp5_bi1=native --route legacy_chain5=legacy --route legacy_cat9_preconvfix=legacy \
  --route legacy_cat9_b1=legacy --route legacy_cat9_convfix=legacy --route replay_cat9_b2=replay \
  --topology native_mtp5_bi1=native_mtp5 --topology legacy_chain5=chain5 \
  --topology legacy_cat9_preconvfix=caterpillar9 --topology legacy_cat9_b1=caterpillar9 \
  --topology legacy_cat9_convfix=caterpillar9 --topology replay_cat9_b2=caterpillar9 \
  --label legacy_cat9_preconvfix="pre-conv-fix caterpillar" \
  --label legacy_cat9_convfix="caterpillar+convfix (bootB)" \
  --label replay_cat9_b2="replay-ON accept-bug confounded (accept 2.02->1.58, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND); BI=1 per run_header boot2 other_env=same-as-boot1" \
  --flags replay_cat9_b2=BATCH_INVARIANT=1 \
  --out output/fr13_speed_tax_gate/backfill_reduce.json
python3 scripts/fr13_speed_tax_gate.py fit \
  --reduce output/fr13_speed_tax_gate/backfill_reduce.json \
  --out output/fr13_speed_tax_gate/backfill_fit.json
```

## ⚠ VALIDITY SCOPE (user challenge 2026-06-10 — binding)
The table above is **NOT the deliverable speed verdict**. Its boots were lossless-debug boots:
**B=1** (not the deployed B=4), **BATCH_INVARIANT=1 on all arms** (known slow-GEMM regime — OFF for speed per
`reference_fr10_speed_measurement_pitfalls`), `FR10_METRICS=1` + LUMO logging envs ON (instrumentation overhead),
mixed capture modes, 4 pinned prompts × 128 tokens (not the SWE-Verified ~1800s workload).
**Valid for:** relative direction at matched contamination (replay −17% vs legacy ≈ the state-traffic prediction;
legacy tax grows with N). **Invalid for:** absolute tax, deployment claims, E5 comparison.

## DEPLOYMENT-REGIME measurement spec (the one that counts; run post-wiring-fix)
- B=4, MAX_NUM_SEQS=4, **FULL CUDA capture proven** (cuda_graph_proof per arm), **BI=0 both arms**,
  **FR10_METRICS=0 + ALL LUMO logging envs unset** (accept counters come from vLLM's native /metrics spec counters,
  which exist regardless — no instrumentation in the serving path), GPU_UTIL deployment value.
- Workload: SWE-Verified agentic shape (`fr12_deliverable_swe4_probe` full-task form, ~30 min/arm class), pinned
  task set, seeds recorded. Arms: native E5 / legacy tree / replay tree (post-fix), same HEAD.
- Basis: /metrics window deltas only (`decode_seconds/spec_drafts`), pairing gate enforced; per-forward ratio +
  wall + accept/event reported together.
