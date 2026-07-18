# FR13 tail6 "accept >5 then, 4.3–4.5 now" — Reconciliation Report

## 1. PROVENANCE TABLE

### A. Measured artifacts (all `bracketed deploy_speed accept_per_event` = num_accepted/num_drafts, decode-only /metrics brackets, prefill excluded — `scripts/fr13_measure.py:602`, counters `:168-170`, bracket `:544/:570/:572`; committed = accept+1 per `:603`)

| accept | committed | Source (artifact) | Regime | Key evidence |
|---|---|---|---|---|
| **5.099** | 6.099 | `output/fr13_tail6_prewarm/tail6_prewarm_pw16/deploy_speed_pw16.json` | tail6, **PREWARM ON** (132-seq corpus), APC OFF, n=16, conc=4, max_num_seqs=4 | mtime 07-16 00:34; `FR13_PREWARM_TRIE` set; launch.log "pre-warm corpus (132 seqs)"; tps_gpu 64.00 |
| **4.954** | 5.954 | `output/fr13_asyncsched/tail6_async_as1/deploy_speed_as1.json` | tail6, **--async-scheduling**, PREWARM OFF, APC OFF, n=16 | mtime 07-17 05:11; `FR13_SERVE_BATCH_FLAGS=--async-scheduling`; no PREWARM_TRIE |
| 4.317 | 5.317 | `output/fr13_tail6b_ab/tail6_b7/deploy_speed_b7.json` | **b7 clean same-session tail6** (plain, tok_per_draft=21), no prewarm/async, n=16 | mtime 07-16 15:02; pf 0.432; tps_gpu 56.89 |
| 4.500 | 5.500 | `output/fr13_tail6b_ab/tail6b_b7/deploy_speed_b7.json` | b7 tail6b (2 branches x 2 depths, tok_per_draft=25), n=16 | mtime 07-16 12:42; `FR13_TAIL_BRANCHES=2`; pf 0.453 |
| 4.809 | 5.809 | `output/fr13_greedy_via_ab/tail6_gv0/deploy_speed_gv.json` | GREEDY_VIA_REJECTION=0 arm, **n=4**, **pf 0.643** (family max) | near-threshold on small subset; mtime 07-17 17:37 |
| 4.184 | 5.184 | `output/fr13_greedy_via_ab/tail6_gv1/deploy_speed_gv.json` | GREEDY_VIA_REJECTION=1 arm, n=4 | mtime 07-17 18:02 |
| 4.386 | 5.386 | `output/fr13_commnative/tail6_gu1/deploy_speed_cn1.json` | **post-change: COMMITTER_NATIVE=1**, n=4 | mtime 07-18 12:42; pf 0.450 |
| 4.363 | 5.363 | `output/fr13_phase1_gate/tail6_gu1/deploy_speed_p1g1.json` | **post-change: phase1 gate, all diag gates 0**, n=4 | mtime 07-18 09:40; pf 0.397 |
| 4.306 | 5.306 | `output/fr13_native_tail6_decomp/tail6_nt1/deploy_speed_recomputed.json` | tail6 decomp, **RECOMPUTED PARTIAL n_tasks=7** (no DONE line) | mtime 07-18 02:30; treat with caution |
| 3.415 | 4.415 | `output/fr13_native_tail6_decomp/nativemtp5_nt1/deploy_speed_nt1.json` | **native MTP-5 comparator** (tok_per_draft=5), n=16 | tps_gpu 75.96; s/fwd 0.05812 |
| 4.462 | 5.462 | `output/fr13_batchfill/tail6_bf_bf1/deploy_speed_bf1.json` | batchfill arm, n=16 (wall-clock tps 1.98 = batch-fill artifact; GPU-basis 61.96 normal) | mtime 07-17 01:58 |
| 4.079 | 5.079 | `output/fr13_batch_ab/tail6_bo0/deploy_speed_bo.json` | COMMIT_BATCH_OUTPUT=0 arm, n=4 | mtime 07-17 14:08 |
| 4.290 | 5.290 | `output/fr13_committer_reloc/tail6_reloc/deploy_speed_reloc.json` | committer-reloc arm, n=4 | mtime 07-17 14:57 |
| 4.093 | 5.093 | `output/fr13_sbr_ab/tail6_sbr0/deploy_speed_sbr.json` | sbr arm 0, n=4 — **PREWARM_TRIE SET yet accept 4.09** | mtime 07-17 16:48; refutes prewarm-as-carrier |
| 4.277 | 5.277 | `output/fr13_tail_g4c/tail6_tailg4c/deploy_speed_tailg4c.json` | g4c geometry, n=4 — **family floor** | mtime 07-15 18:19; BEYOND5:781 reclassifies "LOW outlier" |
| 1.902 / 1.896 | 2.902 / 2.896 | `output/fr13_prewarm_ab/merged_{prewarm,cold}_t33333_pw1/deploy_speed_pw1.json` | **NOT tail6** (merged drafter tok_per_draft=15, APC ON) — not accept-comparable | mtimes 07-15 |

### B. Doc claims (the "5.x" quotes)

| Value | True basis | Regime | Source |
|---|---|---|---|
| ~5.2 headline baseline | accept_per_event, but from old campaign canonical | **STALE** — declared invalid by REGIME NOTE `:274` | `FR13_TAIL6_IMPROVEMENT_PLAN.md:4,:48,:94,:112` |
| 5.23 | **Sigma-survival MODEL** (656-window per-depth aggregate), anchored to stale 5.2 | pre-b7 regime | plan `:29,:33,:107` |
| [5.38, 5.54] band; 5.227 baseline | per-depth conditional-acceptance MODEL units | absolute levels do not transfer (REGIME NOTE) | plan `:190,:195-197,:256` |
| 5.418 → 5.589 interim | **raw-window mean_accept_length == committed_per_event** (accept+1), also diluted by prefill/tool-gap windows | b7 regime, wrong basis — self-corrected | plan `:212,:229`; CORRECTION `:246-256`; `:271-272` "committed ~5.49 − 1 = 4.49 ≈ 4.500 canonical" |
| 6.01 / 5.88 / 6.13 / 9.83 | **vLLM "Mean acceptance length" = accept+1 = committed** (verified: 106/12=8.83, +1=9.83) | diagnostic running-aggregate, never canonical | plan `:252-253`; BEYOND5 `:629-630,:653` |
| 5.237 (cold, 5.201@4→5.237@8), tps_gpu 71.22 | **genuine accept_per_event**, canonical deploy_speed | accept>5 campaign (07-16), high run-to-run variance 4.277↔5.2 (`:768,:786`); ">5 is a repetitive-span WINDFALL" (`:472`) | `FR13_ACCEPT_BEYOND5_DESIGN.md:796,:799` |
| 5.105@10 / 5.109@15 (prewarm) | genuine accept_per_event; cold≈prewarm ⇒ **prewarm ~0 contribution** | same campaign; CAMPAIGN CLOSED | BEYOND5 `:808-810,:680,:742` (plan records same artifact as 5.099 — 0.01 within-doc discrepancy) |
| 5.31 | **committed_per_event, explicitly labeled** (accept ~4.31) | same-campaign decomp, 07-18 | `FR13_TREE_VS_NATIVE_VERDICT.md:14` |
| 5.237 "tree's accept advantage" | accept_per_event but **cross-campaign import of old-regime number** into 07-18 break-even (vs needed >~5.76) | mixes regimes; same-campaign accept is ~4.31 | VERDICT `:29-31,:34` |
| 5.08 / 5.010 live | aggregate accepted/drafts, credits prewarm | 07-17 in-session; **contradicted** by BEYOND5 prewarm~0 A/B and by sbr0 (prewarm set, a=4.09) | `FR13_COMMITTER_UNIFY.md:183-186,:190-193` |
| 4.363 / 4.386 / 4.385 | accept_per_event, canonical | post-phase-1 b7-era, n=4 subset | UNIFY `:367-369,:496,:80` |

## 2. RECONCILIATION

**(a) METRIC BASIS — accounts for every "5.x" quote made in the CURRENT (b7) regime.** committed_per_event = accept_per_event + 1 (`fr13_measure.py:603`), and vLLM's logged "Mean acceptance length" equals committed (plan `:252-253`, verified 106/12=8.83+1=9.83). Every canonical b7 tail6 artifact has committed 5.28–5.50 (b7 pair 5.317/5.500, p1g1 5.363, cn1 5.386, nt1 5.306) — so tail6 **is** "5.x" today on the committed basis while its accept is 4.3–4.5. The plan's interim "5.418/5.589" were raw-window mean_accept_length (== committed, further diluted by prefill/tool-gap windows), self-corrected at `:246-256` and closed at `:271-272`: committed ~5.49 − 1 = 4.49 ≈ the 4.500 canonical. The VERDICT doc's 5.31 is explicitly committed. Basis confusion alone explains ~1.0 of the gap wherever these quotes were read as accept.

**(b) REGIME — accounts for the genuine accept ≥5 readings.** Only two artifacts in the mined set cross 4.9: pw16 **5.099** (the ONLY tail6 n=16 run with PREWARM ON, APC OFF, conc=4) and as1 **4.954** (prewarm OFF but --async-scheduling ON) — neither is plain-tail6. The doc-canonical genuine >5s (cold 5.201→5.237→5.105, prewarm 5.109, tps_gpu ~71) come from the 07-16 accept>5 campaign, a regime the plan's own REGIME NOTE (`:274-279`) declares non-transferable: the tail6 family spans 4.28 (g4c) to 5.10 (prewarm), run-to-run variance 4.277↔5.2 (BEYOND5 `:768,:786`), and BEYOND5 `:472` concedes ">5 is a repetitive-span WINDFALL," not a typical per-task average. Prewarm itself is NOT the carrier: BEYOND5's same-session A/B proved cold 5.105 ≈ prewarm 5.109 (~0 contribution), and sbr0 has PREWARM_TRIE set yet reads 4.093 — refuting UNIFY `:183`'s prewarm credit. accept_per_event is also flagged B-DEPENDENT + TRAJECTORY-BOUND (`fr13_measure.py:638-642`), so cross-boot absolute comparisons are structurally unsound; only same-session deltas transfer (e.g., tail6b − tail6 = +0.183).

**(c) REAL REGRESSION — none; the signal is mildly POSITIVE.** Same-basis, same-regime, across the phase-1 committer changes: b7 tail6 4.317 (07-16, n=16, pre-change) vs p1g1 4.363 (07-18, gates 0), cn1 4.386 (07-18, native committer), dd1 4.385 — all post-change reads are +0.05–0.07 ABOVE the pre-change baseline, well inside the n=4 subset scatter (same-regime n=4 arms span 4.079 bo0 to 4.809 gv0, ~±0.4). The recomputed partial nt1 4.306@7 also matches 4.317. There is no time-ordered same-regime pair where accept declined.

## 3. VERDICT

The canonical current number is **accept_per_event = 4.317** (committed 5.317) for plain tail6 in the b7 SWE-agentic regime (n=16 astropy 12907..14995, conc=4, B=4, APC OFF, no prewarm/async, temp 0.6; `output/fr13_tail6b_ab/tail6_b7/deploy_speed_b7.json`), with post-phase-1 gates reading 4.363–4.386 (no regression, slight positive drift within subset variance); the best genuine historical accept ≥5 is **5.237** (tail6_cold, no prewarm, tps_gpu 71.22, BEYOND5 `:796-799`) with the best artifact-verified read at **5.099** (pw16, prewarm ON, n=16), both from the 07-16 accept>5-campaign regime whose absolute levels the plan's own REGIME NOTE rules non-transferable (family span 4.28–5.10, ">5 = repetitive-span windfall"). One-sentence answer: **we were "above 5" partly because true-accept ≥5 occurred only in a different, higher-variance regime (old campaign trajectory mix, prewarm/async sessions) and partly because committed_per_event and vLLM mean-acceptance-length (both = accept+1) were quoted where accept_per_event was meant — tail6's committed is still 5.3–5.5 today — while same-regime same-basis comparisons show zero regression.**
