# V2 final recommended experiment scope

Updated 2026-09-21 after the numerical-agreement and Bole discussion. The user supports bounded additional experiments; this document finalizes the recommendation. **All new model experiments remain planned, not started.** The draft already uses the archived findings. Original IDs are preserved; E7 is split into E7a/E7b mechanism experiments and deferred E7c howexternal-system comparison.

**P0 completed on 2026-09-21 without model inference.** See [the audit report](p0/P0-REPORT.md). The user authorized the bounded experiment handoff in a new interactive Claude tmux session on the Spark. E7a captured-input arithmetic can start; full-model experiments require fresh runtime qualification and memory preflight. Current handoff status is recorded separately in `p0/handoff-status.json`; authorization is not a measured experiment result.

## Recommended order

**P0 -> E7a -> E2/E7b -> E1**, with E3 inserted before E1 only if its optimizations are to be promoted. This is **three core experiments, one conditional optimization study, and a provenance audit**. Existing E2 fixture mapping can begin alongside P0. E7a is now central because it directly tests the numerical/cost tradeoff raised by Bole and our archived WY work. E7b and E2 share captures and one continuation harness rather than becoming separate campaigns.

| ID | Final recommendation | Existing work to reuse | New question |
| --- | --- | --- | --- |
| P0 | Completed; no inference | Launchers, run manifests, raw measurement files | Source reconstruction completed; historical loaded-binary gaps remain explicit |
| E7a | Core experiment 1 | Archived WY derivation, scan/replay kernels, captured operands where available | How do sequential, triangular-solve, and finite-Neumann realizations differ on identical inputs? |
| E2/E7b | Core experiment 2 | State/KV/conv, sampler, spine, and graph fixtures | Do discrepancies amplify, and does the next iteration preserve the correct accepted prefix? |
| E1 | Main performance experiment | Historical MTP-5/MTP-11/tree controls and reducers | Does the final tree beat strong chains under matched conditions? |
| E3 | Optional before E1 | B4 scan and acceptance-walk implementations | Are individual improvements repeatable and additive in the actual stack? |
| E4 | Defer | Prefix-cache studies and service instrumentation | Does state memory affect admission, cache misses, and latency? |
| E5 | Defer broad task-quality study; exact sampler fixtures stay in E2 | Analytic sampler suite and historical task checks | Is distribution/quality within a declared tolerance beyond the tested prefixes? |
| E6 | Defer | NVFP4 split-K and native probes | Does the extension survive matched controls and composition? |
| E7c | Defer full external port | SGLang calibration and direct related work | Can a fair authors-system comparison be executed? |

No GPU-hour estimate is asserted before a bounded pilot measures the actual cost. Historical multi-hour agent runs should not be used to price a short fixed-prefix timing cell, or vice versa.

## Evidence and decision rules for the selected scope

- Keep one FP8 checkpoint lineage and GB10 for the core. Pin the actual source/build, precision boundaries, compiler settings, and GPU graph path before measurement.
- Freeze an eight-prefix diagnostic pilot selected from native-reference behavior before viewing candidate outputs. Include ordinary and small-margin predictions and short/long contexts. Use the pilot to determine capture cost, numerical scales, and timing variance. Then freeze a separate 32-prefix confirmation set, stratified by native margin and context, with selection seeds and IDs saved. These counts define a bounded diagnostic design, not statistical power for a task-quality or distribution-equivalence claim.
- Reuse archived captures only when the actual files and complete provenance are available. A historical document naming a capture is not the capture itself. Otherwise collect fresh data under the pinned configuration.
- Freeze tolerances, unacceptable decision changes, the timing precision target, and the resampling unit after the pilot and before confirmation. Report pilot and confirmation separately. Native repeat variation calibrates a reference floor; it does not automatically make every candidate error acceptable.
- A failed numerical route produces a localized result and is not promoted to a deployment-speed claim. Small local error alone is not a pass; E2/E7b checks continuation. An isolated counterexample supports a bounded failure claim, not a universal defect of WY, TreeWY, or Bole.
- No claimed hardware-hour estimate includes unmeasured implementation work. Record prototype engineering time separately. Use the eight-prefix pilot to price the fixed confirmation matrix before expanding it; do not launch a large task sweep by default.

## P0 — Configuration and evidence reconciliation

**Purpose:** restore trustworthy labels before interpreting any row as a particular effective-temperature result.

Record source commit, image/build identity, native extension hash, model/tokenizer revision, tree descriptor, requested and effective sampling constraints, graph route, cache flags, batching, and timing denominator. The archive documents a double temperature application on an affected speculative path. Trace exact launch/binary histories instead of inferring behavior from today's source or from the requested `temperature=0.6` alone.

Completed: three-arm numerical accounting was recomputed from raw JSON; source hashes and timing definitions were reconciled. Each July patcher reconstructed against the launchers' pinned image inserts a duplicate target-constraint pass. This compounds temperature scaling and repeats top-k/top-p filtering; it is not uniform temperature-0.36 sampling. H3 is retained as a relabeled historical observation. Original loaded-binary/model-file identity is unavailable and remains unknown. Current image, FA2 binary, model config/tokenizer, and all 66 weight shards are pinned for fresh work. A new run at the intended setting answers a different question.

**Output:** one immutable run manifest per manuscript row and a keep/relabel/omit decision. No model inference is needed for the archive audit.

## E2 — Current-route correctness composition

This is combined operationally with E7b below. Run the baseline fixture subset before introducing a candidate; extend only the missing combinations for a numerically viable reconstruction route.

**Existing coverage:** `tests/test_fr10_tree_rejection_sampler.py`, `test_fr13_attn_kv_remap.py`, `test_fr13_conv_committed_path.py`, fixed32 KV/commit tests, and the spine-layout diagnostics. Do not rebuild a duplicate reference suite.

**Test only missing combinations after a coverage map:**

- A sibling wins; a sibling is rejected; a deep path wins; zero drafts and all drafts are accepted.
- Check recurrent state, convolution history, and attention KV at the same native materialization boundary. Check the next forward too, including the correction token that was emitted but still pending.
- Cold state, prefix-cache hit, eviction, recycled physical rows/pages, actual B1/B4, and the graph path intended for E1.
- Check fixed-spine numerical behavior with a shared prefix and candidate set. Separate exact-byte checks from tolerance-based checks.
- Trace the analytic sampler cases and biased negative control into the actual deployed sampler. The existence of a correct Python reference does not establish the GPU route's behavior.

The August closeout already records a B1 cumulative-sum nondeterminism limitation. Preserve that limitation and avoid asserting a universal byte-exact sampler from narrower B2-B4 fixtures.

**Pass condition:** all supported state paths match their declared reference/tolerance, negative controls fail as intended, and post-boot checks confirm the tested flags. A failed path is either repaired and rechecked or excluded from the claimed scope. This is a targeted validation campaign, not another 16-task quality sweep.

## E1 — Matched final-stack performance

**Suggested core configuration:** keep the FP8 checkpoint lineage for this revision. Freeze a specific tested tree and all optimizations before final measurement. Do not mix FP8 and NVFP4 cells.

**Arms:** native MTP-5, the depth-matched native chain (MTP-11 for tail6), and the selected tree. The old comparison exists; the missing evidence concerns the final stack. A no-speculation arm is optional context. For a pure topology claim, additionally use the same recorded draft candidates and prefixes, because the historical tail6 suffix predictor differs from the native chain's drafting path.

**Small initial design:** three arms x two actual batch conditions (B1 and B4) x at least three paired independent boots = **18 timing cells**. Use the same frozen request/prefix set in each cell, including preselected context-length strata. Balance arm order. Keep capture/profiling separate from timing. Set a practical precision target after a pilot and before final data collection; report the pilot separately if it determines the design.

If E7a/E7b qualifies a reconstruction route for integration, add at most one such route as a fourth arm and retain the sequential tree as its ablation: **24 timing cells**, not an unbounded sweep. Select the candidate using pilot correctness and cost criteria before freezing confirmation data. Do not replace the sequential arm and lose the direct mechanism comparison.

**Report:** physical-step time, request events/step, event-normalized draft/verify/commit/host costs, committed output, per-request TPOT, and aggregate output over the explicitly defined measurement interval. Report interval construction, idle handling, and discarded/mixed steps. Do not equate pure-decode accounting with full-service throughput.

**Decision:** a repeatable win permits a current acceleration claim. A loss supports the scoped cost explanation and should be published as such. A wide interval calls for justified replication, not selecting a favorable run or task.

## E3 — Finish and compose the existing optimizations

Hold the verified attention path fixed. Compare four B4 arms: base; single-pass scan only; batched acceptance walk only; both. The scan has a provisional approximately 8.984 ms observation; the acceptance walk has correctness work but no completed deployment timing in the closeout.

Use **four matched blocks of the four arms (16 timing cells)** as an initial design consistent with the earlier paired validation, with independent boots and balanced ordering. Verify output/state behavior for each arm before timing. If variance is larger than the earlier campaign's, revisit replication from the pilot instead of treating four blocks as automatically conclusive.

**Decision:** retain only improvements whose deployment effect survives pairing and composition. The composed result must be measured; adding historical best deltas is not acceptable. If promoted, freeze the composed stack before E1.

## E4 — Cache capacity and full serving behavior

Replay the same multi-turn requests and arrival schedule for the selected tree and native control; record an offered-load range around the point where memory or admission becomes limiting. Measure physical state/KV allocation, cache hits/evictions, repeated prefill work, admitted requests, TTFT, TPOT, and total output/union wall time.

Use two explicitly different controls if making a causal memory claim: equal total device-memory budget for the deployment comparison, and matched available KV capacity to isolate the role of memory allocation. Do not quietly adjust memory utilization independently to produce a favorable ratio. Use identical warm-up and cache initial conditions.

The historical EXACT_SEED cache machinery was removed, so its old latency result cannot price the current stateless route. The existing SGLang calibration also differs in checkpoint/KV settings. E4 supplies a matched serving test; it is separate from E1's controlled decode timing and from a live agent deciding its own future prompts.

## E5 — Distribution and task quality, only to the extent claimed

**Distribution:** use fixed conditional prefixes, a declared target reference, recorded proposal laws, and negative controls. Verify the deployed sampler on small exact cases first, then measure numerical/probability discrepancies on selected model prefixes. Set the tolerance and analysis before reading the final comparison. Aggregate uncertainty by independent prefix/request, not by dependent token decisions. A nonsignificant test is not proof of equivalence.

**Task quality:** select a held-out task mix before seeing results, with controlled network access and trace-provenance checks. Use paired tasks, declared seed/boot policy, and a practically justified non-inferiority margin. A pilot determines the sample size needed for that margin. Preserve capped, failed, and excluded runs with reasons.

The existing 16 tasks are all Astropy, with eleven historically constant outcomes across heterogeneous arms. Repeating them can detect regressions but does not establish broad quality preservation. The affected gold-patch retrieval observation cannot be silently included as a successful independent solve.

## E6 — NVFP4 extension

Use one chosen NVFP4 checkpoint and the same tokenizer, sampler, KV precision, and native MTP control. Repeat split-K on independent paired boots and compare the complete combination with head/draft optimizations. The existing four native task probes do not replace this timing comparison. Add B4 only if claiming batching generalization. Evaluate quality against that checkpoint's own reference band.

**Decision:** make NVFP4 a clearly labeled extension if the evidence is complete. Otherwise retain it as future work; it is not required for the core FP8 revision.

## E7a — Same-input arithmetic and commitment experiment (core 1)

**Question:** can compact reconstruction remove replay work while meeting a declared numerical reference on GB10? Separate an algebraic identity from its compiled realization.

**Three local mechanisms:** (A) our sequential scan and accepted-path replay; (B) an ancestor-masked triangular-solve formulation with compact state reconstruction, representing the WY family; (C) Bole's finite-Neumann formulation with compact state reconstruction. B and C are our implementations of published mechanisms, not benchmarks of the authors' code or novel algorithms. A CPU-only implementation is an algebraic oracle, not a GPU performance result. Record any deviation from the published operand/storage precision or schedule.

**References:** use high-precision serial recurrence on exactly the same rounded input operands to assess mathematical/implementation error. Separately compare against the pinned native speculative-update kernel for replacement agreement and the pinned one-token recurrent-decode path for continuation agreement. Preserve each reference's actual gate and state-storage boundaries; do not merge their tolerances. Run the existing corrected local WY form where reusable, not the known pre-fix state-write bug as the principal baseline.

**Inputs:** identical captured pre-tree state, Q/K/V, raw gates, topology, and accepted-path descriptor; actual B1/B4; depths 1, 5, and 11 where supported; chains and a fixed branching topology. Include sibling reorder/padding controls. Enumerate valid accepted prefixes from each captured tree so commitment is compared under the same selected path independently of sampler outcomes. Keep all differences in state orientation and scale explicit.

**Isolate the two arithmetic stages:** first compare outputs and correction factors; then feed the same reference factors into the reconstruction implementations to isolate commit arithmetic. Finally compare each method's own verifier factors plus its own commit. This prevents a solver error from being misattributed to the commit kernel and makes a verifier/reconstruction mismatch visible.

**Measurements:** output/state absolute and relative error, relevant ULP counts, gate/decay range, scratch memory, and separately timed verification and commitment. For a candidate with growing errors, a higher-precision diagnostic can distinguish finite-precision sensitivity from an implementation error; it is not silently substituted into the advertised fast arm. Tiny gates also require checking how path decay ratios are evaluated rather than assuming the symbolic expression specifies a safe implementation.

**Decision:** local disagreement is characterized, not dismissed as harmless and not labeled a published-method defect. Only candidates meeting the frozen local criteria proceed to E7b confirmation. A failing method may be used in a bounded teacher-forced diagnostic to locate amplification, without being promoted to serving. If neither reconstruction route qualifies, the existing sequential system still proceeds to E1.

## E7b — Amplification and accepted-prefix continuation (core 2, shared with E2)

**Question:** do local discrepancies remain small in the actual model, or alter logits, acceptance, or later state? This is the direct follow-up to the repository's diffuse-drift history.

Use the shared pilot/confirmation prefixes with fixed teacher-forced candidate inputs. Capture layer outputs and state before allowing generated trajectories to diverge. Use substitution controls: replace only the verifier while replaying the same chosen path through the baseline committer; replace only commitment while retaining baseline verification; then test the combined candidate. Feed identical captured operands to commit-only controls. Distinguish this diagnostic forced path from a separate test of the actual device sampler and its acceptance decision.

Measure per-layer residual growth, final-logit errors, the reference top-token margin, argmax changes, conditional-probability differences, accepted-token counts, and the next step's logits/state. Use the same continuation prefix for short horizons of 1, 8, and 32 tokens where context allows. Report path depth, model-layer depth, and continuation horizon separately. A later free-running divergence is not itself evidence of the stage that caused it.

Run the E2 state-boundary/negative-control matrix on the baseline and any candidate considered for integration: zero/full acceptance, sibling selection, pending correction token, recurrent plus convolution plus attention KV state, warm/cold cache, eviction/reuse, B1/B4, and the actual graph route. Reuse existing tests instead of multiplying the full Cartesian product; publish the coverage map. Numerical agreement and exact small-case GPU sampler validation are separate outputs. If a candidate changes the target distribution beyond the prespecified criteria, retain the finding as a scoped numerical limitation; do not describe it as lossless.

**Deliverable:** one stage-isolation table and residual-by-layer plot, with a keep/repair/exclude decision for each tested mechanism. Promotion means demonstrated agreement in the tested configuration, not a proof over all prompts or devices.

## E7c — Direct external-system comparison (deferred)

First establish whether the relevant implementations are accessible, support the checkpoint/hardware, and can reproduce their own intended configuration. Record source revisions and unsupported features. For a fair run, match weights, precision, KV settings, effective sampling, draft budget/source, context distribution, offered load, and instrumentation where feasible.

Use each engine's own native control alongside any cross-engine measurements. An exact-byte requirement against our fork is not an appropriate correctness criterion for another algebraically equivalent implementation. Report its declared reference and numerical tolerance instead. Do not rank engines using published ratios from different settings. If a matched implementation is unavailable, retain the literature comparison and omit a superiority claim.

E7a/E7b supply the selected mechanism comparison before this larger port. Their findings must retain the label "local reimplementation of the published mechanism." An authors-system superiority claim requires E7c and matched controls.
