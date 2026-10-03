> **Current workload scope (author amendment, 27 September): four attempts total—one predetermined SWE-bench Verified task, one run under each of four methods. The eight-attempt pilot, tuning proposal and larger confirmation campaign below are historical and superseded. Correctness and mechanism comparisons remain required. See [the approved amendment](2026-09-27-four-attempt-workload-amendment.md).**

# Proposed response to the adversarial LumoTree review

27 September 2026. Status: execution approved by the author; campaign initialized under `experiments/review-response-20260927/` with independent scientific gates. The original planning pass launched no GPU experiments. The planning-time manuscript was `COMPLETE_DESIGN_DISTINCTION`; its receipt is preserved under `p0/monitor/review-response-20260927/pre-p0-manuscript/`. The current receipt is `artifacts/FINAL-DELIVERY.json`, status `P0_INTEGRATED_CAMPAIGN_ACTIVE`. This plan supplements, rather than silently replaces, `review-experiments.md`.

The principal gap is evidence linking the implemented design to an advantage. Further prose alone cannot establish that link. Work proceeds from evidence recovery to current-route qualification, mechanism attribution, and then workload confirmation. A claimed gain is removed or narrowed if its corresponding experiment does not support it.

## 1. Triage the review accurately

| Review issue | Assessment and response |
| --- | --- |
| Maximum of three tree revisions on two tasks in the abstract | Valid selection/generalization concern. Treat these as development observations, not replicate estimates or the final confirmatory headline. Any future headline uses the frozen confirmation rule and all attempts. |
| Mean of three rates and spread prove the gain is noise | Unsupported statistical conclusion: these are different revisions and there is one comparator run. Neither a mean of revisions nor their spread estimates within-build uncertainty. Repeated fixed-build task/seed blocks are needed; the arithmetic mean also differs from the paper's pooled estimator. |
| Four-task Sr12 or ten-task Cqc10 below two-task SGLang proves regression | Different task/context populations do not establish a regression or reverse the matched-pair ranking. They do show that the two-task maximum is narrow. A revision bridge on identical recorded inputs is needed if explaining code regression. |
| Engine and configuration underdescribed | Valid. Name patched vLLM explicitly and bind version/commit, model and tokenizer hashes, quantization, attention backend, cache/graph modes, caps, sampling, concurrency and drafter configuration for every arm. |
| No same-stack baseline | Valid for the current reported workload. Native/tree campaigns exist, including a completed older 48-attempt comparison; their route and measurement limitations prevent rebranding them as the missing current-build control. Audit applicability first. |
| SGLang failed because its cap was 24,000 | The excluded earlier pair had a 32,768 cap; the eligible later pair had 24,000. Saved counters put all headline Sr12 and eligible SGLang responses at or below 20,000. The claimed cap-asymmetry cause is not established. |
| Excluded SGLang rate omitted | Valid reporting concern. Saved unfiltered pooled rate is 30.7004947 tok/s. Preserve the author's patch-producing performance comparison, and disclose the excluded rate and failure in a separate all-attempt outcome/diagnostic table. Do not promote its capped thinking-only outcome into a successful-agent speed claim. |
| Tree task times absent from shared-task comparison | Recoverable now. Sr12 pair totals 1038.755 seconds (17.31 minutes); eligible SGLang totals 2043.9 seconds (34.07 minutes). Report per-task times and verdicts, including failed tasks, rather than turning this retrospective pair into a general task-speedup claim. |
| Different generated trajectories invalidate every comparison | Different trajectories are legitimate outcomes of stochastic agents. They prevent attribution of pooled decode differences solely to a kernel. Use full task outcomes/time for system competition and fixed-input replay for causal kernel analysis. |
| No speculative metrics | The paper omits them, but current-route logs and metric snapshots contain draft/acceptance counters. Recover their semantics and aligned populations before requesting new runs. Physical padded slots appear in draft accounting; do not label that fraction as logical-candidate acceptance without verification. |
| Core continuation behavior lacks current-route end-to-end qualification | Valid high-priority gap. Test state publication and next-forward outputs on current binaries; old Cat10 findings cannot substitute. |
| Memory equation unimplemented | The equation describes logical cut-state export bytes, not allocated scratch. The distinction already appears in the paper. Measure both and show actual traffic, peak scratch, registers and spills before claiming a memory advantage. |
| Bole and TreeWY were both executed | Incorrect: TreeWY and Weaver author components were executed; Bole author code was not verified or run. Smoke/component execution is not a matched result. |
| Prior-art overlap is substantial | Valid. Resident paths and replay individually have direct precedents. Compare complete verification/commit policies and identify a measured tradeoff; more workload examples alone do not establish novelty. |
| 4/32 padding is 12.5% runtime waste | 12.5% is the inactive-row fraction, not a measured time or whole-step work fraction. Tensor-core geometry, masking, graph shape and non-row-proportional costs matter. Measure a feasible geometry control. |
| IEEEtran and author identity fail MLSys format | Relevant to a conference submission, not an arXiv preprint. Maintain separate named preprint and anonymous conference builds if MLSys is chosen. |
| Single hardware/model/B1 and broad title | Bound the supported deployment scope and use Gated DeltaNet hybrid terminology explicitly. Additional hardware/concurrency is conditional on the claim; it is lower priority than qualification and attribution. |

## 2. P0: recover evidence and repair presentation without inference

Deliver a source-bound evidence matrix for each reviewer finding: resolved by existing data, requires fresh measurement, or requires narrower claim. Use the current deployment identities and preserve superseded records only as provenance.

- Add the complete configuration table, including the serving engine. Check whether checkpoint conversions/weight bytes actually match; a common display name is insufficient.
- Present every relevant development attempt with date/build, tasks, outcomes, tokens, elapsed agent time, pooled rate and exclusion reason. Keep the failed SGLang run visible in the failure table, with the outcome-conditioned table separately labeled. Rename aliases in displayed tables to dates/builds.
- Recover acceptance/draft-step totals and per-depth/path histograms from pre/post task brackets. Validate whether counters count root/bonus tokens, logical candidates, physical padded slots, or decode attempts. Derive rates from totals, not averages of periodic console means.
- Recover any source-bound draft, verify, acceptance, replay and attention timings already archived. Identify population, warmup, synchronization and instrumentation overhead. Do not reuse old global-token/subset-wall proxies.
- Recover known degeneration, empty patches, incomplete evaluations and budget stops into an explicit outcome table. A trace detector is a diagnostic; task evaluation and completion remain primary outcomes.
- Separate current-build evidence from old native/DSpark/DFlash campaigns before deciding which experiments are genuinely absent.
- Proposed abstract change: lead with the mechanism and replace the max-of-development-runs headline with a fixed-build confirmation result when available. Pending that result, describe the existing shared-task observations as exploratory and make selection scope explicit. This plan does not silently overturn the author's earlier best-number instruction.
- Put algorithm/design explanations in the paper itself; use repository citations for reproducibility, not as substitutes for describing the method. Move author-code smoke details to artifact documentation unless they support a substantive comparison.

Exit condition: every retained quantitative claim has a defined population, source identity and estimator. Missing telemetry is explicitly marked; no inferred acceptance or phase timing is invented.

## 3. E1: current-route continuation qualification before timing

Freeze the deployed path verifier, patched FA2, native committer, model/tokenizer and precision policy. Use current-route captured operands and recorded agent prefixes, with synthetic stress cases labeled separately.

Check an independent sequential recurrence and the actual native sequential implementation as distinct references. Compare candidate outputs, recurrent state, convolution history, KV mapping, next drafter row, and logits after consuming the published continuation. Verifier/committer state differences require investigation; agreement is not assumed from the real-arithmetic formula.

Cover root-only acceptance, long spine, off-spine acceptance, all valid accepted lengths, correction/bonus pending states, padding, physical tile boundaries, repeated continuation, graph replay, cache reuse and request lifecycle transitions. Add deliberately corrupted parent/slot maps as negative controls. Trace the common input and path at each layer to identify the first divergence.

Freeze numerical criteria before timed confirmation, with scales and native repeatability documented. Report exact equality separately from bounded error. Use forced common paths to isolate numerical effects; greedy token agreement and stochastic sampler checks answer additional, separate questions. Equal seeds alone do not require identical stochastic trajectories across stacks.

Exit condition: structural failures and nonfinite values are absent, and predefined numerical/state criteria pass. Otherwise fix the route and rerun qualification before any speed comparison.

## 4. E2: measure the mechanism, including all accepted-state work

Use pinned LumoTree, TreeWY and Weaver components on common shapes, pre-states and inputs. Validate adapters, normalization, casts, padding and consumed-state semantics before timing. Where fusion consumes different representations, include the required conversion/preparation in the measured cycle.

The comparison is verify plus publication, with TreeWY's prior-commit/next-verify accounting and a final deferred flush. Include metadata/stash writes, temporary state handoffs and accepted replay. Run current geometry plus predefined chain and branch controls; vary accepted path length and representative dimensions. B1 is primary; B4 can be a component sensitivity result without implying concurrent-agent performance. Author implementations and local ports must be identified separately. Add Bole/SpecLA execution only after verifying usable author code.

Required measurements: output/accepted-state error; complete-cycle latency and stage breakdown; logical export bytes, allocated and peak scratch; register/shared-memory use and spills; and warmed timing distributions with ordered repeats. Same-input component latency is a mechanism result, not a SWE tokens/s headline.

Ablations target mechanisms, not every table row as an independent invention:

1. Path execution and native replay versus a qualified compact verification/commit policy; if isolating scheduling, use the same recurrence with a dependency-respecting sequential schedule.
2. Spine-first versus another consistent physical permutation, with writes, masks and publication changed together; check numerical/next-forward behavior before timing.
3. GPU acceptance/publication versus a semantically matching host-prepared path; include synchronization cost. Never disable an invariant as an ablation.
4. Fused convolution/preparation and fused selection versus existing reference operations, with parity checks. Measure separately first; combine only when studying interactions.
5. Attention grouping and split-K over predetermined context lengths; test a small split set (for example 1/2/4/8 where supported). Tune on separate data and account for numerical changes.
6. Padding sensitivity only for supported geometries with the same logical candidates. Report inactive rows directly if a clean shape-only control is unavailable; do not invent a runtime penalty.

Existing tests and ablations are reused when they bind the executed route. These measurements can share one captured-input suite. Only mechanisms with an identifiable component effect need expensive task-level ablations.

Exit condition: identify a concrete supported tradeoff or advantage, including its losing cases. If the complete-cycle advantage disappears, narrow the optimization claim and retain a design/compatibility result rather than searching task subsets for a win.

## 5. E3: fair system competition plus same-stack attribution

Use four arms: patched-vLLM native autoregressive; same-stack chain MTP; LumoTree; and SGLang EAGLE. The same-stack controls isolate speculation and tree policy. The cross-stack arm measures complete systems. Use the exact same target checkpoint/tokenizer and precision where feasible; disclose unsupported differences. Keep shared stack optimizations common where possible, and explicitly account for tree-specific attention or state paths that cannot be identical.

Fix hardware allocation, agent harness/tools/network, task/evaluator images, prompt, sampling, task time budgets, response/compaction caps and maximum context. Match supported cache policy and record actual modes. Each method may choose its own depth, candidate budget and kernels under an equal documented tuning budget on disjoint tuning tasks. Equal MTP length is not a prerequisite for fair system competition.

Proposed sequence:

- Qualification/runtime pilot: two predetermined tasks times four arms = eight task attempts, separate from confirmation. Check telemetry overhead, evaluator completion and route engagement. Use this to estimate GPU-hours and choose a feasible fixed budget.
- Confirmation candidate for author review: 16 predetermined tasks spanning at least four repositories, three fixed seed blocks, four arms = 192 attempts. Select tasks without seeing arm outcomes, balance run order, use one qualified build per arm and consistent cache/reset rules. This is a scoped multi-repository study, not enough by itself to establish broad quality equivalence.
- Freeze sample count and stopping rules before confirmation. A smaller pilot-informed budget is legitimate if the claim is narrowed; do not expand or stop based on whether the observed speedup is favorable. Fresh task outcomes stay visible regardless of patch production.

Report all-attempt resolved/failed/incomplete/timeout/degenerate counts, per-task agent time and budget censoring, pooled decode rates, context-length distributions, acceptance, draft/verify/commit timing and memory. Show paired task/seed differences and uncertainty without treating requests as independent benchmark replicates. Rates are pooled from token/time totals. Success-conditional latency is a secondary table with its denominator, never the only outcome.

Different trajectories are expected in real agent competition. For attribution, replay fixed captured requests and token paths through all compatible arms, labeled as controlled diagnostic work rather than a replacement workload benchmark.

Exit condition: the final headline follows the frozen protocol and has corresponding outcome evidence. If tree does not beat same-stack controls, report the result and scope the claim accordingly. Do not substitute the earlier max-of-three observation.

## 6. Paper integration, venue and final review

The main argument should connect a concrete mechanism difference to qualification, complete-cycle tradeoffs, and agent workload outcomes. Retain useful adverse results. Describe only the executed build; keep internal build aliases in the artifact.

Keep the current named arXiv build. If submitting to MLSys 2027, create a separate official-style anonymous build, anonymize identifying artifact links appropriately, and use the official page limits and appendix rules. Official source checked: https://mlsys.org/Conferences/2027/CallForResearchPapers (double-blind research submissions, official style, 10 main pages excluding references, separate appendix). Single authorship itself is not a format violation.

A second model or concurrent-agent setting is a follow-on only if the claim needs that breadth. A GB10-focused single-agent paper can be scoped honestly; added hardware does not repair a missing mechanism comparison.

Run independent claim/evidence review after substantive results, then compile/render and verify the manuscript and source bundle. Approval gates are scientific: unsupported claims are removed or narrowed, and failing methods are fixed before timing. No promise of venue acceptance follows from this checklist.

## Live checks behind this plan

- `results/agent-workload/competitive-rate-audit.json`: excluded and eligible rates, per-task times and totals.
- `results/agent-workload/patch-producing-rate-audit.json`: retrospective selection and exact eligibility records.
- `artifacts/FINAL-DELIVERY.json`: earlier SGLang cap 32768; later cap 24000; executed author components and current paper identities.
- `results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/metrics_after_swe.txt`: speculative counters exist (raw totals require pre/post population validation).
- `notes/comparator-git-history-2026-09-23.md`: prior native, SGLang, DSpark and DFlash experiments and applicability.
- `review-experiments.md`: existing N1/N2/A1 plans; this response prioritizes rather than erases them.


## 27 September execution clarification: full-model decision criterion

Before observing any candidate outcomes, the parent adopted a narrower operational criterion for the full-model target-decision portion of Q1: exact raw next-forward greedy-token agreement after both routes consume the same pending token, with zero disagreements over the frozen cases and cycles. Native repeatability, stable tie handling, finite values, complete coverage, exact structural checks, and separately qualified state/MTP/lifecycle requirements remain mandatory. Centered logit error, TV/KL and top-k differences are diagnostic; no ungrounded global logit tolerance is inferred. This prospective criterion supports tested forced-path greedy-decision consistency, not distribution preservation or task-quality equivalence. The source-bound rationale and native-first order are in `notes/review-response-20260927/q1-fullmodel-reference-design-review.md`; exact terms are in `p0/monitor/review-response-20260927/Q1-OPERATIONAL-DECISION-CONTRACT.json`. This is a design decision, not a passed gate or GPU authorization.


## 27 September execution clarification: paired component numerical criteria

Before any candidate output was observed, independent review found that two-sample calibration maxima and output-scale intervals could not establish useful population coverage. The component policy therefore prospectively uses matched-input per-head, per-depth RMS and maximum-error non-regression against the named native operator, each with multiplier 1.10 and an fp32 arithmetic floor. Calibration maxima remain reported diagnostics. This tests numerical non-regression on the frozen finite corpus; it is not an absolute error theorem. Exact source/coverage/structural/finite checks and powered calibration-only negatives remain mandatory. See `p0/monitor/review-response-20260927/Q1-PAIRED-NUMERICAL-CONTRACT-v2.json`; prior versions and failures remain preserved. No GPU launch is approved by this clarification.


## 30 September final scientific disposition

The original and kernel-pinned native-reference cohorts each completed 336 observations on 84 paths. Independent raw/provenance and categorical review found target greedy agreement on 84/84 but joint target/MTP categorical agreement on only 68/84 in each cohort. The unchanged prerequisite is not met; no candidate subset is admitted. Full-model, continuous, held-out and lifecycle claims remain unqualified. The complete untimed mechanism comparison is retained with all adverse results; timed cells remain zero. The four authorized case-study attempts remain **0/4, unexecuted**, without task verdicts or replacement attempts. Q1/WP remain closed and WC remains superseded. This is supported narrowing under the original claim-removal rule, not a successful qualification or a waiver. See `notes/review-response-20260927/FINAL-CLAIM-DISPOSITION-20260930.md` and `p0/monitor/review-response-20260927/CAMPAIGN-SCIENTIFIC-DISPOSITION-20260930.json`. Delivery verification is separately recorded in `artifacts/FINAL-DELIVERY.json`.
