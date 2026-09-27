# LumoTree integration and author-code experiments — 26 September 2026

The user approved the LumoTree framing and requested use of available author code. This update supersedes earlier statements that no author-code runs have been launched. Existing SWE workload rates and current production implementation are unchanged.

The [mechanism memo](notes/novelty/mechanism-comparison-and-naming-2026-09-26.md) now feeds the background, state-policy comparison table, path-tile method description, and the experiment design below.

| Method | Verified code status | Executed here | Next use |
| --- | --- | --- | --- |
| TreeWY | Author RFC; pinned `b073ed6c` | 39 selected author tests pass on GB10 | Direct same-input output/accepted-state comparison; count fused prior commit and next verify together |
| Weaver | Author SGLang fork; pinned `aeac03f0` | 9 numerical cases and 2 verifier-timing configurations completed; script has no correctness threshold | Direct comparison of pinned fused/chunk verifier and actual replay commit; select route explicitly |
| SpecLA | No author repository verified in the bounded search | Not run | Scheduling/state-lifetime comparison; executable arm conditional on verified code |
| Bole | No author kernel verified in the bounded search | Not run | Finite-Neumann family comparison; author-system execution conditional on verified code |
| FastTree | [Author artifact](https://github.com/PanZaifeng/FastTree-Artifact) verified | Not run | Attention-only comparator if the shared-context/tree-mask interface matches; not a GDN baseline |
| OneLA | No author repository verified in the bounded search | Not run | Adjacent beam-search state-sharing background; not interchangeable with speculative acceptance |

Exact sources, import-only adapters, raw outputs and scope: [author-code README](experiments/author-code-20260926/README.md), [receipt](experiments/author-code-20260926/RUN-RECEIPT.json). Both bounded runs are complete; no model inference was launched. A search failure means unverified availability, not a claim that code does not exist.

## N2 — matched mechanism experiment to prepare next

1. Bind the active two-level LumoTree path kernel and native replay source to the production receipt; do not reuse obsolete Cat10 kernels. Freeze author code and all adapters. Use current topology plus chain/branch controls, B1 and B4, the actual head dimensions, common pre-state/rounded operands, and three predefined synthetic seeds. Mark synthetic inputs separately from any newly captured task operands.
2. First compare every valid output and the consumed accepted continuation against an independent sequential reference, including root-only and off-spine paths, padding, and repeated continuations. Record max/RMS error, finite checks and cast policy. Freeze acceptable numerical criteria before timed confirmation; do not conflate approximation with an intrinsic mathematical defect.
3. Measure complete verify + accepted-state work, including metadata/stash writes and the final deferred flush. TreeWY fuses prior commit into next verify; Weaver's served route replays accepted operands. Excluding either would bias the comparison. Record graph/eager mode and precision, warmup, ordered repeats, CUDA events and physical wall intervals.
4. Report source-level working tiles separately from compiler registers/spills, allocated scratch separately from written cut-state bytes, and total state traffic separately from peak resident memory. A scheduling advantage requires a matched policy ablation; SpecLA already shares the broad scheduling idea.
5. Keep component timing in a mechanism study, never convert it to an agent tok/s headline. Only a qualified route proceeds to a predeclared same-task SWE-bench Verified comparison with equal resource/tuning opportunity, pooled decode rate and all task outcomes.

N1 (layout/continuation attribution) and A1 (matched workload confirmation) below remain the other relevant experiments. No wholesale full-stack port or long task campaign was launched by these bounded code checks.

---

# Novelty-review priorities — 24 September 2026

The completed [novelty review](notes/novelty/novelty-review-2026-09-24.md) found direct path-scheduling overlap with SpecLA and accepted-path replay overlap with Trees from Marginals, as well as established attention and drafting primitives. Merely increasing the SWE sample count cannot establish mechanism novelty.

No experiment is launched by this review. Before any new campaign, inspect existing evidence for exact applicability to the deployed build. If stronger novelty is desired, prioritize:

1. **N1: current-route layout/continuation witness.** Identical logical candidates and captured operands, consistently permuted KV and mask columns, fixed query order, and next-forward state checks. Include off-spine paths, padding, tile boundaries and cache reuse. The current attention gate does not isolate this effect.
2. **N2: closest-mechanism contrast, conditional on claiming a schedule advantage.** Compare current path groups/native replay with SpecLA-style chain decomposition and an available source-pinned compact verifier (Weaver or TreeWY), including publication and temporary state. This diagnoses tradeoffs; component timing is not a paper workload-speed headline. Author-system and faithful local-port results must remain distinct.
3. **N3: workload confirmation only after choosing the claim.** Use A1 below with the relevant qualified comparator or one component toggle. Keep competitive per-method tuning/depths, all outcomes, complete task timing and the same pooled decode metric. A historical completed run is not a current matching ablation.

N2 is not a prerequisite for N1 or its workload ablation. N1/N2 are conditional tests of the narrower proposed contribution, not prerequisites for honestly describing an implementation case study. Native/tree workload comparison alone cannot show that a layout or schedule caused the difference. Bole/TreeWY are no longer the only direct recurrent-verifier neighbors. Bole execution depends on obtaining its implementation. Existing DSpark/DFlash experiments remain recovered history, not automatically suitable evidence for the latest route.

# Current-method correction — 24 September 2026

Reuse the verified production evidence first. The paper now follows Hydra27/fixed32, forked FA2, fused full-vocabulary selection, and captured native replay. September Cat10 diagnostics are archived and are not a reason to repeat already completed production work. The latest current-design audits and `results/current-production/audit.json` supersede the prior qualification framing. No new experiment was launched for this correction.

# Agent-workload experiment proposal for review

## Existing comparisons recovered from Git (23 September correction)

SGLang EAGLE was run through the real SWE-bench Verified harness (commits `128d27b86`, `c27f30207`); DSpark had actual synthetic serving and captured-input experiments (`6530b1f17`, `c0d5550f4`); DFlash2 has a later served smoke on the upstream branch (`92fea914a`); tree/native5/native11 had a complete 48-task-run campaign (`51b7dafdf`). These must not be described as untested alternatives. See `notes/comparator-git-history-2026-09-23.md` for workload, metric, and supersession boundaries.

The experiment proposal below is conditional on the desired current-system claim. Reuse audited applicable records first; the 96-attempt design is not an automatic next step or authorization to repeat completed experiments. No new inference was launched in the history audit.

23 September 2026. The paper targets long-running coding agents. All reported performance must come from named full tasks with outcomes and budgets. This proposal has **not been launched**. The earlier fixed-prefix plan and completed E1/E8 evidence remain audit-only in dated snapshots; superseded Cat10 numerical campaigns remain audit-only. Current component checks are bound to the production FA2 and fused-selection binaries.

Among recorded runs producing nonempty patches on both shared SWE-bench Verified Astropy tasks (12907 and 13033), the best tree run reaches **29.09 pooled tokens/s** versus **26.89 for SGLang EAGLE**, an **8.18% higher rate**. This retrospective subset requires nonempty patches from both tasks for both methods; failed tests remain eligible. One SGLang pair with an empty patch and capped thinking-only response is excluded from the performance comparison, with its failed outcome and raw records retained. The selection ledger is `patch-producing-rate-audit.json`.

## Comparison policy — 24 September clarification

The main comparison is between complete speculative-decoding methods. Their serving stacks, tree topology, MTP depth and draft-token budgets are allowed to differ: these are algorithm and implementation choices whose cost is included in the measured rate. Fair competition fixes the target-model workload, hardware/resource allocation, task harness, requested sampling, quality evaluation and task budgets, and gives each method a documented, comparable opportunity to tune. Equal draft length or candidate count is useful for a separate mechanism ablation; it is not a prerequisite for comparing each method at its competitive configuration.

The current recorded EAGLE comparator uses 3 steps, top-k 1 and 4 draft tokens. No completed shared-task sweep over draft settings is established by the current evidence. Future confirmation should predeclare the tuning budget on separate tasks, freeze the chosen setting for each method, and evaluate all methods on the same confirmation tasks. Predeclare performance eligibility for confirmation and apply it symmetrically; retain all failures in the outcome denominator. The current paper explicitly labels its nonempty-patch filter as retrospective. The cap check finds all 41 headline Sr12 responses at most 20,000 tokens, below either configured cap; keep the actual cap values in the setup rather than a blanket abstract caveat.

## A1 — Matched current native/tree SWE-bench Verified tasks

**Question:** does the final tree configuration reduce agent completion time while preserving useful task outcomes on a declared workload?

1. Freeze one current source/build, exact Qwen3.8-27B NVFP4 weight/tokenizer identity, Qwen Code harness and prompt, evaluator/task images, tool permissions, network policy, task and response budgets, sampling and seeds. Record all active optimizations, physical topology, graph/cache settings, extension hashes, CPU/GPU contention and memory state. Verify comparator compatibility before freezing. Use the same cache policy where both implementations support it; disclose necessary backend differences and avoid claiming an isolated topology effect.
2. Use native MTP-5 as the main comparator and the chosen production tree. Keep the sixteen declared Astropy tasks, including those with earlier incomplete/degenerate attempts; do not select only the ten that previously completed. This is a narrow known-task workload, not a representative benchmark-quality test. B1 is the primary single-agent condition. Native MTP-11 is optional only if a longer-chain claim remains in the paper.
3. Run a small qualification pilot on two predeclared tasks, both arms, to check task/evaluator completion, accounting, reset policy, and actual duration. Choose tasks before outputs. Keep pilot results separate; do not substitute their measurements into confirmation. Freeze the final run manifest after resolving implementation defects.
4. Recommended confirmation: two arms × sixteen tasks × three predeclared seed blocks = **96 full task attempts**, ordered in balanced paired blocks. A task/seed pair is the comparison unit; requests are nested observations, not independent task replicates. Three seeds characterize observed stochastic variation but do not automatically power a quality-equivalence claim. An initial one-seed 32-attempt tranche can establish descriptive paired results; any expansion/stopping rule must be fixed before reading confirmation outcomes.
5. Preserve every attempt, crash, timeout, degeneration, empty patch, evaluation error and rerun. Restore the same repository image and declared serving/cache reset boundary per pair. Treat infrastructure failures separately from agent/test failures; do not silently replace unsuccessful attempts.

**Behavior and comparison headline:** keep the agent harness and task workflow fixed across decoder arms. Report detected degeneration, malformed tool calls, incomplete runs, task outcomes and relative task/service speed together. A run terminating normally is not proof of unchanged task quality. Inspect trace-based failure checks in addition to output/timeout flags; record known detector blind spots. The current abstract already reports sustained agent execution and descriptive pooled decode rates against SGLang. This proposed campaign would add a paired task-completion comparison.

**Primary results:** evaluator verdict per attempt, resolved/failed/incomplete counts over all intended tasks, per-task agent elapsed time, budget-censored outcomes, and paired distributions. Report resolved-pair latency with its restricted denominator and all other outcomes alongside; never call a fast failed or timed-out attempt a success-speed gain. An overall metric such as resolved tasks per allocated agent-hour must state how every failed/capped attempt contributes.

**Secondary service measurements:** requests and output tokens (visible and compaction separately), model-service time, TTFT, tool/evaluator time, and physical decode timing only with matched token/interval support. Use the same pooled decode estimator in both arms: `(sum output tokens - completed request count) / (sum E2E request latency - sum TTFT)`. Verify completed-request populations, counter resets, outstanding requests at bracket boundaries, and timing semantics. Report native/tree task results jointly with service metrics; never multiply gains from different workloads.

**Before launch:** finish compatible-route qualification, freeze the manifest, and review the pilot-derived runtime estimate. No reliable GPU-hour estimate is inferred from old snippet runs. The historical nine-thousand-second task cap gives only a budget ceiling; it is not an expected duration.

## A2 — Targeted optimization attribution, conditional

Only after A1 yields an interpretable matched result, test the smallest mechanism that needs attribution (for example draft-logit reuse or attention split-K). Use the same named full tasks/harness and change one compatible component at a time. Recheck active binary/launch engagement and state/sampler qualification. Reuse A1 as a control only if the frozen design permits that reuse and the configuration is identical. No old E8 percentage or partial control run substitutes for this experiment.

## Deferred unless the paper claims them

- B4 or concurrent agents: add matched task arrival/concurrency experiments only for a multi-agent service claim; B1 token rates cannot establish it.
- Broader task-quality preservation: add held-out repositories/tasks and a separately justified sample-size/equivalence design; the known Astropy cohort cannot establish it.
- Author-system comparisons: execute matched Bole/TreeWY systems only if a competitive performance claim is added. Local same-input arithmetic studies remain distinct.

This scope uses current production-bound component checks and makes any next experiment answer the agentic application claim directly.
