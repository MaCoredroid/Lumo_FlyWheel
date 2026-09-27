# Agent-workload experiment proposal for review

## Existing comparisons recovered from Git (23 September correction)

SGLang EAGLE was run through the real SWE-bench Verified harness (commits `128d27b86`, `c27f30207`); DSpark had actual synthetic serving and captured-input experiments (`6530b1f17`, `c0d5550f4`); DFlash2 has a later served smoke on the upstream branch (`92fea914a`); tree/native5/native11 had a complete 48-task-run campaign (`51b7dafdf`). These must not be described as untested alternatives. See `notes/comparator-git-history-2026-09-23.md` for workload, metric, and supersession boundaries.

The experiment proposal below is conditional on the desired current-system claim. Reuse audited applicable records first; the 96-attempt design is not an automatic next step or authorization to repeat completed experiments. No new inference was launched in the history audit.

23 September 2026. The paper targets long-running coding agents. All reported performance must come from named full tasks with outcomes and budgets. This proposal has **not been launched**. The earlier fixed-prefix plan and completed E1/E8 evidence remain audit-only in dated snapshots; numerical qualification remains useful for mechanism checks.

The existing records now support a common pooled decode comparison: 25.63 tokens/s for the ten-task Cqc10 tree deployment and 26.89 for the two-task SGLang EAGLE deployment, on different SWE-bench Verified Astropy subsets. The calculation pools `(output tokens - completed requests) / (E2E request latency - TTFT)` over all included requests. This descriptive comparison is complete without new inference. A matched campaign is conditional on claiming a causal mechanism effect or task-completion speedup; it is not required to report the existing cross-stack observation.

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

This scope keeps existing numerical checks as supporting evidence and makes the next performance experiment answer the paper's agentic application claim directly.
