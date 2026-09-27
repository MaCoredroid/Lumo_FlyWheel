# Recovered SGLang SWE evidence — 2026-09-23

**SGLang was tested on real SWE-bench tasks.** Commit `c27f30207c8a136efb9f5f1f46c883fe767c584f` banks `results/fr14_nvfp4_port_20260816/sglang16_evidence/{run1,r1}/`. Two task attempts have complete original metadata, evaluator and pre/post metric records. This is neither a never-tested comparator nor a completed sixteen-task trial.

## Completed task records

Paths below are relative to `results/fr14_nvfp4_port_20260816/sglang16_evidence/`; each task lives under `swe_out/verified/per_task/astropy__astropy-<ID>/`.

| Arm / task | Saved evaluation | Agent elapsed s | Evaluator elapsed s | Completed request delta | Generation-token delta |
|---|---|---:|---:|---:|---:|
| run1 / 12907 | Resolved | 426.131 | 53.091 | 18 | 8,314 |
| r1 / 13033 | Failed tests | 1,617.769 | 52.559 | 27 | 39,540 |

Read `runner_metadata.json:agent.elapsed_s` and `eval.elapsed_s` separately. The commit/REDTEAM's 480.5/1,671.5-second values are task-boundary totals, not the agent-only field used for Cqc10. Both evaluator reports match the copy in runner metadata, use SWE-bench Verified and the same recorded x86 evaluator worker hash as Cqc10. A previous partial 13033 in run1 and partial 13236 in r1 have traces but no complete metadata/evaluator/post bracket; they are not completed outcomes.

## What the saved metric brackets permit

Use the per-task files named `vllm_metrics_pre.txt` and `vllm_metrics_post.txt`: their contents are SGLang metrics. Subtract each exact labeled series before pooling labels. Non-streaming deltas are zero; streaming completed requests are 18 and 27. The saved inter-token histograms yield:

| Task | Delta ITL sum (s) | Delta ITL count | Mean ITL (ms) | Inverse mean (/s) |
|---|---:|---:|---:|---:|
| 12907 | 292.3417797149159 | 8,296 | 35.2388837651 | 28.3777433663 |
| 13033 | 1,484.7225807243958 | 39,513 | 37.5755468004 | 26.6130525076 |

The keys are `sglang:inter_token_latency_seconds_{sum,count}`. Counts equal generated tokens minus requests in both brackets. **These are means over the saved inter-token histogram support, not Cqc10's request-weighted mean TPOT.** The corresponding vLLM request-mean-TPOT metric is absent, and both proxy `vllm_request_metrics.jsonl` files are empty. Per-request TPOT cannot be reconstructed from aggregate request E2E/TTFT sums without per-request token/latency pairings. Do not divide Cqc10's 28.197 request-inverse-mean statistic by these numbers. The run-level post bracket also includes work on the interrupted next task and must not replace the completed-task brackets.

This is a narrower conclusion than the old note's assertion that surviving Prometheus brackets cannot reconstruct any decode statistic: inter-token histograms do survive and can be reduced, but they do not recover the missing decode-batch logs or request-weighted TPOT.

**Cross-check of every retained metric family:** all ten SGLang metric snapshots have no request-weighted TPOT histogram. All twenty Cqc10 task snapshots have both request-TPOT and ITL histograms. Cqc10's ITL sums reduce to 9,677.884510502685 seconds over 49,068 observations, mean 197.23413447669938 ms. In each task its ITL count equals the recorded decode-step attempt count; it does **not** equal generated tokens minus requests (248,077 over the cohort). SGLang's ITL counts do equal that token-interval quantity in both completed tasks. Thus the matching ITL metric names also do not establish comparable observation units. These descriptive reductions remain audit-only; no same-estimator speed comparison is justified even before accounting for the disjoint task cohorts.

## Settings and comparison scope

Both recorded command lines use `/models/qwen3.8-27b-nvfp4-radixark-asshipped`, SGLang FlashInfer, EAGLE steps/top-k/draft-tokens 3/1/4, chunked prefill 8,192, memory fraction 0.70, Qwen3 reasoning and Qwen3-coder tool parsers. `models.json` advertises context 262,144. The source launcher names image `lmsysorg/sglang:qwen38-27b`; these retained files do not bind an image digest or all loaded source/weight bytes. An engine seed is not explicit in the saved command. The launch source sets concurrency one and a 9,000-second agent wall limit; the task metadata's campaign-budget field is null, so do not present that field as a matching runtime receipt.

The saved proxy settings agree with Cqc10 on temperature 0.6, top-p 0.95, top-k 20, minimum-p zero, presence penalty 1, 24,000 output cap and disabled auto-continuation. Metadata names Qwen Code 0.19.4, repository-image agent execution, the same network-policy fingerprint and evaluator worker SHA. Unlike Cqc10, it does not retain the full content-addressed agent bundle proof. Model display names alone do not establish identical checkpoint bytes across the as-shipped view and the vLLM port.

Cqc10 uses ten different task IDs, Hydra27/32 physical rows, patched FA2 split-K4, max context 131,072 and a 4,096-token budget. Thus these two SGLang observations cannot supply a paired speed comparison against Cqc10. Earlier tree records for 12907/13033 do exist and agree in saved verdict, but their source/cap lineage differs (including the earlier 32,768 cap); reporting those as an as-executed historical comparison requires that context. Two matching verdicts do not establish quality equivalence. Earlier SGLang four-task/calibration records are separate evidence, not this trial's missing fourteen results.

## Remainder and history

`REDTEAM_20260816.md:6675–6696` records the first boundary preemption. Lines 6792–6814 record the second completion, interrupted 13236 and loss of the final container decode-batch logs. Pass 238 at lines 7050 onward explicitly says **“sglang r2 is DROPPED”** and closes the scorecard at two completed verdicts. That disposition was introduced in commit `5cfa282e54e0e8a69d9eda9fb590b4587302915d`; pass 245 around line 7269 confirms it at pause.

Available `git log --all` searches over SGLang commit messages, evidence paths and the `sglang r2` history found no later completed remainder; `sglang16_evidence` has only the banking commit. This establishes the recoverable repository state, not a claim about untracked or unavailable external records.

## Durable reduction and recommendation

`notes/sglang16-metric-audit-2026-09-23.json` binds each task's evaluator/metadata/bracket hashes, records every relevant metric delta, settings-receipt hashes and the history disposition. Source raw files remain unchanged in Git; no large trace or DCGM copy is needed. The REDTEAM source SHA is `902814907c5642ce917fbda8f7ca955b3d859d4caae2d22570b448c4121d2736`; both command-line receipts have SHA `eb17eb2b7bf3afeeb7fdb8992160c7935b7b9112e937b20e0ad482b5b3da2aad`.

Accurate present statement: **SGLang was tested on SWE-bench agent workloads, including two completed attempts in a later, interrupted sixteen-task comparison; a completed matched comparison against the reported Cqc10 cohort remains unavailable.** Retain recovered historical numbers in this audit under the user's manuscript scope; do not silently replace current workload results with old calibration rates. No inference was run for this review.

The new “Existing comparator experiments” paragraph correctly acknowledges executed SGLang experiments and avoids adding their rates to the current result. Its native wording needed one clarification: both an earlier pre-NVFP4 campaign and later targeted native-MTP5 NVFP4 probes exist. The parent applied this clarification and the final paragraph names both accurately; this does not supply a complete matched Cqc10 comparator. DFlash/DSpark evidence is the other reviewer's independent scope. The abstract no longer implies comparators were never tested.

Final source recheck: `abstract.tex` SHA `df87d7169f48c42173a78720460595950da93b1e48a61b76d74933977332f220`; `results/agent-workload/case-study.tex` SHA `f3a344ae2dc51e4cdd3a1137f866e7429876bfbca11270da4a2a7cb4e36a9d35`. The complete metric-audit JSON, including the twenty Cqc10 bracket hashes and support cross-check, has SHA `6deeffa4ca83f4a864eb1a743f0272d14cb6557fbd1ad2809d9805a871ec4adb`.
