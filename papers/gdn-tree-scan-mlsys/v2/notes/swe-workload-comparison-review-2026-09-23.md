# SWE-bench workload comparison: independent evidence review

2026-09-23. Reviewer: paper_redteam_round1. Read-only local audit plus bounded read-only SSH reads from DGX; no inference, GPU, container, model/repository mutation, or evaluation rerun.

**Decision:** existing records support a dated, task-level agent-workload case study on selected Astropy tasks with the Qwen3.8 NVFP4 port. They do **not** support a fresh September experiment, a homogeneous sixteen-task benchmark score, a matched sixteen-task native/tree comparison, or a causal native/tree task-speed/quality advantage. The correct assembled lineage has **8 evaluator-confirmed resolutions, 5 evaluated test failures, and 3 unevaluated outcomes among 16 intended tasks**. Do not promote 9/16 or change the denominator to 13.

## What was actually measured

`config/fr13_fixed32/subset_b4_sixteen.json` names `princeton-nlp/SWE-bench_Verified`, split `test`, and sixteen `astropy/astropy` instances. Its own selection rationale is reuse of an earlier native task list; this is a selected single-repository subset, not the complete benchmark or a representative random sample. The filename says B4, but the reviewed QC and native probe runs are B1. The executed tasks are full agent attempts with repository edits and offloaded x86 evaluation, not E1/E8 fixed-prefix generation requests.

All reviewed evaluator reports use model ID `qwen3.8-27b-nvfp4-radixark::qwen-code-0.19.4::q38-a`. Each extant `eval_report.json` agrees exactly with `runner_metadata.json.eval_report` and its normalized evaluation outcome; every report has harness exit 0 and no evaluator error. `resolved`/`passed:true` and `tests_passed`, rather than an absent boolean named `resolved`, establish the positive outcomes. This review checks saved scored artifacts; it does not rerun hidden tests or assert an upstream benchmark leaderboard submission.

## Exact run lineage and configuration changes

All paths below are relative to remote `/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/output/` (host `mark@100.103.10.122`). Within a QC run, arm directory is `hydra27_fixed32_promoab_CqcNN`; task directory is `<arm>/swe_out/verified/per_task/astropy__astropy-ID/`.

| Alias | Exact run directory | Source commit | Proxy output cap | Completion / selected lineage |
|---|---|---|---:|---|
| Q16 | `fr14_promoab_Cqc16_20260819T222438Z` | `78a29d3391d02166abf571030a90645ce2375ba0` | 32768 | rc1;12907,13033 evaluated;13236 degeneration then stop |
| Q15 | `fr14_promoab_Cqc15_20260823T193658Z` | `a450f6c3600d088bd259df65e3940a64dc8de199` | 32768 | rc5;reran12907,13033;adds13398 without eval;script collision |
| Q12 | `fr14_promoab_Cqc12_20260824T021301Z` | `60099f0f3a6d2f926de482e611ba057b90d4c168` | 24000 | rc13;13453 evaluated;13579 capped without eval |
| Q10 | `fr14_promoab_Cqc10_20260824T074813Z` | `e7af6b595a8b3a09b7e88d25a3653d381432b99c` | 24000 | rc0;all ten remaining tasks evaluated |

The selected sixteen rows are disjoint, but the **attempts were not disjoint**. Q15 repeats 12907 and 13033: recorded agent times are 670.872 and 2249.684 s versus Q16's 291.437 and 791.870 s; their verdicts agree. The campaign explicitly keeps Q16's earlier outcomes. A separate Q12 attempt `fr14_promoab_Cqc12_20260824T000712Z` generated an unharvested 13453 patch, lacked evaluation/metadata, and was explicitly superseded before the successful Q12 rerun (`promotion_ab_campaign.md:4876–4891,4960–4972`). These are resumed developmental records, not a prospectively immutable one-attempt benchmark campaign. Retain the failed and repeated attempts when packaging.

Actual runtime evidence preserves common major mechanisms: `/models/qwen3.8-27b-nvfp4-radixark`, `modelopt_mixed`, engine seed 0, maximum sequence length 131072, graphs enabled, prefix caching enabled, merged full-vocabulary drafter, tail mode, single-logits reuse, pre-baked tail preparation, and split-K4 GQA-pair dispatch on 32 physical rows. The active FA2 binary remains `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`. Nevertheless, patcher hashes change Q16→Q15→Q12→Q10 (`78206cc4…`, `dd424b05…`, `29fa88e6…`, `8f61b9cd…`), source commits change, and the maximum output cap changes **32768→24000**. The identical broad runtime-attestation digest covers its stated Arctic/FA2/version inventory; it is not proof of byte-identical patched Python or unchanged entire experiment settings. Proxy settings are temperature0.6/top-p0.95/top-k20, so these are stochastic task continuations. These differences bar a single frozen-configuration 50% success-rate headline.

## Recommended existing task table

Caption: **“Dated Astropy agent-workload outcomes from the NVFP4 port's resumed development runs; rows span configurations and are not a matched native/tree benchmark. Agent elapsed time is `runner_metadata.json.agent.elapsed_s`, separate from server boot and evaluation; unevaluated outcomes remain explicit.”** Avoid the campaign's rounded observer wall values where exact metadata is available. A dash is missing authoritative agent metadata, not zero time or an evaluated failure.

| Astropy instance suffix | Run | Saved evaluation | Agent elapsed (s) | Patch bytes |
|---|---|---|---:|---:|
| 12907 | Q16 | resolved | 291.437 | 504 |
| 13033 | Q16 | failed | 791.870 | 1092 |
| 13236 | Q16 | no eval; degeneration | — | — |
| 13398 | Q15 | no eval; empty/incomplete | — | — |
| 13453 | Q12 | resolved | 886.401 | 1047 |
| 13579 | Q12 | no eval; budget-capped | — | — |
| 13977 | Q10 | failed | 1475.104 | 1141 |
| 14096 | Q10 | resolved | 1379.130 | 1321 |
| 14182 | Q10 | failed | 433.624 | 745 |
| 14309 | Q10 | resolved | 100.411 | 574 |
| 14365 | Q10 | failed | 407.009 | 617 |
| 14369 | Q10 | resolved | 1661.955 | 12307 |
| 14508 | Q10 | resolved | 2474.673 | 1124 |
| 14539 | Q10 | resolved | 1114.236 | 662 |
| 14598 | Q10 | failed | 1547.614 | 808 |
| 14995 | Q10 | resolved | 275.154 | 645 |

Thus Q10 itself provides **6 resolved and 4 test-failed tasks in one completed ten-task run**, while the sixteen-task lineage has eight confirmed resolutions and three unscored endpoints. If any count is printed, label it “confirmed resolutions in the recorded developmental lineage,” retain all sixteen intended tasks, and disclose the mixed caps/source versions. Do not present 8/13 or 9/16. The three missing evaluator reports cannot be turned into observed test failures by assuming that an absent/empty patch failed; the deployed-system outcome can be reported as unsuccessful/incomplete with its separate cause.

**Score conflict resolved:** `promotion_ab_campaign.md:5444–5463` agrees with the raw records. `REDTEAM_20260816.md:6058–6064` incorrectly labels 13977 and 14182 resolved and 14539 failed, despite its own 8/16 heading and 6/10 claim. Their actual verdicts are failed, failed, resolved, respectively. Any derivative Pages9/16 statement must be corrected; the raw evaluator is authoritative.

## Native-MTP case comparison: only two targeted tasks

These four successful runs really loaded the NVFP4 model path, with actual engine config MTP5 and `FLASH_ATTN`. The saved purity check says “no side code on the decode path” with the **declared LM-head loading shim exception**, SHA `0a673b5ad0b76467530ef33e8da36500c73b25857eda11281c2e55103ce57a71`; it does not establish an untouched upstream engine. The checkpoint identity `5ec8e240…` in `served_model.json` hashes safetensor filenames/sizes, not all tensor bytes. Do not elevate that weak identity alone into a historical full-weight-byte proof.

| Task / replicate | Exact native run root | Scored outcome | Agent elapsed (s) | Total patch bytes |
|---|---|---|---:|---:|
| 13236_a | `fr14_mtp5_astropy13236_a_20260824T130251Z` | resolved | 1788.385 | 819 |
| 13236_b | `fr14_mtp5_astropy13236_b_20260824T133941Z` | resolved | 5049.512 | 819 |
| 14369_a | `fr14_mtp5_astropy14369_a_20260824T151033Z` | failed | 2769.006 | 12943 |
| 14369_b | `fr14_mtp5_astropy14369_b_20260824T160304Z` | failed | 2640.045 | 13139 |

Each native arm's task path uses `<nativemtp5_probe_astropyID_a-or-b>/swe_out/verified/per_task/astropy__astropy-ID/`. The 13236 patches are byte-identical 819-byte fixes; the 14369 patches differ and both fail. For 14369, campaign prose quotes source-hunk sizes 1110/1274 B, but actual total patch files are 12943/13139 B; generated parser-table content accounts for extra bytes. Q10's total14369 patch is12307 B, whereas its source hunk is1318 B. These are distinct size measures; generated file bulk is not an evaluator success or throughput metric and does not by itself invalidate the saved test verdict.

These were a post-observation degeneration probe, not a preregistered native speed benchmark. Common documented model/path, Qwen-code version, task/base commits, x86 evaluator worker (Q10 and native `4a06bbe937af3c04c3897564d7bc27cdad8d8abc8c13fd156650e203e30994da`), sampling settings and9000 s agent budget support a descriptive case comparison. **Important differences remain:** native prefix caching is OFF versus QC ON; native five drafts versus the merged tree's31; attention backend differs; graph-capture maximum8 versus64 and compile range2048 versus4096; native24000 cap matches only the later QC; implementation/harness source commits vary. Native task metadata also lacks the fixed32 bundle/image-content attestation present on Q10, although both report Qwen-code0.19.4 and same task/base commits. Do not say full harness parity is proven from names alone.

For a paper case table, pair Q10's14369 outcome (resolved,1661.955 s) with the two native14369 failures (2769.006,2640.045 s), **without calculating a speedup or attributing quality to the drafter**. Keep13236's QC degeneration/unevaluated state alongside the two native resolutions, with no fabricated QC runtime. Successful native13236 times alone differ almost threefold; n=2 is neither quality equivalence nor a performance distribution. The probe's absence of observed degeneration is a finite observation, not proof that native eliminates the failure.

## Invalid/superseded observations and metric limits

- Native roots ending010207Z/010422Z/010807Z/013447Z were infrastructure/pre-task refusals; no task outcome. The015818Z arm was the vacuous2.157 s “failure” with wrong served model/name and missing task evidence (`promotion_ab_campaign.md:5114–5204`): exclude from scientific counts. The120604Z attempt reached correct model pin but failed loading the quantized head before tasks (`5536–5574`); it is not a failed SWE task. The130023Z root also has no saved task evaluation. Preserve them as attempts.
- Q16's task3 degeneration, Q15's resident-shell collision, Q12's capped-terminal provenance failure, and the earlier13453 unharvested attempt remain adverse outcomes. Reporting only Q10 as if it were the whole intended cohort would conceal this lineage.
- All reviewed extant task metadata report `vllm_request_metrics_bytes=0` and missing verbose proxy metrics; `vllm_per_turn.json` explicitly has `proxy_request_rows=0` and deferred normalization. Their recorded agent wall times are usable workload-runtime observations, but **not model-only decode time, tokens/s, request latency percentiles, or GPU time**. Separate Prometheus pre/post brackets require their own reducer and scope (reviewer2 is auditing those); do not splice these walls into E1/E8 matched-support rates.
- The existing records are from **August19–24**, audited now. They must not be relabeled as new September measurements or automatically assigned to today's source tree. Calling this model the port's Qwen3.8NVFP4 configuration is supported; “latest everywhere” is outside this file audit.

## Minimal missing experiment, only for a stronger claim

No new model run is needed to publish the accurately dated case ledger above. If the intended central claim is workload-specific native/tree speed or quality under the **current** implementation, the existing data do not discharge it: freeze one current source/model/harness, the named subset, sampling/budgets/cache policy and task-level accounting, then run matched native and tree attempts on that declared subset, preserving all failures/timeouts and reporting outcomes jointly with elapsed task time. One attempt per task gives a descriptive paired workload result; a quality-preservation or robust population-speed claim needs a separately justified replication/equivalence design. Do not require a broad benchmark expansion merely to publish the scoped case study, and do not substitute old E1/E8 snippet rates for missing task evidence. No such new run was launched in this review.

## Reproducible raw paths and hash ledger

For each table row, exact evaluation is `<remote output>/<run>/<arm>/swe_out/verified/per_task/<instance>/eval/eval_report.json`; exact elapsed time is sibling `runner_metadata.json` key `agent.elapsed_s`. No evaluation row was inferred from narrative prose. All available normalized outcomes and metadata evaluation copies were checked for agreement. Full evaluator SHA-256 hashes below bind the selected sixteen-row lineage and all four native probes; `ABSENT` remains absent.

| Run/task | eval_report.json SHA-256 |
|---|---|
| Q16/12907 | `a62f6b555de22602c3e4d3cfc89e9402b75cc0e7ccf29b696428711c6af51a34` |
| Q16/13033 | `5860883b91d7a8b250acf753d530b6dcc397b0af5af52ea9b6472a5d9a7282af` |
| Q16/13236 | `ABSENT` |
| Q15/13398 | `ABSENT` |
| Q12/13453 | `2c9475e7f351372f3a75ee1c974abdace96c5ce7c7fb910664fe89bae7d3d571` |
| Q12/13579 | `ABSENT` |
| Q10/13977 | `1f5b14164712012d82b80c7cbef93f7d9810d8ec0ae72b80f1b83a8e34732fec` |
| Q10/14096 | `320e69da98b54de8f8e2f42f67e920f39bc9987e2a0a333b612aa267a19b3ee0` |
| Q10/14182 | `068635a51846ec266c2ace07980d62b1a933e72a5e87c4ab66fbf2c543ac0831` |
| Q10/14309 | `eb501a85e75d643c3899f4de6726fb5c3832821cca6b38b8b7f8c48be6716963` |
| Q10/14365 | `153bc034b43438d866d3eb8050dfd927144bd432125cb71637b033ab4b2c7c68` |
| Q10/14369 | `676871871a54a99ad05798ba9e0e3c692ad387bc20bf8a702f04ff6fb073b6f2` |
| Q10/14508 | `3e7df6127233c0d0e033ca510dda93f73e5e11144f2f06fc80a6508d1f7866a1` |
| Q10/14539 | `b7645fc557bc487f9ac9a6e39ca178eaa64d7793106cde6071ebe1ae520e0c6d` |
| Q10/14598 | `5e5becfd0fb1333919015978d284ca8b34274d1eb6aa054a66852582ce7f3bd0` |
| Q10/14995 | `1df319b952b53681daa2ad80c7652274f895f455457f06bfee7fd08647fbda6d` |
| 13236_a/13236 | `d07c6d9257c2b0da9c6b0f49c0d76cb04e402925e33e4df6c38c12e23e3a60df` |
| 13236_b/13236 | `29126725f3ac4ad116b3c9b0aa517378ed0615dae5d93e93c2877b8cf0be9dad` |
| 14369_a/14369 | `24c43a24a2613cbaeec5313b354374c6ed46df639a2fde033d5183da54387a29` |
| 14369_b/14369 | `6ff8c118dc0d084f075e26a25705dc7898fe0410bf0815413eb25419ad844481` |

| Reviewed local source | SHA-256 |
|---|---|
| `results/fr14_nvfp4_port_20260816/promotion_ab_campaign.md` | `d3462d1741e230e596aa6799da90be3318872fe16b3b580f0faf0d2f902d357c` |
| `results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md` | `902814907c5642ce917fbda8f7ca955b3d859d4caae2d22570b448c4121d2736` |
| `config/fr13_fixed32/subset_b4_sixteen.json` | `47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c` |
| `results/fr14_nvfp4_port_20260816/promotion_ab_arm_mtp5.sh` | `298a8b9da6a62ecfe618209d3298088b279e823734904ae34b24bcfe50fb1238` |

Detailed read-only extraction (remote paths, raw file hashes, normalized/evaluator records, metadata, configuration receipts and selected log lines) is retained locally for the parent at `/tmp/swe-workload-raw-review-20260923.json` (SHA-256`d96f94d92610633abc9228e621f5b1e44ed55aa02b66ec19e1241093ff0039ed`) and `/tmp/swe-workload-config-review-20260923.json` (SHA-256`1f33917a053e5a925822be221f25ef4390ec06e1240c83b5c36cee2b4ba69fe2`). These are review reductions, not replacements for the raw artifacts. Copy the small original evaluation/metadata/prompt/patch/configuration records into any paper evidence companion before relying on remote-only paths for reader reproduction.
