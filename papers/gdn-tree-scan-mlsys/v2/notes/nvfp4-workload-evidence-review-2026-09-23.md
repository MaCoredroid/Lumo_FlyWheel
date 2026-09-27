# Qwen3.8 NVFP4 workload evidence review — 2026-09-23

Read-only review of the repository, its `Lumo_FlyWheel-gh-pages/only-quantization.html` summary, and small preserved raw records read over SSH. No inference, benchmark, source change, or remote mutation was performed. Here `F = results/fr14_nvfp4_port_20260816`; remote raw paths are relative to `/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816`. This report separates currently applicable agent-workload observations from local numerical qualification and superseded headlines.

## Verdict and eligible manuscript material

**Real Qwen3.8 NVFP4 SWE-bench Verified agent evidence exists and should not be discarded merely because it predates the fresh FP8 numerical work.** It is a different, graph-enabled Hydra27 serving configuration. The latest complete single-configuration cohort located is **Cqc10, August 24: ten selected Astropy tasks, six resolved and four test failures**. It supports a descriptive workload case study with actual task runtimes and request-weighted decode statistics. It does not establish a full SWE-bench Verified score, quality equivalence, or a native-versus-tree speedup.

**Final manuscript disposition:** lead only the latest completed Cqc10 segment. Keep the older Cp1/Sr12 pair and its percentage change in this audit/artifacts, not the manuscript. The older **Cp1/Sr12 split-K comparison remains an inspectable component case study on four selected SWE-bench Verified tasks**, with one serving process per arm and no replicated causal estimate. Its control stopped on the fourth task; keep that fact beside the statistics. Later promoted-default runs establish continued use of split-K, but do not turn this pair into a replicated measurement or replace its censored control.

| Evidence | Eligible statement | Required scope |
|---|---|---|
| Latest Cqc10 | 6/10 selected Astropy tasks resolved; 10/10 terminated with evaluations; individual agent runtimes below; mean request TPOT 35.4643 ms (inverse 28.1974 tok/s) | One server run on August 24, one configuration, no native timing comparator; selected subset, not all 500 Verified tasks |
| Cp1 vs Sr12 | Mean request TPOT 40.9975 vs 34.6995 ms; inverse statistic 24.3917 vs 28.8188 tok/s; observed inverse-statistic difference +18.1500% | One run per arm, stochastic agent trajectories differ; Cp1 fourth task censored/no eval, rc13; not a causal end-to-end task-speed result |
| Exact16 continuation ledger | Eight confirmed resolves, five test failures, three no-evaluation outcomes across four continuation roots | Not one frozen-config 16-task run: source revisions and output caps change; no claim of quality parity. Pages' 9/16 is contradicted by raw evaluations |
| Four native MTP5 probes | Same-model native task13236 resolved twice; task14369 failed twice; actual runtimes recorded | Degeneration probes, not matched 16-task baseline or throughput comparison; native APC/graph settings differ |
| Head/kernel isolated microbenchmarks | Can explain implementation provenance in artifacts | Exclude their rates from the workload performance headline under the latest user constraint |
| E1/E8 FP8 local-document timings | Preserve raw artifacts and source qualifications | They are not SWE-bench/agent workload performance. E8 studies head reuse and does not supersede E1's different system comparison, but neither supplies this paper's requested workload rates |

## Latest complete workload observation: Cqc10

Raw arm: `output/fr14_promoab_Cqc10_20260824T074813Z/hydra27_fixed32_promoab_Cqc10/`. Required files are `deploy_speed_promoab_C.json`, `health.json`, `fixed32_final_flush.json`, `logs/fr13_fixed32_work_census.jsonl`, `container_env.txt`, `docker_full.log`, `offload_proxy_env.txt`, and each `swe_out/verified/per_task/<id>/{runner_metadata.json,vllm_metrics_pre.txt,vllm_metrics_post.txt,eval/eval_report.json}`.

The 265 completed engine-request observations comprise **256 normal requests plus nine successful internal compaction requests**. All ten task provenance records state a 24,000-token normal ceiling and 20,000-token compaction ceiling. Recorded completion reasons are265 stop, zero length, and zero other; failed compactions zero. This is not a count of visible user turns or all possible API request types.

Independent reconstruction of all ten metric brackets found their selected pre/post counter vectors contiguous, with no overlap/reset. Sum of bracket deltas exactly reproduces the stored reduction (floating tolerance 1e-7):

- `vllm:request_time_per_output_token_seconds_count = 265` and `_sum = 9.39803353270284 s`; inverse = **28.19738821721218**, arithmetic request-mean TPOT = **35.464277481897504 ms**.
- `fr13_decode_step_wall_seconds_total = 9677.623664388899`, retained `...steps_total = 49067`; mean physical start-to-start interval **197.23283804571093 ms**. Preserve 49,068 attempted intervals and one rejected interval. This is not task wall time.
- `fr13_decode_forward_gpu_seconds_total = 5667.758015617326`, `...steps_total = 49333`; mean pure-decode GPU forward **114.88776307172331 ms**. It is a component statistic measured during the workload, not full iteration or end-to-end latency.
- `generation_tokens_total = 248342`; `aggregate_window_wall_s = 11375.017803192139`. The stored aggregate 21.8322295663 uses an earliest-pre/latest-post file-mtime envelope; do not substitute it for directly recorded task durations or compare it with E1 retained-interval throughput.
- The raw reduction's work-census gate says PASS with 49,333 physical steps/events, every event B1. `health.json` records orchestrator rc0 and 11,421 s workload window.

The following times are **`runner_metadata.agent.elapsed_s` / health's `codex_elapsed_s`**, not model-only time or evaluation duration. The harness actually used Qwen Code 0.19.4; `codex_*` is a legacy field name.

| Astropy task | Agent elapsed seconds | Evaluation |
|---|---:|---|
|13977|1475.104|failed|
|14096|1379.130|resolved|
|14182|433.624|failed|
|14309|100.411|resolved|
|14365|407.009|failed|
|14369|1661.955|resolved|
|14508|2474.673|resolved|
|14539|1114.236|resolved|
|14598|1547.614|failed|
|14995|275.154|resolved|

Sum **10868.910 s**, range **100.411–2474.673 s**, mean **1086.891 s**. These are all ten tasks, including failures. Do not silently restrict runtime reporting to solved cases. The broader elapsed arm/window includes boot or evaluation/coordination components and has a different boundary. Source narrative: `F/promotion_ab_campaign.md:5413–5440`; use the exact raw times above rather than rounded narrative minutes.

## Exact served method: agent route, not Cat10

Actual boot settings and receipts for Cp1/Sr12/Cqc10 establish this route:

| Surface | Agent-serving configuration and evidence |
|---|---|
| Model | `/models/qwen3.8-27b-nvfp4-radixark`, `RadixArk/Qwen3.8-27B-NVFP4`; `modelopt_mixed`; BF16 model dtype, KV `auto` (BF16 here); MTP tensors BF16; quantized NVFP4 target/shared MTP LM head. Cp1/Sr12 boot lines66–67 and83–84 explicitly select `ModelOptNvFp4LinearMethod` |
| Checkpoint identity | Download pin `554ebba9b5f1b79dc11246341960360e6ef05ef4`; `F/radixark_download_verification.json` records all22files present and LFS hashes matching **before KV-config surgery**, with `.pre_kv_surgery.bak` preserving original config. Actual runs bind the local modified model directory, not a pristine HF checkout; keep the loader/config adjustments explicit. Do not claim that equal architecture isolates quantization from the model refresh |
| Runtime | vLLM `0.19.2rc1.dev134+gfe9c3d6c5`; engine seed0; max sequence131072; max batch tokens4096; max sequences1; block/mamba block1024; GPU memory utilization0.7; GDN prefill Triton; mamba stateFP32 |
| Graph/cache | `enforce_eager=False`, `FULL_AND_PIECEWISE`, prefix cache ON, chunked prefill ON; graph max64, compile range4096. This is not the synchronous eager/cache-off E1 path |
| Tree | Hydra27: **27 active draft nodes (root excluded)**,31 physical draft slots,32 verifier rows including root; max draft depth11. Five MTP head depths (three candidates each), six main Arctic suffix tokens, rank1 rescue4 and rank2 rescue2 =27. Four remaining physical rank2 suffix slots are invalid padding. Four **post-root** MTP forward calls; do not call these four total head evaluations. Authority `scripts/fr13_fixed32_topology.py:100–109`, physical tree in actual boot's `speculative_token_tree`; `F/promotion_ab_arm_{c_prime,s_served}.json:/census` confirms shape and replays |
| Drafting | Full vocabulary `FR13_DRAFT_VOCAB_K=0`, `ROOT=0`, single-logits reuse1, device multidraft1, tail mode1, fused draft top-k1; suffix pass gate0. The task-specific suffix predictor is active in this route, unlike fresh Cat10. Static packed/merged draft graph: one replay per pure step, four post-root MTP calls and main tail6 in the census |
| Attention | Configured vLLM backend is `TREE_ATTN`, **but patched served full attention uses forked FA2**, not E1's stock flat-map TREE_ATTN implementation. Cp1 actual retag receipt: `gqa_pair`, binary `3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae`. Sr12/Cqc10: `gqa_pair_splitk`, four splits, binary `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`. Engagement says candidate served at all16 full-attention layers, no fallback |
| Other held mechanisms | Host-tail PREP_BAKE1; EAGER_PACK1; TREE_CONV_FUSED1; conv pregather1 and fused/batched writeback1; fixed32 accept packing1; fixed32 remap16=1. Boot records conv emulation engagement; forward graph registry records unfused convolution pregather and48 state-row consumes. Do not infer that every environment flag is a newly measured individual optimization |
| Inactive candidates | `FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0`; do not credit the experimental single-launch GDN optimization to these observations. Separate B4 kernel candidates are not an actual served NVFP4 B4 workload result |
| Sampling/harness | Qwen Code0.19.4, actual SWE-bench Verified repository tasks in per-instance x86 evaluation images; agent+proxy offloaded to Alienware, model on GB10. Proxy forces temperature0.6, top-p0.95, top-k20, min-p0, presence penalty1.0. Cqc10 output cap24000; earlier Cp1/Sr12/Cqc16 cap32768. Engine seed is0; this is stochastic, not greedy. User task budget9000s; environment/context/harness effects remain part of task times |
| Containment | Per-task completed records carry the nine-rule internal bridge fingerprint `e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4`; the earlier golden-patch network leak was repaired before this pair. Check the actual boundary evidence, not only the denylist settings. Cp1 censored fourth task has no final metadata, so its provenance failure must remain explicit |

Cp1 and Sr12 repository revisions differ (`f7fde8e1b455c4baafc477e3699ad69e59e3265c` and `bdca0bd50cbf5643205c87e4e3e2fe668895c364`); Cqc10 is later still. Runtime/source attestations are useful but do not justify saying their entire emitted Python sources are identical except the kernel. The later source baseline preserves this implemented route; implementation date alone does not make an older applicable observation invalid. Conversely, a later helper fix does not retroactively establish an earlier unrecorded correctness gate.

## Split-K exact4: raw rates, task outcomes, and censorship

Control `output/fr14_promoab_Cp1_20260818T081918Z/hydra27_fixed32_promoab_Cp1/`; treatment `output/fr14_promoab_Sr12_20260819T043506Z/hydra27_fixed32_promoab_Sr12/`. Both have four metric brackets. Their selected count/sum, generation and forward-GPU vectors are contiguous. Independent sums reproduce the stored reductions:

| Metric / exact raw support | Cp1 control | Sr12 split-K |
|---|---:|---:|
| TPOT histogram completed-request count |194|124|
| Sum of request-mean TPOT (seconds) |7.953518185805249|4.302743886812845|
| Mean request TPOT (ms) |40.99751642167654|34.699547474297134|
| Inverse mean request TPOT (tok/s) |24.391721432942017|28.818819632755332|
| Recorded generation-token counter delta |255984|101923|
| Pure GPU forward seconds / forward count |6891.252737350475 / 52500|2289.3166711044437 / 19901|
| Mean pure GPU forward (ms) |131.2619569019138|115.03525808273171|
| Physical wall seconds / retained intervals |11232.740012879716 / 52304|3884.4578990789596 / 19776|
| Mean retained physical interval (ms) |214.75871850871283|196.42283065730985|
| Work-census steps |52507|19901|
| Final serving/orchestrator status |13; fourth task censored|0; four task evaluations|

The histogram is **request-weighted**: each request contributes its own mean TPOT, then the displayed number is the reciprocal of the arithmetic mean of those request means. It is neither token-weighted throughput nor the mean of per-request rates. Source `scripts/fr13_measure.py:2018–2022,2330–2337`. The counts do not include an in-flight terminal request as a completed histogram observation. Do not divide generation tokens by this TPOT sum, or substitute structural accepted-plus-bonus counts.

Control's raw census has seven extra terminal events relative to the closed metric brackets (52,507 vs52,500); the original campaign already discloses the capped-bracket gap (`F/promotion_ab_campaign.md:916–919`). Preserve it; do not call full event coverage exact. The independently matched wall ratio above uses its own retained physical support and does not recreate H3's mismatched token numerator.

| Task | Control agent seconds / eval | Split-K agent seconds / eval |
|---|---|---|
|12907|365.058 / resolved|407.188 / resolved|
|13033|1267.008 / failed|631.567 / failed|
|13236|1431.243 / failed|238.166 / resolved|
|13398|9000s task cap; no final metadata/eval|3143.315 / failed|

Cp1 `swe_orchestrator.log:17–41` fails because the capped trace lacks the Qwen terminal result needed to independently count completed model requests. The reduced health has only three completed tasks. The campaign's `:543` and wording such as “drained4/4” must not override raw status. Its split-K section `:2403` says “2/4, the same rate as the promoted control”; **this is false for the actual Cp1 comparator**, which has one confirmed resolve, two failures, and one no-evaluation task. Do not carry that quality comparison into the paper.

The +18.1500031% inverse-TPOT difference and −8.5379015% mean-interval difference are arithmetic descriptions of these two recorded workloads. Task trajectories/output lengths differ greatly, and the experiment has one boot per arm. The old ±10% “variance doctrine” is not a confidence interval or an equivalence test. Do not describe acceptance “unchanged within variance,” infer task speedup, or multiply these observations with other component gains. `F/promotion_ab_pair_splitk_vs_control.json` remains auditable as raw derived evidence, not an inferential trial.

## Corrections to the public summary and broader ledger

`only-quantization.html:1494–1511` says9/16. `F/promotion_ab_campaign.md:5442–5463` says8/16; reviewer1 independently read actual eval reports and resolved the contradiction in favor of **8 confirmed resolves**. In particular13977 and14182 failed,14539 resolved. The 16-task collection spans Cqc16/Cqc15/Cqc12/Cqc10, not four independent replicates: the later boots continue unfinished IDs, with different source revisions and output caps (32768 then24000). Report the full task lineage in artifacts and prefer the single-configuration Cqc10 table in the body. Reviewer1's report `notes/swe-workload-comparison-review-2026-09-23.md` provides the exact eval/source/config closure.

The same-model native MTP5 loader issue was repaired; blanket wording “no native run exists” is obsolete. Successful roots are `fr14_mtp5_astropy13236_a_20260824T130251Z`, `_b_20260824T133941Z`, `fr14_mtp5_astropy14369_a_20260824T151033Z`, `_b_20260824T160304Z`. Actual agent times are1788.385/5049.512s for13236 (both resolved),2769.006/2640.045s for14369 (both failed). These were two selected degeneration probes with two attempts each. Their native launcher uses a loader-only NVFP4 head shim and APC OFF, graph max8/compile range2048 versus the tree's APC ON/max64/range4096. Do not call them pristine-upstream or matched whole-cohort throughput evidence. Source narrative `F/promotion_ab_campaign.md:5564–5677`.

The older25.261 stock headline is a structural acceptance/event proxy with a different wall support; the26.692 headline is likewise not the same inverse-request-TPOT statistic as28.819. Those earlier Unsloth/RadixArk/shape/config headlines must not appear as a common “tokens/s” progression. Earlier network-enabled quality runs could fetch gold patches and are unsuitable quality evidence. `F/ablation_a_step2.md` explicitly labels its early SGLang cross-workload comparison non-citable; the Pages link to that file does not establish a final16-task score. No served NVFP4 B4 workload or completed full16 native parity trial was established in this review.

## Minimal remaining work, without another experiment

1. Preserve the retrieved raw bracket/config/health/eval receipts beside the manuscript results and link the exact current case table. No new inference is necessary to report the scoped Cqc10 observation or censored Cp1/Sr12 comparison.
2. Correct9/16 and the false Cp1 quality-parity sentence; label the mixed-source continuation ledger and native probes honestly. Keep incomplete/failed outcomes in the denominator.
3. Use the actual Hydra27/FA2 graph/cache/sampling method for agent-workload results; retain the FP8 Cat10 branch only as separately scoped local correctness support. Neither qualifies the other route's untested sampling/state properties.
4. A **causal native-versus-tree agent speed/quality claim** would require a prospectively matched task/harness/config comparison with complete task outcomes and replicated serving blocks. A repeated split-K causal speed claim would similarly need matched prospective blocks. Those additional claims are not required for a clearly labeled descriptive case study; do not add broad experiments just to repeat currently inspectable observations.

## Evidence identities and replay

Temporary extraction files below contain raw-file SHA256 values plus JSON records or explicitly selected line extracts. They are a read-only audit convenience, not complete portable run archives; preserve original files if packaging. The raw source roots above remain authoritative. Only allowlisted nonsecret settings belong in the deliverable bundle. The older-pair temporary extraction was redacted locally after collection; its listed hash now binds that redacted extraction, while per-file hashes still identify the original remote files. Do not distribute unfiltered environment records.

| File | SHA256 |
|---|---|
|`/tmp/nvfp4-splitk-raw-review-20260923.json`|`aad45e8670dba8c584ea5ea0de591caba10b0d4f5007c03e6af71a4ebd18ff57`|
|`/tmp/nvfp4-splitk-extra-review-20260923.json`|`269c27ff8fb70a8836a55d9e1160b3b7ab2f0c7ecaca3dc7202be9c70c4f41e4`|
|`/tmp/nvfp4-cqc10-metrics-review-20260923.json`|`db97431425afade4065d5dd910d9d2cee9e9486507464afb1f5c02e14ce607a0`|
|`/tmp/swe-workload-raw-review-20260923.json`|`d96f94d92610633abc9228e621f5b1e44ed55aa02b66ec19e1241093ff0039ed`|
|`/tmp/swe-workload-config-review-20260923.json`|`1f33917a053e5a925822be221f25ef4390ec06e1240c83b5c36cee2b4ba69fe2`|
|`results/fr14_nvfp4_port_20260816/promotion_ab_campaign.md`|`d3462d1741e230e596aa6799da90be3318872fe16b3b580f0faf0d2f902d357c`|
|`results/fr14_nvfp4_port_20260816/promotion_ab_pair_splitk_vs_control.json`|`74cd7fc19845ef226d7088c0bfdf529d4ce9515ccdb391d3c914586624b79d33`|
|`results/fr14_nvfp4_port_20260816/promotion_ab_arm_c_prime.json`|`368ea01d796d9f9f8c49441af2f008acefbd668ddc565dc8cb5fbc6ae4c4d873`|
|`results/fr14_nvfp4_port_20260816/promotion_ab_arm_s_served.json`|`c455a72bf823e60be6e66edad0e7c27249229d80352773741608c30f2a26984b`|
|`results/fr14_nvfp4_port_20260816/radixark_download_verification.json`|`9c542db2a74010fc2f9368138a0733c19f74e3facc478d5d310ad57e43083623`|
|`scripts/fr13_measure.py`|`8340cc6593260531174f847eb5229e82801600a3cd8bbe0c0d1e4a8c4540ccc1`|
|`scripts/fr13_fixed32_topology.py`|`c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc`|
|`config/fr13_fixed32/subset_b4_four.json`|`0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5`|
|`config/fr13_fixed32/subset_b4_sixteen.json`|`47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c`|
|`/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel-gh-pages/only-quantization.html`|`dbfe085a657a977263aec1d9dcb22e4494f328681d2dfcc7cdbec2d31ff5d3d4`|

Independent arithmetic can be reproduced without inference by parsing each raw metric line into its exact metric-name/label key, subtracting each task's pre from post, verifying adjacent task origins agree, and summing disjoint deltas. Then use `1000*TPOT_sum/TPOT_count` for request-mean TPOT, `TPOT_count/TPOT_sum` for its inverse, and `1000*wall_seconds/wall_steps` for retained physical-interval time. Hashes bind the records; retain metric labels rather than collapsing histogram buckets during any general reimplementation. The checks in this review selected the exact count/sum/seconds/steps fields, and did not use buckets to derive a result.

## Durable Cqc10 closure and current manuscript check

The safe original Cqc10 bundle is assembled by reviewer1 at `results/agent-workload/raw/cqc10/`; unfiltered `container_env.txt` is omitted in favor of allowlisted values bound to its original hash. The independent CPU-only `results/agent-workload/cqc10_reduce.py` reads those original bracket/metadata/eval files, requires the exact ten task IDs/order, rejects missing/duplicate/unexpected metric labels or noncontiguous/counter-reset supports, reconciles per-task TPOT counts with completed/normal/compaction records, and verifies the stored reduction. Running it from any working directory prints deterministic JSON; `cqc10-metric-audit.json` records the result and input hashes.

Actual Cqc10 source revision is `e7af6b595a8b3a09b7e88d25a3653d381432b99c`; FA2 patch-source SHA256 `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`; ten-task subset SHA256 `716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b`. These are distinct kinds of identities: do not call the patch-source hash a Git commit. `git_head.txt`, `arm_meta.txt`, original boot lines and actual engagement supply the chain. Proxy `AUTO_CONTINUE=0`, sampling settings, and output24000 were directly checked against actual Cqc10 proxy snapshots.

Current `main.tex` Hydra27 paragraph, `abstract.tex` Cqc10 count/rate/duration language, and `results/agent-workload/case-study.tex` settings were checked against the evidence above. No material numerical/configuration mismatch found. With parent authorization, only case-study.tex was amended to specify disabled auto-continuation, the recorded internal-network boundary, and the265=256+9 completed-request population. Main/abstract were not edited. The quoted28.20statistic is explicitly separated from task-completion speed; source geometry matches the active Hydra27 route.

- `results/agent-workload/cqc10_reduce.py`: `59f6d35aad7a409e9a00b551697da3205bb5fa8706ef4a54977c62089fb4e8a3`
- `results/agent-workload/cqc10-metric-audit.json`: `d46418013862609dd494f1b2bcbe0dba82dada578cfeb2ce01816e539fabc761`
- `results/agent-workload/case-study.tex`: `4acf538efb64f6d27ce0abc6069fce3371a4d67454343f4db0f5b904195d86c4`
- `main.tex`: `bf8f1f47a95e95e0a4732691dbcf56d86f2150b4a9da70d6b762005baf10e90c`
- `abstract.tex`: `018538d3d3938a177950976e49f99662e38027cc12a34a2693d5b136abeb9c80`
