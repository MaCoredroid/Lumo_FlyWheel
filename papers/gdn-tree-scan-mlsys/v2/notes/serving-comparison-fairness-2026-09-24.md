# Serving-comparison fairness audit — 2026-09-24

**Verdict:** comparing these complete speculative-decoding systems is legitimate. Their different implementations and draft budgets do not invalidate a system comparison. SGLang EAGLE 3/1/4 is a documented MTP configuration with working GB10 calibration; I found no evidence it was deliberately weakened. The recovered records do **not** establish an optimal SGLang configuration or a common tuning sweep. Crucially, a newly recovered earlier SGLang observation on the same two tasks is faster than the tree's best recovered shared-task observation. Retain both SGLang observations and the earlier empty-patch failure. No new inference is necessary for this explicitly scoped recorded-results comparison.

## Draft budgets and tuning evidence

- The pinned upstream SGLang checkout is `sgl-project/sglang`, commit `ace7314173c8221ecf5f213575302eab98f4e84f` (shallow checkout, read via `git show`). Its [Qwen3.8 cookbook](https://github.com/sgl-project/sglang/blob/ace7314173c8221ecf5f213575302eab98f4e84f/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx#L174-L177) explicitly specifies EAGLE, three speculative steps, top-k one, and four draft/verification-budget tokens using the in-checkpoint MTP head. This supports a reasonable documented baseline, **not** a proven best setting on SWE workloads. The same page, lines159–166, marks its Spark recipe unvalidated; the config snippet lines19–32 says measured data were not carried over. Do not describe this as independently validated optimal vendor tuning.
- Actual local launch: [sglang_calibration.sh](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/results/fr14_nvfp4_port_20260816/sglang_calibration.sh:24) lines24–36 uses FlashInfer, chunk8192, memory fraction0.70, EAGLE3/1/4, qwen3/qwen3_coder parsers and the as-shipped NVFP4 view. The cookbook Spark recipe's memory fraction0.95/disabled-prefill-graph differs; the whole launch is not a verbatim cookbook command. Lines51–60 calibrate random1024/1024 traffic at concurrency1/8, not multiple draft lengths. [REDTEAM pass40](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md:1089) documents a second calibration of the same recipe, not a tuning search.
- `git log --all -G 'speculative-num-steps|speculative-eagle-topk|speculative-num-draft-tokens' -- results/fr14_nvfp4_port_20260816` returns `c27f30207`, `128d27b86`, `f902010ba`, `ad52fbce6`, `d85b01e67`; the relevant served configuration remains3/1/4. No stronger deeper/wider SGLang SWE configuration was located in this bounded records/history audit. This is a search result, not proof that no possible better setting exists.
- [fr13_fixed32_topology.py](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/scripts/fr13_fixed32_topology.py:62), lines62–84/96–129, defines Hydra27 as27 valid draft nodes in31 physical draft slots plus the root. There are21 valid nodes at depths1–5 and six main-spine continuation nodes at depths6–11; four physical slots are masked. [The contemporaneous correction](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md:1098) binds these to MTP-fed shallow nodes and suffix-fed deep nodes. Thus27 is a candidate-node count, **not**27 sequential MTP calls. Likewise SGLang's3/1/4 parameters are step/top-k/budget parameters, not four independent chains. Avoid directly equating the two integers as computational depth.
- Equal depth/candidate budget would answer a mechanism-ablation question. A best-configuration serving comparison may use different budgets, with their actual proposal cost, acceptance, task behavior and timing included. A claim of best-tuned competition would require tuning evidence; the present recorded system observations need only disclose the missing common sweep. A large equal-budget sweep is not required to report them honestly.

## Newly recovered same-task SGLang observation

Original remote root: `/home/mark/shared/tmp-scratch/fr14_ablation_a/step2/swe_out/verified/per_task/`. Only completed shared tasks `astropy__astropy-12907` and `astropy__astropy-13033` were copied. For each, the bundle retains original `vllm_metrics_pre.txt`, `vllm_metrics_post.txt`, `runner_metadata.json`, and `eval/eval_report.json`; four contemporaneous launch/note files are also retained. The older note's31.71/29.22 telemetry estimates are **not** reused.

| Task | Completed requests R | Output tokens N | Sum request E2E s | Sum TTFT s | Pooled decode (N−R)/(E−F), tok/s | Recorded outcome |
|---|---:|---:|---:|---:|---:|---|
|12907|11|5,895|230.3327858532|20.3665699260|28.0235559517|Resolved;504-byte patch|
|13033|6|41,955|1,360.5710431901|12.4841883241|31.1174312312|Failed;empty patch|
|Shared two pooled|17|47,850|1,590.9038290433|32.8507582501|**30.7004946729**|One resolved,one empty-patch failure|

The unchanged common `Audit.task(...,'sglang')` passes on both brackets: running/waiting queues empty before and after, token histogram sum equals generation counter delta, request/token/E2E/TTFT histogram counts align, no reset, and nonstreaming contributions are zero. The safe trace projection independently yields the same11/6 positive-usage responses and5,895/41,955 tokens. Summed request latency counts overlapping request exposure separately; this is not aggregate machine-wall throughput. The runner serializes **agent tasks** (`ablation_a_step2_run_swe.sh:22–31`), but contemporaneous telemetry reports occasional physical running-request count3 (`ablation_a_step2.md:119–120`). Do not label SGLang a strict physical-B1 run.

Earlier shared-task SGLang is30.7004946729 versus later SGLang26.8939560242. Best recovered tree Sr12 is29.0934427002: **8.1783679350% above the later SGLang observation and5.2346126337% below the earlier observation**. The old+8.18% comparison was arithmetically correct for its specified later comparator; the new recovery prevents treating it as a best-versus-best win. These two SGLang observations are not a confidence interval or a tuning envelope. They are recorded runs of the same recipe with different response caps/trajectories.

Preserve the adverse behavior precisely:

- `13033/runner_metadata.json`: `patch_bytes=0`; `empty_patch_retry.cause="agent_gave_up"`; agent exit0, `timed_out=false`, `budget_capped=false`. This last flag concerns the agent budget, **not** absence of a response-token cap.
- `13033/eval/eval_report.json`: `synthetic_no_patch=true`, `harness_invoked=false`, `error="empty_patch"`, failed `patch_apply_failed`. No evaluator tests were invoked for this empty patch; do not describe it as a measured test failure.
- Original `13033/qwen_trace.jsonl:24`: final response has `usage.output_tokens=32768`, one thinking block with136,053 characters, no text/tool-call block. Its raw `stop_reason` is null. The live proxy projection records `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768`; say a **final32,768-token thinking-only response at the configured cap**, not a literal recorded `finish_reason=length`. The contemporaneous note lines80–85 calls this a capped runaway. The rate includes it; it is not filtered away to favor either system.
- Live sampling projection records temperature0.6, top_p0.95, top_k20, presence penalty1.0,min_p0, auto_continue0. Source hash is retained without copying environment secrets.
- Task13236 has five nonstreaming requests/22,214 tokens within its all-request deltas, so its all-request TTFT clock does not meet this reducer's streaming first-token rule. It is excluded from this shared-pair pool for that concrete reason;task13398 was unfinished. Neither is silently counted in the two-task outcome denominator.

## Durable replay and source binding

Run from the repository root:

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/results/agent-workload/sglang_step2_reduce.py
```

The new reducer verifies all12 raw-manifest file hashes, then invokes the unchanged common boundary/population audit. No attempted run, old reducer, old manifest, or original evidence was changed. No GPU/model operation was performed.

| Evidence | SHA-256 |
|---|---|
|`results/agent-workload/raw/sglang-step2-20260924/manifest.json`|`a524f1e25270a9ca87f737b6f99ff5ae3e814ac265b3c0781d7824caffe5d02e`|
|`results/agent-workload/sglang_step2_reduce.py`|`9315e2fb93252234c316fde3e684839cf7ba58c047b1d89f1e36546cdc5a237f`|
|`results/agent-workload/sglang-step2-rate-audit.json`|`c1e01be7b5a1bc11cce67666a112fe5c4c6478ae10b3a3315d95eeb8c5e39787`|
|`results/agent-workload/sglang-step2-behavior-projection.json`|`8ffb89b13304bf4a3005725085234f0af3bb53b8a79059b929c4b55488866115`|
|`results/agent-workload/sglang-step2-config-projection.json`|`5ebbd60c77b000efc5890a6e97bd33c249f30583a7162b6258e05b6ba13055dc`|
|Unchanged `results/agent-workload/shared_rate_reduce.py`|`c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f`|
|Repository `scripts/fr13_fixed32_topology.py`|`c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc`|
|Repository `results/fr14_nvfp4_port_20260816/sglang_calibration.sh`|`47fe550a7455998a61a456c48c344eba2c310b1f7417beaa5d75ec37290af1f1`|
|Upstream cookbook at pinned commit above|`c8b48306cfa38e7d9c8328e41812ef5d59426ced9fbc8dcd2fa82ac031e3d46f`|
|Upstream config snippet at pinned commit above|`c968812a6afdbf2a3aa1475df08df169fc14fa11846213e98e9203583072b610`|

**Disposition:** evidence review complete. Publication wording should compare the complete systems, disclose actual budgets/tuning limits, report both recovered SGLang observations and retain failure behavior. No supported conclusion of superior quality, optimal SGLang tuning, universal speedup, or a best-versus-best tree win follows. Parent owns manuscript integration; this note does not certify a moving manuscript hash.
