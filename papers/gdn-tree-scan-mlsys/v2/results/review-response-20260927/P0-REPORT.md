# P0 evidence recovery: 27 September 2026

Completed locally without inference or changes to manuscript, raw results, or runtime code. The reducer replays the earlier rate/eligibility reducers byte-for-byte, verifies their input bindings, then recovers acceptance and phase counters from the same per-task boundaries.

## Recovered populations and results

`all-attempts.csv` contains **22 unique completed-bracket attempts**: four Sr12, two Cqc16, two Cqc15, ten Cqc10, and two attempts in each SGLang pair. It retains empty patches and failed evaluations. The shared Sr12 pair is a subset of the four-task run, not two extra attempts. This is the recoverable current-route comparison population, not an assertion that only 22 attempts ever existed in the repository.

Six additional comparison-lineage attempts are explicitly retained in `additional-attempt-ledger.json` without invented pooled rates: the earlier SGLang third task reported resolved, its fourth task stopped at the GPU budget, two partial later SGLang attempts, Cqc16 task 13236 flagged by the degeneration screen, and Cqc15 task 13398 without completed evaluation. The Cqc16 adverse trace has zero visible characters, zero tools and 70,755 thinking characters; its idle engine bracket does not establish a completed agent outcome. Cqc15 has 147 tool calls and is not flagged by that screen, but its post bracket retains one running request and mismatched generation/completion populations. Neither record is silently discarded or assigned a completed-task rate. Seven additional Cp1/Cqc12/unharvested records from the source-bound bracket projection remain in a separate provenance-only ledger; no rates from superseded revisions are promoted. The incomplete campaign must not be described as a completed sixteen-task study. Old native-MTP, DSpark and DFlash records remain provenance, as documented in `notes/comparator-git-history-2026-09-23.md`; they do not become current-build controls.

| Recorded build / cohort | Tasks | Resolved | Nonempty patches | Agent minutes | Pooled decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aug. 19, tree bdca0bd, shared pair | 2 | 1 | 2 | 17.313 | 29.093443 |
| Aug. 19, tree 78a29d3, shared pair | 2 | 1 | 2 | 18.055 | 28.283047 |
| Aug. 23, tree a450f6c, shared pair | 2 | 1 | 2 | 48.676 | 27.268008 |
| Earlier SGLang, shared pair | 2 | 1 | 1 | 26.703 | 30.700495 |
| Later SGLang, shared pair | 2 | 1 | 2 | 34.065 | 26.893956 |
| Aug. 24 tree, separate cohort | 10 | 6 | 10 | 181.149 | 25.633622 |
| Aug. 19 tree, full cohort including shared pair | 4 | 2 | 4 | 73.671 | 26.205365 |

The earlier SGLang pair remains excluded from the retrospective patch-producing table as the author requested, with its **30.70 tok/s** retained visibly in the all-attempt diagnostic table. Exclusion is at pair level, based on one empty patch; test failure alone does not exclude a task. Three tree revisions are development observations, not statistical replicates.

For the headline shared pair, task 12907 takes **407.188 s** with Sr12 and **426.131 s** with later SGLang; both resolve and produce 504-byte patches. Task 13033 takes **631.567 s** versus **1617.769 s**; both fail evaluation and produce 1017- and 912-byte patches. Total agent time is **1038.755 s** versus **2043.900 s**. These are recorded agent times, not task-boundary time including evaluation, and not a causal kernel speedup.

The rate remains `(sum output tokens - completed requests) / (sum request E2E latency - sum TTFT)`. It includes all completed model requests inside each task bracket, including compaction where present. It is a ratio of pooled totals; requests that overlap contribute separate latency intervals. It is not machine wall throughput and cannot be replaced by mean console rates, inverse mean request TPOT, or accepted-token counts.

## Acceptance: what is actually recoverable

For every tree task, the pre/post snapshots have all 31 position series, integer nonnegative deltas, no reset, monotonically decreasing position counts, a position-count sum equal to the accepted-token scalar, and exactly **31 physical draft slots per event**. The task boundary explicitly binds `hydra27_fixed32`: **27 active logical draft candidates plus four padded drafts**, with the root separate, giving 32 physical verifier rows.

The source has two different acceptance definitions. The scheduler observes `len(generated_token_ids) - 1` after the sampler output is parsed; the statistics class increments position `i` whenever the observed accepted count exceeds `i`. Thus the position series are **survival counts of output-side accepted draft length**, not counts for logical node IDs, physical slots, branches, or off-spine selection. The reducer differences adjacent survival counts to recover an exact output-side accepted-length histogram. It does not infer a branch histogram or normalize those counts by 27 to manufacture a logical-candidate acceptance rate.

`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:173–201` explicitly distinguishes this output-side counter from the commit-side `accepted_lens` ladder. Filtering, discarded rows, and finished/aborted requests can separate the two; bonus/root conventions must not be silently interchanged. The scheduler's `-1` excludes one sampled correction/bonus token. The already-pending root is not one of the 31 draft slots. `accepted + draft events` also does not reproduce API output totals in these records, so it must not be used as the throughput numerator.

| Tree cohort | Spec events | Output-side accepted drafts | Accepted drafts/event | Accepted/physical slots |
| --- | ---: | ---: | ---: | ---: |
| Sr12 shared pair | 4483 | 20648 | 4.605844 | 0.148576 |
| Cqc16 shared pair | 4800 | 21532 | 4.485833 | 0.144704 |
| Cqc15 shared pair | 13054 | 57515 | 4.405929 | 0.142127 |
| Cqc10 ten tasks | 49333 | 199714 | 4.048284 | 0.130590 |

The final column is padding-dependent and should stay an artifact diagnostic. For the manuscript, accepted drafts/event is more interpretable, with the output-side definition stated. The recovered length support is 0–11; all 31 raw position series remain in the artifact, including trailing zeros. `output-acceptance-histograms.csv` preserves task-level histograms.

SGLang exposes `spec_verify_calls_total` (recovered task deltas are in JSON), but its saved `spec_accept_length` and `spec_accept_rate` are gauges. They describe recent logging windows; neither differencing endpoints nor averaging them yields an acceptance statistic for the task. No comparable task-total acceptance numerator or depth histogram was recovered. Equal-looking console means would conceal this mismatch.

### Source binding and limitation

The parent extracted `vllm/v1/spec_decode/metrics.py` and `vllm/v1/core/sched/scheduler.py` into `source-semantics/current-deployment-image/`, with file hashes and image identity. This immutable image matches historical Cqc10: image `ffa30d66...`, configured digest `3dbe092e...`, installed version `0.19.2rc1.dev134+gfe9c3d6c5.cu130`. The reducer fails if that image/digest identity differs. An initial older-image extraction is retained for provenance only; it was caught and corrected before experimental launch. The matched-base scheduler confirms that it counts scheduled drafts minus declared invalid speculative tokens; the observed delta is exactly 31/event, so those four padded slots remain in the denominator.

The reducer separately inspects each recorded tree revision's git-object patcher and records all functions writing `SCHEDULER_PATH`. These implement decode-mode draft selection, diagnostic tracing, request-state cleanup, Mamba block alignment, and (where present) NVTX wrapping; no rewrite of `SpecDecodingStats.observe_draft` or of the `len(generated_token_ids)-1` observation was found. This closes the obvious runtime-patch interpretation error, while retaining the limitation that historical installed scheduler bytes were not individually attested. The counter conservation checks are independent observational evidence, not a substitute for that attestation.

## Phase measurements already available

These are **own-span** means: summed recorded seconds divided by that same timer's recorded span count. They are not new experiments and are not substitutions for workload rates.

| Cohort | Target forward ms/span | Drafter ms/span | Committer dispatch ms/span | Admitted step wall ms/span |
| --- | ---: | ---: | ---: | ---: |
| Sr12 shared pair | 113.892 | 51.953 | 20.734 | 193.347 |
| Cqc16 shared pair | 114.566 | 52.493 | 20.684 | 194.442 |
| Cqc15 shared pair | 115.902 | 54.614 | 20.455 | 198.502 |
| Cqc10 ten tasks | 114.888 | 54.763 | 20.146 | 197.233 |

Every tree task boundary reports zero pending forward/drafter/committer CUDA-event samples. Forward, drafter, and committer span counts each match spec-event counts in these brackets. The boundary forward interval and complete-work-census counter also match the forward count. This supports task population alignment; it does not establish matching numerical inputs across revisions.

The forward timer brackets the pure-decode target model forward. The drafter timer brackets proposal generation. The committer timer brackets the **whole rejection-sampler dispatch**, including acceptance/path/bonus handling and commit work; **20.7 ms must not be labeled pure recurrent replay time**. Events are asynchronous, drained once their end event completes; task-boundary flush acknowledgements exclude pending tails from the next task. The instrument's code comments claim low overhead, but no same-input on/off overhead measurement is in these brackets, so zero overhead is not claimed.

The wall timer admits intervals only between compatible consecutive pure-decode steps, with a fixed request set and an interval cap. Its span population is smaller: Sr12 pair **4441 wall intervals versus 4483 spec events**. It omits request transitions, broken chains, and rejected intervals. Do not divide all accepted tokens by this subset wall sum, add it to CUDA-event totals, or label it total application time. Warmup is outside the task brackets.

Missing matched task-bound telemetry: isolated acceptance time, isolated native-replay time, attention-only time, convolution-only time, per-event latency distributions, and instrument overhead controls. GPU subspan means cannot establish a critical-path decomposition or each optimization's causal benefit.

## Configuration and identity gaps

The tree engine is **patched vLLM**, recorded version `0.19.2rc1.dev134+gfe9c3d6c5`. All three shared-pair revisions record the same patched-FA2 binary hash `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857` and runtime-attestation digest, but different repository commits. The SGLang launch uses FlashInfer and EAGLE steps/top-k/draft-tokens **3/1/4**. These are different complete serving configurations.

Requested sampling is temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, auto-continue off. Shared tree runs and earlier SGLang use response cap **32768**; later SGLang and Cqc10 use **24000**. Compaction has its own recorded cap. The earlier empty-patch thinking-only task was therefore not caused by a smaller 24000 cap in that run. Per-task cap/finish-count evidence remains bound through the existing projection.

The model display name is identical, but this does **not** prove identical checkpoint tensors, conversions, tokenizer bytes, KV precision, or generation configuration. SGLang's path is explicitly `...-radixark-asshipped`; the older calibration documentation describes aggressive quantization/auto-FP8 KV versus another conservative conversion. That older description is not sufficient to assign the same precision difference to every current tree build. The source table records the known identities and leaves the unresolved fields explicit. E3 must bind exact checkpoint/tokenizer/precision identities rather than inheriting a common display name.

Task concurrency is one. Tree `MAX_NUM_SEQS=1` is recorded, but earlier SGLang telemetry observed up to three physical requests, so "one agent task" must not automatically be described as identical engine concurrency. Historical SGLang per-container loaded code/image hashes are not recovered, even though the retained image and launch command are known. Current code execution should freeze and attest these before using historical results as causal controls.

## Independent reproduction

From the repository root:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 papers/gdn-tree-scan-mlsys/v2/results/review-response-20260927/reduce_p0.py
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s papers/gdn-tree-scan-mlsys/v2/results/review-response-20260927 -p 'test_*.py' -v
```

The reducer writes only this output directory; it does not change original evidence. Eleven tests include the extracted engine observation class and reject counter reset, missing position, nonmonotone survival, accepted-total mismatch, wrong model, duplicate series, one-sided counter availability and logical/physical denominator confusion, and distinguish pooling from averaging. Existing rate reducers also enforce quiescent boundaries and matching token/request populations. `MANIFEST.json` binds delivered files; `p0-evidence.json` binds consumed source inputs and historical git objects.

Paper-ready fragments are `all-attempts-table.tex` and `paired-task-outcomes.tex`. No fragment has been inserted into the manuscript by this delegated P0 task. Fresh continuation qualification and complete-cycle measurements remain necessary; recovered counters reduce missing instrumentation work but do not supply those missing experiments.
