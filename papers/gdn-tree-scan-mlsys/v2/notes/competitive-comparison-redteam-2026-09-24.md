# Competitive comparison red-team — 2026-09-24

**PASS for the reviewed manuscript source, recovered two-task SGLang evidence, and combined CPU reduction.** The stronger recorded SGLang comparator is now retained. No remaining material issue was found in this bounded check. This is a source/data review; final PDF rendering, supporting-document updates, and final archive extraction remain the parent's delivery checks. No inference was run and no manuscript, attempted run, or production source was modified.

## Numerical reconstruction and selection

I independently verified all 12 files in `raw/sglang-step2-20260924/manifest.json` against their recorded sizes and SHA-256 values, then called the unchanged `shared_rate_reduce.Audit.task` directly on tasks 12907 and 13033. Both pass idle-boundary, completed-token histogram, request-count, E2E/TTFT population, and zero-nonstreaming-contribution checks. The result retains the failed task.

| Earlier SGLang task | N output tokens | R requests | Sum E2E (s) | Sum TTFT (s) | Pooled tokens/s |
|---|---:|---:|---:|---:|---:|
| 12907 | 5,895 | 11 | 230.332785853185 | 20.366569926031 | 28.023555951694 |
| 13033 | 41,955 | 6 | 1,360.571043190081 | 12.484188324073 | 31.117431231217 |
| Pooled pair | 47,850 | 17 | 1,590.903829043265 | 32.850758250104 | **30.700494672912** |

The pair has 47,833 conventional post-first-output intervals over 1,558.053070793161 summed request seconds. The best recovered tree pair, Sr12, is **29.093442700159**, or **5.234612633689% below** this SGLang pair. It is **8.178367935126% above** the later SGLang pair's 26.893956024190. All three tree observations and both SGLang observations are kept. These complete-system comparisons may include their respective algorithm/serving choices; no generic software-method mismatch invalidates the observed rates.

`competitive_rate_reduce.py:29–45` replays both saved child reductions, checks identical shared task IDs for all five rows, selects tree maximum only from Sr12/Cqc16/Cqc15 and SGLang maximum from both recovered SGLang runs, and retains the separate full-four-task Sr12 and ten-task Cqc10 populations. This correctly chooses `Sr12` and `SGLang_earlier`. The claim is the strongest among the explicitly recovered shared-task observations, not proof of an optimum over every possible setting. No outcome filter or token-cap filter removes the unfavorable SGLang task. The incomplete fourth step-2 task and the third task with mixed streaming/nonstreaming timing are not relabeled as members of this two-task streaming-only population.

## Configuration, completion, and patch scope

The recorded PID-1 command identifies FlashInfer, EAGLE speculative steps/top-k/draft-token settings 3/1/4, the as-shipped RadixArk Qwen3.8-27B NVFP4 path, and an 8,192-token prefill chunk. I independently checked the safe configuration projection against the original recorded proxy-environment hash and all 12 allowlisted values, plus all three cited source-file hashes. The source explicitly requests temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, auto-continuation off, and a 32,768-token response cap. No raw environment or credential was copied or printed.

I independently read the two original remote traces without copying their contents, checked their hashes/sizes, and reproduced the projection's assistant IDs, stop reasons, output usage, text/thinking character counts, and tool-call counts. Nonzero usage records reconcile exactly with the metrics: 11 records / 5,895 tokens for 12907 and 6 / 41,955 for 13033. Task 12907 resolved and its metadata records a 504-byte patch. Task 13033 records a zero-byte patch and a synthetic failure with `harness_invoked=false`; it is not an executed-test failure. Its last response has 32,768 output tokens, 136,053 thinking characters, zero text characters and zero tool calls. The recorded `stop_reason` is null: the evidence establishes reaching the configured token cap, not a literal `length` finish-reason receipt. The manuscript uses the supported wording.

Both task runners exited zero without task timeout or campaign-budget capping; that does not imply absence of a per-response cap. The recorded harness runs one task at a time while occasional auxiliary model requests overlap (the archived note reports physical running-request maximum three). The current body correctly states task concurrency and explains that overlapping requests remain in summed request time; it does not assert every SGLang physical step is B1.

## Manuscript source disposition

- `abstract.tex:2` retains both 26.89 and 30.70 beside tree 29.09 under one estimator; its abstract environment is intact.
- `results/agent-workload/case-study.tex:36–45` defines the pooled estimator and excludes first-output latency, with an accumulated-request-time denominator.
- `:47–54` gives both SGLang populations, the early response cap, synthetic empty-patch failure, auxiliary overlap, all selected tree runs, and both signed comparisons.
- `:57–75` marks each system's best recorded shared-task rate, preserves all five two-task rows, and separates the larger Sr12/Cqc10 populations and adverse lineage records. The Hydra27 description states the current selected tree geometry, rather than importing Cat10 into the agent deployment.
- `main.tex:330–332` reports both SGLang observations and makes no general-superiority or full-model-equivalence claim.

The prior weaker-baseline-only headline is superseded by this expanded comparison. Original raw files and old reducer outputs remain immutable audit history.

## Replay and artifact dependencies

The final competitive receipt binds **94 input files**, **5 configuration/behavior projections**, **4 reducer scripts**, and **2 saved child audit receipts**. I copied exactly these **105 files** into an otherwise empty temporary repository layout and ran:

```sh
python3 /var/folders/xc/sy7ktq0n42d1n78zg10b8p_r0000gn/T/competitive-rate-review-20260924-qnfp1mbh/papers/gdn-tree-scan-mlsys/v2/results/agent-workload/competitive_rate_reduce.py
```

The output was byte-identical to `competitive-rate-audit.json`. Independently executed negatives in disposable copies rejected (1) a changed early-SGLang raw metric file via its manifest check, and (2) a changed saved child audit via byte-for-byte replay comparison. Restored clean replay passed again. The standalone `sglang_step2_reduce.py` also reproduces its saved output byte-for-byte. The final combined audit now binds both new SGLang projections in addition to the three tree projections, closing the initially noted receipt-identity omission. Paths outside the paper subtree must retain their repository-relative layout on extraction; this was exercised by the isolated replay.

## Reviewed SHA-256 identities

| File, relative to paper v2 | SHA-256 |
|---|---|
| `main.tex` | `1d8d5f381176a5d30d2152a8cfcf870ae77ad85ed7ee376369a35fef0231dea8` |
| `abstract.tex` | `ec48b81b0f535167202dd976a39803c1f04c0db27544df1e9ac52740b0daaee3` |
| `results/agent-workload/case-study.tex` | `0e553bc129a8139125b7928af5b457f60d6f7b38f0a47057b3ae4b3bd0f569d6` |
| `results/agent-workload/competitive_rate_reduce.py` | `a2d2a669cf63061d5fabddcd39778dadd0dff9c031fec3b9d99ccd9d713c310c` |
| `results/agent-workload/competitive-rate-audit.json` | `e30060ec1c4ade6f4b3765e4a0c55c25cff3ba121715e4ad26f2f2a44abecac7` |
| `results/agent-workload/sglang_step2_reduce.py` | `9315e2fb93252234c316fde3e684839cf7ba58c047b1d89f1e36546cdc5a237f` |
| `results/agent-workload/sglang-step2-rate-audit.json` | `c1e01be7b5a1bc11cce67666a112fe5c4c6478ae10b3a3315d95eeb8c5e39787` |
| `results/agent-workload/shared_task_reduce.py` | `d007110971ed8d3d0b2829a4f1f8e617ff12faaf88ca6231263997cfff4881ca` |
| `results/agent-workload/shared_rate_reduce.py` | `c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f` |
| `results/agent-workload/raw/sglang-step2-20260924/manifest.json` | `a524f1e25270a9ca87f737b6f99ff5ae3e814ac265b3c0781d7824caffe5d02e` |
| `results/agent-workload/sglang-step2-config-projection.json` | `5ebbd60c77b000efc5890a6e97bd33c249f30583a7162b6258e05b6ba13055dc` |
| `results/agent-workload/sglang-step2-behavior-projection.json` | `8ffb89b13304bf4a3005725085234f0af3bb53b8a79059b929c4b55488866115` |
