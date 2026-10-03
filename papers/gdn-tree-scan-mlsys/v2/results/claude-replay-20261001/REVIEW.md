# Claude experiment audit and paper integration

Reviewed October 1, 2026, America/Los_Angeles. This is the new September 30/October 1 replay campaign, not the closed September 27 adversarial-response campaign or the older E8/Cat10 measurements. The October 1 human “Go!” in the Claude session authorizes its new follow-on plan. Historical scope files, failures, release tags, and the arXiv upload package have not been changed.

## Evidence and reconstruction

- Session: `4b42a19b-dcd2-4f56-b934-a8a23ca4ec19`. `CLAUDE-EXCERPTS.json` preserves bounded text records with original line numbers, timestamps, source path, and a snapshot byte count/SHA256. The live JSONL is append-only and may grow after that snapshot.
- Remote worktree: `/home/mark/shared/lumotree-v2exp-20260930`. Raw root: `/home/mark/shared/lumotree-v2exp-runs`. The older `/home/mark/lumotree-review-20260927` remains historical.
- `SYNC-MANIFEST.json`: 169 transferred harness/result/log files, all independently checked for byte size and SHA256. `CORPUS-MANIFEST.json` identifies all 43 incoming request files by SHA256; request contents remain on the remote machine.
- `RUN-SELECTION.json` freezes the complete, current sampled-protocol runs. `audit_replay.py` independently requires exactly 43 unique corpus requests per run, no request errors, positive finite decode durations, the sampled regime and 1,024-token cap, and no negative vLLM counter deltas. Run it with Python 3; it performs no inference.
- `KERNEL-SNAPSHOT-MANIFEST.json` records the later prepared kernel harness, separately from the completed replay evidence. It was copied at remote commit `b5d15a209`; no GPU result existed at review time.
- Native vLLM and LumoTree launcher/patcher/kernel/template hashes were compared with the newer deployment code. The two host model-directory aliases resolve to the same directory and configuration inode. The selected tree runs confirm `gqa_pair_splitk` engagement. SGLang uses a separate immutable image and attention/parser stack.

## What Claude added

There are **18 complete baseline/configuration runs plus 3 complete single-run variants**, each with 43 requests: 903 completed request records in the paper reduction. These are serving replays, not SWE-bench task attempts or full-model qualification cases.

| Configuration | Complete runs | Pooled tokens/s | Individual run range |
|---|---:|---:|---:|
| Plain vLLM | 1 | 10.54 | — |
| vLLM MTP-1 | 1 | 15.37 | — |
| vLLM MTP-3 | 1 | 22.06 | — |
| vLLM MTP-5 | 3 | 26.12 | 26.04–26.23 |
| vLLM MTP-7 | 1 | 25.72 | — |
| LumoTree | 3 | 28.79 | 28.38–29.41 |
| SGLang EAGLE 3 steps / 4 draft tokens | 1 | 26.32 | — |
| SGLang EAGLE 5 / 6 | 3 | 29.28 | 29.23–29.36 |
| SGLang EAGLE 7 / 8 | 3 | 30.05 | 29.18–30.69 |
| SGLang EAGLE 9 / 10 | 1 | 28.81 | — |
| LumoTree, grouped split-K disabled | 1 | 27.11 | — |
| LumoTree, fused draft top-k disabled | 1 | 29.40 | — |
| LumoTree, precomputed tap preparation disabled | 1 | 29.02 | — |

Each rate is **sum(completion tokens − requests) / sum(client decode seconds)** across the same requests and repetitions. Claude's summary averages per-run rates; that gives 28.81 for LumoTree and 30.06 for EAGLE 7/8. The paper instead uses the matched-total reduction, giving 28.79 and 30.05. Ranges are observations, not confidence intervals.

The corpus comes from Astropy 12907/13033 agent sessions, batch one, 23,024–68,557 reported vLLM prompt tokens. Settings are temperature 0.6, top-p 0.95, top-k 20, presence penalty 1, response cap 1,024, no per-request seed. Earlier greedy/per-request-seeded protocols and incomplete/failed runs are preserved outside the selected cohort. The attempted 31-node variant fails at runtime and contributes no successful tree-size result.

## Red-team verdict on completed experiments

**Admit a bounded serving-rate observation.** On these fixed incoming requests and this vLLM stack, LumoTree's pooled rate is 10.24% higher than MTP-5. It is not evidence of faster task completion, improved task quality, or distribution preservation. Native invocation and specialized tree attention differ; the comparison measures the integrated configurations, not the isolated recurrent tree algorithm.

**Do not admit SGLang parity/superiority as a controlled comparison yet.** Every one of the 43 request names has a reported SGLang prompt count exactly 146 above its vLLM count. Shared request files and a shared template file do not prove equal rendered token IDs. Tool parsers also differ (`qwen3_coder` versus `qwen3_xml`), as do attention and chunked-prefill implementations. Determine whether the 146-token discrepancy is actual serialization or accounting, and preserve rendered prompt IDs/hashes for both stacks. The 30.05 rate is descriptive until that issue is resolved.

**Reject the significance and causal-ablation claims.** The setting sweep is exploratory and unevenly replicated. Do not report Claude's p≈0.09, “marginally significant,” “split-K adds 6%,” or “other optimizations have no measurable effect.” All three variants have one stochastic run, with different output lengths. The observations cannot establish either a causal effect or absence of one.

**Keep acceptance counters distinct from committed paths.** Pooled output-side accepted-draft counts are 4.47 per event for LumoTree and 3.17 for MTP-5. Adding one produces 5.47/4.17 but does not turn the counter into committed-path length. SGLang exposes window averages rather than compatible cumulative totals.

**Treat stream timing as a proxy.** The client starts timing at the first nonempty parsed SSE delta, which can contain multiple speculative tokens; subtracting one token does not exactly remove that first chunk. Stream buffering, reasoning/tool parsing, finite response caps, cold/hit histories, effective KV capacity, and unmatched instrumentation may affect the comparison. Equal memory-fraction flags do not force equal KV-block counts.

## Ongoing plan and required corrections

### 1. Distribution diagnostic: running, not a correctness certificate

The current queue is AR → MTP-5 → LumoTree → AR → SGLang 7/8. Each arm samples 20 requests × 40 continuations × 24 output tokens. The first AR run completed all 800 requests at 08:50:05 Pacific, after 36.0 minutes of sampling plus startup/warmup. MTP-5 began at 08:50:28 and was booting at the 08:54 check. No arm has a reviewed distribution result. This is one completed diagnostic collection, not a passed full-model qualification.

Blocking issues in the prepared analyzer:

1. It retokenizes `reasoning + NUL + content + NUL + concatenated tool-call text`. Those are parsed response fields with artificial separators, not the actual generated token sequence. Capture token IDs before parser transformation; otherwise label the experiment a parsed-text diagnostic and choose observables that do not depend on backend parsing differences.
2. Both native AR launches use server seed 0 and the same request order. They are not established independent noise-floor samples. Freeze distinct server RNG seeds before the reference replicate runs; the tree route's refusal of per-request generators is a separate constraint.
3. Dropping sequences shorter than position j conditions on output length. Include EOS consistently or predefine a valid length-conditioned question.
4. Fisher aggregation assumes independent p-values, but token positions within the same continuation are dependent. Use request/continuation-level resampling or a permutation-calibrated aggregate, with a predefined multiple-testing policy.
5. A failure to reject difference is not an equivalence result. Specify effect-size margins, achievable power, and a planted negative control before interpreting candidate outcomes. Forty samples per prompt over a large vocabulary provide limited sensitivity.
6. Matching token-position marginals does not prove conditional or joint sampling preservation. This diagnostic cannot replace connected full-model state and next-forward checks.

The analyzer was untracked when inspected. Freeze its source and protocol before collecting or interpreting candidate results. Preserve current samples if the analysis changes; do not relabel them as a predeclared equivalence test.

### 2. Kernel timing and memory: prepared, zero measured cells

The new harness covers fixed32 LumoTree scan/native replay, TreeWY author default, Weaver author default and an FP32-preparation port, plus two materialize-all-node-state baselines. It correctly distinguishes measured allocator statistics from logical state exports and explicitly includes a TreeWY final deferred flush and Weaver accepted replay. A smoke run is preparation, not a completed timing experiment.

Before making a ranking:

- Restore a numerical admission rule or label results as timing/error tradeoffs. The new harness makes error informational, whereas the prior frozen M1 study rejected three author policies under its comparators. Timing alone cannot override those failures.
- Keep author-default normalization separate from aligned ports. TreeWY's norm clamp differs from Lumo/native's epsilon-regularized norm, especially for near-zero keys; byte-identical raw operands do not establish identical mathematics.
- Include required Weaver topology rebuilding in a claimed complete GDN step, or explicitly condition on static reused topology. Include TreeWY's final flush in finite-session totals; a difference of timings is an estimate, not a measured standalone commit kernel.
- Label the measured boundary **GDN verification and publication**. Convolution, attention/KV, sampling, drafting and pending-token materialization are excluded. Capturing 48 adjacent synthetic GDN calls is not the full deployed forward's cache behavior.
- The tree-size sweep uses a generic route, not the deployed fixed32 route. It is not evidence that the failed 31-node serving variant works.
- Compare common persistent-state/storage policies when attributing memory savings. Lumo's page storage, author pools, naive state banks, graph pools, allocated scratch, and logical bytes written have different meanings.

The stated 20–30-minute full-run duration is an estimate, not an observed GPU duration. No timed mechanism cells or measured-memory results are added to the paper yet.

### 3. Ten-task agent study: planned, zero new task attempts

The latest Claude follow-up orders kernel smoke/full after the distribution queue, then ten Astropy tasks each under LumoTree, MTP-5 and SGLang 7/8: **30 planned attempts**. Plain autoregressive task decoding is not in that queue. This new human-approved plan is separate from the earlier closed four-attempt study.

Before launching, verify exact task IDs/subset hashes, fresh task workspaces, effective response and context caps, agent/proxy/parser versions, sampling settings, tool budgets, evaluator images/timeouts, and matched treatment of task failures. Save generated patches, agent transcripts, evaluator results and matched token/time totals for every task. Confirm backend tokenized inputs and actual KV capacity where a controlled comparison is claimed. Do not select tasks or discard failed patches after observing outcomes. The existing 25.63-token/s ten-task deployment remains its own cohort.

The proposed 3.3 hours per arm / 10 hours total is not yet measured for these new arms. Thirty attempts have 9,000-second agent budgets each: that ceiling alone is 75 agent-hours, before startup/evaluation. Use observed per-task durations as results arrive; do not promise a fixed ten-hour completion.

At **08:59:26 Pacific**, exactly one experiment container (`v2exp-mtp5`) was active. AR had completed 800/800 unique prompt/sample pairs without request errors; MTP-5 had collected 16/800 after a 331-second boot and warmup. Completed new full-model qualification cases: **0**. Completed new timed mechanism cells: **0**. New agent task attempts: **0/30**. The AR collection took about 43 minutes including startup; only that duration is observed for this diagnostic queue. A remaining 1–3-hour collection window is a rough projection, not a commitment. Kernel first-run duration and task-arm durations are still unmeasured, and the methodological corrections prevent a defensible overall scientific-completion ETA.

### 4. Full-model qualification remains the central missing claim

The deployed route rejects temperature-zero requests, and native MTP/plain greedy disagreement is not proof that LumoTree is correct. Qualify a stable native reference with equal KV capacity and one common long prompt; then check all 84 path cases with connected recurrent/convolution/KV publication, pending-token handling and subsequent full-forward logits over multiple cycles. Freeze numerical rules before candidate outcomes. The new distribution diagnostic and isolated GDN benchmark do not close this gap.

## Paper changes

Added separate replay protocol, a complete baseline/configuration table and three single-run variants; updated the abstract, introduction, discussion, conclusion and Limitations. The paper uses pooled rates, states the SGLang prompt-count discrepancy, and admits no significance, causal-ablation, task-speed or full-model-equivalence claim. The historical ten-task outcomes and component results remain era-bound. The stable existing release tag and yesterday's arXiv upload package are preserved.
