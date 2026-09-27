# Stronger B1 NVFP4 workload records — 2026-09-24

## Finding and recommended use

**Yes: stronger existing B1 SWE observations are recoverable.** The current Cqc10 pooled 25.63 tokens/s is a ten-task cohort result, not the best observed tree rate. More usefully than selecting fast unrelated tasks, the repository contains three tree runs of exactly the two completed SGLang task IDs, `astropy__astropy-12907` and `astropy__astropy-13033`. All three exceed SGLang's 26.8939560242 pooled tokens/s under the same `(N−R)/(ΣE2E−ΣTTFT)` estimator.

For a latest-applicable shared-task comparison, prefer **Cqc15, August 23: 27.2680081503 versus SGLang 26.8939560242, +1.3908408482% descriptive**. It is the latest recovered tree execution of those IDs, rather than the fastest older run. Keep Sr12/Cqc16 in the chronology, and retain Cqc10 as the broader latest completed ten-task segment. Do not call any of these a current-HEAD rerun, a statistically established speedup, or a general win over frontier systems. A specific served SGLang EAGLE configuration is the actual external comparator.

No inference, rerun, source change or paper edit was performed. Read-only SSH collected original metric lines, safe metadata/evaluation fields, exact source hashes, loaded-extension engagement and trace-screen summaries. Parent independently recomputed the three shared-task rates from the retrieved lines.

## Shared-task records, all retained

Both tasks have completed agent metadata and evaluation in every listed row. Verdicts agree across all four rows: 12907 resolved; 13033 failed tests.

| Deployment / date | N | R | ΣE2E seconds | ΣTTFT seconds | Pooled tokens/s | vs SGLang | Agent seconds summed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sr12 split-K, Aug19 | 25027 | 41 | 989.2611684799194 | 130.44221544265747 | 29.093442700158857 | +8.1783679351% | 1038.755 |
| Cqc16 split-K, Aug19 | 26239 | 35 | 1048.7886128425598 | 122.29720187187195 | 28.28304686877344 | +5.1650669888% | 1083.307 |
| Cqc15 split-K, Aug23 | 70424 | 53 | 2849.883605480194 | 269.1670525074005 | 27.26800815027045 | +1.3908408482% | 2920.556 |
| SGLang EAGLE, Aug24 | 47854 | 45 | 2013.6344224798959 | 235.9487611737568 | 26.893956024189762 | reference | 2043.900 |

The latest tree pair generates more tokens and takes more total agent time than SGLang despite its slightly higher decode rate. This directly prevents interpreting +1.39% as a task-completion gain. The pairs are the complete intersection of recovered completed SGLang tasks with each listed tree run, not a speed-selected intersection. These are developmental runs with stochastic trajectories, different source revisions and output budgets, not prospectively paired replicates. The original larger campaign includes unsuccessful later attempts; see below.

## Full cohorts and per-task extrema

The complete **Sr12 four-task** run (12907/13033/13236/13398) pools to **26.2053649279**, from N=101923, R=124, E=4300.757935524 s, F=416.095375776 s. All four have complete metadata/evaluation, two resolved and two failed tests. This is a complete selected cohort, modestly above Cqc10's 25.6336224659 but not a matched four-task SGLang comparison.

The remembered **28.8 @ 196 ms** headline traces to this Sr12 run's inverse mean request-TPOT statistic (28.8188196328), not its new pooled 26.2053649279. Public lineage includes `11b0eeb0d345b03d35fe6d3ebdf4cba09b4c2d15` and `60a76da627cb0629a22659e1b36aa7cf8a714d93`; upstream reuse is `b8c23b4d04f72d957ba652e17384199b5d36cdf2`. Those numbers cannot be relabeled as pooled decode. Sr12 is a still-applicable split-K mechanism observation; the estimator and cohort must be explicit.

Within current Cqc10, identical reduction gives 14309=30.6707354723 (resolved), 14995=30.2165108794 (resolved), and 14182=28.3394456401 (failed). These prove 25.63 is not a maximum. Reporting only these favorable tasks as the system's cohort rate would be selection-biased. All ten remain in the current shared-rate audit.

## Route, caps and trace evidence

All three actual `logs/fr13_fa2_qrow32_b1_production_engagement.json` records independently state:

- `status=ENGAGED`, `candidate_served=true`, `batch_size=1`, 32 physical rows.
- `arm=gqa_pair_splitk`, `num_splits=4`, 16 full-attention layers, `fallback_allowed=false`.
- Full vocabulary (`draft_vocab_k=0`, `draft_vocab_root=0`).
- Identical loaded FA2 binary SHA-256 `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`; runtime-bundle canonical hash `1289c82bd59f6f44a2d80a200e6d96c0b50d5c83d51bfecd339a0489d7d603b9`.

The model/route is the NVFP4 Hydra27 deployment, not Cat10/stock TREE_ATTN. Sr12/Cqc16 boot excerpts explicitly identify `/models/qwen3.8-27b-nvfp4-radixark`, ModelOpt mixed quantization, BF16 execution, and quantized NVFP4 target/MTP LM heads. Source/patch versions differ; shared binary identity does not imply every emitted Python file or model-file byte was identically attested:

| Run | Served source commit | FA2 patch source SHA-256 |
|---|---|---|
| Sr12 | `bdca0bd50cbf5643205c87e4e3e2fe668895c364` | `e80ed4ea84f3259cbcde270d35898aa5c7a9f84b6b0da6343ac830b160f0e18b` |
| Cqc16 | `78a29d3391d02166abf571030a90645ce2375ba0` | `78206cc447e112686c068fb3a73916ac7f755e82ec58e5507504d4ca17788d6f` |
| Cqc15 | `a450f6c3600d088bd259df65e3940a64dc8de199` | `dd424b0524b3520958d1ca323f6285bf80a710fa3e148b33310b7db08e19b93e` |

All three saved proxy environments explicitly set output cap **32768**, versus SGLang's **24000**. Temperature .6, top-p .95, top-k20, min-p0, presence1 and disabled auto-continuation agree. Every selected task's recorded network fingerprint matches SGLang/Cqc10: `e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4`.

All selected requests finish `stop`; length, abort, error and repetition deltas are zero. Every selected request in Sr12/Cqc16 and Cqc15/12907 is at most 20000 output tokens by the saved histogram. Cqc15/13033 has 27 of 28 requests at most 20000, with one above 20000; the available bucket does not prove that one is below 24000. Thus no recorded 32768 cap was hit, but do not assert the changed 24000 counterfactual budget is irrelevant.

The same source-hashed trace screen used in the current paper was rerun read-only (`promotion_ab_eyeball.py` SHA `f4cc87ade9f4e6866dafcdf8f2429085ede0dce5aceb96db380e0cc808aa2ea0`). All six shared-task traces have tool calls, nonempty saved patches, zero malformed tool calls, and no degeneration flag. This bounded heuristic does not prove task quality or absence of all loops.

## Adverse and excluded records, explicit reasons

- **Cqc16 third task13236:** 0 visible characters, 0 tools, 70755 thinking characters; the recorded degeneration heuristic fires. No final agent metadata/evaluation. Its engine bracket itself is idle/population-aligned and would yield a three-task engine pool of 27.9331124870; it is retained as adverse, not presented as a completed-task result. The good shared pair does not erase this failure.
- **Cqc15 third task13398:** no final metadata/evaluation; trace has 147 tool calls and is not flagged by the same heuristic. Post-bracket has one running request, generation counter205470 vs completed-token histogram203612, TTFTcount150 vs completed149. It cannot enter a completed-request comparison using the global generation delta. Its evidence is retained, not mislabeled as degeneration.
- **Cp1 fourth task13398:** no final metadata/evaluation; one request still running, generated189933 vs completed188089, TTFTcount113 vs completed112. The original four-task control is censored. Its first three completed tasks pool24.5710107717; its shared two pool24.7401875088. Do not turn Sr12's whole-four versus Cp1's completed-three into an isolated component gain.
- **Cqc12 second task13579:** one request remains running, generated211476 vs completed200319, TTFT86 vs completed85; no final evaluation/metadata. First completed task13453 pools27.7933668248, resolved. Earlier unharvested13453 attempt pools26.6676920495 at engine level but has no final metadata/evaluation and is not a completed-task observation.

## Exact raw paths and independent extracts

Remote base: `mark@100.103.10.122:/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/output/`.

| Run | Root / arm |
|---|---|
| Cp1 | `fr14_promoab_Cp1_20260818T081918Z/hydra27_fixed32_promoab_Cp1` |
| Sr12 | `fr14_promoab_Sr12_20260819T043506Z/hydra27_fixed32_promoab_Sr12` |
| Cqc16 | `fr14_promoab_Cqc16_20260819T222438Z/hydra27_fixed32_promoab_Cqc16` |
| Cqc15 | `fr14_promoab_Cqc15_20260823T193658Z/hydra27_fixed32_promoab_Cqc15` |
| Cqc12 | `fr14_promoab_Cqc12_20260824T021301Z/hydra27_fixed32_promoab_Cqc12` |

Under each arm, task inputs are `swe_out/verified/per_task/astropy__astropy-ID/{vllm_metrics_pre.txt,vllm_metrics_post.txt,runner_metadata.json,eval/eval_report.json}`. Config/engagement files are named above. SGLang originals are already in the repository at `results/fr14_nvfp4_port_20260816/sglang16_evidence/{run1,r1}/swe_out/verified/per_task/astropy__astropy-{12907,13033}/` respectively.

Local audit extracts retain original paths and SHA-256s. These are source-bound projections, not full raw archives; no raw environment secrets were copied:

- `/tmp/nvfp4-b1-cohort-brackets-20260924.json`: `5eb43689e82c402ac800c34777855c2f10e74eea90a7e91b74dcfdf2f9a9fba2`
- `/tmp/nvfp4-b1-cohort-rates-20260924.json`: `4df1578b8def6361e33f18a973d2c9b3f64d07aca76fac07e4ba904e2abd3d05`
- `/tmp/nvfp4-b1-cohort-config-behavior-20260924.json`: `d25ca9d36dbdcb94e701ed6dda1a879212702a49666072d84598786b31d55a77`
- `/tmp/nvfp4-b1-cohort-engagement-20260924.json`: `cd3de566e6f14d6dcd98414f240a17ec04bb497f4b16a70858efcc6b1108fae4`
- `/tmp/nvfp4-b1-cohort-caps-20260924.json`: `8efcf7a07982f1eda49c2eade9bd62fd1b754180cf2eb62a3dd8c35004602a2e`

DSpark's actual external baseline remains synthetic random requests; DFlash2 evidence includes captured-input work and an FP8 serving smoke. Neither supplies an additional SWE pooled baseline. Their exact history is in `notes/external-baseline-history-audit-2026-09-23.md`. Existing results suffice to answer the stronger-number question; no new benchmark is necessary to expose these observations.

## Authorized durable raw recovery — 2026-09-24

The user subsequently authorized reporting the best recorded shared-task result. This changes the manuscript selection policy, not the evidence: Sr12's best shared-two 29.0934427 must still be distinguished from its full-four 26.2053649 and the later Cqc16/Cqc15 records. No manuscript file was changed by this reviewer.

Recovered 32 original files (eight complete task records, four files each) into `results/agent-workload/raw/shared-tasks-20260924/`. Layout is `<Sr12|Cqc16|Cqc15>/swe_out/verified/per_task/astropy__astropy-ID/`. Sr12 includes all four tasks; Cqc16 and Cqc15 each include both shared task IDs. Every file's exact original path, size, source hash, copied hash and projection flag is in `MANIFEST.json`, SHA-256 `2e20393e2fa238d489215a11d3b3315a08e9516609f341713ecd1ae62ed4dbd2`.

All 32 originals match the previous read-only raw extract hashes byte for byte. A credential-pattern/key scan inspected their content without printing sensitive values before writing the new snapshot; **no unsafe metadata was found and zero raw files required projection or redaction**. No raw environment or full trace was copied. Three already-safe, explicitly labeled config/behavior, engagement and cap projections were copied unchanged into `audit-projections/`, with source and copied hashes in the manifest. These remain projections, not full originals.

The existing common `Audit.task` validator was independently applied to the recovered files without creating source/test files. All eight records pass its label, integer-count, no-reset, idle-boundary, histogram/counter population, task-identity, terminal metadata and evaluation checks. Direct raw-file pooling reproduces:

- Sr12 all four: 26.20536492791583; shared two: 29.093442700158857.
- Cqc16 shared two: 28.28304686877344.
- Cqc15 shared two: 27.26800815027045.

The parent's dedicated shared-task reducer/manuscript revision is pending at this recovery checkpoint. This record does not pre-approve a not-yet-reviewed implementation or final archive. Original experiment/source files and earlier manifests were untouched; no inference ran.
