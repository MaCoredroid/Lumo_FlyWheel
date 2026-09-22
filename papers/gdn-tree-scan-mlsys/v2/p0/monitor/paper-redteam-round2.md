# Independent manuscript red-team, round 2

Reviewed the current corrected candidate in `p0/monitor/paper-build-review1`, directly against source, historical raw JSON and historical instrument source, current E7a result JSON, and Bole/TreeWY primary papers. No previous review conclusions were used. No inference, source changes, tmux operations, or experiment mutations were performed. No applicable ancestor or paper-subtree `AGENTS.md` was found locally or in the remote worktree.

**Verdict: one remaining material accounting finding.** The E7a corrections checked below are supported by the raw results. Failed frozen checks do not need to become passing checks to support the explicitly bounded characterization. E2/E7b and E1 remain pending, as the manuscript already states.

## Reviewed identity

Candidate root: `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/p0/monitor/paper-build-review1/`.

| File | SHA-256 |
|---|---|
| `main.tex` | `ca191a41d714854bb8afd478e169adc11eb98d949ab2749e838b492021dcfeb5` |
| `abstract.tex` | `e2aa41e1a353592aa0bb8a1d5dc2821bc312c9f24295ba204dbcfe2e0e6523af` |
| `ref.bib` | `51c788f1ffcaef1cc1b1a7846be1e862cbbc56e58d041c37bfc267d703e22905` |
| `results/historical-macros.tex` | `228e73de3c30abb84d728c7a86f814bb31f93f6f3b2970394956c7ab881b6c09` |
| `figures/cost-breakdown.tikz` | `f5c53a1d8c30802c1e13e448a198b3ba2f7e77daf6230892f2e7c4130507f098` |
| `main.pdf` | `d06e7d7188c8251fb0c1e249bb555286db05037d6d8391b4e22e712e6da9d4f7` |

The PDF hash identifies the associated build; this review checked substantive source claims, not a full visual-layout pass. Source hashes were unchanged when rechecked at the end of evidence gathering.

## P2 — H3 still presents a mixed-population accounting proxy as matched decode accounting

**Locations:** `main.tex:209–220`, `247–285`, `396`; `abstract.tex:2`; `figures/cost-breakdown.tikz:15–16`. The most direct unsupported implications are the equation's actual-emitted-token numerator, the interpretation of the residual as host work, and the cost figure's implied decomposition of the measured wall time.

The per-physical-step to per-request-event conversion repairs units, but it does not make the underlying event populations equal. H3's reported rate is exactly

`(global accepted tokens / global draft events + 1) / (retained pure-decode wall seconds / retained pure-decode wall events)`.

The numerator includes events on mixed prefill/decode steps; the wall denominator excludes those and also drops intervals where the consecutive-pure-step chain breaks. The verification GPU timer has a third support set: all measured pure-decode forward events. Drafter and committer span timers explicitly have **no pure-decode gate**. Their average per-span costs are divided by occupancy measured on pure-decode forward steps. These are reproducible estimates/proxies, not one matched set of emitted tokens and complete wall intervals.

### Direct historical evidence

All paths below are relative to `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel`. Each raw file's hash matches `v2/notes/evidence-sources.json`:

| Arm/raw path under `output/fr13_kvremap_tail6/` | SHA-256 |
|---|---|
| `native5_control_kvr1/deploy_speed_kvr1.json` | `88deadb9f1ffac9985f263b985dae0dae16a8b2be09da30b2b076efc22cdee87` |
| `native11_control_kvr1/deploy_speed_kvr1.json` | `acd116d76f6f58ccd6d8f72b668b90f2fabd75cb2da1292a04ed6fa6e2caeafc` |
| `kvremap_tail6_kvr1/deploy_speed_kvr1.json` | `bd8b1d65054c852053bf4492754af879769339c639f6d7a43f4e99e7bb41f2d8` |

Counters are in `raw_counter_delta_aggregate`, with the common `vllm:` prefix:

| Counter | MTP-5 | MTP-11 | Tree |
|---|---:|---:|---:|
| `spec_decode_num_drafts_total` | 115,663 | 113,620 | 103,456 |
| `spec_decode_num_accepted_tokens_total` | 395,775 | 559,400 | 443,372 |
| `fr13_decode_forward_gpu_drafts_total` | 89,030 | 72,168 | 67,525 |
| `fr13_decode_forward_gpu_steps_total` | 31,923 | 26,293 | 24,129 |
| `fr13_decode_step_wall_drafts_total` | 85,805 | 69,077 | 65,488 |
| `fr13_decode_step_wall_steps_total` | 30,703 | 25,097 | 23,363 |
| `fr13_drafter_gpu_spans_total` | 42,824 | 44,366 | 39,771 |
| `fr13_committer_gpu_spans_total` | 42,550 | 43,923 | 39,339 |

Thus retained wall events cover 74.19%, 60.80%, and 63.30% of global draft events. Drafter/committer spans cover 1.33–1.69 times the pure-forward step counts. The mismatch is arm-dependent. The raw counters do not supply accepted-token totals for the retained wall subset, so they cannot establish the missing support equality retrospectively.

This was verified against **each historical revision**, not inferred only from current code:

- MTP-5: `103cedccd75b3149c87b465c9201416abff7533a`.
- MTP-11: `2fc0faf624eb44dc1bfd43cf48e915c936b72804`.
- Tree: `224c5db735dae01bd5a6b4d1dd1599db31640175`.

At all three revisions, `scripts/fr13_measure.py` has SHA-256 `b8fd03920debea2902666efb943beb04c075188fac93bb9f756ca4ac9faee8b9`: lines 1586–1587 compute global accepted/drafts plus one; 1605 computes the normalized component estimate; 1620–1626 compute the wall proxy; 1628–1630 define `overhead_other_ms_per_event` by subtraction. `scripts/fr10_phase4_patch_vllm_tree_gdn.py` has SHA-256 `31cbd3fbab034755c76eb457ee915606d6d8e86afb5b84bc4115565ac6e269e2`: lines 15926–15947 define consecutive-pure-step wall intervals and break on mixed/prefill; lines 16101–16109 define the pure-forward gate; lines 16151–16156 explicitly document ungated component spans. These are reconstructed historical sources, subject to the already disclosed unresolved loaded-binary identity.

The numerator is also a **structural accepted-plus-one count**, not an observed actual-emitted count. Accepted tokens plus draft events equal 511,438 / 673,020 / 546,828, while `generation_tokens_total` is 510,460 / 671,097 / 544,713. Those totals differ by 978 / 1,923 / 2,115 tokens (0.19% / 0.29% / 0.39% of the structural counts). The aggregate does not identify the entire cause; do not attribute this difference exclusively to EOS/truncation or replace the numerator mechanically with `generation_tokens_total`, which still lacks matched pure-wall support.

**Why material:** the 42.74/39.95/32.85 numbers are exactly reproducible, but they do not directly measure tokens emitted during the timed decode intervals. Similarly, subtracting differently supported component averages from wall time does not isolate genuine host cost. The `1.017–1.025` alignment ratio is not independent evidence that these populations align; it compares estimates sharing the same structural numerator, and the residual is defined to close their sum. The conclusion may retain the historical proxy ordering; this review does not show that the ordering reverses.

**Minimum fix, no new historical GPU experiment required:** retain the original files/numbers; label H3 explicitly as the archived structural-token/pure-wall accounting proxy; state the distinct populations and unverified representativeness assumption; change the cost figure's `Host/other` to `Residual estimate` and describe component values as normalized estimates. Qualify the abstract, three-arm heading/conclusion, 19.5% gain, and break-even calculation accordingly. The latter remains a conditional threshold for that proxy. Avoid asserting a matched pure-decode throughput or measured host decomposition.

**Experiment implication:** directly matched per-event actual emitted tokens and retained wall intervals are required if a future E1 claim is measured decode throughput; component attribution also requires spans matched to those events. E1 is already planned, so this is a measurement requirement for existing work, not a newly demanded campaign. Honest historical-proxy labeling resolves the retained historical claim.

## Checks that did not produce another material finding

- Read Bole's closed-form/state lifecycle, kernel, and setup from [Bole v1](https://arxiv.org/html/2608.01651v1), especially Sections IV and VI-A/VI-D. The manuscript correctly recognizes finite-Neumann propagation, compact accepted-state reconstruction, SGLang, unquantized Qwen3.5-27B/GB10, and session replay. It does not infer a comparative system ranking from the local reimplementation.
- Read [TreeWY v1](https://arxiv.org/html/2608.20961v1), Sections 3–5. The triangular/pseudo-value description, greedy verification, disabled prefix caching, B200/model distinctions, finite-precision caveat, and wider-tree throughput limitation are supported. No matched external run is necessary for the currently disclaimed superiority claim.
- Independently reduced all 9 historical/synthetic, 8 fresh-pilot, and 31 primary confirmation JSON result files under the remote worktree `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/`. The reported fp32/tf32 ranges, compact/replay absolute extrema, minimum bfloat16 agreement, and native-seam characterization checked against the corresponding raw fields. The confirmation compact maxima are `7.814217335422313e-7` against oracle and `1.9073486328125e-6` against native; replay maxima are `1.3365283662025718e-6` and `2.384185791015625e-7`. These are **absolute** errors; adding that word at `main.tex:352` would avoid ambiguity after earlier relative-error discussion, but this is not a separate material finding.
- Confirmation C's minimum bfloat16 agreement is `0.9997721354166667`, on p005 (`out-20260922T051611Z-e7a-fresh-confirmation/result_29_layers_62_linear_attn.json`), exactly 14 of 61,440 elements differing. Served-vs-oracle maximum absolute error is `0.00048772463622170914`, consistent with the rounded text.
- Recomputed timing dispersion from each raw cell's `(sync_p90_us-sync_p10_us)/(2*sync_median_us)` for the four frozen verify-class kernels: the fresh-pilot-v2 has 7 failing cells across 6 prefixes, maximum 20.2251%; confirmation has 14 across 12, maximum 32.0861%, plus 3 probe-drift failures. The manuscript's corrected counts match. These are dispersion measures, not confidence intervals.
- Inspected original `capture_provenance.json` records across all 32 confirmation prefixes: 31 pass and p087 fails. Its original failure remains separate from retrospective reverdicts. No requalification is inferred.
- H1's 15/15 to 0/15 fixture and 0/84, 0/61 script observations, H2's later acceptance figures, and H4's 27.03 ms/step summary match their specifically identified archive documents. Their limitations are retained.

Scope limits: this was an evidence/claim review, not a fresh tensor execution, proof of sampler law, rerun of historical binaries, or validation of ongoing E7b/E1 instrumentation. Packaging fresh evidence and finishing the already tracked qualification work remain outstanding; they are not new findings in this review.

## Resolution recheck — H3 finding closed

Rechecked the revised canonical local `v2/` sources and supporting files. An intermediate source (`main.tex` SHA-256 `d9ddf00edc23eecec4345016d2dec23469b8638f16d039f00399680b42d44b1b`) resolved the substantive population/residual interpretation but retained an overbroad evidence-table label and omitted definitions for the new barred quantities. The coordinator corrected those before this final recheck.

**Closed for the final source hash below. No remaining material H3 rate/host claim or algebra inconsistency was found.** The abstract, introduction (`main.tex:38`), evidence table (`190`), explicit proxy equation and population accounting (`209–219`), results/table/figure (`246–271`), conditional threshold (`275–284`), and conclusion (`395`) now consistently retain an archive proxy. The figure labels its subtraction term `Residual estimate`; its caption explicitly denies a measured-host interpretation. The threshold defines `bar a = A_all/D_all` and `bar t = W_pure/D_wall` and is limited to proxy parity. The E1 requirement (`388`) now explicitly matches actual emitted tokens, wall intervals, and component spans by event identifiers. No new inference is necessary to support these corrected historical claims.

Independently compared all nine support counters per arm in `results/historical-supports.json` with the three original raw JSON files, checked both derived support fields, and recomputed the CSV's structural numerator, retained wall mean, rate, and residual. They agree. Recomputed proxy rates are 42.743932408417045 / 39.94662568433149 / 32.85450376098399; residual estimates are 2.551107869979141 / 2.237077181826475 / 2.672956194343656 ms/event. Conditional threshold outputs are 5.876613631417309 accepted drafts/event, 123.65753554598925 ms/event, and a 23.136450228630935% reduction. The audit script's arithmetic checks are now accurately described as such, and the erratum explicitly supersedes stronger original P0 interpretations without altering the historical evidence.

Final reviewed SHA-256 identities, relative to canonical `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/`:

| File | SHA-256 |
|---|---|
| `main.tex` | `5e579cbf64ed4211efafbbdd9e15acc8ba99a53574162fdc12e6069178255b76` |
| `abstract.tex` | `2952d72ce3c00fa1f425b4ef63abbb8324dd2f5180261402c24576ace10f7e19` |
| `ref.bib` | `51c788f1ffcaef1cc1b1a7846be1e862cbbc56e58d041c37bfc267d703e22905` |
| `figures/cost-breakdown.tikz` | `a29baf437e9f19ee1c33111a07645daed2ace8857492b6daf29468452ee62fee` |
| `scripts/audit_evidence.py` | `e801bce52db5ecf761514b1aae6a346c4b50ed5ee368c459a2a3156693050433` |
| `README.md` | `a4c80e4afb7f545aaa6e7a323fc51bf62face07151ddf8c7080525f3335efd60` |
| `p0/P0-MEASUREMENT-ERRATUM.md` | `83ce2026e5bb7e544b210ddd96d3a039989a1f631f8e2b5c6ffd190b98c50d8a` |
| `results/historical-supports.json` | `2a85237b412479febae72886597e4f64aa0a6e18f8ac3dd3b66a1799837794e7` |
| `results/historical-metrics.csv` | `ab8bd3bdc07bf5a8eba168dcbb9844bcacf53121136e68d6656cfdff81016e94` |
| `results/derived-accounting.json` | `0ae3f4c186da6ff19c4bbf07caad40ff88f18f87fc02537b460f4f9c01d41d95` |
| `results/historical-macros.tex` | `228e73de3c30abb84d728c7a86f814bb31f93f6f3b2970394956c7ab881b6c09` |

The final canonical `main.tex` and `p0/monitor/paper-build-review1/main.tex` were byte-identical. The rebuilt PDF's visual QA was in progress separately and is not certified by this source-level closure. This closes the H3 finding only; it does not close pending E2/E7b, E1, packaging, or submission gates.
