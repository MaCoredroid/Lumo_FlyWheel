# E8 manuscript integration: independent review

2026-09-22. Reviewer: paper_redteam_round1. **PASS. No unresolved material result, interpretation, or cross-section consistency finding remains in the reviewed E8 integration. No additional experiment is indispensable for its stated bounded claims.** CPU-only source/evidence review; no manuscript or campaign mutation, inference, or GPU activity. PDF rendering is the parent's separate check.

## Findings and claim checks

- **E8 methods and qualification (main.tex:340–353):** the two untimed boots, eight 32-ID requests per arm, 103 proposals and 515 paired head comparisons per arm agree with the independently reviewed actual qualification. The same-input full-logit, ordered-candidate, hidden/RNG and watched-mutation checks are correctly bounded; the text does not claim an archived full-tensor offline replay or full-model state equivalence. The clean five-versus-ten head census, exact six-cell order, warmup/timed request counts, actual engine/API seed, fixed route and sampled ownership language agree with the frozen timing record.
- **E8 numerical result (main.tex:355–364 and results/e8):** all six CSV rates, numerators, durations, retained intervals and per-prefix minima agree with the prior independent raw reconstruction. All three table rows agree with the frozen aggregate: OFF/ON 9.686/12.591, 9.703/12.595, 9.697/12.887 tokens/s, with 29.99%, 29.81%, 32.90% paired changes. The primary is the mean paired relative difference, 30.8988813971%, with the declared 10,000 paired-block draws and coarse 95% interval [29.8050350761%, 32.9031987150%]. The manuscript's 30.90% [29.81%, 32.90%], arm means 12.691/9.695, 298–314 intervals and at least 29 per prefix round correctly. The copied aggregate is byte-identical to the sealed campaign's aggregate, and the summary retains the primary-versus-ratio-of-means distinction.
- **E8 adverse outcomes and interpretation (main.tex:356,364,374,381,437):** 16/24 within-pair and 32/48 within-arm stream matches remain explicit. All eight third-ON streams differ; its p021 EOS after 101 IDs remains in the estimate. Neither equal-output latency nor quality preservation follows. The manuscript preserves every fixed pair without favorable replacement; the source-controlled component contrast is not an isolated head-kernel timing or a universal deployment guarantee.
- **E1 unchanged and correctly bounded (main.tex:309–338,372,379):** the fresh three-boot means remain native-5/native-11/tree = 13.6704141/11.4385719/12.6949033 at B1 and 55.3750507/42.1505198/47.2839395 at B4. All reported rounding, rate differences, intervals and 4.82% maximum precision ratio match results/e1/aggregate.json. The tree trails depth-comparable MTP-5; the longer MTP-11 comparison is not presented as best tuned performance. Engine-seed/backend differences, 14/144 across-arm and 72/144 repeat-stream equality, the 101-ID EOS case, matched support rather than full-service latency, and three-block uncertainty remain disclosed.
- **Cross-section consistency:** abstract, introduction (46–50), optimization map (214), evidence table (254–255), results, discussion, conclusion and validation appendix (409–437) consistently separate 18 E1 boots from six E8 timing boots and two E8 qualification boots. E1 already uses reuse: main.tex:362 explicitly forbids multiplying its Cat10 rate by the E8 increment; abstract and conclusion repeat that reuse was already active. No new composed gain, maximum-performance, full-model-equivalence, B4-ablation or quality claim appears. Existing compact-candidate nonpromotion and bounded E2 limitations remain intact. Historical quantitative results remain excluded; no removed historical rate has re-entered through this integration.
- **References and presentation inputs:** all main-file label references resolve and all 24 cited BibTeX keys exist. E8 uses the checked generated three-row table, not a selectively plotted subset. The E1 figure is the existing fresh-campaign figure with the hash below; visual layout is outside this source review.

The first pre-container infrastructure failure is correctly an artifact/provenance event, not an additional scientific qualification or timing measurement. The paper need not inflate the experimental count by including it.

## Reviewed identities

The source changed during the pass only for the parent's stated final wording/abstract-environment corrections. This disposition applies to the final hashes below, not the earlier 8569255a/06031410 snapshot.

| File | SHA-256 |
|---|---|
| `main.tex` | `8a39045767c72099fd9678d2867eca466e3f01277e0fc7e2cb0066a1abd5d461` |
| `abstract.tex` | `224eb1a5505205b0a4319ebea0c346fc38f8c6197ef90342cd1c27780218850b` |
| `ref.bib` | `88ceccadbd0376d694f6037181ebf0dacd71f0936558d6c88f856666ccc05fc8` |
| `figures/e1-timing.pdf` | `9fc36e22047056751bb540da8ce27983f1683baf4ec69ebad787c72cd06bdf61` |
| `results/e1/aggregate.json` | `dbd4542ee9899da3eb592fdf8a0b935010f5c8035de9d5cb75e581343c7deb1d` |
| `results/e8/aggregate.json` | `74b0988b191bb68fad43d41f76f520d49d529e86753ac0f18535721f6c3757fb` |
| `results/e8/summary.json` | `4b8f8aca86891c932779b812ca5ef8bd3f301ce39dc5e4d393cf5a3581fdde8d` |
| `results/e8/blocks.csv` | `8fa6a03ae43173c9dcf38685d8f4429f2eab3393a43b2a2af85ba2cd9f99849c` |
| `results/e8/cells.csv` | `2245308540e18438e8a4dc2a76e5154f45c74a2508dd0f81196b417081179665` |
| `results/e8/paired-rates.tex` | `c5af31c6e91fa80e61becefb370693c47f8c8e87559098f5b2b0bdcba2d17d89` |
| `p0/monitor/e8-qualification-results-redteam.md` | `e8e02700cad9337bfa61f1446b3f1b090404f144c8acfe193fbad197291cfca3` |
| `p0/monitor/e8-timing-final-redteam.md` | `8f246e660beca4d5d601c7ba23a6e2040861b33052a6ade52fad8bc576b26364` |
| `p0/monitor/e8-timing-final-independent-review.json` | `274566f0a0efec3c108931a329a54cc514bd0c5e783e7ec72415210669574544` |

Evidence of record: `p0/monitor/e8-qualification-results-redteam.md`, `p0/monitor/e8-timing-final-redteam.md`, and its independent JSON; actual six-cell campaign `experiments/out-20260922T215254Z-e8-timing`. Those prior independent reviews include raw receipt, source/configuration, API reconstruction, physical support, ownership and frozen-reducer replays. This pass reconciled the new manuscript values to those results and directly checked its result tables/aggregate and existing E1 aggregate. No additional experiment is requested.
