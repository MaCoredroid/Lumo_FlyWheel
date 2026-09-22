# Final E1 integration: independent paper/evidence review

2026-09-22. **PASS for the bounded claims after one stale-status correction; no unresolved material finding and no indispensable additional experiment.** This is a changed-section/evidence review plus a whole-paper consistency pass, building on the closed pre-E1 review. It is not an approval of unseen artifact packaging or the final rebuilt PDF.

Initially reviewed `main.tex` SHA-256 `21fe849360f601459961f4c026e4a0298d3a7eb88e8f1bdb182dfcca551d6d62`. During review the parent corrected the finding below and removed punctuation from three paragraph headings. The corrected source was rechecked at **`9de0d3464994c6392ce0991f8e062005ccd41dbf48635aa42fb0134d30ae86c7`**. Abstract remained **`9c4061adc247f22c297a8f05627958a4f564e1c38b5803a79ab91df5a7cae0a5`**. The starting 15-page PDF was `097717f776a3879004d21683344aa62edbf4dc54a713c138bcb61a5cc6010ca0`; the parent will rebuild and visually verify it after source closure. I inspected the rendered E1 figure and its source binding, not every PDF page.

## Finding and resolution

**F1, closed — stale E1 status in historical optimization table (`main.tex:302`).** The initial row said “Combined final stack vs. chains / Planned E1/E3; no measured conclusion,” contradicting the completed E1 campaign. H4 optimization composition remains unmeasured, so the parent changed only that row to “Composed optimization stack vs. chains / Conditional E3; no composed result.” The corrected row preserves the unresolved composition claim and no longer calls completed E1 planned. No data or experiment was required.

## Independently checked E1 evidence

Read `p0/monitor/e1-remaining-cells-redteam.md` and its complete JSON, including prior cells 1–6, all cells 7–18, raw/source hashes, finite preflight qualifications, and retrospective diagnostics. Recomputed the numerical claims directly from the recorded per-cell numerator and denominator; did not import or execute the campaign aggregator. Recomputed all six contrasts using a separate 10,000-resample implementation with the prescribed seed and enumerated all 27 ordered three-block resamples. CSVs, aggregate, independent-review JSON, support JSON, and figure sidecar agree exactly. This pass did not rerun the earlier full event-join audit; it checked its identities against the actual remote artifacts.

Remote read-only verification rebound **188 recorded raw/source hashes** to the completed campaign and independently checked **all 144 timed API streams** against their raw `capture_request.json` bytes, response IDs, seed/temperature, finish reasons, and token-ID sequences. Remote campaign root: `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells/`. No inference, GPU use, boot, replacement cell, or source/campaign modification was performed.

| Claim and manuscript location | Recomputed result / disposition |
|---|---|
| Completed/support counts, line 402 | All 18 initial cells VALID and sealed. B1: 249–314 retained intervals/cell, minimum 24/prompt. B4: 51–64/cell, minimum 25/cohort. Zero eligible intervals above 1.5 seconds. Matches. |
| Mean rates, line 404 and abstract/conclusion | Native-5 / native-11 / tree: B1 **13.67041408 / 11.43857190 / 12.69490328**; B4 **55.37505070 / 42.15051977 / 47.28393948** tokens per retained wall second. All printed rounding matches. |
| Percentage gaps, line 404 | Tree relative to native-5 mean: **−7.13592724% B1**, **−14.61147415% B4**. Correctly printed −7.14% / −14.61%; no mean-of-percentages substitution. |
| Tree − native-5, line 404 | B1 −0.97551080, interval [−1.08486237, −0.78687482]; B4 −8.09111122, [−9.68011932, −5.83011139]. Matches. |
| Tree − native-11, line 404 | B1 1.25633138, [1.15742430, 1.44473578]; B4 5.13341972, [4.17945055, 6.81124475]. Matches. |
| All six precision decisions, lines 404/445 | All meet 0.10. Maximum half-width/native-5 mean **0.0481685559**, or **4.82%**, for B4 native-11 − native-5. Exact 27-outcome extremes coincide with the recorded percentile endpoints; this does not enlarge the three-block evidence. |
| Retrospective stream counts, line 414 and abstract | Across-arm same-block **14/144**; repeated-block within-arm **72/144**. B1 native-5, native-11, tree: **24/24, 24/24, 8/24**. B4: **7/24, 5/24, 4/24**. All reproduced from the complete streams, with dependent pairs correctly disclosed. |

The 101-token termination is specifically verified, not inferred merely from `finish_reason=stop`: cell 16, p021 ends in token **248046** in both the API stream and the final ledger row. The read-only checkpoint's `generation_config.json` declares EOS IDs `[248046,248044]`; the request has no explicit stop-string setting. Its capture is `cell_16_b3_tree_B1_a1/cohort/req_t3_p021/capture_request.json`, SHA `2d68273c85aad00b288e29c125a35f92dd3f8c72ea8c1299c14a7a13da79b25c`. All other timed streams contain 128 IDs. The EOS request retains eight-prompt cell coverage. Generation configuration SHA: `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.

The figure's source hash equals the reviewed aggregate and its entire `plotted` object equals `aggregate.by_batch`. Its gray block lines, blue arm means, and paired-difference intervals represent the supplied data consistently with the caption at line 409. No rate or uncertainty was inferred from the image.

## Claim-scope and whole-paper consistency

- **Methods, lines 385–399:** the integrated text preserves the already closed methods: implemented configurations and different attention backends; exact workload and EOS policy; finite warm-up/preflight checks; timed-only physical-step support; one denominator per B4 interval; incomplete/invalid-cell reporting; mean paired differences; coarse three-block uncertainty; and the genuine engine-seed deviation. The retrospective diagnostic does not alter eligibility or the frozen analysis.
- **Abstract, introduction 47–49, evidence table 181–198, limitations 420–447, conclusion 454:** completed E1 is now consistently described as an instrumented, retained-decode configuration comparison. The measured lower rate against native-5 and higher rate against native-11 remain explicit. These central observations were checked, not removed to obtain closure. Their scope excludes equal continuations, isolated algorithm causality, full-service latency, preserved quality, and full-model equivalence. Stream divergence and seed/backend differences are disclosed where the new headline is introduced.
- **Earlier evidence:** H3 remains a historical mixed-population proxy, distinct from E1's aligned counts; H4's 27.03 ms is not added to E1 or declared composed. E7a's failed criteria, distinct native references, diagnostic fp32 stores, and local method-family reimplementations remain explicit. E7b's one-prefix substitutions and fixed-input continuation do not become a general serving oracle. Selected-route B1/B4 evidence retains its 53-computation/49-link/32-block and 85-structural/83-visible/128-total boundaries and the legal duplicate-sibling/premature-stop distinction. New E1 text does not promote those checks into a proof they did not provide.
- **External comparison:** the new material makes no measured comparison with Bole, TreeWY, or ReplaySSM. Previously reviewed attribution/novelty limits are unchanged; no additional literature search or external-system experiment is needed for this integration.

No new experiment is indispensable for these stated claims. An equal-continuation throughput claim, quality-preservation claim, seed-controlled causal effect, full-model sequential equivalence, broad sampler guarantee, or composed/external-system superiority would require additional evidence; none is asserted here. Current claims remain supported by executed failure witnesses, repaired bounded route checks, and the complete, honestly labeled E1 observations. Final artifact identity/export and rebuilt-PDF checks remain the parent's separate delivery work.

## Final reviewed evidence hashes

All SHA-256; paths relative to the local paper directory unless stated otherwise. Raw campaign identities are reproducible through the per-cell `hashes` and `loaded_hashes` objects plus the diagnostic `capture_path`/`capture_sha256` fields in the evidence-hashed reviewer JSON.

| Artifact | SHA-256 |
|---|---|
| `p0/monitor/e1-remaining-cells-redteam.md` | `6e97dab8031d9c35270d59120132c341f8039232aadf984fe6ed554246e910ca` |
| `p0/monitor/e1-remaining-cells-redteam.json` | `347ef2f472908b70c5e7f7cb20fe2f7e0efdb553d064a66239bb6698b6f0a8f8` |
| `results/e1/aggregate.json` | `dbd4542ee9899da3eb592fdf8a0b935010f5c8035de9d5cb75e581343c7deb1d` |
| `results/e1/independent-aggregate-review.json` | `04ecfdddc27229956d831c77df6fc85975479fd9aeb67759ba900561621a550d` |
| `results/e1/api-stream-diagnostic.json` | `4f7b8346f065e224ee8cb59a26636a24b9f427f26cce25dd539b7b0bace2131b` |
| `results/e1/support-exclusions.json` | `dfddf42002e234da24057645416a69196ee851376db2f195f741360e0471de95` |
| `results/e1/cells.csv` | `35cd67f1682ea62c8700e7dcfe8f77a9879cf7753a6c689c845b81c70a617094` |
| `results/e1/contrasts.csv` | `c464e06fb7528a35aaf967a4f9f94447cbcd81ab9ab5282f1fc0941ee561d359` |
| `figures/e1-timing.json` | `b1e2f4bb8bbb0ad269a8d7df4d3a21be4bff55e3083155698d68bf5595b15e3f` |
| `figures/e1-timing.pdf` | `9fc36e22047056751bb540da8ce27983f1683baf4ec69ebad787c72cd06bdf61` |
| `figures/e1-timing.png` | `3bf52a3535c5028acfa1c2a670c821e442c5f00bb9b3d4565c933fa9e4043904` |
| `scripts/plot_e1_results.py` | `9d833763e6f2c8178744c08c4eb444633bea3c73ebf9e9fb025d7776de988588` |
