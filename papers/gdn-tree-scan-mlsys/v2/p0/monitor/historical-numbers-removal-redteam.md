# Historical-number removal: independent closure

2026-09-22 19:41 UTC. **PASS: no historical numerical-result leakage, orphan references, or unresolved material claim issue found in the settled manuscript and its included figures.** No experiment was performed or is required for this edit. Final rebuilt-PDF and packaging checks remain the parent's responsibility.

Final reviewed SHA-256 identities:

- `main.tex`: **`09d5fbd626233c5f363aaa94f3fce6dd9e06164d01b419c63d1d1fa8e2285d92`**.
- `abstract.tex`: **`36983c20ae3fd466dc485abb4e59baf84cbfd364df9011d27095bfdb70a04899`**.
- `ref.bib`: **`64ca0fc74b5185f2f61358945a2f1250de76a7ec7c2ad793504dff0fc22693c0`**.

Read the full settled main/abstract, recursively inspected all loaded text/figure sources, checked numerical context manually, and independently checked labels/citations and the old-value/include inventory. The initial settled removal version was main `2a17a76448ade359646089a09e268f10bea19371d8190d006be19379db8ec380`; two narrow points found in that version were corrected and rechecked below.

## Removal and retained-evidence boundary

Historical H1/H2 regression/script counts and acceptance values, historical WY error magnitudes, H3 rates/components/coverage/break-even quantities, H4 optimization timings and uncertainty, H5 arm/task counts, old B1 throughput values, and their abstract/introduction/table/conclusion repetitions are absent. The historical macro input and cost-breakdown plot are no longer loaded. The mixed nine-set archived-input numerical table, its repeated extrema, legacy-WY row, and archived-input kernel costs are also absent. Numeric historical-settings/P0 reconstruction details were conservatively removed. The uncited historical optimization bibliography entry is gone.

Fresh v2 remeasurement on archived inputs is not the same as an old campaign measurement; nevertheless, the parent elected to exclude both from the manuscript's quantitative results. Lines 177 and 212 now state that boundary explicitly. The preserved historical discussion is qualitative and cited. Dates, method/model names, equations, schematic indices, and references are not experimental historical-number claims.

Fresh synthetic tiny-gate evidence, the eight fresh pilot/31 provenance-passing confirmation results and frozen failures, E7b substitutions and limited continuation, E2 B1/B4 state/path/API checks, and the complete E1 observations remain. The scope statements continue to separate numerical agreement, actual-route checks, full-model equivalence, and as-executed timing. Historical deletion does not turn the fresh adverse findings into successful compact-route promotion or a general correctness/speedup claim.

## Findings resolved during this check

1. **Historical configuration numeral, closed (`main.tex:37`).** “Archived batch-four instrument” was changed to “an archived instrument.” Although this was configuration rather than a result, removal honors the user's broad wording without losing the accounting explanation.
2. **Lost dtype/precision context, closed (`main.tex:218`).** Deleting the old table also removed context for the tiny-gate error range. The paragraph now explicitly specifies **fp32 dense products**, **diagnostic fp32-output-store outputs**, **committed fp32 states**, and normalization by the corresponding oracle maximum. This clarification is necessary: in the fresh chain-11 raw result, `stage1_verify.A_prod_out32_vs_oracle.max_rel_to_refmax` is approximately `9.2025e-8`, whereas the deployed out16 error is approximately `1.2811e-3` and the tf32-product B out32 error is approximately `1.6042e-3`. The fresh caterpillar B fp32-product out32 error is `5.820985769509085e-7`, supporting the printed upper endpoint. The corrected paragraph no longer suggests that bf16 output stores or tf32 products share the fp32 diagnostic accuracy.

These are wording repairs from existing evidence. The no-nonfinite observation remains separate from the narrower precision claim.

## Loaded-source and reference check

Only `abstract.tex`, three schematic TikZ files, and the fresh E1 timing PDF are included. Schematic coordinates, node subscripts, and conceptual state-surface counts carry no old campaign result. The E1 PDF and source sidecar are byte-identical to the previously reviewed fresh-data figure.

- **27 cited keys / 27 bibliography entries**, no missing or uncited key.
- No undefined cross-reference or duplicate label.
- No references to removed historical tables, numerical macro file, or cost-breakdown figure.
- Fresh error/failure/timing claims retain their previously reviewed scope; no new inference or experiment was introduced.

| Included figure / checked fresh raw evidence | SHA-256 |
|---|---|
| `figures/state-contract.tikz` | `50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898` |
| `figures/pipeline.tikz` | `d120d94850a3b300e7b5919365a608f69f532540dae52ba14b6da8f8aa4742a6` |
| `figures/spine-layout.tikz` | `390a6d64b34627e93a2d63cae64d971abe54569be978fa5477984e3228d5579c` |
| `figures/e1-timing.pdf` | `9fc36e22047056751bb540da8ce27983f1683baf4ec69ebad787c72cd06bdf61` |
| `figures/e1-timing.json` | `b1e2f4bb8bbb0ad269a8d7df4d3a21be4bff55e3083155698d68bf5595b15e3f` |
| Tiny-gate `result_01_synthetic_chain11_tiny-gates.json` | `59dc7575a93d0fd11e941df6a408677603dd93804498a849cc62dbf1ac322925` |
| Tiny-gate `result_02_synthetic_chain5_tiny-gates.json` | `5ec340e83930886721e99dcf37b6793198b89a35d100cea42fad2bcb5958fd7b` |
| Tiny-gate `result_03_synthetic_caterpillar_tiny-gates.json` | `a08de54a36ea0af347e3d050d0329940780b40969b1765ca3d3851c6074d3f95` |

Tiny-gate files were read from the preserved local evidence snapshot under `artifacts/e7a-evidence-20260922T0700Z/experiments/out-20260921T231030Z-e7a-tinygates-v3/`. Old evidence remains preserved outside the manuscript; this check neither deletes nor relabels it as fresh.
