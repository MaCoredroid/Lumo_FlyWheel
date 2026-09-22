# Final current-design / E8 manuscript review

2026-09-22. Reviewer: `paper_redteam_round2`. **PASS for the final source identities below: no unresolved material freshness, method, E8-result, or claim-scope finding.** No additional experiment is required for the bounded claims. This is a read-only manuscript/evidence review; only this report was written. No GPU, inference, serving-source modification, or frozen-experiment mutation occurred. Final PDF rendering and delivery metadata remain the parent's separate checks.

The full manuscript was read, including the abstract, related-work boundary, mathematical/state interface, scan/replay algorithms, optimization map, E7a/E7b/E2 results, E1 measurement and limitations, E8 qualification/timing, conclusion, and appendices. Included figures/table sources were checked for agreement with their surrounding claims. The earlier freshness audit is `notes/latest-design-supersession-audit-2026-09-22.md`; the present review closes its carry-forward items against the completed E8 data.

## Final checked source

All paths below are relative to `papers/gdn-tree-scan-mlsys/v2`. Final main source has 444 lines.

```text
8a39045767c72099fd9678d2867eca466e3f01277e0fc7e2cb0066a1abd5d461  main.tex
224eb1a5505205b0a4319ebea0c346fc38f8c6197ef90342cd1c27780218850b  abstract.tex
88ceccadbd0376d694f6037181ebf0dacd71f0936558d6c88f856666ccc05fc8  ref.bib
8c9dbd43516be8703d4ba841a06d1f2d17caad2c434a61ec29ede40035f0762b  figures/pipeline.tikz
50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898  figures/state-contract.tikz
affc55cd6972d929de38dc8ebdd86c6dcea77fb0beb73c96551010bfaffadcfe  figures/scan-replay.tikz
9fc36e22047056751bb540da8ce27983f1683baf4ec69ebad787c72cd06bdf61  figures/e1-timing.pdf
4b8f8aca86891c932779b812ca5ef8bd3f301ce39dc5e4d393cf5a3581fdde8d  results/e8/summary.json
74b0988b191bb68fad43d41f76f520d49d529e86753ac0f18535721f6c3757fb  results/e8/aggregate.json
c5af31c6e91fa80e61becefb370693c47f8c8e87559098f5b2b0bdcba2d17d89  results/e8/paired-rates.tex
8fa6a03ae43173c9dcf38685d8f4429f2eab3393a43b2a2af85ba2cd9f99849c  results/e8/blocks.csv
2245308540e18438e8a4dc2a76e5154f45c74a2508dd0f81196b417081179665  results/e8/cells.csv
8f246e660beca4d5d601c7ba23a6e2040861b33052a6ade52fad8bc576b26364  p0/monitor/e8-timing-final-redteam.md
```

## Freshness and mechanism disposition

- **Current route leads.** `main.tex:111,120,227–229,241` describes root-plus-nine-draft Cat10, stock `TREE_ATTN`, flat read/write positions, selected-path KV remapping, synchronous eager/cache-off execution. Recurrent, convolution, attention KV and pending-token publication remain distinct contracts. The removed FA2 spine-first figure, old launcher repair and old WY prototype bug are not presented as the current method.
- **Source availability and engagement stay separate.** `194,199,204,214–221` identifies device probability arithmetic with host loops/readbacks, active vectorized convolution, stacked eager buffers without claiming all-layer replay, and inactive guarded FA2/fixed-shape candidates. The corrected head explanation at `192` accurately states that the duplicate projection serves spine selection, while candidate extraction already uses the first logits in both arms. The pipeline, abstract and conclusion now use device probability arithmetic consistently.
- **No stale passing-pilot headline.** E7a opens with 31/32 provenance-passing confirmation prefixes and failed promotion criteria (`265`). The pilot supplies freeze provenance (`275`); final numerical/control/timing failures and the original incomplete checker's defects remain explicit (`277–283`). The retrospective p087 inspector does not replace the frozen denominator. Fresh triangular/Neumann characterization remains distinct from the removed older prototype history.
- **Diagnostics do not become full serving qualification.** The earlier unremapped E7b boots retain only their local numerical claims and explicitly point to the separately corrected route (`293`). E2's finite state/accepted-path evidence, helper-versus-live-byte boundary, duplicate-sibling choice, and stochastic/full-model limits remain (`297–305`).
- **E1 stays intact and scoped.** The fixed qualified eager-configuration comparison, depth-comparable native5 versus longer native11, different attention backends, engine-seed deviation, complete initial cells, support matching and observed continuation divergence are preserved (`309–338`). The abstract explicitly says three-boot means, not maximum tuned rates. E8 is not treated as a replacement or a multiplier for E1.
- **E8 is integrated consistently.** The abstract, introduction (`48`), mechanism map (`214`), evidence table (`255`), dedicated result (`340–364`), discussion/conclusion (`374,381`) and validation scope (`409,423,437`) all describe six B1 timing boots after two untimed qualification boots, with bounded component evidence rather than a composed system or identical-output acceleration claim.

The newest repository integration branch and historical Pages volumes do not supersede the measured route merely by date. The prior audit verified no `scripts/src` difference between serving base `984f613d…` and local paper HEAD `84daa5519…` or integration branch `855f6cf03…`. Measured-source assertions remain bound to the preserved repaired dependencies and loaded modules; the manuscript does not claim that the historical public pages or unmodified HEAD alone reproduce the fresh serving bytes.

## Independent E8 result binding

Source run: `experiments/out-20260922T215254Z-e8-timing`. The paper's `results/e8/aggregate.json` is **byte-identical** to the original closed run aggregate, SHA `74b0988b…`. The actual-data review at `p0/monitor/e8-timing-final-redteam.md` establishes the full raw-ledger, configuration, source, ownership and terminal audit; this pass independently checked the manuscript-to-evidence reduction, rather than rerunning the entire serving qualification.

Executed local `python3 -B` with standard-library JSON/CSV/hash/math/random operations. For all six original `cell_*/cell_result.json` files: checked terminal seals and VALID status, exact receipt hashes against the closed aggregate, exact embedded result agreement, retained tokens divided by unique wall seconds, per-prefix interval/token sums, and the paper CSV rate/support fields. Recomputed three paired relative changes and all 10,000 paired-block bootstrap draws with seed 20260921 using an independently written calculation. Recomputed stream equality from the stored complete per-prefix streams, checked all eight final-ON divergences and p021's 101-ID length. No data file was modified.

| Independent quantity | Reproduced result |
|---|---:|
| Block 1 `(ON−OFF)/OFF` | 29.98841040008109% |
| Block 2 `(ON−OFF)/OFF` | 29.805035076057614% |
| Block 3 `(ON−OFF)/OFF` | 32.903198715014454% |
| Primary mean paired relative difference | 30.898881397051053% |
| Frozen 95% percentile interval | [29.805035076057614%, 32.903198715014454%] |
| Mean ON rate | 12.69119070946222 tokens/s |
| Mean OFF rate | 9.695410530690497 tokens/s |
| Total retained support | 5,940 API-bound tokens / 1,868 intervals |
| Within-pair equal streams | 16/24 |
| Across-block within-arm equal streams | 32/48 |

The table's three rate pairs and relative changes round correctly. The paper uses the prespecified mean of paired relative changes, not the slightly different diagnostic ratio of arm means. The three-block resampling uncertainty is explicitly coarse. The last ON cell's different continuations and p021 EOS/stop at 101 tokens remain included, without favorable replacement. The paper does not claim identical-output latency, quality preservation, pure LM-head kernel timing, B4 benefit, or a generally optimal system.

The exact qualification gate `experiments/e8-single-logits-qualification-v2/e8_head_gate.py:58–60,152–160,185` uses a byte-view comparison for whole logits and checks argmax, ordered top-two, and full ordered candidate reconstruction. Thus the manuscript's **bitwise** same-input head wording (`344`) is supported, not inferred from approximate equality. Its 103 proposals / 515 head comparisons per arm and zero timed support agree with the closed actual qualification review. Clean ON five-head versus OFF ten-head census is distinct from the untimed paired gate; the manuscript preserves that distinction. Complete cross-boot qualification streams matching does not imply all timing streams match.

## Findings caught and closed during this pass

1. **Abstract environment regression:** initial `abstract.tex` SHA `06031410…` contained ordinary text only, while `main.tex:27` used a bare `\input{abstract}`. Reported immediately. Final abstract SHA `224eb1a5…` has exactly one `\begin{abstract}` and one matching `\end{abstract}` around the text; closure verified from source. The parent rebuilt and owns visual QA.
2. **Two leftover older-context references:** initial main SHA `8569255a…` still mentioned “our measured replay cost” and “shared-spine byte checks in our particular route” after removing those old narratives. Final `63,67` instead states that the replay implementation's cost is not a lower bound on alternatives and refers to the actual local numerical comparisons.
3. **Subsection scope shorthand:** final `191` now says “keep probability arithmetic on device,” matching the documented host control/readbacks rather than implying an entirely device-resident walk.

A lightweight citation/label check found no undefined citation keys, duplicate labels, or unresolved `\ref` labels in the final manuscript structure. Bibliographic content and external author-system claims were not freshly re-researched in this bounded final pass; no new external performance claim was added. The accepted E1/E2/E7 raw audits and independent E8 actual-data review remain the numerical/serving evidence boundary.

**Final result: PASS.** All actionable findings raised in this pass are closed in the exact final source hashes above. No new inference, expanded benchmark, replacement cell, or stronger equivalence claim is needed for this manuscript's current bounded result.
