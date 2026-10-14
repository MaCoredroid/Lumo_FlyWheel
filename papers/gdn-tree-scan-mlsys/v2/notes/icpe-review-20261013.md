# ICPE 2027 draft review — 13 October 2026

Draft: `icpe2027/main.tex` at 07e020e4a (13 pages; body ends on page 9, so about one page of slack). Every number below marked *computed* was derived on 13 October from records already in `results/` or on GB10; no new GPU runs.

## Verdict

The draft is careful and every claim is bounded, but three things would decide a performance-engineering review: (1) the paper never says where the tree's gain comes from, and the records say most of it is the suffix tail, not tree width; (2) every rate difference is disclaimed as carrying "no significance claim" when the per-request records support confidence intervals that exclude zero; (3) the results are tables only, with no model, no end-to-end latency and no energy, which is what ICPE readers look for. All three are fixable before 16 November, mostly without GPU time. Six stale sentences left over from the 4–5 October updates also need fixing.

## A. Errors to fix (stale or inconsistent text)

1. `main.tex:168` still says "Three serving arms share this checkpoint" and "once per task and arm". The study now has four arms (plain decoding added) and two attempts.
2. `sections/workload-icpe.tex`: "below the 3.15 of its fixed-request replays". The paper elsewhere gives 3.17 (first corpus counter); the disjoint corpus is 3.27. 3.15 is Table 6's tokens/step minus one, a different quantity.
3. Table 6 caption defines tokens per step as "accepted drafts plus the correction or bonus token", but the column is computed as output intervals over speculative events (5.32 vs 4.35 + 1 = 5.35; 4.15 vs 3.17 + 1 = 4.17). Either change the caption to the computed definition or recompute.
4. `sections/sampling-validation-icpe.tex`, last paragraph: "the cause of the live agent study's token-volume difference remains unresolved". Section 5.2 now says the first attempt's ordering did not repeat; this sentence contradicts it.
5. Table 8 "Task study" column (4.21 drafts/event, 115.43 ms target, …) is the first tree attempt only. The second attempt's values (4.41, 114.92, 54.52, 20.19, 197.10 ms) are in `AUDIT.json`; pool both or label the column.
6. Threats, Internal: "Each arm has three replay repetitions per corpus" is false for plain decoding and MTP-1 (one run each).

## B. Substantive gaps, ranked

### B1. Attribution: the gain is mostly suffix depth, not tree width (highest risk)

Per-position acceptance counters are in every replay record. Accepted drafts per speculative event, pooled over three runs (*computed*):

| Corpus | MTP-5 | TreeHost depth 1–5 | TreeHost depth 6–11 | Extra over MTP-5 from width (depth ≤5) | Extra from depth >5 |
|---|---|---|---|---|---|
| First | 3.17 | 3.36 | 0.99 | 0.19 | 0.99 |
| Disjoint | 3.27 | 3.49 | 1.24 | 0.22 | 1.24 |

Depth 6–11 can only come from the six-token suffix tail hung off the depth-5 MTP spine (the rescue branches stop at depth 5). So about 84–85% of the tree's extra acceptance is retrieval depth beyond the neural drafter's reach, and 15–16% is width. The tree accepts more than the chain at depth 1 (0.956 vs 0.850) but slightly less at depths 4–5 (0.426 vs 0.463), because accepting a sibling ends the path early.

Per request, the tree/MTP-5 rate ratio correlates with deep acceptance at r = 0.77 (first) and 0.84 (disjoint), with the shallow width gain at 0.58 and 0.55, and not at all with prompt or output length (*computed*).

Why it matters: a speculative-decoding reviewer will ask "is this tree verification or suffix decoding?" and propose a chain that concatenates MTP-5 with a suffix tail. A rough projection from these counters (chain of 11 drafts, no siblings, step cost near MTP-5's since suffix tokens need no drafter pass) lands at or above the tree's rate. The current framing ("A chain verifier cannot host tree drafters at all") is true but does not match where the measured gain comes from.

What to add (no GPU): the table above, one sentence on the per-request correlation, and a reframing of contribution 1 as "one verifier hosts a neural spine, neural siblings and a retrieval tail at once; on this workload the retrieval depth carries most of the gain and width adds 0.2 tokens per step". Position SuffixDecoding and Arctic's suffix+speculator switching (already cited) as the nearest alternatives, and state the concatenated chain as an untested baseline in Threats.

### B2. Quantified uncertainty instead of "no significance claim"

Request-level bootstrap (resample the 43 requests, pool the three runs, 4,000 resamples) on the tree/MTP-5 pooled-rate ratio (*computed*):

| Corpus | Ratio | 95% CI | Tree faster on |
|---|---|---|---|
| First | 1.077 | 1.041–1.117 | 34/43 requests |
| Disjoint | 1.118 | 1.051–1.205 | 34/43 requests |

Per-request ratios range 0.86–1.47 and 0.91–1.65. Both intervals exclude 1. Report them, state that they capture request heterogeneity (the three runs share a seeded generator, so they are not independent sampling replicates), and soften Threats from "none of the differences carries a significance claim" to that narrower caveat. Do the same for the per-step cost ratio. ICPE's audience (Hoefler & Belli's twelve rules; the SPEC RG reproducibility principles) expects CIs.

### B3. End-to-end latency and first-token time are missing

The pooled estimator excludes first-output time by design, which is fine as a decode metric but leaves out what an agent user waits for (*computed*):

| Corpus | Arm | Median first-token time | Summed request time per run |
|---|---|---|---|
| First | TreeHost | 2.06 s | 857 s |
| First | MTP-5 | 1.89 s | 884 s |
| First | SGLang | 1.01 s | 847 s |
| Disjoint | TreeHost | 1.99 s | 628 s |
| Disjoint | MTP-5 | 1.81 s | 686 s |
| Disjoint | SGLang | 0.95 s | 602 s |

End to end, the tree is 3.1% and 8.5% faster than MTP-5, and 1.2% and 4.3% slower than SGLang on both corpora; the disjoint corpus's +0.9% decode-rate edge over SGLang becomes an end-to-end deficit because SGLang's prefill (8,192-token chunks vs 4,096) is about twice as fast to first token. Report both metrics in Tables 4 and 5; otherwise a reviewer computes this and reads the omission as selective.

### B4. Energy per token (data already collected)

NVML GPU power at about 10 Hz exists for all 80 agent attempts (`dcgm_samples.jsonl`, excluded from the local sync). Integrated over agent wall time, including tool idle (*computed*):

| Arm | J per output token (attempt 1, 2) | Mean power |
|---|---|---|
| TreeHost | 2.13, 2.07 | 50 W |
| MTP-5 | 2.02, 2.34 | 49 W |
| SGLang | 1.41, 1.72 | 39–43 W |
| Plain decoding | 4.34, 4.34 | 44 W |

Speculation halves energy per token against plain decoding; the tree costs the same per token as MTP-5. This is a natural ICPE addition (one table row or a column in Table 7) for a desktop-class device. Caveat to state: board power from NVML, whole-task integration including tool time; replay runs have no power telemetry.

### B5. A cost model and one data figure

The results section has no figure. The per-step data support a small analytical model that ICPE reviewers value:

- Chain MTP-k step time fits t ≈ 113 + 9.5k ms (k = 1, 3, 5, 7); plain decoding at 94.9 ms implies about 18 ms of fixed overhead for enabling speculation.
- The tree's residual attribution implies about 0.35 ms per extra verification row against about 10 ms per neural drafter pass: on this memory-bound batch-1 platform, width is roughly 30 times cheaper than neural depth. That is the actionable guidance for the tree drafters the paper says should follow (caveat: the row cost is a residual, not a direct measurement).
- Break-even: at 189 ms per step the tree must reach 4.15 × 189.0/159.0 = 4.93 tokens per step to match MTP-5; it reaches 5.32 (8% margin) and 5.68 on the disjoint corpus.

One figure would carry this: ms/step against tokens/step for every arm in Table 6, with iso-throughput lines and the break-even point marked. A second, optional panel: per-request speedup against deep suffix acceptance (B1).

### B6. The native spread needs an explanation

Table 10 shows two native processes differing by up to 0.434 relative L2 in recurrent state and 0.479 in newly committed KV, and the maximum absolute next-token logit difference is 5.81 (`AUDIT.json`). The paper never says why identical native processes diverge this much. A reader will question a metric on which the controls differ by 43%. Add one or two sentences on the source (cross-process nondeterminism in long-prefix prefill and reductions, compounded through the recurrent state; cite the existing determinism reference) and, if the records hold per-layer values, report the median or 90th percentile beside the maximum. The 121/126 greedy agreement and the 3.19–50× fault margins already answer "is the test discriminating"; say so in that paragraph.

### B7. Turn the methodology claim into evidence with a pitfalls table

Contribution 2 would be much stronger as "here is what goes wrong without these instruments", with magnitudes. The records already contain four instances:

| Pitfall | Symptom | Caught by | Effect |
|---|---|---|---|
| Flattened penalty history in the tree sampler | Rates look normal | Captured-row audit | Node TV up to 0.421; inflated the first-corpus advantage from 7.7% to 10.2% |
| Shortened drafting passes duplicated principal tokens into inactive siblings | Misleadingly low acceptance | Exact-output checks | Variant results invalidated and rerun |
| Seeded process-wide generator | Repeated boots look like replicates | Ten-boot distribution diagnostic failed its sensitivity check | Runs are not independent samples |
| Live-agent token volume | Tree emitted 2× MTP-5's tokens | Second attempt and identical-prompt check | Ordering did not repeat; trajectory variance |

The first row is the paper's best methodological result and is currently one subordinate clause in Section 5.6; the 7.7% vs 10.2% inflation is not stated anywhere in the draft.

### B8. Related work from the performance-engineering community

The references are all ML-systems papers. ICPE reviewers will look for: Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC'15); Papadopoulos et al., "Methodological Principles for Reproducible Performance Evaluation in Cloud Computing" (IEEE TSE, SPEC RG); Agrawal et al., Etalon/Metron on LLM-inference metrics (directly relevant to the pooled estimator and end-to-end time); Xia et al., Spec-Bench (speculative-decoding evaluation). Verify each entry before citing.

## C. Optional GPU experiments (GB10 is idle), ranked by value per hour

1. **Retrieval-only chain baseline (about 3–4 h).** The native image supports vLLM's `ngram` prompt-lookup proposer today (`suffix` is listed but needs the `arctic_inference` package, which the image lacks). Run ngram with k = 5 and 10, three runs per corpus. If the tree beats both MTP-5 and the retrieval-only chain, the "hosts heterogeneous proposers at once" claim is measured; if not, better to know before submission. Check first whether the tree's suffix source is request-local or a prebuilt per-task index, so the baseline is comparable.
2. **Order-permuted replicates (about 2.5 h).** Rerun tree and MTP-5 three times each on the first corpus with shuffled request order. The seed contract stays unchanged, but each request draws different random numbers, giving independent sampling replicates and closing the Internal threat.
3. **Spine-plus-tail-only valid mask (1–2 days engineering, about 3 h GPU).** A new pinned mode that masks the MTP siblings and rescue branches while keeping the 32-row verifier isolates width at identical verifier cost. Most direct answer to B1, but the costliest, given the digest-pinned mode gates.

Batch above one stays out of scope (batch-one split-K kernel).

## D. Presentation

- The abstract (about 230 words, 15 numbers) and the contribution paragraph repeat each other almost verbatim. Shorten the abstract to the three claims with four or five numbers; make the contributions a bulleted list.
- Section 3 is long relative to ICPE's interests; Table 2 and the four implementation paragraphs could shrink by a third to make room for B1, B5 and B7 within the 10-page body.
- Use the "measured cost of hosting tree verification" framing consistently, but pair it with the end-to-end numbers (B3) so the SGLang comparison is not read as decode-rate-only.
