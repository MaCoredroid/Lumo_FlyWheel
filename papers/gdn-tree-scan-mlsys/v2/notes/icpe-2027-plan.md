# ICPE 2027 submission plan — 4 October 2026

**Venue facts (checked 4 Oct 2026).** 18th ACM/SPEC ICPE, Gothenburg, 24–28 May 2027. Research track: abstract 9 Nov 2026, paper 16 Nov 2026 (AoE), notification 25 Jan 2027, camera-ready 12 Mar 2027; proceedings in the ACM Digital Library with DOI. Papers "must not exceed 10 pages, including all figures and tables, but not including references and appendices", ACM double-column proceedings format, double-anonymous. Prior arXiv release is permitted if the submission uses "a sufficiently different title"; no links that reveal authors; associated code must be anonymised; prior own work in the third person. Artifact Evaluation track: details TBA (chairs Lishan Yang, Gabriele Russo Russo); Emerging Research track dates TBA. Listed topics that fit: "Measurement and evaluation of AI/ML systems, LLM inference pipelines, and agentic workflows"; "Runtime management of ML/AI inference, LLM serving, batching, routing, caching"; "Benchmarks for AI/ML systems, LLM serving, agentic systems"; "Reproducibility, repeatability, and reusable benchmark artifacts"; "Controlled experiment design ... diagnostics".

## Framing

The verifier makes tree speculation possible on a recurrent-hybrid model at the cost of a tuned chain-speculation stack. Three claims, in this order:
1. A path-parallel verifier with coordinated recurrent/convolution/KV commitment hosts 27-candidate trees on Qwen3.8-27B in vLLM; a chain verifier cannot host tree drafters at all.
2. A measurement and audit methodology for speculative serving on hybrid models: the pooled completed-request decode-rate estimator with explicit exclusions; two frozen recorded-request corpora plus a disjoint-task generalisation check; a frozen native-envelope numerical rule with ten planted faults; a captured-row and Monte-Carlo sampler audit with planted faults; identical-prompt output-length comparability. Reusable for any speculative method on any hybrid model.
3. Evidence: +7.7% and +11.8% over same-stack MTP-5; within 5% of SGLang EAGLE (top-k one, a seven-step chain) with the sign changing between corpora; bounded numerical agreement (max ratio 1.34 of a 2.0 bound); sampler exact on its inputs up to the top-k tie convention; a descriptive ten-task agent study.

Parity with the chain stack is stated as the cost of hosting tree verification, not as a shortfall; tree drafters are named as the untested next step.

## Title (must differ from the arXiv title)

Preferred: "Hosting Tree Speculation on a Recurrent-Hybrid Model: Verifier Design, Audit Methodology, and Serving Evidence". Alternatives: "Measuring Tree Speculative Decoding for Hybrid Language Models: Path-Parallel Verification and Its Audit"; "Tree Verification at Chain Cost: Measurement Methodology for Speculative Serving of Hybrid LLMs".

Rename the system in the submission (e.g. "TreeHost"); the rule only requires a different title, but the system name is one search away from the preprint. A single macro change.

## Outline and page budget (10 ACM pages; the current 15 IEEE pages are roughly 13 ACM pages, so cut about a quarter)

| § | Content | Pages | Source |
|---|---|---|---|
| 1 | Introduction: the three claims above; one paragraph on why hybrid models need a state-aware verifier | 0.75 | main.tex I, rewritten |
| 2 | Background and closest designs; Table I kept, mechanism comparison text halved | 1.0 | II |
| 3 | Design: state boundary, descriptor, path scans and native replay, the five implementation changes (Table II); Fig. 1, Fig. 2, Algorithm 1 | 2.25 | III–V merged; Algorithm 2 to appendix |
| 4 | Measurement and audit methodology: Eq. 5 and exclusions; corpora and replay protocol; frozen numerical envelope and fault controls (Eq. 7); sampler audit protocol; identical-prompt length check | 1.5 | VI, VII-E/F protocol text, Appendix C condensed |
| 5 | Results: Tables III, IV, VII, IX (telemetry), X (component), XI (continuation), XII (faults), length comparability | 3.0 | VII-A–F |
| 6 | Discussion and limitations, consolidated; each caveat once | 1.0 | VIII + IX |
| 7 | Conclusion | 0.25 | X |
| App. | Tables V, VI, VIII, XIII–XVII; Algorithm 2; Appendix C protocol; serving configuration | not counted | |

Cuts: Section VII-G/H and Tables XIII–XV move whole to the appendix with one sentence in §5 ("isolated GDN timings under distinct numerical policies are in Appendix D; they are not a ranking"). Tables V, VI, VIII to the appendix with one sentence. Repeated hedges ("descriptive", "does not establish") stated once in §6.

## Anonymisation checklist

- Author block anonymous; e-mail removed; no acknowledgements.
- Reference [37] (artifact) replaced by an anonymised repository (anonymous.4open.science or a Zenodo anonymous record) holding the paper source, `results/claude-penfix-20261003/`, the reducers and the patcher diff; author name and GitHub path removed from every file name and README inside it. Host names and paths (`GB10`, `/home/mark/...`) scrubbed from included logs or logs excluded.
- Do not cite the arXiv preprint. No "our earlier volumes"; no Lumo FlyWheel.
- System name changed by macro; figure labels checked.
- Supplement references ("in the supplement") point to the anonymised repository.

## Optional strengthening, ranked by reviewer impact per GPU day

1. Batch 2–4 replay sweep on both corpora, tree vs MTP-5 (one GPU day; answers "batch 1 only"). Worth doing if GB10 is free in the week of 12 Oct.
2. Plain-decoding agent control for the task study (about 3 hours; gives Table III a non-speculative row).
3. Not for this deadline: Bole comparison, second model, second GPU.

## Timeline

| Week | Work |
|---|---|
| 5–11 Oct | Upload arXiv v3. Create `papers/gdn-tree-scan-mlsys/icpe2027/` with the ACM template, system-name macro, and the appendix moves; first cut to about 11 pages. |
| 12–18 Oct | Optional batch sweep on GB10; write §4 (methodology) as the centrepiece; rewrite §1 around the three claims. |
| 19–25 Oct | Full 10-page draft; third red team by a fresh session (anonymity, numbers against `AUDIT.json`, claim scope). |
| 26 Oct–1 Nov | Anonymised artifact packaged and tested from a clean checkout; submission-site account and conflicts. |
| 2–8 Nov | Final pass; abstract text frozen. |
| 9 Nov | Abstract registration. **16 Nov: paper.** |
| After | Artifact track when its call appears; 25 Jan decision; TACO extended version the week after; EuroMLSys as floor if rejected; camera-ready 12 Mar. |

In parallel, independent of the venue: post the TreeWY comment/RFC on vllm-project; reconcile the resume's "3.04x"; Scale §2870 disclosure.
