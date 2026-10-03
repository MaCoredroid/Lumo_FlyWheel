# Third academic review pass

This pass follows the author-provided review of the 11-page manuscript. It changes presentation and citation metadata only. No inference, workloads, numerical thresholds, or scientific gates changed.

- Removed the older two-task rates, their selection/exclusion footnote, and the SGLang serving column. Earlier runs, adverse records, and source metadata remain in the prior snapshots and experiment artifact; no raw evidence was deleted.
- The ten-task NVFP4 deployment remains descriptive, with 25.63 pooled tokens/s, 265 completed requests, six resolved tasks and four failed-test patches. It has no matched native-decoding or chain-MTP baseline. No speed or quality advantage is claimed, and no baseline was invented.
- Removed development-history and cross-stack statements from Limitations after their comparison was removed. Retained the numerical, task-selection, timing, and memory evidence gaps.
- Removed the window-gauge sentence along with the SGLang comparison. Retaining even the suggested plain-language replacement would discuss telemetry for a deployment no longer reported.
- Explained context summarization requests in the pooled-rate definition; counts and durations are unchanged.
- Relabeled Weaver aligned-local as Weaver (FP32 preparation), preserving all numerical counts and preprocessing distinctions.
- Standardized NVIDIA DGX Spark (GB10) in the abstract and Evaluation Setup; subsequent device mentions use GB10. Expressed the exact attention bound as 2^{-8} and removed a duplicated word.
- Moved the native-reference study and its table unchanged in numerical content to Appendix C. Limitations explicitly states that full-model tree equivalence remains untested and points to the appendix; the 68/84 joint-choice result is not hidden.
- Removed the MLSys publisher fields from FlashInfer and FastTree. Removed the repeated year from Flash-Decoding's date.
- Verified EAGLE-3 against the official NeurIPS 2025 proceedings and publisher BibTeX; upgraded the entry to the venue version with unchanged key and authors.
- Replaced the dated tooling branch in the artifact citation with the versioned tag lumotree-v2-artifact-r1. This tag is a manuscript/artifact snapshot, not a public submission or proof of completion of the larger experiment plan. It must never be moved; later versions require new tags.

The pre-edit source closure and PDF are preserved in artifacts/before-academic-review3-20260930/MANIFEST.json. Clean compilation, standalone dependency-closure compilation, and full rendered-page review are required before delivery. The release tag is verified against the pushed artifact tree before reporting completion.
