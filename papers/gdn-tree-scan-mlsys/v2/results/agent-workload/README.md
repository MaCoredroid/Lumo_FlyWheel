# Cqc10 coding-agent evidence

This directory preserves one completed B1 Qwen3.8 NVFP4 ten-task Astropy segment, recorded on 2026-08-24 and reviewed on 2026-09-23. It is a selected segment of a resumed development campaign, not a new September run, complete SWE-bench score, or matched native/tree trial.

`raw/cqc10/MANIFEST.json` records the remote source paths, byte counts and SHA-256 hashes for the compact original evaluator/metadata/normalized-output/patch/prompt records, per-task and campaign metrics, and launch/runtime receipts. All raw bytes are original except two explicitly named `SANITIZED` configuration projections: credential values are omitted/redacted, with original file hashes retained. Do not substitute these projections for a claim of byte-identical original configuration receipt. The omitted originals are not included here.

`cqc10-task-audit.json` independently reconciles every saved evaluator record with the metadata copy and normalized outcome and derives all ten agent elapsed times. Evaluation duration and server boot are separate from this agent field. The service metric is handled by the independent metric audit/reducer; task wall must not be used as model-only decoding time.

The full model weights, container images, Qwen agent runtime bundle, complete tool traces/workspaces, and evaluator implementation bytes are not in this compact bundle. Runtime identities and trace/worker hashes are retained where recorded. This bundle supports offline reading and reduction of saved evidence; it is not a promise to reproduce generation or rerun hidden tests without those external dependencies.

## Earlier lineage retained for auditing

`audit/swe-lineage-and-native-review-extract.json` and `audit/swe-lineage-config-review-extract.json` are clearly labeled review reductions from read-only remote originals, with source paths and hashes. They retain earlier interrupted segments, repeat attempts and targeted native observations; they do not supply the manuscript's performance headline. The original complete records remain at their remote paths.

Before Cqc10, the intended first six tasks were spread across Cqc16, Cqc15 and Cqc12. Cqc16 stopped on degeneration; Cqc15 repeated already attempted tasks, then stopped in a script-edit collision with another task unscored; Cqc12 included an earlier unharvested attempt and later stopped at an unclassified budget cap. Source/patcher revisions changed, and the response cap changed from32768 to24000. The Cqc10 remainder therefore has a completed, distinct single-run ledger but does not erase the interrupted endpoints or form a homogeneous complete-cohort score with them. See `notes/swe-workload-comparison-review-2026-09-23.md` for the corrected raw evaluator lineage and invalid native attempts.
