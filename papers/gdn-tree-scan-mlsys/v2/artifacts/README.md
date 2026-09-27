Design-distinction update (26 September 2026): Section IV now names the specific scheduling and commit-policy differences from the closest methods. Section V explains four concrete implementation changes, their nontrivial constraints, and established supporting techniques; Table II summarizes the changes. The 11-page PDF was rendered and checked. See `notes/novelty/design-distinction-2026-09-26.md` and `p0/monitor/2026-09-26-design-distinction-build.json`. Empirical results are unchanged.

Algorithm readability update (26 September 2026): both algorithms now span the page width, use normal-size text and descriptive operations, and separate verification, selection and state publication. Algorithm 2 explicitly marks parallel work and preserves per-layer replay. The 11-page PDF was rendered and visually checked, including both algorithm pages (5 and 6). Build receipt: `p0/monitor/2026-09-26-algorithm-readability-build.json`. Experimental results and citations are unchanged. Earlier formatting checkpoints below are superseded.

Algorithm formatting update (26 September 2026): both algorithms now use compact `algpseudocode` with explicit inputs/outputs, aligned line numbers and nested loops. The state-publication boundary is unchanged; a notation paragraph defines helper outputs. The 11-page PDF has been rendered and checked; build details are in `p0/monitor/2026-09-26-algorithm-format-build.json`. Existing experimental evidence is unchanged.

Current LumoTree revision (26 September 2026): approved title and framing; mechanism memo integrated into background/method/comparison and experiment planning. Pinned TreeWY and Weaver author components executed on GB10; exact sources, licenses, adapters and raw records included. No new SWE performance claims. FINAL-DELIVERY.json identifies the current 11-page PDF and private source archive. Earlier checkpoints below remain historical.

Current patch-producing revision (24 September 2026): conditional best tree29.09 versus SGLang26.89; one empty-patch SGLang pair excluded from the performance table with raw evidence preserved. Run `results/agent-workload/patch_producing_rate_reduce.py`; FINAL-DELIVERY.json binds current artifacts. Entries below are dated earlier checkpoints.

Current pooled decode revision (23 September 2026): the common tree/SGLang comparison and its 48 hash-bound input files are packaged with the current paper. Run `results/agent-workload/shared_rate_reduce.py` from the extracted paper directory. See `FINAL-DELIVERY.json` for final source/PDF/archive identity. Older checkpoints below retain their original scope and superseded estimator labels.

Current history-correction revision (23 September 2026): executed comparator experiments are acknowledged; superseded rates remain excluded. The current PDF and source checkpoint are identified by `FINAL-DELIVERY.json`. See `../notes/comparator-git-history-2026-09-23.md`.

> **Current scope — 23 September 2026:** Paper performance now uses only named coding-agent workloads. E1/E8 local-document rate results below are audit history, not current manuscript claims. Latest eligible completed workload case is dated Cqc10 NVFP4; matched current native/tree task comparison remains proposed, not launched. See `review-experiments.md` and current `FINAL-REVIEW.md`. Earlier completion statements apply only to their original bounded scope.

# Private review checkpoints

These files are private review artifacts. The latest manuscript uses the qualified design and final current results; obsolete narratives and pilot-only summaries have been removed. Core E1/E2/E7 evidence remains unchanged. E8 is complete and independently verified, with two qualification and six timing boots. No further GPU experiment is required for the bounded claims. The delivery receipt identifies the latest reviewed PDF/build and paper archive, which binds five evidence companions. Dated entries below describe preserved earlier checkpoints.

The new E8 companion is `e8-single-logits-20260922T2255Z.tar.gz`: 702 payload files, 16,237,576 bytes, SHA-256 `b4c0a5cbcd2592ce5ddfdc823243f0e99dc15ef412d238a096068d6bae11ebf4`. It preserves every attempted E8 source/run and the final raw review. Extract to an empty directory treated as v2, then run `python3 -B scripts/export_e8.py --paper . --verify-extracted`. This verifies member hashes, two qualification and six timing joins, and the entire aggregate with only relocated path metadata normalized for comparison. No model weights or Docker image are redistributed. Review `../p0/monitor/e8-artifact-final-redteam.md` and the delivery receipt for final extraction verification.

Reviewed manuscript/E7a checkpoint (2026-09-22; predates the added model diagnostics):

- `paper-review-20260922T0710Z.tar.gz`: manuscript/PDF,80hash-manifested files, historical accounting sources, P0 audit sources, exact older E7a source versions. SHA256 `3676083f2e7b3c7b31ce1648ddebbf9dfcb804bf80363bb43261e8c79ed6c8ea`.
- `e7a-evidence-20260922T0709Z.tar.gz` and `.manifest.json`:2859files, all8frozen pilot inputs/provenance,31primaryconfirmation inputs plus original failure/retry records, native selection pool/scores,4historical tensors, result JSON and frozen-analysis erratum. SHA256 `776ddcaf5d938cc665bafca20f46e29d1e9a550a34914d798e78f70636cd180b`.

All companion file hashes were verified after transfer. Targeted independent review closed the missing-file findings in `../p0/monitor/artifact-redteam-round1.md`. Original absolute paths are preserved and mapped explicitly; hash verification and raw-result inspection are supported, while a full unmodified GPU rerun on a different host is not certified. The pinned image and model are external dependencies; model weights and JIT cache bytes are excluded.

The earlier0700E7a and0702paper pair is superseded because it omitted3pilot captures,4historical tensors,selection data and older source/P0 records. Preserve it as audit history, use the newer pair for review. `review-smoke-*` directories are extracted verification workspaces, not authoritative evidence.

Rebuild with `../scripts/build_review_bundle.py --output <new filename>` after final experiments/claim ledger stabilize; do not overwrite old snapshots.

Additional completed diagnostic evidence:

- `e7b-diagnostics-20260922T0801Z.tar.gz` and `.manifest.json`: 727 files, 2,179,735,658 archive bytes. Includes both completed three-boot batches, raw operands/states/hidden/logit captures, executed snapshots, and the reviewed CPU continuation source/dependencies. SHA-256 `c573b5064d944547df280e314520a97dd0c10678fb877c8d75b3a898d548ad05`. Every member size and hash was independently verified after transfer (see `../p0/monitor/2026-09-22-e7b-artifact-checkpoint.json`). These six boots had plain-route attention-KV remap disabled; they support the stated local diagnostics, not full-route qualification.

The twelve-page manuscript/PDF at08:37 UTC included the independently reviewed verifier/commit diagnostics and explicit KV-off limitation. It was newer than the0710 manuscript archive. The final manuscript archive must be rebuilt with this E7b companion and the eventual corrected-route/timing evidence; no current archive should be represented as the completed v2 release.

Latest intermediate manuscript checkpoint:

- `paper-review-20260922T0837Z.tar.gz`: 98 manifest-listed files, current twelve-page manuscript/PDF, updated scope, diagnostic results and independent review reports. SHA-256 `745ca13026dfff18f9cd1b50b7547d2e600477ce41af5098d2fa146e7c059111`. Its manifest binds both the 0709 E7a and 0801 E7b companions. Every member hash passed, archived historical accounting reproduced, and the extracted manuscript rebuilt without warnings (`../p0/monitor/2026-09-22-paper-checkpoint0837.json`). This supersedes the 0710 manuscript checkpoint for reviewing the current text, but still precedes completed serving-route qualification and E1 timing. Retain all earlier checkpoints as audit history.


## Selected-route evidence checkpoint, 2026-09-22 09:34 UTC

`selected-route-20260922T0930Z.tar.gz` and its `.manifest.json` contain840 files /1,735,458,237 bytes. Archive SHA-256: `7f127f0c2e8198f067eb0f59737e1a58afc34991aa562dacbc11d7d147d25a30`. All member hashes and the embedded manifest were verified locally; proof is `p0/monitor/2026-09-22-selected-route-artifact-checkpoint.json`.

The three captured roots preserve incompatible policy A (including the failed async recorder) and the selected synchronous flat-slot B1/B4 pilots. The B4 loop's original failed verdict is retained together with the reviewed v3e offline gate, independently derived exact API-ID evidence, and corrected numerical-label summary. Later offline analyzers and repository helper dependencies are copied under `p0/monitor/selected-route-source-20260922T0928Z`; its README records the path mapping and distinguishes reviewed sources from other preserved dependencies. These are finite continuation checks, not warmed timing or full-model sequential-equivalence evidence.

The current selected-route manuscript text has passed independent review; native live preflight,18 E1 confirmation cells, and a final paper bundle remain pending. The0837 paper archive above is still an older intermediate checkpoint. No file has been published or submitted.


Latest manuscript archive (09:48 UTC): `paper-review-20260922T0948Z.tar.gz`,110 files, SHA-256 `7899411a6ded18ffd6f94f6da72a1de66a126720a5774d03505f1152fd174279`. It contains the13-page paper at source1857d760 / abstract9a29ea33 / PDF93610803 and binds allthree companions. Every member size/hash passed; the canonical manuscript was built and visually checked immediately before packaging. This supersedes0837 for the selected-route text, while native preflight and E1 timing are still pending. Proof: `p0/monitor/2026-09-22-paper-checkpoint0948.json`.


Working manuscript checkpoint10:16 UTC: `paper-review-20260922T1016Z.tar.gz`, 118 files, SHA-256 `6c5560a1c518f6cdad65d81fbff81721815d3ee38033d9f93f22a602e7c883f7`. Includes current13-page paperf52a828f/abstract0eddd199/PDFae619da4, closed whole-paper review and first-native-cell audit. Every member size/hash verified; allthree earlier evidence companions bound. E1 campaign is still running; this is not a final result package. Proof: `../p0/monitor/2026-09-22-paper-checkpoint1016.json`.


## Final E1 evidence companion, 2026-09-22

`e1-timing-20260922T1225Z.tar.gz` and `.manifest.json`: 1,482 payload files; 12,955,447 archive bytes; SHA-256 `bf50b33386eb9614cf7874039f60cb62a484ee62bc0882e32ab32731857318d6`. All member sizes/hashes and the exact member set were checked in `../p0/monitor/2026-09-22-e1-artifact-checkpoint.json`. Independent essential-evidence/source/reducer/qualification completeness review is `../p0/monitor/e1-artifact-final-redteam.md`.

The archive contains the original 18 cells, 230 phase captures, 90 loaded modules, direct API IDs, event records, the frozen aggregate, support/exclusion tables, full-stream diagnostics, exact guarded dependencies and T1 deviation records. It retains the native/tree engine-seed difference, the EOS outcome, and continuation divergence. The measured rates compare as-executed instrumented configurations on fixed prefixes; they are not matched-output, quality-preserving, seed-controlled causal speedups.

The final paper package built by `../scripts/build_review_bundle.py` binds all four evidence companions. Extract the paper archive at a scratch repository root and companions at a separate evidence root. Original DGX paths remain in provenance; replay needs explicit path mapping. For raw E1 accounting, use the archived direct-ID mapper/joiner with a cell's event/API/join inputs and a separate output path. The frozen aggregate writes `aggregate.json`, so use a writable scratch copy. Fresh serving additionally requires the pinned external model/image/runtime. Copied-byte verification does not certify an unmodified cross-host GPU rerun.


## Historical-number removal revision, 2026-09-22 19:45 UTC

The current 11-page manuscript reports quantitative results only from fresh v2 captures or fresh synthetic controls. Historical result numbers, plots and mixed archived-operand remeasurements were removed at the author's request; historical raw files and earlier immutable checkpoints remain audit evidence. `paper-final-20260922T194605Z.tar.gz` supersedes the 12:39 paper archive for manuscript review and binds the same four unchanged raw-evidence companions. See `FINAL-DELIVERY.json` for archive hashes and member checks; the independent removal review and canonical build/visual proof are included in the archive. The earlier extracted numerical replay checks remain applicable to the unchanged raw evidence, not a claim of a repeated GPU campaign.


## Attention provenance correction — 24 September 2026

Earlier entries calling the recorded Cat10 qualification attention stock are superseded by the loaded-source audit in `notes/attention-backend-provenance-correction-2026-09-24.md`. These jobs use patched Triton unified attention selected as TREE_ATTN; the recorded agent deployment uses patched FA2. Backend identity is specific to each launch, not determined by FP8 versus NVFP4. Historical records remain intact, and no existing qualification is relabeled as an FA2 execution. The current build is recorded in `p0/monitor/2026-09-24-attention-correction-build.json`.


## Superseded-route removal — 24 September 2026

The paper describes the latest deployed NVFP4 Hydra27/fixed32 method located in the production history: forked FA2 GQA-pair split-K4, spine-first KV slots with matching mask columns, full-vocabulary single-logits drafting and fused top-three selection, two-level GDN path scans with transient cut states, fixed32 device acceptance, and one captured committer replay containing 48 native per-layer GDN updates. Prefix caching and serving/drafter/committer graphs are enabled. Query row and position order remain logical. The single-launch GDN and Hydra31 candidates are not the executed workload method.

The September 22 Cat10/E2/E7/E8 jobs revisited a superseded route. Their original data, adverse outcomes and dated analyses remain intact in the archive, but all their numerical results and current-method qualification claims are removed from the paper. A later run date does not make an old implementation current. The current scan and native committer are not claimed to share one update body or to be bit-identical by construction.

Current validation is bound to the deployed binaries: fused-selection parity over 6,840 configurations and 24 graph replays, and the split-K numerical credential with 16 determinism cases across two processes and all nine declared checks passing. The attention probe uses synthetic tensors at scales measured from model operands; it is not a captured-task/full-model equivalence test. Runtime receipts positively identify the engaged attention arm and captured device route.

Current receipt: `artifacts/FINAL-DELIVERY.json`. Build: `p0/monitor/2026-09-24-current-production-build.json`. Earlier attention-provenance wording is accurate only as an archived run description and no longer defines the paper method.

## Novelty review — 24 September 2026

The current claim set has now undergone primary-source review and two independent mechanism reviews. See `notes/novelty/novelty-review-2026-09-24.md`, `claim-matrix.csv` and `source-register.json`. Direct overlap requires attribution to SpecLA, Trees from Marginals and Snakes and Ladders; attention/drafting/caching precedents were added. The paper is positioned as an implemented systems design with scoped component and workload evidence. A new general verifier, replay mechanism, chain schedule or split-K algorithm is not established. The most specific layout intervention remains a candidate for controlled current-build validation.

This review did not change workload arithmetic or run inference. Earlier implementation-audit PASS statuses are not novelty approvals. The dated review reports describe their exact input snapshots.
