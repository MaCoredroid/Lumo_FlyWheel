# Private review checkpoints

These files are private and have not been published or submitted. The bounded v2 work is complete: P0; two E7a kernel campaigns, eight pilot inputs and 31 eligible confirmation inputs (one original provenance failure); six corrected E7b diagnostic boots; selected B1/B4 route checks; and all 18 original E1 timing cells. Independent final manuscript and essential-artifact reviews have no unresolved material finding or necessary additional experiment for the stated claims. The current 15-page PDF is built and visually checked. Final delivery archive identities and verification are recorded in a separate delivery receipt to avoid a self-referential manifest. Dated entries below describe superseded checkpoints.

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
