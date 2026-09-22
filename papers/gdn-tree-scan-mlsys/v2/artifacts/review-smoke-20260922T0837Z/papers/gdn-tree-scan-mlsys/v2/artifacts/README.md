# Private review checkpoints

These files are local and have not been published. The current paper and experiment state continue evolving; final release/submission is not implied. E2/E7b qualification and E1 timing remain incomplete.

Reviewed manuscript/E7a checkpoint (2026-09-22; predates the added model diagnostics):

- `paper-review-20260922T0710Z.tar.gz`: manuscript/PDF,80hash-manifested files, historical accounting sources, P0 audit sources, exact older E7a source versions. SHA256 `3676083f2e7b3c7b31ce1648ddebbf9dfcb804bf80363bb43261e8c79ed6c8ea`.
- `e7a-evidence-20260922T0709Z.tar.gz` and `.manifest.json`:2859files, all8frozen pilot inputs/provenance,31primaryconfirmation inputs plus original failure/retry records, native selection pool/scores,4historical tensors, result JSON and frozen-analysis erratum. SHA256 `776ddcaf5d938cc665bafca20f46e29d1e9a550a34914d798e78f70636cd180b`.

All companion file hashes were verified after transfer. Targeted independent review closed the missing-file findings in `../p0/monitor/artifact-redteam-round1.md`. Original absolute paths are preserved and mapped explicitly; hash verification and raw-result inspection are supported, while a full unmodified GPU rerun on a different host is not certified. The pinned image and model are external dependencies; model weights and JIT cache bytes are excluded.

The earlier0700E7a and0702paper pair is superseded because it omitted3pilot captures,4historical tensors,selection data and older source/P0 records. Preserve it as audit history, use the newer pair for review. `review-smoke-*` directories are extracted verification workspaces, not authoritative evidence.

Rebuild with `../scripts/build_review_bundle.py --output <new filename>` after final experiments/claim ledger stabilize; do not overwrite old snapshots.

Additional completed diagnostic evidence:

- `e7b-diagnostics-20260922T0801Z.tar.gz` and `.manifest.json`: 727 files, 2,179,735,658 archive bytes. Includes both completed three-boot batches, raw operands/states/hidden/logit captures, executed snapshots, and the reviewed CPU continuation source/dependencies. SHA-256 `c573b5064d944547df280e314520a97dd0c10678fb877c8d75b3a898d548ad05`. Every member size and hash was independently verified after transfer (see `../p0/monitor/2026-09-22-e7b-artifact-checkpoint.json`). These six boots had plain-route attention-KV remap disabled; they support the stated local diagnostics, not full-route qualification.

The current twelve-page manuscript/PDF includes the independently reviewed verifier/commit diagnostics and explicit KV-off limitation. It is newer than the 0710 manuscript archive. The final manuscript archive must be rebuilt with this E7b companion and the eventual corrected-route/timing evidence; no current archive should be represented as the completed v2 release.
