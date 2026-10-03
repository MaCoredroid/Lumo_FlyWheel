# Working source checkpoint — bounded independent review

Reviewed 2026-09-29 UTC. **PASS: no material archive, receipt, or claim-boundary defect found.** This disposition applies to `artifacts/working-source-20260929T002220Z`, not campaign completion, publication, or full-model qualification.

## Independently checked

- The gzip archive contains exactly **19 unique regular files**: the **17 recorded source members**, `BUILD.txt`, and `SOURCE-MANIFEST.json`. No absolute/traversal member names or links occur. Every archived byte matches the staged source. All 17 recorded hashes and sizes also match both the extracted clean-build copies and the current manuscript workspace; no member mismatch was found.
- Recomputed archive, source-manifest, build-log and clean-PDF hashes match `VERIFICATION.json`. The manifest's builder and originating recorder hashes match the actual current files. The builder reads the recorder's manuscript dependency closure, preserves previous artifacts by requiring a new output directory, copies regular source files, extracts its archive, checks hashes, and invokes the declared `latexmk` command with shell escape disabled.
- The saved clean build records successful compilation to **14 pages / 306,710 bytes**. Its recorder resolves all non-system manuscript inputs inside `clean-build/`; all 254 external inputs are under the installed TeX distribution. There is no fallback to the original paper or another workspace. Apart from generated `main.aux`/`main.out`, local TeX inputs are packaged members. The bibliography sources and generated bibliography are included.
- Independent `pdftotext -layout` extraction of the clean and current working PDFs produces the same SHA-256, matching both saved text files and `RENDER-COMPARISON.json`. The saved clean/working page-11 and page-12 PNG pairs are byte-identical. This verifies the stated comparison, not an independent all-page visual review. The log has no unresolved references/citations, multiply defined references, or horizontal overflow. It retains three small vertical-box warnings (1.01–1.73 pt); the receipt correctly limits its overflow assertion to horizontal overflow.

This review inspected the existing build artifacts and extracted PDF text; it did not rerun the compiler or builder, change source, invoke a model/GPU, or use the network. The archive is explicitly a manuscript-source closure requiring an installed TeX toolchain. It does not promise to contain the private raw experiment archive or independently reproduce scientific measurements.

## README and M1 scope

`README.md:5–13` consistently describes Codex direct ownership, the 30-minute watchdog, **0/4** workload attempts, pending candidate full-model qualification, and zero completed timed mechanism cells. The old `FINAL-DELIVERY.json` is expressly limited to the earlier delivery; the subsequent historical narrative does not authenticate this newer working draft. Operational status is reviewed here as wording, not independently re-queried runtime state.

The README's M1 summary agrees with the included `m1-qualification.tex` and the accepted result/prose reviews: 285,696 unique cells, 571,392 observations from two repeats in one process; Lumo passes its frozen comparator, while each of the three other configurations fails its own predeclared comparator. The manuscript preserves the distinctions between author preparation and the local aligned variant, software comparator conventions, and the original failed collection versus its lifetime-only repair. It does not turn finite-corpus component agreement into full-model/agentic equivalence, algorithmic incorrectness of other methods, or a qualified four-way timing advantage. The archived M1 TEX is byte-identical to the previously closed prose-review hash.

## SHA-256 bindings

| Artifact | SHA-256 |
|---|---|
| `lumotree-working-source.tar.gz` | `670cc0ac90113473adf6c95e352bf2a0900fa89452cd44e6496951a33338f3db` |
| `source/SOURCE-MANIFEST.json` | `7a18043084c0d25604c6c8fc03c61b966757a62e157cd290fa6383ddeeb22c5d` |
| `VERIFICATION.json` | `cd8e29ed3bfa8c55546ce133175349c8b9d344b48218dbfb8971c8a0882b0391` |
| `RENDER-COMPARISON.json` | `ebfda4e34f6f3cca551e8561c4cacd3521257b9f21dbf38846489d419f23e026` |
| `clean-build/main.pdf` | `4207af8cb9fb77fd689f2e7af24ad48d532786fecb17d5984c22f51beb3ac2dd` |
| Clean and working extracted text | `a731b441957612b88195d46bcf7cb55832f52e96efc6ed2b41f49ad81e6e64a9` |
| `scripts/build_working_source_checkpoint.py` | `ed3cda5a46a4ffda255bf048ba9ebf8ef22c8c9ee4bad043f132f5d6a5968f57` |
| `main.tex` | `a99e869d8af56353ec4b74d9f8feacb20053affb5ac3da7104abe5c8a459af82` |
| `README.md` | `94a8fe3583dbc2091472d7f19c3794fed7c4f61efc9e5f800229085be50d99e0` |
| `results/review-response-20260927/m1-qualification.tex` | `a47d54a8e55a2cda867c9494a59c74adf0e44399991356d7c9241571a21570df` |
| `notes/review-response-20260927/m1-q-c0-repaired-result-review.md` | `e2d9f8d6ab787e1a517d12985a6ba9fe1d11443414a2d1e840543c930521fe2e` |
| `notes/review-response-20260927/m1-qualification-paper-prose-closure.md` | `3dbccb499c73c61c6e9bca8870baf45010e09bddd59305982b814d7d655fe3fe` |

The manifest preserves the complete 17-member identities, including the unchanged abstract. No additional scientific experiment is required to close this bounded source-checkpoint review; the separately pending campaign gates remain pending.
