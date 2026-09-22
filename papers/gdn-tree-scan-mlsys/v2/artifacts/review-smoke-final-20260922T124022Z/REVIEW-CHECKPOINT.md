# Private LumoFlyWheel v2 review checkpoint

This checkpoint contains manuscript sources/PDF, the explicit historical evidence needed to reproduce its accounting, review reports, and a SHA-256 manifest. It is NOT a public release and has not been uploaded. Read MANIFEST.json's scope and the closure ledger for current completion status and limits. The cited historical GitHub revision identifies historical evidence, not the unreleased fresh v2 experiment files.

From the extracted repository root:

    python3 papers/gdn-tree-scan-mlsys/v2/scripts/audit_evidence.py
    cd papers/gdn-tree-scan-mlsys/v2
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first command reproduces archived proxies, not matched throughput. It runs no inference. Build requires LaTeX (IEEEtran, TikZ/PGFPlots and standard packages listed in main.tex); the included class and bibliography support an offline build. PDF byte identity is not expected across TeX versions/timestamps.

The four companions listed with archive and manifest hashes in MANIFEST.json contain raw experiment evidence. The E7a companion contains completed E7a sources, result JSON, frozen records/erratum, provenance records and captured tensors. The E7b diagnostic companion contains both stage-isolated three-boot batches and their raw operands, states, hidden activations, logits, exact source snapshots and reviewed continuation/reducer dependencies. These six diagnostic boots had KV remapping disabled; they do not qualify the corrected serving route. The selected-route companion contains the failed policy A and bounded corrected B1/B4 qualification. The E1 companion contains all 18 original timing cells, raw events/direct API IDs, preflight/qualification records, loaded sources, exact frozen reducer and independent numerical review. Read artifacts/README.md and the closure ledger for the retained failures and limits.

Extract companions into a separate evidence directory; archive entries are relative to the original v2 directory. All original absolute provenance paths are retained unchanged and refer to the DGX, so execution requires an explicit local path mapping. This package certifies copied bytes and enables raw-result checks; it does not claim an unmodified full GPU rerun on another host. The historical tensor mapping is recorded in the E7a companion manifest. Byte-exact ladder-v3/tiny-gate core and device sources are in the paper archive under p0/monitor/snapshots/20260921T2308Z/experiments/e7a; the E7a companion contains the matching production kernel in the earlier source_snapshot. Native prefix-selection pool/scores and all eight pilot captures are included. JIT caches/model weights are excluded; recorded hashes are retained. Keep all companions private pending a separate release/provenance review.
