# Private LumoFlyWheel v2 review checkpoint

This checkpoint contains manuscript sources/PDF, the explicit historical evidence needed to reproduce its accounting, review reports, and a SHA-256 manifest. It is NOT a public release and has not been uploaded. E2/E7b qualification and E1 remain incomplete; read the closure ledger for current limits. The cited historical GitHub revision identifies historical evidence, not the unreleased fresh v2 experiment files.

From the extracted repository root:

    python3 papers/gdn-tree-scan-mlsys/v2/scripts/audit_evidence.py
    cd papers/gdn-tree-scan-mlsys/v2
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first command reproduces archived proxies, not matched throughput. It runs no inference. Build requires LaTeX (IEEEtran, TikZ/PGFPlots and standard packages listed in main.tex); the included class and bibliography support an offline build. PDF byte identity is not expected across TeX versions/timestamps.

The companion `e7a-evidence-20260922T0700Z.tar.gz` and its manifest contain original completed E7a sources, result JSON, frozen records/erratum, provenance records and captured tensors. Extract that archive into a separate evidence directory; archive entries are relative to the original v2 directory. All original absolute provenance paths are retained unchanged and refer to the DGX, so execution requires an explicit local path mapping. This package certifies copied bytes and enables raw-result checks; it does not claim an unmodified full GPU rerun on another host. JIT caches/model weights are excluded; recorded hashes are retained. Keep the companion private pending a separate release/provenance review.
