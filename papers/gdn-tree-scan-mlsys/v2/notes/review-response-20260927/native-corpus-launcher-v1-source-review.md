# Native corpus launcher v1: bounded source review

**Final disposition: the two concrete findings below are closed on the reviewed successor. No remaining blocker identified within the stated sole-parent, serial execution workflow.** This is a source/CPU review, not launch authority, memory readiness, model qualification or a corpus result. A fresh exact gate and admitting readiness consumer remain parent obligations. The corpus manifest must be regenerated against the final source hashes before gating; its exact source/build equality check rejects an older manifest.

Reviewed final sources:

| File | SHA-256 |
|---|---|
| `tools/run_q1_native_corpus_v1.sh` | `1464009d5115c1a55addc5b3fe31c972a999366ee4add4d16d209f4f2d141243` |
| `tools/q1_native_corpus_v1.py` | `d240befad2a424c86674031b0d724adf4c1a56ba68c4c2da714f021ddbbdef10` |
| Accepted smoke `tools/run_q1_native_smoke_v2_4.sh` | `df5e0c1982f12060a4277bcd79d9762c371ed009c8a5ed47d3cfc7062903a721` |
| Reused cleanup library `tools/q1_native_smoke_cleanup_v2_3.sh` | `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524` |

## R1 — Unproven creation was treated as no container; closed

The first read of the new wrapper returned `absent` whenever `OWNED_CID` was empty. It assigned that variable only after successful Docker return, a full returned CID, and a matching CID/name/image inspection. Consequently, an ambiguous creation result or a failed initial ownership inspection could leave a created engine behind while cleanup recorded `no_owned_container` and rc 0.

The parent added `CREATION_ATTEMPTED` before issuing the creation command and a retained attempt record (final lines 178–195). The final override (lines 90–98) now distinguishes an empty CID before creation from an empty CID after attempted creation. Extracting the actual function and sourcing the unchanged cleanup library under a recording shell stub produced:

- Before creation: `cleanup_rc=0`, `no_owned_container`.
- After creation attempted with ownership unresolved: `cleanup_rc=8`, `not_verified docker_query_failed`; no deletion was issued.

These controls call only shell functions and a stub, not Docker. Exact-CID cleanup is armed only after the creation CID matches the intended container name and image. Ambiguous creation now remains an explicit failed cleanup requiring parent resolution rather than false absence.

## R2 — Corpus module executed before its gate/source check; closed

The intermediate launcher SHA `1edc1e5b48734bf365cfd1e53370d1579579e23af8920992536b86e917b4b62e` imported `q1_native_corpus_v1` before checking the gate or its `corpus_tool` hash. Executing the actual first heredoc against a temporary sentinel-only module and a missing gate created the sentinel before failure. No real campaign module or operational action was invoked by that negative control.

The final heredoc uses only stdlib before verifying the gate is open and belongs to this run, then checks the corpus module SHA and both caller/gate manifest SHAs before importing (final lines 39–48). The exact final heredoc now refuses a missing gate, a closed gate, and a wrong module SHA under an otherwise open matching gate without executing the sentinel. A full positive heredoc control using actual local fixture/source paths and a clearly synthetic byte-binding fork fixture succeeded. Repeating it with the same run ID refused with `FileExistsError` at the exclusive per-run binding file. That positive control checks the gate's byte-binding logic; it is not validation or execution of an actual GPU binary.

## Scope, preserved configuration and admission boundary

CPU construction through the actual corpus/job builders and arm renderer produced the four declared combinations: `aligned_nonpacked` A/B and `native_default_packed` A/B. Each has exactly 84 unique calibration cases, 3 prefixes × 28 paths, R=2, 168 requests, full state and KV archival, and the correct packed flag (0 for primary, 1 for the separate diagnostic control). There is one arm/process per wrapper invocation; it does not silently start all four. The corpus build enforces cycle 0 and excludes held-out cases. No model request or tensor computation was executed in these checks.

Compared with smoke v2.4, the wrapper still calls the same spec-off renderer, hooks/patcher v2.1, FA2 installer and reference driver. Its current gate additionally binds the corpus tool/manifest, driver object helper, readiness consumer and memory runner. The accepted renderer retains the same model, patched FA2 fork, cache layout/dtypes, memory utilization and common serving arguments; selecting the named packed arm changes its declared packed operator as intended. All calibration-prefix bytes are checked, not just the smoke's shortest prefix.

The wrapper requires exact reviewed scope/source hashes, `launch_limit=1`, an exclusive per-run binding and a new output directory. The final readiness call uses the gate-bound authority namespace and accepted v2.8 producer verifier; it does not issue a cache operation or capacity query. Existing failure, lock, hold or receipt mismatch refuses through that verifier. The named parent `no_reclaim_boot_approved` disposition remains an explicit alternative already implemented by the shared consumer, not an implicit fallback.

Container/compute/port checks are point-in-time checks. They do not acquire a cross-run GPU lock: continued sole-parent serial ownership is part of this bounded disposition. The per-run namespace prevents a duplicate invocation of the same run; it should not be described as a global exclusion lock for different arm/process run IDs. No additional framework is requested for the current sole-executor workflow.

`bash -n` passed. Review used source comparison, extracted shell/Python controls and CPU job construction only. No Docker, CUDA, GPU query, cache write, SSH, model boot, scientific gate modification or implementation edit was performed by this reviewer.
