# Native MTP raw auditor: repair closure

Bounded disposition: accept the repaired auditor's source/CPU preparation for the three findings in `q1-native-mtp-raw-audit-v1-review.md`. Reviewed final SHA-256 is `d7ca5a80415f00c27acdbd4e44b94cbe6f00cac3b231992dfc400691cc87393c`. No auditor code was edited by this reviewer; no launch or scientific gate is approved by this note.

The final byte-preserved source and independently constructed fixtures/controls are under `p0/monitor/review-response-20260927/native-mtp-raw-audit-v1-final-review-20260929/`. The original failing snapshot and intermediate repair remain in their separate review directories.

All 35 CPU controls matched their intended outcomes: four valid synthetic histories/variants passed and 31 malformed cases refused. In particular, all 21 originally accepted malformed documents now refuse, and the eight original rejection controls remain effective. The original full-sized vocabulary/hidden/KV geometry is retained in the fixtures; these are not reduced-dimension stand-ins or model outputs. Tests ran locally with bundled Python and NumPy 2.3.5, without Torch native execution, GPU, or remote operations.

F1 is closed: snapshots now require the native 64×4×256 BF16 layout, complete ordered full/tail coverage, exact allocation maps, stable positive storage metadata, finite authenticated raw bytes, recomputed logical digest, and exact preservation of the previously materialized K/V prefix. A focused old-tail-to-new-full-block corruption refuses. A new suffix row with different finite content still passes, demonstrating that the check preserves only already materialized bytes.

F2 is closed: every allocation is joined to record generation, group, extent, columns, physical bounds and recomputed slots. Follow records join the preceding first pass, the next real target allocation and the independently reconstructed scratch row. The final small correction also preserves all old allocated columns as `History._follow` requires, including unused columns. A changed old unused column refuses; a legitimate appended new column passes.

F3 is closed: same-input schema/API, distinct derived first/follow rows, scratch location, and both native score SHA/greedy/dtype entries are checked against the corresponding history records. BF16 target-hidden geometry and existing finite full-vocabulary score checks remain active.

Caller responsibility is now documented in the auditor: authenticate the parent seal, job/fixture identity and target O0/O1/O2 first. Parent reports this is mandatory in the prospective `q1_reference_driver_v3_mtp.py`; this note does not independently approve that separate driver. The audit verifies internal raw/metadata consistency. Actual native execution and cache restoration remain bound to the pinned runtime checks; absent raw pre/post restoration images are not retroactively manufactured. Qualification remains false, and experiment denominators do not change.

Reproduction: run `controls.py` and then `boundary_controls.py` in the final review directory with `PYTHONDONTWRITEBYTECODE=1` and `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`. `RESULTS.json` retains the 31 original controls under repaired expectations; `BOUNDARY-RESULTS.json` retains the four focused boundary controls.
