# Verification

- The authenticated one-task real SWE-Verified evaluation returned one
  `resolved` verdict and one `tests_passed` failure-mode classification.
- The gate and live-pass summaries report source-only pass status, reference
  always served, no fallback, and production disabled.
- Independent aggregation of the record stream found 22,080 passing records,
  48 layers, both required surfaces on every record, zero byte differences,
  zero shape or dtype mismatches, and 30,749,491,200 compared bytes.
- Source, runtime, and external launch/end manifests are pairwise identical.
- A post-run host census found zero running Docker containers, zero GPU compute
  processes, and zero MiB GPU memory in use.
- Independent packed-source decoding yields historical tap counts 23, 28, and
  31, for 82 total. The corrected traffic math is analytical, not measured.

This B1 run is not timing-eligible or floor-acceptance-eligible. A canonical
exact4 B4 or exact16 campaign remains mandatory for acceptance.
