# SM121 FP8 Device/Build Sanity

This artifact proves only that the pinned vLLM image can quantize the exact
`[65536,5120]` K64 drafter-head weight and execute its SM121 CUTLASS block-FP8
path for B1 and B4 with the required tensor layouts.

It is invalid for performance measurement, tuning, lossless qualification,
acceptance, or production admission. The synthetic tensor timings in
`result.json` are retained as raw sanity output and must not be compared with
real SWE-Verified full-step measurements.

Required next evidence is the canonical one-real-SWE B1 integrity and
engagement gate. Exact4 full-step timing is allowed only after that gate passes.
