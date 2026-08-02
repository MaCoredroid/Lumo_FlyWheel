# SFWD prior-reuse K64/root1 B1 gate

This reduced package records the completed real SWE-Verified B1 correctness
diagnostic for `fixed32_sfwd_prior_reuse_rowgroup32_c64_v1` at source commit
`b6572a9ab91f281d7c1f84bfb41c24329e6323da`.

The gate passed: one authenticated task resolved, 25,056 records covered all
48 layers and both compared byte surfaces, and every comparison was byte
equal. The reference result was always served. The K64 shim and root gather
each engaged exactly once, with no full-vocabulary or linear fallback. Source,
runtime, and external launch/end manifests were byte identical, and teardown
left no Docker container or GPU compute process.

This is a source-bound B1 correctness credential only. It contains no timing or
acceptance result, does not serve candidate bytes, and does not enable
production by default. The cited 119.658015414 ms floor is an optimistic
mandatory-weight-read lower bound, not a complete measured hardware-floor
step. Candidate-served B1 timing and exact4 B4 remain required.

The earlier missing-cache launch is recorded separately in
`preflight_rejection.json`. It failed before container, GPU, or task execution
and is not a kernel result.

Only derived counts, verdicts, and hashes are included. Raw task, model,
request, response, patch, process, environment, container, and comparison logs
are excluded.
