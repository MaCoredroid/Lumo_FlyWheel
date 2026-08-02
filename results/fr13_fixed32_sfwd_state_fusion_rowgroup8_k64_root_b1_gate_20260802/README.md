# SFWD rowgroup8 K64/root1 B1 gate

This reduced package records the completed real SWE-Verified B1 correctness
diagnostic for `fixed32_sfwd_state_fusion_rowgroup8_v3` at source commit
`b2c5f4ab71c5b91401c5459f4a5478ba6cbd3e84`.

The gate passed: one authenticated task resolved, 18,672 records covered all
48 layers and both compared byte surfaces, and every comparison was byte equal.
The reference result was always served. The K64 shim and root gather each
engaged exactly once, with no full-vocabulary fallback. Launch/end runtime and
external manifests were byte identical, and teardown was clean.

This is a source-bound B1 correctness credential only. It contains no timing or
acceptance result and does not enable production by default. The cited
119.658015414 ms floor is an optimistic mandatory-weight-read lower bound, not
a complete measured hardware-floor step.

The earlier relative-FA2 launch is recorded separately in
`preflight_rejection.json`. It failed before container or task execution and is
not a kernel result.

Only derived counts, verdicts, and hashes are included. Raw task, model,
request, response, patch, process, environment, and container logs are excluded.
