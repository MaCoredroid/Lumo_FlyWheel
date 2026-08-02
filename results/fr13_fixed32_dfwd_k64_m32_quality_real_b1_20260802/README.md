# Fixed32 K64 M32 quality route: real B1 diagnostic

Status: `KEEP_FOR_B4_VALIDATION`

This is a one-task, real SWE-Verified diagnostic for the fixed K64/root-on
Hydra27 route with the draft-head GEMM padded to 32 output rows. The task
resolved and the strict runtime markers show that the M32 route was ready and
engaged without a reported fallback.

The point retains 4.662337662338 accepted draft tokens per event and reaches
24.917947737874 wall tokens/s at 227.239326524918 ms/full step. Relative to the
earlier non-simultaneous stock B1 reference, wall latency is 5.9864% lower and
wall TPS is 2.2610% higher despite 4.6557% lower acceptance. It dominates the
rejected pair8 diagnostic on acceptance and TPS at essentially equal wall
latency.

This is not acceptance evidence. B1 is diagnostic-only, the run overlapped
host compilation, and neither byte equality nor a clean paired baseline was
measured. The observed wall latency is still 1.899073x the mandatory-weight
read lower bound and 1.651368x the one-sided 1.15x cap.

Only sanitized aggregate evidence is published here. Raw task identifiers,
requests, responses, patches, environment values, process/container identity,
and raw logs are intentionally excluded.
