# Fixed32 DFWD K64 M1 pair8 real B1 diagnostic

This reduced artifact records the first real SWE-Verified B1 run of the
default-off `pair8bits` BF16 K64 draft-head candidate. The task resolved and
the underlying serving arm returned zero. The outer launcher returned four
only because its post-run check searched the redirected launcher log instead
of the captured container runtime log; commit `abbd87372` fixes that evidence
path and has focused regression coverage.

The diagnostic is not gate-eligible and is not hardware-floor evidence. It
measured 227.276 ms per full step, 3.653 accepted drafts per event, and 20.472
wall TPS. The prior stock B1 diagnostic measured 241.709 ms, 4.890 accepted
drafts per event, and 24.367 wall TPS. Pair8 therefore reduced step wall by
about 6.0 percent while reducing end-to-end wall TPS by about 16.0 percent.
It is rejected as a dominated B1 point. The run overlapped host compilation,
so the wall result is also not suitable for close-call promotion.

Only sanitized aggregates are published. No task, prompt, response, patch,
environment, process, container, or raw log material is included.
