# Fixed32 DFWD K64 M32 quality-point source

This artifact records the source-ready direct M32 BF16 draft-head quality
point for exact fixed32 B1 with the drafter vocabulary fixed at K64/root1.
It runs one 32-row padded GEMM, returns row zero, and leaves verifier,
rejection-sampling, and committer logic unchanged. Drafter logits and therefore
acceptance may change; that is the quality/speed tradeoff this point is meant to
measure.

The runner enables only M32, pins K64/root1, records the candidate in the
fixed32 runtime manifest, and fails closed unless both static-buffer readiness
and an eager executed-GEMM marker appear in the final runtime log. The capture
state query is limited to the first invocation, so attestation does not add a
per-step host query after engagement.

Fifty-three focused host tests, Python compilation, Bash syntax, the manifest
self-test, and the git diff check passed. No GPU, Docker, real-task,
acceptance, throughput, B4, output-equivalence, production, or hardware-floor
claim is made here. The next eligible action is one real SWE-Verified B1
diagnostic after the current live GPU owner releases the machine.
