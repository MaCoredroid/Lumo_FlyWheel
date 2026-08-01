# SFWD B1 eager timing preflight

This CPU/source-only preflight makes the post-byte-gate stock versus SFWD
timing pair lifecycle-correct. Both arms are exact full-vocabulary fixed32 B1,
run eager, execute the real `astropy__astropy-12907` SWE-Verified task, and are
classified as `eager_kernel_timing_diagnostic`.

The runner now rejects graph acceptance artifacts, validates one eager task
boundary plus terminal and traffic-skip evidence for each arm, and requires the
candidate arm to install and engage the authenticated SFWD live PASS. The pair
is diagnostic only: one task cannot satisfy the standing exact4/exact16
acceptance rule.

No GPU kernel, Docker container, live timing pair, or acceptance campaign was
started by this preflight. Run `prepared_command.sh` only after the B1 every-byte
gate has emitted its final live PASS.

Code commit: `b234e66cf3e81c5e4b5d45f62119e6bbad9c642a`.

