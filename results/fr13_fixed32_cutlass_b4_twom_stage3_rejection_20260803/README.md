# Fixed32 B4 two-M Stage3 rejection

This reduced artifact records a host-only SM121a compile and code-generation
audit of a three-stage mainloop on the retained B4 two-M kernel. The experiment
kept the `64x128x128` ping-pong tile, identity epilogue, exact fixed32
projection gate, and two-M static scheduler unchanged.

## Result

The candidate compiles for FP16 and BF16, but each kernel has an 8-byte
per-thread stack frame. Focused SASS contains three `LDL` and one `STL`
instruction per dtype, grows from 1,040 to 1,064 instructions, and adds four
branches. Register use remains 168, so the extra buffering does not improve
the register-limited residency bucket and adds spill traffic.

The three-stage candidate is rejected and was removed from the branch. The
retained two-stage kernel remains zero-stack and zero-local-traffic.

No GPU kernel, Docker service, synthetic workload, real task, timing run, or
hardware-floor measurement was used. The object, cubin, raw SASS, raw resource
dump, generated dispatch, and build logs are not published; their reduced
identities and counts are recorded here.

## Files

- `manifest.json`: pinned source/build identities and rejection verdict.
- `kernel_resources.tsv`: candidate and retained-kernel resource comparison.
- `sass_summary.tsv`: reduced instruction comparison.
- `verification.txt`: checks and scope boundary.
- `SHA256SUMS`: checksums for the reduced artifact.
