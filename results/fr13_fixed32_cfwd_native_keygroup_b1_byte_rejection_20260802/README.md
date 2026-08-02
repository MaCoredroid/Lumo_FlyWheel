# Native CFWD B1 byte rejection

The real SWE-Verified B1 qualification reached server health, opened the
authenticated task interval, and invoked the native key-group precompute
candidate. The first byte comparison rejected the candidate on all 48 model
layers.

This is a kernel-correctness rejection. The later engine-liveness and outer
runner exit statuses are consequences of the fail-closed byte gate, not the
primary result. No SWE task completed, and no timing, TPS, or hardware-floor
claim is valid.

A host CUDA compile overlapped the authenticated task interval. That overlap is
recorded for provenance, but it cannot explain a deterministic 48-of-48 byte
mismatch and no timing result is being retained.

This directory contains no task identifier, prompt, model request or response,
patch, raw log, environment, process identifier, or container identifier.
