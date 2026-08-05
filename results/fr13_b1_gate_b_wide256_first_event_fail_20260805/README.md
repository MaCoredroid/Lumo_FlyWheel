# B1 Gate B wide256 first-event failure

Status: **FAIL** for the authenticated byte-correctness gate.

Gate B ran from exact source commit
`f81a1c774b55a7f76d30d30ed0fac2be73665be9` with the fixed Hydra27,
physical32, K64/root1, B1 contract. The real SWE-Verified task was
`astropy__astropy-12907`.

The first target-GEMM comparison used shape `M32 x N16384 x K5120`. The
`identity_wide256_fullgrid_b1` candidate differed from the stock result in 22
of 1,048,576 compared bytes; the first differing byte was offset 125,412. The
fail-closed gate terminated before the task completed, so no production
credential was issued.

The candidate binary was 119,979,144 bytes with SHA-256
`85937b5c35ec87bce12e4b5d677dd67f63004f9a9d9fb6d64473a5bd3b53b2da`.
The launch and end source manifests were byte-identical, with SHA-256
`79d39ea3e911d8fdbf1f342d265538d2fad5cb8bc0285c337f0a1c7776abcb8a`.

This result rejects the wide256 candidate for production and timing. It makes
no timing, TPS, acceptance, speedup, or hardware-floor claim. The reduced
artifact contains only the sanitized comparator record, source identity,
manifest, checksums, and this explanation. It excludes prompts, responses,
patches, traces, logs, environment dumps, process identities, binaries, and
tensors.
