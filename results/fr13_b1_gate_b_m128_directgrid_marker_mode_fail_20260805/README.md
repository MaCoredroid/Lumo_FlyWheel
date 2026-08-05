# B1 Gate B M128 marker-mode failure

Status: **INCOMPLETE** for the authenticated byte-correctness gate.

Gate B ran from exact source commit
`97a0e596f81ca5cb4ae8946e44138f87636c4646` with the fixed Hydra27,
physical32, K64/root1, B1 contract. The real SWE-Verified task was
`astropy__astropy-12907`.

The corrected M128 direct-grid target kernel passed its first comparison at
`M32 x N16384 x K5120`: all 1,048,576 compared bytes were equal to the stock
result. The run then failed closed before a second comparison because ingress
published the authenticated B1 event marker with mode `0400`, while the SFWD
conv/post-prep consumer requires mode `0444`. The task did not complete and no
production credential was issued.

The candidate binary was 119,781,296 bytes with SHA-256
`7d762dfa793671d75d1e353bd37d76fc07370cbe387ad1e315e32584d27927d4`.
The launch and end source manifests were byte-identical, with SHA-256
`6f6d2e058a00835bfd905cb48e8208aa9176988bcf15a1429120b6d597b6b111`.
The marker contract was corrected by commit
`8508c7a6607e589700f12ad19e8ed5e424742f98`.

This result is positive first-invocation evidence only. It does not qualify or
reject the M128 candidate and makes no timing, TPS, acceptance, speedup, or
hardware-floor claim. The reduced artifact contains only the sanitized
comparator record, source identity, checksums, and this explanation. It
excludes prompts, responses, patches, traces, logs, environment dumps, process
identities, binaries, and tensors.
