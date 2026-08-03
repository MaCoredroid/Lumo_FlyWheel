# Fixed32 B1 N5120 full-grid K64/root byte PASS

Status: `PASS` for the bounded byte-correctness diagnostic.

The authenticated real SWE-Verified task `astropy__astropy-13236` resolved.
For `identity_onen_n5120_fullgrid_b1`, all 320 shadow comparisons were
byte-equal to stock at M32. The comparisons covered the five audited `(N,K)`
projection shapes: `5120x6144`, `5120x17408`, `14336x5120`, `16384x5120`,
and `34816x5120`. There were zero mismatching comparisons and zero differing
bytes.

The canonical gate is bound to source commit
`c49c8eb5370e4d4035aceffaa8476aea31f921f5`, the `k64_root` qualification
profile, and the `astropy13236` diagnostic profile. The repository validator
returned 0 against the pinned 118,836,392-byte candidate with SHA-256
`1024adda7fe4d314f31779206b5b6a7691ef1eee6c7c15ce5c837cced99a3584`.

The diagnostic served stock; the candidate remained default-off. This result
is not acceptance, timing, TPS, hardware-floor, or production-performance
evidence. It makes no candidate-caused acceptance claim.

This reduced artifact retains only the canonical gate, this README, a concise
manifest, and checksums. Raw prompts, responses, patches, logs, environment
payloads, task output, process identifiers, and container identities are not
published. The candidate binary is identified by digest and size but is not
included. `SHA256SUMS` covers every retained artifact file except itself; the
artifact integrity and claim boundaries are checked by the corresponding test
under `tests/`.
