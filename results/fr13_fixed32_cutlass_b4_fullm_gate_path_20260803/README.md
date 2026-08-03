# Fixed32 B4 full-M gate path

Status: **host-ready, credential-gated, default off; real SWE-Verified Tail23
and Hydra27 byte gates pending**.

`identity_fullm_b4_byte_ab` is admitted only as an eager exact-B4 diagnostic.
`identity_fullm_b4` is installable only with a same-source dual-topology
production credential. The credential requires independent authenticated
four-task SWE-Verified PASS records for Tail23 and Hydra27 at K64/root1,
physical rows 128, batch size 4, and concurrency 4.

The candidate is the already compiled full-tile stable-libtorch binary with
SHA-256 `85937b5c35ec87bce12e4b5d677dd67f63004f9a9d9fb6d64473a5bd3b53b2da`.
No live credential is included. No GPU run, timing result, or hardware-floor
claim is represented by this artifact.

The live gate binds the patch, gate, timing harness, qualification logic,
binary installer, launcher, server wrapper, vLLM patcher, SWE runner, fixed32
contract, proxy, exact-four subset, and K64 block map to the clean source
commit. A later code change therefore invalidates the credential.
