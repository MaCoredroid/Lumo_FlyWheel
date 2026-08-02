# Fixed32 CUTLASS divisor K64 B1 timing path fix

Status: **READY_FOR_ONE_REAL_SWE_VERIFIED_B1_K64_ROOT_RETRY**.

The failed direct divisor session was a pre-task harness failure, not a kernel
or timing result. Container startup reached the qualified CUTLASS production
installer, where the K64 sidecar verifier tried to resolve the pinned
draft-vocabulary block map through its host-relative default. The repository
was mounted at `/workspace`, but the container had no matching working
directory, so installation stopped before the server or task ingress started.

Source commit `5de0781a067f1576c5d1e9137bee51715c7dce9e` passes the exact pinned block-map
path through the binary installer and supplies the `/workspace` path in the
container command. The sidecar schema, candidate identity, block-map SHA-256,
K64/root1 profile, and all byte-gate qualifications remain unchanged.

An exact host replay of the failed container-side qualification passed from a
non-repository working directory using the original qualified sidecar and
candidate. The stopped failed container was removed only after its full
recorded identity, terminal state, and zero restart count matched; aggregate
Docker state is now empty.

No GPU run, service, task, timing sample, throughput result, or acceptance
claim was produced. This package contains no raw task/model content, request,
response, patch, log, environment dump, process/container identifier, binary,
PTX, SASS, credential, or secret.
