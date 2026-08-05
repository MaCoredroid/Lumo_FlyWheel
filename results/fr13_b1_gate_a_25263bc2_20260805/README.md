# FR13 B1 Gate A pre-task Qrow32 profile rejection

Status: **REJECTED BEFORE HEALTH OR AUTHENTICATED TASK WORK**.

The combined Gate A launch ran from clean, pushed source commit
`25263bc26c0df5c3e92d0465e48d1f3be0bca142`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

The preceding fixed32 profile-scope lifecycle rejection was repaired: this
launch entered `profile_cudagraph_memory` and reached the Qrow32 live A/B hook.
The hook then applied final real-event geometry validation to a throwaway
profile invocation and failed closed with the Qrow32 B1 live-gate geometry
guard. The production Qrow32 path already distinguishes profile capture from
final capture; the live A/B path did not.

The launch did not reach health, authenticated task work, or either byte
comparator. It issued no Qrow32, GQA-group3, or DFWD credential and produced no
timing, TPS, acceptance, kernel qualification, or hardware-floor evidence.

This reduced package contains only the status summary, structured manifest,
and package checksums. Raw logs, environment data, task content, requests,
responses, workspaces, process or container identity, runtime manifests,
binaries, tensors, and credentials are excluded. No digest of an excluded raw
artifact is published.
