# FR13 B1 Gate A pre-task capture rejection

Status: **REJECTED BEFORE HEALTH OR AUTHENTICATED TASK WORK**.

The combined Gate A launch ran from clean, pushed source commit
`4dc01e59f1c29e57192ea2e0341c4b18b95a8714`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

The engine stopped during `profile_cudagraph_memory`. The fail-closed fixed32
lifecycle detected that a throwaway profile graph's capture scope had not
closed cleanly. The launch did not reach health, authenticated task work, or
either byte comparator. It issued no Qrow32, GQA-group3, or DFWD credential.

This attempt produced no timing summary, TPS, acceptance, kernel qualification,
or hardware-floor evidence. The failure condition is established, but its
root cause is not yet verified. A source fix must preserve the strict capture
lifecycle, pass focused host tests, and rerun Gate A from a new clean, pushed
commit.

This reduced package contains only the status summary, its structured
manifest, and package checksums. Raw logs, environment data, task content,
requests, responses, workspaces, process or container identity, runtime
manifests, binaries, tensors, credentials, and raw failure payloads are not
published. No digest of an excluded raw artifact is published.
