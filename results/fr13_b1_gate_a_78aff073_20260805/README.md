# FR13 B1 Gate A pre-task final-capture rejection

Status: **REJECTED BEFORE HEALTH OR AUTHENTICATED TASK WORK**.

The combined Gate A launch ran from clean, pushed source commit
`78aff073d77f9c8bc9dc2528fb808644de202ef0`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

This attempt cleared model loading, initial profiling, the throwaway
profile-memory scope, and PIECEWISE graph capture. It then failed closed while
capturing the final FULL B1 graph because the Qrow32 live hook rejected the
observed attention geometry. The two preceding profile lifecycle failures did
not recur; this rejection is in the later final-capture path.

The launch did not reach health, authenticated task work, or any byte
comparator replay. It issued no Qrow32, GQA-group3, or DFWD credential and
produced no timing, TPS, acceptance, kernel qualification, or hardware-floor
evidence.

This reduced package contains only the status summary, structured manifest,
and package checksums. Raw logs, environment data, task content, requests,
responses, workspaces, process or container identity, runtime manifests,
binaries, tensors, and credentials are excluded. No digest of an excluded raw
artifact is published.
