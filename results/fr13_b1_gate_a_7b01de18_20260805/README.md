# FR13 B1 Gate A pre-task unresolved final-capture rejection

Status: **REJECTED BEFORE HEALTH OR AUTHENTICATED TASK WORK**.

The combined Gate A launch ran from clean, pushed source commit
`7b01de18c47b0f23addfba092ac81bcd0c70f685`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

This source included the repair that permits a valid fused-QKV query view to
have row stride `8192` while retaining the Qrow32 kernel contract on query head
stride `256` and element stride `1`. The launch cleared model loading, initial
profiling, profile CUDA-graph memory, and mixed graph capture. It then failed
closed in the Qrow32 live hook during final FULL CUDA graph capture with the
same aggregate geometry-drift rejection.

The row-stride repair therefore did not resolve the remaining geometry drift.
The aggregate rejection does not identify which remaining exact-geometry field
differs, so this record does not assign a new root cause or justify relaxing
another guard. A field-specific, non-secret geometry diagnostic and real Gate A
rerun are required.

The launch did not reach health, authenticated task work, or any byte
comparator replay. It issued no Qrow32, GQA-group3, or DFWD credential and
produced no timing, TPS, acceptance, kernel qualification, or hardware-floor
evidence.

This reduced package contains only the status summary, structured manifest,
and package checksums. Raw logs, environment data, task content, requests,
responses, workspaces, process or container identity, runtime manifests,
binaries, tensors, and credentials are excluded. No digest of an excluded raw
artifact is published.
