# FR13 B1 Gate A qrow32 split-2 byte rejection

Status: **REJECTED DURING THE FIRST REAL TASK REPLAY**.

Gate A attempt 8 ran from clean, pushed source commit
`49b1ecd37fa8a4618c6cfd3946069bddf865cdeb`. Its fixed contract was
Hydra27, physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL
graph mode. The candidate tuple was qrow32 split2, GQA-group3, and mapped K64
top3.

The v4 binary repaired the attempt-7 split scratch defect. Model and drafter
loading, profiling, mixed graph capture, FULL decode graph capture, health,
authenticated ingress, and the canonical task request all completed. The first
real decode replay reached and executed the private qrow32 split2 path, then the
fail-closed live comparator rejected it against the incumbent qrow16 output:
3,104,943 output bytes and 9,551 LSE bytes differed.

This proves only that split scratch allocation and dispatch now reach CUDA. It
does not qualify the kernel. The mismatch is a kernel or split-combine semantic
defect. The task did not complete, no candidate credential was issued, and the
run produced no valid timing, TPS, acceptance, or hardware-floor evidence.

The next source change must explain and repair the split2 output/LSE contract,
retain fail-closed guards, rebuild the pinned SM121a binary, and rerun the same
real Gate A from a clean pushed commit.

This package contains only the status summary, structured manifest, and package
checksums. Raw logs, task content, requests, responses, workspaces, environment
data, process or container identity, runtime manifests, binaries, tensors, and
credentials are excluded. No digest of an excluded raw artifact is published.
