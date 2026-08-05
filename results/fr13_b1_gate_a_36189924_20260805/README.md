# FR13 B1 Gate A pre-task interleaved K/V stride rejection

Status: **REJECTED BEFORE HEALTH OR AUTHENTICATED TASK WORK**.

The combined Gate A launch ran from clean, pushed source commit
`361899240f84d7c5fdf65549ca7ddaa2a4531219`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

This source added a field-specific, non-secret geometry diagnostic without
relaxing the Qrow32 guard. Model and drafter loading, initial profiling,
profile CUDA-graph memory, and mixed graph capture completed. The first final
FULL decode graph then failed closed before health or task execution.

The diagnostic reported one mismatched field. The live BF16 K cache had shape
`(606, 1024, 4, 256)` and stride `(2097152, 1024, 256, 1)`, while the private
Qrow32 candidate was specialized for block/page stride `1048576`. The V cache
passed the exact equality-to-K stride check. Source audit verified the live
layout as interleaved K/V pages: each K and V view advances by two physical
pages between block-table entries, while their base pointers select the K or V
plane.

This is a kernel address-specialization defect, not grounds for relaxing only
the Python selector. The candidate translation units, shared static page-stride
assertion, launcher guards, selector, source closure, binary identity, and
address-mapping tests must be rebuilt for stride `2097152` before Gate A is
rerun.

The launch did not reach health, authenticated task work, or any byte
comparator replay. It issued no Qrow32, GQA-group3, or DFWD credential and
produced no timing, TPS, acceptance, kernel qualification, or hardware-floor
evidence.

This reduced package contains only the status summary, structured manifest,
and package checksums. Raw logs, environment data, task content, requests,
responses, workspaces, process or container identity, runtime manifests,
binaries, tensors, and credentials are excluded. No digest of an excluded raw
artifact is published.
