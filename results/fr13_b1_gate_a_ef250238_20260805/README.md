# FR13 B1 Gate A split-2 wrapper rejection

Status: **REJECTED DURING THE FIRST REAL TASK REPLAY**.

The combined Gate A launch ran from clean, pushed source commit
`ef250238a3a609ea5256ee292854d13e7c7aa6c6`. Its fixed contract was Hydra27,
physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL graph
mode. The intended candidate tuple was Qrow32 split2, GQA-group3, and mapped
K64 top3.

The interleaved K/V page specialization cleared model and drafter loading,
profile and mixed capture, final FULL graph capture, server health, fixed32
ingress, and the start of the authenticated real task request. On the first
qrow comparator replay, the Qrow16 reference call returned and the candidate
call reached `flash_attn_varlen_func` with `num_splits=2`. The installed FA2
Python interface retained its generic `num_splits > 1` rejection and raised
before dispatching the private custom C++ operator.

This is a runtime routing defect, not a kernel output or performance result.
The Qrow32 split2 kernel did not launch, byte comparison did not run, the real
task did not complete, and no Qrow32, GQA-group3, or DFWD top3 qualification
credential was issued. The run produced no timing, TPS, acceptance, kernel
qualification, or hardware-floor evidence.

The repair must keep generic FA2 split counts fail-closed while allowing only
the exact private Qrow32 B1 split2 selector tag to reach the custom tree-bias
operator. Substituting `num_splits=1` is invalid because it would exercise a
different kernel. After static verification, the same real Gate A must rerun
from a new clean, pushed source commit.

This reduced package contains only the status summary, structured manifest,
and package checksums. Raw logs, environment data, task content, requests,
responses, workspaces, process or container identity, runtime manifests,
binaries, tensors, and credentials are excluded. No digest of an excluded raw
artifact is published.
