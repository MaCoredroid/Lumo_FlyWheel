# FR13 B1 Gate A split-2 scratch rejection

Status: **REJECTED DURING THE FIRST REAL TASK REPLAY**.

Gate A attempt 7 ran from clean, pushed source commit
`372b2bdeae1580c13f2af72917a0ad89a7f46ebe`. Its fixed contract was
Hydra27, physical32, K64/root1, B1, one canonical SWE-Verified task, and FULL
graph mode. The candidate tuple was Qrow32 split2, GQA-group3, and mapped K64
top3.

The exact private Python route cleared its split-count guard, and the request
entered the private C++ Qrow32 split2 launcher. That launcher rejected the
parameters before launching CUDA because the FA2 varlen implementation only
called `set_params_splitkv` when `seqlenq_ngroups_swapped` was true. The
canonical qlen-32 geometry makes that predicate false. Because
`set_params_fprop` zero-initializes the parameter block, the launcher received
`num_splits=0`, `oaccum_ptr=null`, and `softmax_lseaccum_ptr=null` despite the
private wrapper requesting split count 2.

This is a source-proven scratch-allocation defect, not a kernel-output or
performance result. The split-2 CUDA kernel did not launch, byte comparison did
not complete, the real task did not complete, and no candidate credential was
issued. The run produced no timing, TPS, acceptance, kernel qualification, or
hardware-floor evidence.

The repair widens scratch allocation only for the exact private selector
sentinel while retaining the stock predicate for every ordinary FA2 call. It
uses the existing `set_params_splitkv` allocation and tensor lifetime path;
the hidden launcher still revalidates split count and both scratch pointers.
After rebuilding the pinned binary, the same real Gate A must rerun from a new
clean, pushed source commit.

This package contains only the status summary, structured manifest, and package
checksums. Raw logs, task content, requests, responses, workspaces, environment
data, process or container identity, runtime manifests, binaries, tensors, and
credentials are excluded. No digest of an excluded raw artifact is published.
