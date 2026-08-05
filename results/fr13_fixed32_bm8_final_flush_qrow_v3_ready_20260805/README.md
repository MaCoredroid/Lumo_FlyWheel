# FR13 BM8 final-flush accounting and Qrow v3 repin

Status: **SOURCE READY; LIVE QUALIFICATION BLOCKED BY QROW SPLIT2 WRAPPER**.

BM8 production engagement is no longer published after the first measured
graph replay. The runtime retains the capture record until the authenticated
final flush, reconciles the complete event census against the graph lifecycle,
requires one measured replay per complete B1 event and zero unmeasured
replays, then publishes the engagement with the final-flush binding. The
composed smoke gate and exact4 timing reducer validate that binding and use
the observed replay count instead of a hard-coded count of one.

The branch also contains the Qrow32 v3 interleaved-K/V specialization from
main. Its external binary was rechecked offline at 300,140,712 bytes with
SHA-256 `07e02c0a53185c48d745fb221e7c807f97bfe40f61354e4242e9271e743e13c1`.
It is hash-pinned in the runtime and gate paths, but it is not qualified.

The latest real one-task B1 Gate A reached the first Qrow comparator replay,
then the generic FA2 Python wrapper rejected `num_splits=2` before entering
the private C++ candidate operator. That is a routing rejection, not a kernel
correctness or performance result. The exact private split2 route must be
fixed and the same real Gate A rerun before BM8 composed timing can proceed.

No GPU, Docker, synthetic probe, real task, byte comparison, timing, TPS,
acceptance, or hardware-floor measurement was run for this source artifact.
It makes no kernel-qualification or performance claim.
