# Fixed32 DFWD K64 M1 full-warp R32 build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row and packs 32 rows
into a 1,024-thread CTA. Relative to the 16-row warp32 candidate, it keeps the
same 160 dependent FMA iterations per lane, contiguous full-warp weight access,
and total arithmetic, but halves the launch from 4,096 to 2,048 CTAs. A 1,024
thread block may permit one resident CTA where the 512-thread parent may permit
two; active-thread capacity can therefore remain equal. Only a real-task route
comparison can determine whether lower CTA scheduling overhead wins.

The changed lane partition relative to the incumbent draft head can change BF16
logit rounding and draft argmax choices. That is allowed only through the
deterministic proposal contract: fixed32 requires `draft_probs=None`, current
MTP proposals use argmax, and the deterministic multi-draft committer is the
one-hot rejection-sampling specialization. Its seven reference tests pass.
Runtime integration must still prove that candidate logits exclusively produce
proposal tokens and that no stale probability or reference-token path is used.
This artifact does not claim that closure.

Static codegen is clean: 18 registers/thread; zero stack, local, and shared
memory; zero barriers, local load/store, atomics, or calls; five shuffle-down
and five FP32 reduction-add instructions. Its operational SASS is identical in
count to the 16-row warp32 parent. The immutable 113,640-byte binary has SHA256
`23e300504db1704042a1498a142f9b714e92353af6780e24163246e6653104f5`
and a GLIBC ceiling of 2.32.

Source commit: `3f0e98f2270c96cbf8b33d6176a75cc41521796f` on
`agent/fixed32-dfwd-k64-m1-warp32-r32-20260802`. Full-step speed, acceptance,
task quality, and hardware-floor impact remain unmeasured.
