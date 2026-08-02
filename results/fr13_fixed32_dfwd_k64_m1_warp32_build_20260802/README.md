# Fixed32 DFWD K64 M1 full-warp build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row. Relative to the R32
exact-order candidate, it halves the dependent per-lane FMA chain from 320 to
160 iterations and changes two distant half-warp weight segments into one
contiguous full-warp segment. It launches 4,096 CTAs of 512 threads rather than
R32's 2,048 CTAs of 512 threads. Total weight elements and FMAs are unchanged;
the tradeoff is shorter dependency depth and fuller warp memory access against
twice as many CTAs and warps.

The changed lane partition and added stride-16 reduction can change BF16 draft
logit rounding and therefore draft argmax choices. That is allowed only through
the deterministic proposal contract: fixed32 requires `draft_probs=None`, vLLM
generates current MTP proposals by argmax, and the deterministic multi-draft
committer is the one-hot rejection-sampling specialization. Its existing seven
statistical/reference tests pass. Runtime integration must prove that candidate
logits exclusively produce the proposal tokens and that no stale probability
or reference-token path is used. This artifact does not claim that closure.

Static codegen is clean: 18 registers/thread; zero stack, local, and shared
memory; zero barriers, local load/store, atomic, or call; five shuffle-down and
five FP32 reduction-add instructions. The immutable 113,616-byte binary has
SHA256 `4e50f356480bbf2395cb163b096f84a008d3848d7e6c525b609594fc4111fd37`
and a GLIBC ceiling of 2.32.

Source commit: `09389b86937af751b9601128d828ba2e203e94d5` on
`agent/fixed32-dfwd-k64-m1-warp32-20260802`. Full-step speed, acceptance, task
quality, and hardware-floor impact remain unmeasured.
