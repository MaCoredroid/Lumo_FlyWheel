# K64 physical32 B1 stack review

Status: corrected and statically verified; no Docker, GPU, SWE-Verified task,
timing, TPS, acceptance, or hardware-floor measurement was run.

Reviewed base: `27e4cfb1e1fe3e5f34c321ade42728e84c5fd45b`.

## Findings

1. The fixed32 closeout called the current authenticated traffic-audit API
   without its required `concurrency` argument. Every non-eager fixed32 run
   would finish its real tasks and then fail closeout with rc16. The serve
   runner now passes and validates `SWE_CONCURRENCY` as exactly 1 or 4.
2. The fixed32 runtime manifest omitted the SFWD production controller, SFWD
   PASS validator, qrow16 sidecar issuer, and integrated timing runner. They
   are now covered by launch/end source-drift comparison.
3. The B1 runner relied on the canonical K64 default block-map path without
   pinning its SHA. It now checks the exact host bytes, exports and verifies the
   container path, and publishes both identities in the timing summary.
4. Phase reporting converted SFWD seconds to milliseconds correctly, but did
   not reject a negative or non-finite other-wall residual or independently
   reconcile full-wall TPS. The reducer now requires
   `SFWD + DFWD + CFWD <= full wall`, checks TPS from committed tokens and wall
   time, and records the residual.
5. The proposed B1 runner is not an all-parent committer stack. Both TAW
   selectors are deliberately zero. Its metadata now says so explicitly.
   Source-v7 all-parent remains default-off pending the exact4 shadow PASS and
   a combined co-candidate gate.

The qrow16 qualification is not graph-execution-only: its live FULL run
retained real graph operands, synchronized after replay, and recalled stock and
candidate FA2 directly outside CUDA capture. The eager bridge invokes that
same exact-geometry candidate path. This review found no binary or shape drift
in that bridge.

## Next real gate

After integrating this review branch and the pinned Qwen turn-cap bundle fix,
run the source-v7 all-parent Tail23 shadow gate first:

```bash
cd /home/mark/lumoFlyWheel-k64-m128-b4
TAG="source_v7_k64_root1_$(date -u +%Y%m%dT%H%M%SZ)"
RUNROOT="$PWD/output/fr13_tail23_all_parent_exact4_b4_${TAG}"
FORKED_FA2_SO=/home/mark/lumoFlyWheel-node32-nonscaling/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
RUNROOT="$RUNROOT" TAG="$TAG" FORKED_FA2_SO="$FORKED_FA2_SO" \
  bash scripts/fr13_run_b4_tail23_all_parent_live_gate.sh
```

The stock FA2 input is pinned to 299,183,936 bytes and SHA256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
On PASS, the gate publishes a source-v7 bundle with independent B1-B4 records.
The detailed host validation and credential-consumption flow is in
`results/fr13_fixed32_tail23_all_parent_exact4_b4_gate_ready_20260801/README.md`.

Do not add `FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1` to the qrow16 +
SFWD timing runner. That runner correctly rejects the unqualified
co-candidate. A combined exact real-task gate is required before serving all
three candidates together.
