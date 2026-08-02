# Fixed32 B1 M32 static-linear K64/root gate handoff

Status: **host codegen and end-to-end gate wiring pass; live byte gate not run
from this branch yet**.

This candidate preserves the stock B1 swapped `128x32x128` collective, scale
layout, ordered K traversal, FP32 accumulation, and epilogue. It replaces only
CUTLASS's generic static-persistent coordinate decode with the exact fixed32
one-dimensional M-tile schedule. The selector is default off, the diagnostic
always serves stock bytes, and direct production installation remains blocked.

## Immutable candidate

- Kernel source commit: `5aae84c25b30d472a4c014e9343783b902d446ba`
- Gate wiring commit: `bbe534024abcb51c9debdc6738f008af6c85a2ea`
- Patch source SHA-256:
  `59fb1d8a39a6b634fde9895db3efd32e2921636ef017e30167dd94748e718e39`
- Patched dispatch SHA-256:
  `9709adac0a029a57de58229c8dc9478bd647491988749bd23697fb40131d42ff`
- Candidate SO:
  `/home/mark/fr13_m32_static_linear_build/bin/_C_stable_libtorch.static_plus_m32_v4_079d82d60426411b.abi3.so`
- Candidate SHA-256:
  `079d82d60426411bf403eb96f4869cb8d3872a4a68d49e9c336a55a90d571f91`
- Candidate size/mode: `113809232` bytes, `0555`
- ELF build ID: `dcb1fa96503b8bff225a7590da310248ccb24eae`

The direct FP16 and BF16 device bodies each use 168 registers, 1,024 bytes of
static shared memory, zero stack, zero local memory, zero `LDL`/`STL`, and no
device calls. Each has 648 SASS slots, versus 936 for the generic static
scheduler and 1,176 for stock. Mandatory math and data movement are unchanged:
32 `QMMA`, 32 `FFMA`, 24 `FMUL`, 24 `LDSM`, four `STSM`, eight output packs,
and 38 `SYNCS` for both static variants. These are compiler facts, not latency
or throughput evidence.

## Real B1 gate

Run only after the active GPU owner confirms that the previous arm has fully
ended and the GPU/container are clean:

```bash
TAG=m32_static_k64_$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=/home/mark/fr13_b1_m32_static_k64_runs/$TAG
env \
  RUNROOT="$RUNROOT" \
  TAG="$TAG" \
  FORKED_FA2_SO=/home/mark/lumoFlyWheel-b1-wide256-k64-root-profile/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so \
  CUTLASS_STREAMK_SO=/home/mark/fr13_m32_static_linear_build/bin/_C_stable_libtorch.static_plus_m32_v4_079d82d60426411b.abi3.so \
  FR13_STREAMK_GATE_CANDIDATE=m32_static_linear \
  FR13_STREAMK_QUALIFICATION_PROFILE=k64_root \
  bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh
```

The required result is one authenticated real SWE-Verified B1 task, 320 or
fewer contiguous comparisons, all five projection shapes, and zero differing
bytes. A PASS remains a one-task correctness diagnostic. Full-step TPS,
acceptance, phase timing, and hardware-floor claims require the standing real
exact4 or exact16 task campaign after production qualification is explicitly
wired.

No GPU kernel, container, real task, synthetic timing probe, or performance
measurement was launched while preparing this artifact.
