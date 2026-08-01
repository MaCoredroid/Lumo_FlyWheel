# FR13 fixed32 full-vocabulary M32 real-B1 byte-gate readiness

This is a CPU/source readiness record. The real SWE-Verified B1 gate was not
run, no GPU measurement was made, and this artifact is not a probe, timing,
floor, production, or acceptance result.

The gate is pinned to the single `astropy__astropy-12907` task. It exercises
one physical M32 BF16 full-vocabulary GEMM for each of the root plus four loop
heads, compares every served full-logit BF16 element against the incumbent as
raw bits, and serves the incumbent logits. Any mismatch, incomplete five-call
event census, task/provenance drift, missing terminal evidence, input mutation,
or launcher failure returns nonzero.

## Exact command

Run from a clean checkout of this branch with the ignored runtime closure in
place. This command intentionally enables only the M32 live diagnostic:

```bash
cd /home/mark/shared/lumoFlyWheel-fullhead-m32
TAG="draft_head_m32_b1_$(date -u +%Y%m%dT%H%M%SZ)"
RUNROOT="output/fr13_${TAG}" \
TAG="$TAG" \
PYTHON_BIN="$PWD/.venv/bin/python" \
FORKED_FA2_SO="$PWD/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
FR13_GATE_DRAFT_HEAD_M32=1 \
FR13_GATE_QROW16=0 \
FR13_GATE_TAW_NATIVE=0 \
FR13_GATE_DRAFT_HEAD_PAD=0 \
FR13_GATE_BM8=0 \
FR13_GATE_GDN_BV=0 \
FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
FR13_FA2_QROW16_PRODUCTION=0 \
FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
FR13_FIXED32_BATCH_GDN_BV8_TIMING=0 \
FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
FR13_FIXED32_CUTLASS_WAVE=stock \
FR13_FIXED32_CUTLASS_WAVE_SO= \
FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON= \
FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256= \
FR13_FIXED32_ATTRIBUTION_ONLY=0 \
bash scripts/fr13_run_b1_kernel_live_gate.sh
```

## Traffic accounting

One BF16 full head reads `248320 * 5120 * 2 = 2,542,796,800`
weight bytes. Five calls therefore require 12,713,984,000 weight bytes/event
for both the stock full-vocabulary B1 incumbent and M32 candidate. M32 saves no
mandatory bytes; it changes GEMM concurrency. Including nominal input reads and
output writes but excluding copies and runtime overhead, incumbent M1 is
12,716,518,400 bytes/event and candidate M32 is 12,795,084,800 bytes/event, a
78,566,400-byte increase (0.617829%). The diagnostic executes both paths, so
its nominal combined head operand traffic is 25,511,603,200 bytes/event and is
not performance evidence.

The deployed root-64K five-head workload reads 3,355,443,200 head-weight
bytes/event. Full vocabulary adds 9,358,540,800 head-weight bytes/event
(278.90625%) and raises total mandatory step weight bytes from 32,666,638,208
to 42,025,179,008 (+28.648619%). At 273 GB/s, the full-vocabulary mandatory
weight-only floor is 153.938385 ms/event. There is no defensible positive M32
timing-savings estimate and no measured saving yet; a rejected candidate has
zero production saving. The five head reads are DFWD traffic already included
in the total floor and must not be added to or subtracted from an SFWD
projection.

See `readiness.json` for hashes, exact geometry, and verification results.
