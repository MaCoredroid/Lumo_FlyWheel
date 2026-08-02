# Fixed32 B4 FA2 qrow32 gate source

Status: **code complete and default off; CPU/static gates pass; GPU build,
resource inspection, real exact4 byte qualification, and timing pending**.

## Implemented

The candidate is a hidden FA2 `BM32/BN64`, two-warp BF16/HD256 launcher. For
B4 physical32 tree attention it keeps one CTA per batch/query-head pair and a
complete ordered K loop per real row, while removing the 32 out-of-extent query
rows in the stock `BM64` tile.

The ordinary path remains stock. The hidden launcher is reachable only when a
diagnostic tree-bias tensor carries the private batch-stride tag and the C++
gate confirms the pinned B4 geometry, including:

- batch 4, total query rows 128, 24 query heads, 4 KV heads, head dim 256;
- BF16 paged K/V with page size 1024 and FP32 `32 x 32` tree bias;
- noncausal full-window attention with no ALiBi, softcap, append-KV, or dropout;
- real paged-dispatch state `params.num_splits == 0 && force_split_kernel`.

There is no qrow32 production selector. A mismatch after the private tag is
present raises instead of falling back.

## Build Contract

Both source flags are mandatory:

```text
--tree-bias-tile-earlyout --fixed32-query-tile32
```

The pinned FA2 CMake file uses a non-`CONFIGURE_DEPENDS` source glob. The build
gate therefore accepts exactly one of these discovery routes:

1. Fresh configure discovers the generated qrow32 TU, followed by exactly one
   qrow32 CUDA-object build and one FA2 shared-library relink.
2. An explicit compile produces the named qrow32 object and the explicit FA2
   shared-library link appends that object exactly once.

Both routes require the pinned source and stock-object hashes, the exact
generated TU/API gate, and a final Ninja dry run with no work remaining.

## Real Byte Gate

`scripts/fr13_run_b4_fa2_qrow32_live_gate.sh` is the only live qualification
runner. It uses the canonical real SWE-Verified exact4 set at concurrency 4,
K64 with root reduction on, and `FULL_AND_PIECEWISE` graph execution. After the
first real stock B4 replay, it recalls stock and qrow32 on retained live paged
operands for all 16 tree-attention layers and all four slots. BF16 output and
FP32 LSE must match byte-for-byte. The captured stock graph output is always
served, and the run emits no timing samples.

Tail23 and Hydra27 require independent fresh run roots. No live run was
executed in this audit worktree because the canonical exact4 config fixture and
an inactive B4 GPU container were not available here.

After the candidate SO passes `fr13_fa2_qrow32_gate.py verify-build`, the two
real gates are launched independently:

```bash
FR13_QROW32_FIXED32_MODE=tail6_fixed32 \
RUNROOT=output/fr13_qrow32_tail23_exact4_20260802a \
TAG=tail23-20260802a \
FORKED_FA2_SO=/absolute/path/to/_vllm_fa2_C.abi3.so \
scripts/fr13_run_b4_fa2_qrow32_live_gate.sh

FR13_QROW32_FIXED32_MODE=hydra27_fixed32 \
RUNROOT=output/fr13_qrow32_hydra27_exact4_20260802a \
TAG=hydra27-20260802a \
FORKED_FA2_SO=/absolute/path/to/_vllm_fa2_C.abi3.so \
scripts/fr13_run_b4_fa2_qrow32_live_gate.sh
```

## Static Verification

- Ruff: pass.
- Focused/regression pytest: `24 passed, 1 skipped`.
- Shell syntax, Python byte compilation, and `git diff --check`: pass.
- No GPU command, container launch, synthetic probe, or timing command ran.

This artifact contains reduced source and aggregate verification only. It has
no prompts, responses, traces, patches, credentials, or performance samples.
