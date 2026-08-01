# FR13 BF16 B1-B4 full-head source readiness

Status: `SOURCE_AND_EXACT4_SHADOW_ROUTE_READY_BUILD_UNAVAILABLE`

This package binds the smallest B4 extension of the qualified M1 draft-head
idea. It is not a byte-equality result, timing result, production credential,
or hardware-floor acceptance result.

## Credential separation

The already-qualified B1 files remain byte-for-byte unchanged:

- `csrc/fr13_bf16_gemvx_m1.cu`:
  `26ea8aad9f891b5e758a39464209d6f82008a10fac8da4c02ee052e839218a54`
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`:
  `c4b5550cac2bbb5b213d76de3551e3ea61c1a0b5e5db93064404711f6313332d`

B4 uses the separate source
`csrc/fr13_bf16_gemvx_b1_b4.cu` and separate patcher
`scripts/fr13_phase4_patch_vllm_tree_gdn_b1_b4.py`. The launcher chooses them
only when `FR13_DRAFT_HEAD_M1_MAX_BATCH=4`.

## Candidate

- One CUDA launch per full-vocabulary head for actual batch sizes B1-B4.
- Grid `[31040,1,1]`, block `[16,8,1]`, one CTA per eight vocabulary rows.
- Each thread loads its weight element once, then updates one FP32 accumulator
  per request row. B4 therefore performs one logical read of the
  2,542,796,800-byte BF16 head weight, not four independent reads.
- Each request row retains the M1 scalar FMA chain and shared-memory reduction
  order. This is a source property only; byte equality still requires the real
  exact4 gate.
- The B4 operator is `fr13_bf16_head::gemvx_b1_b4_out`.

## Runtime boundary

`FR13_DRAFT_HEAD_M1_MAX_BATCH=4` is reference-first and shadow-only. The
launcher pins Hydra27 fixed32, four engine sequences, concurrency four,
full-vocabulary `root=0/K=0`, graph mode, and the canonical four SWE-Verified
task IDs. Candidate results are compared over raw BF16 bits and the stock
tensor is always served. B4 production is hard-disabled.

The Docker argument regression verifies that the candidate SO bind is added
before fixed32 ingress arguments, ingress appends to the same array, and the
final `docker run` expands the accumulated array.

## Verification

The focused source/runtime/M32 suite passed with `36 passed`. `bash -n`, Python
bytecode compilation, and `git diff --check` passed. Ruff is unavailable in
this host environment. Runtime-manifest generation reached its fail-closed
dataset check and stopped because the pinned SWE-Verified cache blob is absent.
See `source_validation.json` for exact commands and identities.

## Exact next commands

Build in the pinned Torch/CUDA environment:

```bash
PYTHON_BIN=/path/to/torch-2.10.0-cu130/bin/python \
  bash results/fr13_fixed32_bf16_gemvx_b1_b4_source_ready_20260801/prepared_build.sh
```

After integrating commit `0f2a31ed2` (runner-owned concurrent B4 endpoint
metrics), restoring the pinned SWE-Verified cache, and obtaining the build
outputs, run the real exact4 shadow gate:

```bash
PYTHON_BIN=/path/to/torch-2.10.0-cu130/bin/python \
FORKED_FA2_SO=/absolute/path/to/fr13_fork_fa2.so \
FR13_DRAFT_HEAD_M1_SO=/absolute/path/to/fr13_bf16_gemvx_b1_b4.abi3.so \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION=/absolute/path/to/build_attestation.json \
  bash results/fr13_fixed32_bf16_gemvx_b1_b4_source_ready_20260801/prepared_exact4_gate.sh
```

Only after zero raw-BF16 mismatches and complete exact4 provenance may a
candidate-only timing route be added. Full-wall TPS and floor/U95 remain
unmeasured.

The preparation host exposes Torch `2.4.1+cpu`. No candidate binary was built
and no GPU/container was used. A host-only `nvcc -c` attempt stopped before
source compilation because the CPU Torch headers omit
`c10/cuda/impl/cuda_cmake_macros.h`; it emitted no object.
