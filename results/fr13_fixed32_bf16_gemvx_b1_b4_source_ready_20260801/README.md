# FR13 BF16 B1-B4 full-head source readiness

Status: `SOURCE_AND_SHADOW_RUNTIME_READY_BUILD_UNAVAILABLE`

This package binds the smallest source-level B4 extension of the qualified M1
draft-head idea. It is not a byte-equality result, timing result, production
credential, or hardware-floor acceptance result.

## Candidate

- One CUDA launch per full-vocabulary head for actual batch sizes B1-B4.
- Grid `[31040,1,1]`, block `[16,8,1]`, and one CTA per eight vocabulary rows.
- Each thread loads its weight element once, then updates one FP32 accumulator
  per request row. The B4 specialization therefore has one logical read of the
  2,542,796,800-byte BF16 head weight, not four independent weight reads.
- Each request row retains the M1 scalar FMA chain and shared-memory reduction
  order. This is a source property only; byte equality with the stock B4 path
  still requires the real exact4 gate.
- The original `gemvx_m1_out` symbol is unchanged. The new symbol is
  `gemvx_b1_b4_out`.

## Runtime boundary

`FR13_DRAFT_HEAD_M1_MAX_BATCH=4` is accepted only with reference-first,
shadow-only exact4 B4 traffic. The launcher pins Hydra27 fixed32, four engine
sequences, concurrency four, full-vocabulary `root=0/K=0`, graph mode, and the
canonical four SWE-Verified task IDs. Candidate results are compared over raw
BF16 bits; the stock tensor is always served.

B4 production remains hard-disabled until a real exact4 byte gate passes. The
Docker argument regression verifies that the candidate SO bind is retained
when fixed32 ingress arguments are appended and is expanded into the final
`docker run` invocation.

## Verification

The focused source/runtime/M32 suite passed with `34 passed`. `bash -n`, Python
bytecode compilation, `ruff` on the changed tests/build script, and
`git diff --check` also passed. See `source_validation.json` for exact commands
and identities.

## Remaining gates

1. Build with the pinned Torch `2.10.0+cu130`, CUDA 13.0, and SM `12.1a`
   environment using `prepared_build.sh`.
2. Integrate the runner-owned B4 campaign endpoint-metrics fix before launching
   exact4. It is not an ancestor of the source commit bound here.
3. Run canonical real SWE-Verified exact4 at B4 with four physical fixed32
   request trees, `root=0/K=0`, and reference-first full-logit comparisons.
4. Only after zero raw-BF16 mismatches and complete campaign provenance may a
   candidate-only timing route be added. Full-wall TPS and floor/U95 remain
   unmeasured.

The host available for this preparation exposes Torch `2.4.1+cpu`, so no
candidate binary was built and no GPU/container was used.
An additional host-only `nvcc -c` attempt stopped before compiling the source
because that CPU Torch installation omits
`c10/cuda/impl/cuda_cmake_macros.h`; it emitted no object. The exact failure is
recorded in `build_attempt.json`.
