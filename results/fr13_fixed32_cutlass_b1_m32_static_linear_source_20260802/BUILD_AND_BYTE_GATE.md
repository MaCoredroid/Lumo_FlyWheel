# Build and byte-gate handoff

This handoff is intentionally not executed by the source-candidate commit.
Review `0001-feat-fixed32-add-M32-linear-static-scheduler.patch` first and recheck
disk. Source closeout saw only 4.4 GiB available.

## Pinned host build

Use a new isolated vLLM source and build tree at commit
`fe9c3d6c5f66c873d196800384ed6880687b9e52`; share only the immutable fetched
dependencies and pinned CUTLASS checkout. Apply:

```bash
/home/mark/fr13_streamk_build/venv/bin/python \
  /home/mark/lumoFlyWheel-sfwd-b1-wide256-recompute/scripts/fr13_patch_cutlass_fixed32_wave.py \
  /home/mark/fr13_m32_static_linear_build/vllm-source \
  --cutlass-root /home/mark/fr13_streamk_build/cutlass-source
```

Configure for SM121 and build the complete stable extension:

```bash
env TORCH_CUDA_ARCH_LIST=12.1a cmake \
  -S /home/mark/fr13_m32_static_linear_build/vllm-source \
  -B /home/mark/fr13_m32_static_linear_build/vllm-build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=/home/mark/fr13_streamk_build/venv/bin/python \
  -DVLLM_CUTLASS_SRC_DIR=/home/mark/fr13_streamk_build/cutlass-source \
  -DNVCC_THREADS=1 \
  -DCMAKE_MAKE_PROGRAM=/home/mark/fr13_streamk_build/venv/bin/ninja \
  -DFETCHCONTENT_BASE_DIR=/home/mark/fr13_streamk_build/vllm-build/_deps

env TORCH_CUDA_ARCH_LIST=12.1a \
  /home/mark/fr13_streamk_build/venv/bin/ninja \
  -C /home/mark/fr13_m32_static_linear_build/vllm-build \
  -j2 _C_stable_libtorch
```

Do not import or execute the extension during this gate. Preserve the linked
binary read-only under a content-addressed name and record its SHA256, size,
mode, RUNPATH, dynamic symbols, and embedded cubins.

## Static reject conditions

Reject before GPU execution if either BF16 or FP16 candidate has:

- more than 168 registers;
- nonzero stack, local memory, spills, `LDL`, `STL`, `LD.LOCAL`, or `ST.LOCAL`;
- static shared memory other than 1,024 bytes;
- changed mandatory counts relative to stock: 32 QMMA, 32 FFMA, 24 FMUL,
  24 LDSM, 4 STSM, and 8 dtype-specific output packs;
- any stock device-function body change;
- an additive device-body set other than the predicted BF16 and FP16
  `m32_static_linear` kernels.

Compare the new candidate with both stock and the separate generic-static
candidate. The generic-static source/artifact commits are `0adf68ed9` and
`668c06e0d`; do not merge its selector/config into this source branch.

## Real SWE-Verified byte gate

Before launch, add `m32_static_linear` and `m32_static_linear_byte_ab` to the
pinned binary-loader, server-launcher, live-gate, and pass-sidecar allowlists.
Pin those wiring changes and the immutable candidate binary by commit and
SHA256. Do not use a synthetic GEMM probe as the correctness gate.

Run `m32_static_linear_byte_ab` only after the authenticated real-task arm is
present. It runs stock and candidate on the same process and stream, logs up to
320 projection comparisons, and serves the stock result. The valid campaign
uses the established real SWE-Verified B1 task set with K64/root1 and the fixed
physical 32-row topology. One-task diagnostics are informative only and do not
qualify for acceptance.

Required pass conditions:

- every logged row has `byte_equal=true` and `mismatch_count=0`;
- all five allowlisted projection pairs are covered as expected;
- no non-M32 or non-allowlisted call dispatches the candidate;
- both Tail23 and Hydra27 arms resolve cleanly under the pinned task set;
- runtime/source/external manifests and binary identities are unchanged from
  launch through final flush.

Only after the byte gate passes may `m32_static_linear` be timed against stock
with paired full-step TPS and phase breakdowns. Acceptance still requires the
one-sided U95 to be at most `1.15x` the hardware floor. Commit and push the raw
gate reduction and timing artifacts separately.
