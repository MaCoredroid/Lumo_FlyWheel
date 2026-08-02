# FR13 B1 wide256 recompute stack-zero cap-320 build

Status: compiled and static-codegen qualified; eligible for the one-real-task
K64/root1 byte gate. No GPU task, byte-equivalence, timing, or hardware-floor
claim has been made from this artifact.

This rebuild raises the B1 diagnostic comparison cap from 256 to 320. The
established real-task comparison histogram consumes 256 calls on the first four
mandatory projection shapes (`69 + 68 + 51 + 68`), so the prior immutable
binary could not reach the fifth required shape. The new cap leaves 64 bounded
comparison calls for that fifth-shape coverage. The candidate still always
serves the stock result during byte A/B.

## Static result

- Candidate FP16 and BF16 kernels: `REG=168`, `STACK=0`, `LOCAL=0`,
  `SHARED=1024`, `CONSTANT[0]=2944`.
- Both candidate SASS bodies contain zero `LDL`, `STL`, or device-call
  instructions.
- Each candidate body has 4,952 instructions and 15 `SR_TID.X` reads.
- All six stock device-kernel symbols and resource records exactly match the
  pinned stock-symbol-exact reference: `REG=168`, `STACK=0`, `LOCAL=0`,
  `SHARED=1024`, `CONSTANT[0]=2560`.

## Identity

```text
source_commit=9ae5e07229a824ca97bfa0c48cb7e55c9ba5822e
vllm_commit=fe9c3d6c5f66c873d196800384ed6880687b9e52
cutlass_commit=da5e086dab31d63815acafdac9a9c5893b1c69e2
patch_sha256=1119c135b0828f70e4be289fed670a57c19d4429e8397a75b7feedb3514475cc
patched_dispatch_sha256=f3a3d8191d1f64bf7f63c4816ca1b979c042c6d511d134e1794f3e3330178b11
binary_sha256=503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a
binary_bytes=113079680
binary_mode=0555
```

Immutable candidate:

```text
/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_recompute_stack0_k64_root1_gate_ready_cap320.abi3.so
```

The host build used CUDA 13.0.88 and GCC 13.3.0. The extension RUNPATH is
`/home/mark/fr13_streamk_build/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:`.

## Scope

The next permitted action is one real SWE-Verified B1 K64/root1 byte gate that
proves all five mandatory projection shapes were compared and every output was
byte-identical. This artifact contains no task prompt, patch, model output,
token sequence, or other raw SWE-Verified material.
