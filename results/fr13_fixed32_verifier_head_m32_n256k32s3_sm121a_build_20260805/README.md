# Fixed32 verifier-head M32 N256/K32/stage3 full build

Status: **host full-SO build and static audit pass; live qualification pending**.

This artifact records a loadable Torch extension for the default-off
N256/K32/stage3 BF16 verifier-head candidate at source commit `93a9d882e`.
The kernel preserves the full `248320`-token vocabulary and the exact
BF16 `[32,5120] x [5120,248320] -> [32,248320]` contract.

## Build isolation

The extension was compiled in a new external directory under a private
Torch `2.11.0+cu130` environment. CUDA visibility was empty,
`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_CACHE_DISABLE=1`, and `MAX_JOBS=1`.
No Docker command or GPU tensor was used. CUTLASS was the clean pinned commit
`da5e086dab31d63815acafdac9a9c5893b1c69e2`.

The builder resolves NVIDIA package headers from the active pinned Python
environment instead of assuming the container-only `/usr/local` site-package
path. A wrong-Torch invocation was rejected before creating any output.

## Full shared object

- external path:
  `/home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/fr13_bf16_verifier_head_m32_n256k32s3_sm121a.abi3.so`
- SHA-256: `03f5d07a7f4029d7bc4a6a271a3c7e34f433c2f139d9adb374fcd0a80d1b91a7`
- bytes: `235328`
- mode: `0555`
- registered schema:
  `fr13_verifier_head::bf16_m32_n256k32s3_out(Tensor(a!) output, Tensor hidden, Tensor weight) -> ()`

The binary remains outside Git. Its exact identity, source binding, build
attestation, dependency audit, and rebuild commands are retained here.

## Static audit

The shared object contains one `sm_121a` cubin. The kernel uses 128 registers,
zero stack/local bytes, 1024 static shared bytes, and 55296 source-locked
dynamic shared bytes. Its SASS is byte-identical to the staged codegen SASS:
32 BF16 HMMA instructions, 27 asynchronous 128-bit global-to-shared loads,
eight ordinary 128-bit global loads, sixteen 128-bit stores, and no `LDL`,
`STL`, or `CALL` instructions.

Fresh CPU-only loading resolved every dynamic dependency and registered the
expected CUDA-only operator schema. The focused source, artifact, M32, and B4
suite passed 45 tests, and the broader inherited regression set passed 84.
No output comparison, real task, timing, TPS, or hardware-floor claim was made.

## Admission boundary

This candidate is built but unqualified and is not wired into Gate A. The
current M32 Gate A source, launcher, and binary identities were unchanged.
The next admissible step is one real SWE-Verified B1 shadow task comparing
every raw BF16 output byte while always serving incumbent logits.
