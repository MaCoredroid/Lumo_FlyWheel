# Fixed32 B1 BF16 verifier-head M32 full build

Status: **offline SM121a build and static audit pass; live qualification pending**.

This artifact records the full loadable extension build for the default-off
BF16 verifier-head candidate. The source byte-matches corrected main commit
`b3cd2c5a9`, including the row-major `M=32` orientation. It preserves the
full `248320`-token verifier vocabulary and all BF16 input, weight, and output
types.

## Build isolation

The extension was compiled and linked in pinned image
`vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
with no network, one CPU, `MAX_JOBS=1`, `NVIDIA_VISIBLE_DEVICES=void`, an
empty `CUDA_VISIBLE_DEVICES`, and no NVIDIA device nodes. The source and
CUTLASS mounts were read-only. The build log explicitly reported that no CUDA
runtime was found.

The pinned image does not contain Git. A Git 2.34.1 binary from the local
Ubuntu 22.04 SWE base image was mounted read-only solely so the builder could
verify the clean pinned CUTLASS checkout. The source image ID and binary hash
are recorded in `manifest.json`.

## Full shared object

- path: `/home/mark/fr13_bf16_verifier_head_m32_sm121a_20260805/fr13_bf16_verifier_head_m32_sm121a.abi3.so`
- SHA-256: `5b5e8c3051f29bc4f65ef93c96ed22ef38ef07a1754e9c36a167e5158f71f4b7`
- bytes: `186048`
- mode: `0555`
- offline registered schema:
  `fr13_verifier_head::bf16_m32_out(Tensor(a!) output, Tensor hidden, Tensor weight) -> ()`

The binary itself is intentionally outside Git. Its identity and the complete
rebuild command are retained here.

## Static audit

The exact shared object contains one `sm_121a` cubin. Its candidate kernel
uses 158 registers, zero stack bytes, zero static local bytes, 1024 static
shared bytes, and 61440 source-locked dynamic shared bytes. SASS contains 64
BF16 HMMA instructions, 60 asynchronous 128-bit global-to-shared loads, eight
ordinary 128-bit global loads, and sixteen 128-bit stores, with no `LDL`,
`STL`, or `CALL` instructions.

The focused CPU suite passed: 7 tests, plus Ruff and Python byte-compilation.
No GPU tensor, synthetic workload, real task, timing, or output comparison was
run.

## Admission boundary

This candidate remains default-off and unqualified. It requires a real
SWE-Verified B1 shadow task comparing every raw BF16 output byte while serving
the incumbent logits. Only a zero-mismatch gate permits frozen-source exact4
B1 timing. B4 byte and timing gates follow only after B1 passes.

There is no byte-equality, verifier-distribution, latency, TPS, speedup, or
hardware-floor claim in this artifact.
