# Fixed32 FA2 qrow32 SM121a checkpoint

Status: **kernel resource pass; shared-object ABI not qualified**.

This is a host-only checkpoint for the hidden fixed32 B4 FA2 qrow32
candidate. It contains no GPU execution, synthetic performance probe, real
task output, byte-parity result, timing result, or production authorization.

## Kernel result

The exact generated block-M-32, block-N-64, two-warp translation unit was
compiled as one SM121a cubin with one NVCC worker at niceness 19. The target
kernel reports:

- 244 registers;
- zero-byte stack frame;
- zero static local bytes;
- zero ptxas spill loads and stores;
- zero SASS `LDL` and `STL` instructions;
- 1,024 static shared bytes and 80 KiB launch-time dynamic shared memory;
- four internal SASS calls to the compiler-generated integer-division
  subroutine, with no stack or local-memory traffic.

The qrow32 kernel keeps one complete ordered K loop per physical query row,
uses 64 threads per CTA, launches one CTA per B4 batch/head pair, and does not
use split-K or a combine kernel. Because the first source already met the
zero-stack admission rule, no arithmetic, mask, reduction, or template
specialization was made.

## ABI blocker

A host relink was attempted only to test the shared-object boundary. It is
rejected. `DT_NEEDED` matched the pinned exact-safe FA2 binary, and the qrow32
launcher remained local and absent from `.dynsym`, but the available host
GCC13 API object did not reproduce the pinned GCC11.4 dynamic ABI:

- defined dynamic records: 633 stock, 673 candidate; 40 added;
- undefined dynamic records: 168 stock, 175 candidate; 9 added and 2 removed;
- `DT_NEEDED`: 10 records on each side, zero differences.

The failed relink is not a candidate binary and must not be installed or used
for a byte gate. ABI qualification remains false until the API object is
rebuilt with the pinned GCC11.4 and matching Torch headers and all three ABI
record sets compare byte-for-byte.

## Resume boundary

After the live B4 timing arm has torn down and host memory has recovered:

1. Re-run the single SM121a qrow32 compile at niceness 19 with one NVCC worker,
   retain the ptxas output, and require the same source and object hashes.
2. Compile only the patched `flash_api.cpp` with the pinned GCC11.4 and
   matching Torch headers, then relink the unchanged stock object set plus the
   qrow32 object.
3. Run `scripts/fr13_fa2_qrow32_static_gate.py`; any resource or defined,
   undefined, or `DT_NEEDED` ABI drift rejects the build.
4. If and only if the static gate passes, run the canonical real
   SWE-Verified exact4 B4 raw-byte gates for both Tail23 and Hydra27 with stock
   output always served. Timing remains forbidden until both gates pass.

The qrow32 route remains hidden, default-off, byte-unqualified, and
timing-ineligible.
