# FR13 fixed32 CFWD native key-group suffix-zero v5 host proof

## Classification

`READY_FOR_REAL_B1_BYTE_GATE`

This is a host-only compile and regression receipt. It is not a correctness,
timing, B1, B4, exact4, or exact16 acceptance result.

## Repair

The rejected v4 real B1 byte gate showed a raw FP32 running-bank mismatch in
all 48 layers. Host-side elimination found no drift in launch indexing,
butterfly reduction order, active recurrence order, or gate-scalar lowering.

The remaining source-level difference was the inactive suffix. The incumbent
runs fixed length 16 while the native candidate runs only the active root plus
accepted steps, at most 12. Incumbent inactive rows have zero K and V. For
finite state, their final FMA boundary collapses to `state + +0`, including the
required negative-zero to positive-zero normalization. V4 normalized the
initial load but stored the post-active state without that final boundary.

V5 adds one explicit round-to-nearest `FADD` per final FP32 register-state
element. The binary stays default-off and diagnostic-only.

## Verification

- Source commit: `2572fa82abe11db04e899c711133678ed16cbbbe`
- CUDA source SHA-256:
  `aaafab67be3754825109879b14033a892a87c4ae0b32da9d8d5106f903a2b0f9`
- Focused unit and gate-binding tests: 77 passed.
- Pinned target: SM121a, CUDA 13.0.
- Registers per thread: 64, unchanged.
- Stack, local memory, spill loads, spill stores, and calls: zero.
- Cuobjdump shared bytes: 7592, unchanged.
- FADD count: 225 to 257; the exact 32-instruction increase is the repair.
- `FADD register, RZ, register` count: 32 to 64.
- FFMA, shuffle, RSQ, RCP, and EX2 counts: unchanged.
- No GPU execution and no container launch were used for this receipt.

## Next gate

Rebuild the source-bound v5 binary and run the fail-closed real SWE-Verified B1
raw-byte gate. Only a clean B1 result permits timing work or the formal exact4
B4 campaign. A mismatch must remain a rejection and be localized by changed
byte count and first mismatch offset before further optimization.
