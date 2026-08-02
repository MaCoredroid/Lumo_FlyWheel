# Fixed32 CFWD native full-value compile qualification

Status: **SM121a resource qualification passed; default off; real-task raw-byte
and timing qualification pending**.

## Result

The exact full native translation unit was compiled for the non-stable vLLM
`_C` extension with the pinned CUDA 13.0 / PyTorch 2.10.0+cu130 toolchain. The
post-fix kernel uses:

| Resource | Native CUDA result |
| --- | ---: |
| Registers/thread | 64 |
| Stack frame | 0 bytes |
| Spill stores | 0 bytes |
| Spill loads | 0 bytes |
| Local memory | 0 bytes |
| Source static shared memory | 548 bytes |
| Reported shared memory | 1,572 bytes |
| Barriers | 1 |
| SASS `LDL` / `STL` | 0 / 0 |

The 1,572-byte resource report comprises the source kernel's 548 bytes plus
1,024 bytes reserved by the SM121 target. At 64 registers/thread and 512
threads/CTA, the kernel reaches the required register ceiling for two-CTA/SM
eligibility. This is a compiler resource result, not an observed occupancy or
latency measurement.

The first exact compile exposed an 8-byte spill load and 8-byte spill store
caused by a 64-bit state-bank inner offset living across the recurrence. The
source now keeps inner offsets 32-bit, separates load/store address lifetimes,
and reconstructs the final-store indexing after the recurrence. The exact
recompile retained 64 registers and eliminated the stack, local memory, and
spill traffic.

## Incumbent comparison

A host-only Triton 3.6.0 compile of the exact incumbent full-value BV128
layer-batch kernel at its fixed B4 geometry reported 255 registers/thread,
424 bytes of stack, and 53 `LDL` plus 53 `STL` instructions. The native
candidate therefore has a strong compiler-resource advantage. Neither compile
is a real SWE-Verified timing measurement.

## Byte-equivalence audit

One concrete mismatch was corrected: the native state-bank initialization now
performs `load + 0.0f`, matching the incumbent Triton `zeros; += load`
sequence and its negative-zero normalization. Exact native SASS retains the 32
corresponding add-zero instructions.

Raw-byte equality remains **high risk and unlikely on the first real attempt**:

- K-norm reduction uses a CUDA down-shuffle tree rather than Triton's
  FMA/butterfly and shared inter-warp tree.
- State-dot reduction groups and combines terms in a different order.
- CUDA and Triton transcendental paths and fusion/scheduling differ.

The logical ordered recurrence is preserved, but that is insufficient for the
required FP32 raw-byte credential. The candidate must remain default-off and
ineligible for timing until it passes all-bank raw-byte gates on authenticated
real SWE-Verified B1 and canonical exact4 B4 campaigns. If it fails, the next
kernel revision must deliberately match the incumbent operation trees and
transcendental ordering before any performance conclusion is valid.

## Scope and verification

- Candidate plus adjacent lifecycle/task-boundary suites: `90 passed`.
- Ruff, Python byte compilation, and `git diff --check`: pass.
- The full translation unit and exported host wrapper compiled; the operator
  was not linked into or launched by the serving image.
- No GPU query, CUDA launch, Docker mutation, synthetic/probe performance run,
  real-task campaign, or timing measurement was performed.

This directory contains reduced compiler/resource facts only. It contains no
prompts, responses, patches, traces, raw logs, process/container identities,
credentials, object files, or timing samples.
