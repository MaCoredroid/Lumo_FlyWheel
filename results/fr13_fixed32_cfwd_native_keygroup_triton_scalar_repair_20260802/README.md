# Fixed32 CFWD native key-group Triton-scalar repair

Status: **source repair and offline SM121a compile pass; real-task byte
qualification pending; default off and timing-ineligible**.

## Authenticated rejection

The prior `fixed32_cfwd_native_keygroup_precompute_cuda_v3` binary reached the
first authenticated real SWE-Verified B1 K64/root-reduction event after a clean
server boot and graph capture. The paired gate completed both the incumbent and
candidate replays, then rejected the candidate because all 48 FP32 running-bank
rows differed at the raw-byte comparison. The engine exited through the
intentional fail-closed exception. No task completed and no timing sample is
valid.

This rules out a pointer crash: candidate addressability was sufficient to
finish the replay and comparison. The v3 source contract had explicitly left
CUDA `logf` versus Triton `tl.log` lowering as its remaining byte-risk.

## Repair

The v4 repair removes softplus, exponential, sigmoid, and gate-ring access from
the native CUDA kernel. A graph-captured Triton kernel materializes the active
root-inclusive decay and beta scalars using the same expression as the
incumbent byte-gate kernel. The native kernel consumes those FP32 scalars and
retains only K normalization and ordered FP32 state recurrence.

The event path is now two captured launches: one scalar precompute and one
native recurrence. At B1 the persistent scalar buffer is 221,184 bytes; at B4
it is 884,736 bytes. Only the active accepted-depth prefix is written and read.

## Offline verification

- Focused selector, integration, binary, and gate tests: 99 passed.
- Ruff, Python byte compilation, and `git diff --check`: pass.
- SM121a native compile: 64 registers/thread, 0-byte stack/local, 0 spills,
  7,592 bytes reported shared memory.
- Native SASS: no `LDL`, `STL`, or `CALL`; zero `MUFU.EX2`; 70 `FFMA`.
- No GPU query, CUDA launch, Docker mutation, synthetic/probe measurement, or
  performance claim was used for this repair.

The host-compiled object is evidence for source/resource shape only. A full
pinned-image extension rebuild and a fresh authenticated real SWE-Verified B1
all-depth byte gate are mandatory before B4 or timing.

The artifact excludes tasks, prompts, responses, patches, environment dumps,
raw logs, process/container identities, credentials, and timing samples.
