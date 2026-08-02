# Fixed32 B4 M128 static-scheduler host build

Status: pinned host compile/link and static binary audit pass. The candidate is
still default off and is not acceptance-valid.

## Result

`persistent_b4_m128_static` compiled for `sm_121a` and linked into the complete
`_C_stable_libtorch` extension from the pinned vLLM and CUTLASS sources. This
closes the source-construction and host-build gates for the scheduler-only M128
candidate.

The build was isolated from the active B4 campaign. It did not launch a GPU
kernel, import the extension, use Docker, or run a synthetic/probe workload.
The 113 MB linked binary, raw build logs, raw symbol lists, cubins, and raw SASS
are not published; their paths and cryptographic identities are reduced into
this artifact.

## Kernel audit

The BF16 and FP16 candidate kernels each have:

- 168 registers, zero stack, zero local memory, and zero detected spills;
- 1,024 bytes static shared memory;
- 384 threads per CTA, or 12 warps;
- 1,792 bytes of kernel parameters and 2,688 bytes in `CONSTANT[0]`.

The incumbent dynamic-persistent M128 kernels have the same registers, stack,
local memory, shared memory, and CTA shape. Their parameter and `CONSTANT[0]`
sizes are 1,664 and 2,560 bytes respectively. The static scheduler therefore
adds only 128 bytes to the parameter/constant bank in the reported resource
tuple.

The incumbent linked binary contains 307 CUDA resource records. The candidate
contains those same 307 records without a removal or changed tuple, plus the
two BF16/FP16 static-M128 kernels. The incumbent M128 SASS embedded in the new
binary is byte-identical to the incumbent comparison binary.

For both output dtypes, static scheduling reduces the candidate kernel from
1,696 to 1,440 SASS instructions and from 27,136 to 23,040 text bytes: 256
instructions and 4,096 bytes fewer, or 15.094% by either count. Both forms
retain the same audited projection instruction counts: 128
`QMMA.16832.F32.E4M3.E4M3`, 128 `FFMA`, 72 `FMUL`, 48
`LDSM.16.M88.4`, 16 `STSM.16.M88.2`, and 32 dtype-specific output packs.
Neither form contains `LDL`, `STL`, `LD.LOCAL`, or `ST.LOCAL`.

This static-code reduction is compile evidence only. It does not establish a
latency, throughput, occupancy, or hardware-floor improvement.

## Link and ABI audit

The candidate has an additive dynamic-symbol delta:

- 1,297 normalized defined symbols versus 1,295 in the incumbent;
- exactly two added weak static-M128 `get_grid_shape` definitions;
- no removed definitions;
- the same 182 undefined symbols, with no additions or removals;
- the same nine `DT_NEEDED` entries and the same normalized `RUNPATH`;
- the same 17 embedded cubins: 16 `sm_121a` and one `sm_89`.

The normalized candidate is 311,496 bytes larger than the incumbent linked
extension, a 0.276% file-size increase. No runtime import was attempted, so
this is an additive static ABI classification rather than a runtime load or
compatibility claim.

## Scope boundary

The build does not establish raw-byte equivalence. Source construction, equal
math-instruction counts, and an unchanged per-output K order are insufficient
to replace a real-task output gate.

The next valid gate is authenticated real SWE-Verified exact4 B4 raw-byte
comparison for both Tail23 and Hydra27 at the fixed K64/root1, physical
root-plus-31 configuration. Full-step timing is permitted only after both byte
gates pass. Until then there is no B4 TPS or hardware-floor claim.
