# Fixed32 SFWD lane-pair codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_GATES_REQUIRED**.

The default-off
`fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_u32x2_s20_lane2_v1`
candidate preserves the explicit split20 arithmetic and memory surfaces while
mapping two channel elements to each Triton thread. B1 retains C128 and uses
two warps; B2-B4 retain C256 and use four warps. CTA counts and logical global
bytes are unchanged from the split20 baseline.

Host-only SM121a codegen completed with CUDA visibility empty. B1 C128/W2 and
B4 C256/W4 both report 80 registers, 1,743 static and 1,760 encoded SASS
instructions, 75 `LDG`, 136 `STG`, and zero shared memory, barriers, spills,
local memory, stack, or calls.

After weighting each static instruction by launch CTAs and warps per CTA, the
candidate is 2.84% below the split20 schedule in static SASS and 3.51% below
in encoded SASS. Weighted `LDG` falls 1.32%; weighted `STG` is unchanged. These
are compiler-work proxies, not measured latency or throughput.

No Docker container, GPU kernel, task, request, timing run, or acceptance run
was launched. Real SWE-Verified byte correctness and isolated B1/B4 timing are
required before any performance, hardware-floor, or production claim.

This package excludes raw SASS, PTX, compiler IR, binaries, compiler logs,
task/model content, requests, responses, patches, environment values,
process/container identifiers, credentials, and secrets.
