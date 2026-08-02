# Fixed32 SFWD tail-prior pair experiment

Status: **OFFLINE_CODEGEN_REJECTED_NO_STATIC_IMPROVEMENT**.

Source commit `96406cdafab1118ff50d0b670a2a9c4eaca0a373` widens the
remaining scalar BF16 `prior_2` load to an aligned 32-bit load of
`prior_2/prior_3`, then recovers only `prior_2` by exact bitcast. The fixed
contiguous state contract makes the access aligned, and focused source tests
pass.

Matched fresh-cache SM121a B1/B4 builds showed no instruction or resource
improvement over parent `6254582fc8b2000f00bc3fa425e8d55df11b3216`:
55 registers, 391 static instructions, 408 encoded instructions, 19 LDG,
12 STG, 6 LDS, 6 STS, 3 barriers, and 4,096 launch-shared bytes in both cases.
There were no spills, local memory accesses, stack bytes, or calls.

The candidate widens 128 useful bytes per CTA to 256 bytes without removing
an instruction. At 160 CTAs per request that is 20,480 additional logical
bytes per layer, or 983,040 bytes across 48 layers. Hardware sector traffic
may remain unchanged, but there is no static evidence of a benefit.

The variant is rejected and was not bound to a runtime candidate. No GPU,
Docker, service, synthetic probe, real task, timing run, or acceptance run was
started for this experiment.

This reduced package excludes raw compiler output, binaries, PTX, SASS, IR,
logs, task/model/request/response content, environment values, credentials,
process identifiers, container identifiers, and secrets.
