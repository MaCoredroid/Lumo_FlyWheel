# Fixed32 SFWD weight-matrix codegen rejection

Status: **REJECTED_OFFLINE_NO_REAL_TASK_RUN**.

This experiment replaced the C128/W4 candidate's two aligned 32-bit weight
loads with a `[BLOCK_C, WIDTH]` BF16 tile followed by reshape/split operations.
The intent was to keep the weight bytes contiguous without integer unpacking.

Offline SM121a code generation rejected the idea: Triton's layout conversion
introduced one barrier, four shared loads, and two shared stores. Registers
rose from 60 to 64, global loads rose from 38 to 40, and cubin size rose from
63,496 to 66,360 bytes. B1 and B4 generated the same rejected binary.

No GPU, Docker, service, task, timing, throughput, or acceptance run was used.
The included patch applies to source commit
`d3143f9163b4094ea0ae4d8c34d523103bc4eb0b` and exists only to make the
negative experiment reproducible.

No raw compiler IR, SASS, PTX, binary, task/model content, request, response,
environment value, process/container identifier, credential, or secret is
included.
