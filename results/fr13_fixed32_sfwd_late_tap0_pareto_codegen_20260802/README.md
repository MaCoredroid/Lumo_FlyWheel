# Fixed32 SFWD late tap-0 reload Pareto codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_GATES_REQUIRED**.

The default-off
`fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_u32x2_firstuse_tap0n17_24_v2`
candidate loads every current row at first use and reloads only the tap-0
ancestor for nodes 17 through 24. This preserves the fixed32 topology, the
ordered BF16 products, every BF16 rounding boundary, and the left-to-right FP32
accumulation order. It changes no output or state surface.

Fresh host-only SM121a codegen from source commit
`375fa3bfc4eb1fb6495e9aadddca5de3ab6339da` reports the same kernel resources
for B1 C128/W2 and B4 C256/W4: 44 registers/thread, 91 `LDG`, 1,920 static and
1,936 encoded SASS instructions, and zero stack, local, shared, spill, barrier,
or call resources.

The known threshold-12 reload schedule also reaches 44 registers, but costs 99
`LDG`, 1,965 static, and 1,976 encoded instructions. The promoted subset keeps
the register result while reducing those counts by 8.08%, 2.29%, and 2.02%,
respectively. The prior one-edge schedule remains smaller at 77 `LDG` and 1,818
static instructions, but uses 48 registers. This is therefore an offline
compiler-resource Pareto point, not a latency result.

Deletion and same-load swap checks did not find a zero-spill schedule below
eight late reloads that retained the 44-register ceiling. That search was
targeted, not an exhaustive proof over every possible kernel transformation.

No Docker container, GPU kernel, task, request, timing run, or acceptance run
was launched. A real SWE-Verified byte gate is required before timing, and the
standing exact4 or exact16 campaign is required for acceptance or hardware-floor
claims.

This package excludes raw SASS, PTX, compiler IR, binaries, compiler logs,
task/model content, requests, responses, patches, environment values,
process/container identifiers, credentials, and secrets.
