# Fixed32 SFWD channel-serial codegen readiness

Status: **OFFLINE_CODEGEN_PASS_REAL_BYTE_GATE_REQUIRED**.

The unbound `fixed32_sfwd_channel_serial_r32_c64_w2_v1` prototype assigns one
channel to each lane across two warps, issues an explicit coalesced C64 load
for every physical row, and retains the fixed32 rows in registers. Direct
fixed-topology tuples replace all local gathers and layout conversions.

Offline SM121a code generation removes the shared-memory path completely:
`BAR=0`, `LDS=0`, `STS=0`, and launch/ELF shared allocation are both zero. The
kernel reports 64 registers with no spills, local memory, stack, or calls.
B1 and B4 generate the same binary, and a second fresh-cache build reproduces
all binary and text hashes.

| Metric | Tap-mask W16 | Channel-serial W2 | Delta |
|---|---:|---:|---:|
| Threads/CTA | 512 | 64 | -448 |
| Reported / allocated registers/thread | 49 / 56 | 64 / 64 | +15 / +8 |
| Allocated registers/CTA | 28,672 | 4,096 | -24,576 |
| Static / encoded SASS | 371 / 392 | 896 / 912 | +525 / +520 |
| Warp-weighted static / encoded | 5,936 / 6,272 | 1,792 / 1,824 | -4,144 / -4,448 |
| LDG / warp-weighted LDG | 19 / 304 | 36 / 72 | +17 / -232 |
| STG / warp-weighted STG | 12 / 192 | 68 / 136 | +56 / -56 |
| LDS / STS / BAR | 6 / 6 / 3 | 0 / 0 / 0 | -6 / -6 / -3 |
| Launch shared bytes | 4,096 | 0 | -4,096 |

The higher static code size and exact 64-register boundary remain live risks;
only a real byte gate and real SWE-Verified timing can determine whether the
lower warp work and absent synchronization improve full-step throughput.

No GPU, Docker, service, task, timing, or acceptance run was used. This reduced
package excludes binaries, compiler IR, raw SASS/PTX, logs, task or model
content, requests, responses, environment values, credentials, process
identifiers, and secrets.
