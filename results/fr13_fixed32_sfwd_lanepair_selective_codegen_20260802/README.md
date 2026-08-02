# Fixed32 SFWD lane-pair selective reload codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_GATES_REQUIRED**.

The source-bound
`fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_u32x2_firstuse_r21_v1`
candidate moves each current-row load to first use and reloads only the long
row-21-to-row-4 dependency. That extra row-4 load shortens register lifetimes
without changing the fixed32 topology or arithmetic order.

Offline SM121a codegen reports 48 registers/thread for both B1 C128/W2 and B4
C256/W4, with no stack, local memory, spill, shared memory, barriers, or calls.
The previous lane-pair candidate used 80 registers/thread. The selective
reload adds two `LDG` and modest static code relative to that candidate, but
retains its one-CTA-per-channel-tile launch geometry.

Against split20, the candidate uses the same 48 registers and half as many
warps per CTA. Its warp-weighted static-instruction proxy is 1.3378% higher at
B1 and 1.5642% higher at B4; the encoded-instruction proxy is 0.4386% higher,
the load proxy is 1.3158% higher, and the store proxy is equal. Those are
compiler proxies, not latency claims.

The kernel remains default-off. A strict real SWE-Verified B1 byte gate and
then isolated B1/B4 timing are required before performance, hardware-floor,
or production claims.

No raw SASS, PTX, compiler IR, binary, task/model content, request, response,
environment value, process/container identifier, credential, or secret is
included.
