# Fixed32 FA2 qrow16 division-free SM121a checkpoint

Status: **static resource admission passed; runtime qualification remains open**.

The hidden, default-off exact-B1 qrow16 translation unit now specializes the
deployed 1024-row paged-KV geometry. It resolves each 64-row K block with
`n_block >> 4` and `(n_block & 15) << 6`, removing the compiler's four signed
64-bit division helper calls. A private exact kernel wrapper instantiates the
same ordered attention body and template flags without changing stock FA2 or
the qrow32 route.

Paired host-only CUDA 13.0 SM121a compilation against pinned FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95` gives:

- registers: 244 to 224;
- stack, local memory, spill loads, and spill stores: zero on both sides;
- SASS calls: 4 to 0;
- SASS instructions: 5,440 to 5,040, down 400 (7.353%);
- kernel text: 87,040 to 80,640 bytes;
- static shared memory: 1,024 bytes on both sides;
- launch-time dynamic shared memory: 73,728 bytes on both sides.

The ordered attention pipeline is unchanged at the opcode-count boundary:
512 BF16 HMMAs, 132 FFMAs, 264 FMULs, 336 `LDGSTS`, 288 `LDSM`, 76 global
loads, and 38 global stores on each side. The candidate removes five reciprocal
operations and associated control work with the division paths.

The candidate remains byte-unqualified, timing-ineligible, and unauthorized
for production. A loadable pinned shared object, real SWE-Verified B1 paged-KV
byte comparison, and clean real-task timing are still required. No GPU,
container, synthetic timing probe, real task, or raw runtime data was used for
this checkpoint.

