# Fixed32 SFWD wider gate-row tile: offline SM121a audit

This artifact binds the default-off SFWD conv/post-prep source at
`1a86df82dbe6e704e472d2a770d3290917ca57e2` against the first gate-pack
revision at `0bf56d9d4d024129c2ff485c1802546dd518da30` and the fusion baseline at
`e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea`. It targets fixed32, K64 B1/B4
specializations and does not use a GPU API.

The candidate packs four gate rows per B1 program and eight per B4 program.
Relative to the first gate pack, dynamic programs across 48 layers fall from
4,608 to 4,224 for B1 and from 9,216 to 8,448 for the whole B4 batch. Kernel
launches remain 48. Exact source-address requested gate bytes fall from
1,105,920 to 995,328 for B1 and from 3,981,312 to 3,760,128 for B4. These
counts are not measured DRAM or HBM traffic.

Offline SM121a codegen keeps both profiles at 56 registers and zero stack,
local, shared, LDL, STL, and CALL. Relative to the first gate pack, encoded
SASS grows by 200 instructions for both profiles; static SASS grows by 194 for
B1 and 197 for B4. Static LDG and STG each grow by four. Two independent
cold-cache builds produced byte-identical summaries.

The source retains the same row-by-head Cartesian mask, loads, FP32 softplus
and gating algebra, and stores. CPU tests enumerate every fixed32 output and
prove that the wider B1/B4 tiles cover it exactly once. This static and CPU
evidence does not establish device byte equality, runtime correctness, or
speed.

The next required gate is lossless byte equality on real SWE-Verified B1 and
B4 tasks, followed by full-step timing on the same workload set.
