# Fixed32 SFWD gate-row packing: offline SM121a audit

This artifact binds the default-off gate-row packing source at
`0bf56d9d4d024129c2ff485c1802546dd518da30` against the current fusion
baseline at `e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea`. It targets the fixed32,
K64, B1/B4 deployment specializations and does not use a GPU API.

The candidate packs two gate rows per B1 program and four per B4 program.
Across 48 layers, dynamic programs fall from 5,376 to 4,608 for B1 and from
13,824 to 9,216 for the whole B4 batch. Kernel launches remain 48. Exact
source-address requested gate bytes fall from 1,327,104 to 1,105,920 for B1
and from 5,308,416 to 3,981,312 for B4. These byte counts are not measured
DRAM or HBM traffic.

Offline SM121a codegen keeps both profiles at 56 registers and zero stack,
local, shared, LDL, STL, and CALL. Relative to the baseline, encoded SASS grows
by 96 instructions for B1 and 104 for B4; static LDG and STG each grow by two.
This static gate does not establish runtime correctness or speed.

The next required gate is lossless byte equality on real SWE-Verified B1 and
B4 tasks, followed by full-step timing on the same workload set.
