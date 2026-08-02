# Fixed32 DFWD K64 M1 shuffle R32 source checkpoint

Status: source-only, default off, compile and real-task qualification pending.
This candidate was developed while another kernel build was running and uses no
GPU, Docker workload, synthetic probe, or task data.

The candidate doubles output rows per CTA from 16 to 32. Its launch changes
from 4,096 blocks of 256 threads to 2,048 blocks of 512 threads. Total threads
remain 1,048,576, but block scheduling work is halved. Each output row still
uses the same 16 K-partition lanes, 320 dependent scalar `__fmaf_rn`
operations per lane, width-16 `8+4+2+1` shuffle reduction, FP32 add order,
alpha-one/beta-positive-zero epilogue, and BF16 round-to-nearest conversion.

There is no cross-row arithmetic, shared memory, CTA barrier, atomic, split K,
or output fixup. Changing block ownership cannot change a row's expression.
The 512-thread block may reduce resident block count, so this source structure
does not establish a speedup; compile/resource audit and real SWE-Verified
timing are required.

Source commit: `fc5effa3fb5cc20b78829c2aac363d62ec2bfb37` on
`agent/fixed32-dfwd-k64-m1-shuffle-r32-20260802`.

Five focused source tests, Python compilation, Ruff, and diff checks pass. The
next gates are a pinned CUDA 13 `sm_121a` build with SASS/resource audit, then a
default-off B1 K64/root real SWE-Verified diagnostic. B1 is not acceptance;
formal timing remains the canonical exact4 B4 or exact16 campaign.
