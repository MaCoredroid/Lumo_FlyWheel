# Fixed32 SFWD 8/16-row gate tile: rejected offline evaluation

This artifact records a rejected follow-on to the accepted default-off SFWD
4/8-row source at `1a86df82dbe6e704e472d2a770d3290917ca57e2`. The experiment changed the
gate-row multiplier from two to four, producing eight gate rows per B1 program
and sixteen per B4 program. The experimental source SHA-256 is
`e7a927bcd6c1da3f98403ee2cb23e55038f1cb4a28fdf906e8208cf65d45fcf9`.
The source experiment was restored after evaluation and is not proposed for
merge.

All 34 focused CPU tests passed, including exact-once coverage of every
fixed32 row-by-head output, and the generated kernel matched its generator.
Two independent cold-cache SM121a builds were byte-identical. Both profiles
had zero stack, local, shared, LDL, STL, and CALL usage.

The experiment is rejected because both profiles cross from 56 to 64
registers. Relative to the 4/8-row source, encoded SASS grows by 384
instructions for each profile, and static SASS grows by 374 for B1 and 379 for
B4. In exchange, total programs across 48 layers fall by only 192 for B1 and
384 for B4. Exact source-address requested gate bytes fall by only 55,296 and
110,592 respectively. Those byte counts are not measured DRAM or HBM traffic.

This is an offline rejection record. No GPU API, device correctness run, real
SWE-Verified task, or timing path was used, and the artifact makes no runtime
performance claim.
