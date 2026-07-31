# Fixed32 GDN one-launch feasibility

Verdict: do not implement either one-launch topology before the two-launch
batch candidate is measured on the standing real SWE-Verified exact4 set.
Terminal-prefix recompute is exact-capable and is the only plausible follow-up,
but the source arithmetic does not establish a likely wall-time win. The
monolithic shared-prefix CTA is a likely regression.

This is a source/math result only. No GPU was acquired.

## Fixed dimensions

The deployed fixed32 scan uses:

```text
nodes=32, KH=16, VH=48, DK=128, DV=128, BV=8, warps=8
base CTAs = VH * ceil(DV/BV) = 48 * 16 = 768 per path/request
```

Inputs q/k/v are bf16. Per logical recurrent node, the current CTA tiling issues:

```text
q = 48 * 16 * 128 * 2 = 196,608 B = 192 KiB
k = 48 * 16 * 128 * 2 = 196,608 B = 192 KiB
v = 48 * 128 * 2      =  12,288 B =  12 KiB
total                            = 405,504 B = 396 KiB
```

These are issued load bytes before cache effects. Unique logical q/k/v bytes
are much smaller, so they must not be presented as measured DRAM traffic.

## Exact work comparison

| Topology | Launches/layer | Paths/request | Recurrent nodes/request | CTA-node steps/request | Critical chain | q/k/v issued/request/layer | CTAs/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current two-level | 2 | 12 | 32 | 24,576 | 5+7=12 | 12.375 MiB | 9,216 |
| Terminal recompute | 1 | 11 | 62 | 47,616 | 12 | 23.9765625 MiB | 8,448 |
| Monolithic h-cache | 1 | 1 | 32 | 24,576 | 32 | 12.375 MiB | 768 |

Multiply the CTA counts by B for B1-B4. Thus the current B4 grids are
`3,072 + 33,792` CTAs, terminal recompute is `33,792`, and monolithic is
`3,072` long-lived CTAs.

## A. Terminal-path recompute

The root path is `(0,1,4,9,14)`. Prepending the root-to-parent prefix to each
of the eleven terminal paths produces lengths:

```text
(12, 6, 8, 3, 3, 4, 4, 5, 5, 6, 6), sum=62, max=12
```

The exact combined paths are:

```text
(0,1,4,9,14,19,24,26,28,29,30,31)
(0,2,7,12,17,22)
(0,3,8,13,18,23,25,27)
(0,1,5)
(0,1,6)
(0,1,4,10)
(0,1,4,11)
(0,1,4,9,15)
(0,1,4,9,16)
(0,1,4,9,14,20)
(0,1,4,9,14,21)
```

An exact owner map is straightforward: terminal path 0 owns stores for shared
prefix nodes `(0,1,4,9,14)`; every path suppresses stores for its recomputed
prefix and owns its unique suffix. Output and ring destinations are therefore
covered exactly once without cross-CTA communication. Every program starts
from its request's h0 and preserves root-to-leaf `_gdn_node_step` order.

Cost versus current:

- Recurrent work and q/k/v issued loads increase `62/32 = 1.9375x`.
- The increase is 23,040 CTA-node steps and 11.6015625 MiB q/k/v issued loads
  per layer/request. Across 48 GDN layers at physical B4, that is 4,423,680
  extra CTA-node steps and 2,227.5 MiB additional issued q/k/v loads.
- Critical dependency length remains 12, CTA count falls 8.33%, and one launch
  per layer is removed.
- It removes the current fp32 handoff: five 3 MiB parent-state stores plus
  eleven 3 MiB parent-state loads, or 48 MiB issued per layer/request. Across
  48 layers at B4 that is 9 GiB of issued handoff traffic.
- The handoff is an immediate producer/consumer over only 15 MiB of unique
  state, so a large fraction can hit cache. Counting all 48 MiB as DRAM would
  overstate the benefit. Conversely, the extra recurrent reductions, outer
  products, normalization, and transcendental work are real even when q/k/v
  hit cache.

Register state remains one `[BV,DK]` fp32 tile: 4 KiB/CTA, four fp32 values per
thread before other live values. The principal unresolved risk is performance,
not capacity. Exactness still requires a byte gate because the combined
`MAX_PATH_LEN=12` specialization and owner predicates can change Triton
instruction scheduling versus the current length-5/7 kernels, even though the
mathematical operation order is the same.

## B. Monolithic shared-prefix CTA

The existing monolithic form computes each of the 32 nodes once and avoids the
handoff scratch. It loses all subtree parallelism: the per-CTA critical chain
rises from 12 to 32 (`2.667x`). Its fixed32 h-cache is:

```text
32 * 8 * 128 * 4 = 131,072 B = 128 KiB/CTA
```

At 256 threads that is 128 fp32 registers/thread for state alone, before q/k/v,
reductions, indices, and gates. The current path kernel carries only the 4 KiB
working tile. The 128 KiB cache also exceeds the documented 99 KiB shared-memory
capacity, so it cannot simply be moved there. An older pre-fixed32 B1 result
reported subtree parallelism at +4.7%; that result is stale as a magnitude but
supports the structural direction. Saving one launch does not justify the
32-step serialization and high register pressure.

## Decision gate

First run the committed two-launch batch candidate through production compile,
byte equality, CUDA graph replay, then the standing real exact4 campaign. On
that same real-task capture, measure path-kernel duration, L2 hit rate, and DRAM
bytes for the 15 MiB export scratch. Build terminal recompute only if those
measurements show that the second launch plus handoff is larger than the cost
of 30 extra recurrent nodes/request. Do not build the monolithic option.
