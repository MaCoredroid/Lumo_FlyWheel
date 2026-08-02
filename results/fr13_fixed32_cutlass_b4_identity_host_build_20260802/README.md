# Fixed32 B4 identity CUTLASS host build

This reduced artifact records the host build and static audit for the paired
`identity_stockshape_b4` and `identity_divisor_b4` candidates. Both selectors
are carried by one pinned extension binary and remain diagnostic-only.

The divisor-balanced candidate preserves the stock-shape 64x128x128 tile and
ping-pong mainloop, but maps the five real K64 projection shapes to exact
logical grids: 80x2 CTAs for N=5120, 224 CTAs for N=14336, 256 CTAs for
N=16384, and 544 CTAs for N=34816. Static inspection found the same arithmetic
site counts as the dynamic-scheduler control while reducing BF16 SASS slots by
25.46% and branch sites by 49.47%.

This is not a performance or acceptance result. Required next gates are exact
Tail23 and Hydra27 raw-byte comparisons on the real SWE-Verified exact4 B4 set,
followed by paired full-step timing only if both byte comparisons pass.

