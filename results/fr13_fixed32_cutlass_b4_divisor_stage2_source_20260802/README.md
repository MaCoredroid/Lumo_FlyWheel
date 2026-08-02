# Fixed32 B4 divisor stage2 source checkpoint

This source-only checkpoint adds a B4 CUTLASS candidate that combines the
64x128x128 stock tile, identity epilogue, divisor-balanced complete-tile
scheduler, ping-pong mainloop, and an explicit two-stage mainloop pipeline.

The candidate does not use split K and does not change an output tile's FP32
accumulation order. It remains disabled by default and has not been compiled,
run on GPU, checked for byte equality, or timed. Its next decision point is a
host compile/resource audit after the active real SWE-Verified B4 gate releases
the GPU and disk lanes.

