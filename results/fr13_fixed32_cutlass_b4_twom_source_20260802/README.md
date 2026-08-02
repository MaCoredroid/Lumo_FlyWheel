# Fixed32 B4 two-M scheduler source checkpoint

This source-only checkpoint adds a B4 CUTLASS candidate for the fixed physical
shape `M=128` with a 64-row output tile. The scheduler maps each persistent
linear work index directly to the two M tiles and the N tile, avoiding the
generic batch, cluster, and raster-order division/modulo path on each tile.

The candidate keeps the stock 64x128x128 tile, identity epilogue, ping-pong
mainloop, divisor-balanced physical grid, complete output tiles, and no split K.
It is disabled by default. It has not been compiled, run on GPU, checked for
byte equality, or timed; those gates remain mandatory before any performance
claim.
