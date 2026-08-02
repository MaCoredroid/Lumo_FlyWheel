# Fixed32 B4 two-M Stage2 minimal-state source checkpoint

This source-only revision removes the two-M scheduler's cached grid-size and
problem-tile fields. The grid stride is read from launch geometry and the bound
is read from the inherited CUTLASS scheduler parameters, leaving one 32-bit
candidate-owned work-index field instead of three.

The candidate mainloop is now explicitly fixed at two stages, matching the
stock-scheduler Stage2 control. A later byte and timing comparison therefore
isolates the direct `M=linear&1`, `N=linear>>1` mapping rather than mixing the
scheduler change with a different pipeline depth. This checkpoint has no
compile, GPU, byte-equality, timing, or hardware-floor claim.
