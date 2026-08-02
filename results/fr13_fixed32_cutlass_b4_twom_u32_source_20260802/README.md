# Fixed32 B4 two-M u32 scheduler source checkpoint

This source-only revision narrows the two-M scheduler's hot persistent work
index, grid stride, advance, and bounds arithmetic from 64 bits to 32 bits. The
candidate is host-gated to `M=128` and `N<=65536`, so the 64x128 output tiling
contains at most 1,024 logical tiles. Larger or non-B4 shapes fall back to the
stock kernel.

The direct `M=linear&1`, `N=linear>>1` mapping, divisor-balanced physical grid,
complete-tile accumulation, identity epilogue, and no-split-K contract are
unchanged. The candidate remains disabled by default and still requires CUDA
compile, resource/SASS audit, GPU runtime, real byte equality, and real-task
timing before any performance claim.
