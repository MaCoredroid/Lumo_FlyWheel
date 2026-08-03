# Fixed32 B4 two-M one-dimensional host audit

This artifact records a host-only SM121a build and static audit for the
default-off B4 two-M scheduler revision at `61263d0cf`. The exact B4 projection
allowlist has two scheduler-M tiles and at least forty scheduler-N tiles, so
the pinned CUTLASS heuristic always launches an X-only grid. The revision uses
`blockIdx.x` and `gridDim.x` directly instead of flattening unused Y/Z grid
coordinates.

Both FP16 and BF16 candidates compile at 168 registers with zero stack, local
memory, `LDL`, `STL`, or `CALL`. Against the source-bound predecessor object,
the candidate falls from 1,040 to 1,032 SASS instructions per dtype while
retaining 39 branches and the same 128 QMMA, 128 FFMA, 48 LDSM, and 16 STSM
instructions. The linked extension imports successfully.

This is not a runtime speed or correctness result. No GPU kernel, synthetic
probe, SWE-Verified task, timing campaign, or hardware-floor acceptance run
used this revision. It needs fresh Tail23 and Hydra27 exact4 raw-byte gates
before it can replace the qualified predecessor or enter paired B4 timing.
