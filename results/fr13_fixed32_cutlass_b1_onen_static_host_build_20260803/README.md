# Fixed32 B1 one-N static scheduler host audit

This artifact records the host-only SM121a build and static audit for the
default-off `identity_onen_b1` CUTLASS candidate. The candidate specializes the
fixed32 K64 B1 swap-AB geometry: one scheduler-N tile, one batch plane,
cluster `(1,1,1)`, and five exact projection shapes.

The scheduler is stateless. The current `WorkTileInfo.M_idx` is the persistent
linear cursor, so the next tile is `M_idx + gridDim.y` and maps directly to
`{M_idx, 0, 0}`. This removes generic batch, cluster, swizzle, and raster
divmods without changing the incumbent two-stage schedule selection:
`N == 5120` remains cooperative and the other audited shapes remain ping-pong.

Both FP16 and BF16 candidate schedules compile at 168 registers with zero
stack, local memory, `LDL`, `STL`, or `CALL`. Versus the matching generic
two-stage scheduler, ping-pong falls from 968 to 744 SASS instructions and
cooperative falls from 864 to 568. Exact counts and pinned identities are in
`build_manifest.json`.

The qualification path is hard-pinned to the explicit `k64_root` profile.
The diagnostic runner and production launcher reject an omitted profile or
`full_vocab` before GPU or Docker work, and sidecar/direct-install verification
enforces the same contract centrally. The source commit must equal runtime
`HEAD`, the tracked worktree must be clean, and `git show <commit>:<path>` bytes
for the patcher, runner, credential, binary registry, and launcher must equal
the runtime files. That source identity is carried through the live result,
sidecar, and production attestation.

No GPU kernel, synthetic probe, SWE-Verified task, timing campaign, or
hardware-floor acceptance run was performed. The byte A/B selector remains
diagnostic and stock-serving; the direct selector requires a bound K64/root
production sidecar.
