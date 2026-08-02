# Fixed32 CFWD metadata-copy fusion source checkpoint

Status: source/static only. The candidate is default-off and is not qualified
for serving or acceptance measurement.

## Change

The armed direct-conv commit kernel also writes the existing B-specific
committer `accepted_paths` and `accepted_lens` buffers. One program per request
(`pid_l == 0 && pid_c == 0`) owns the metadata transfer. The source path loads
and destination stores are masked to that writer.

An exact pointer/batch/shape/stride/dtype/device/CUDA-stream lease connects that
conv enqueue to one immediately following committer replay. A matching lease
skips the two redundant high-level `copy_` calls and the duplicate committer
metadata validation chain. An absent lease falls back to the incumbent copies
and guards. A present-but-mismatched, malformed, or stale lease remains latched
and fails closed, preventing a cross-stream writer/copy race.

## Static invariants

- The default direct-conv kernel is source-identical to parent `59bc4332`.
- The armed kernel's conv-state body is source-identical to the default body.
- The ordered CFWD recurrence is unchanged; its established normalized source
  hash remains `d16ad65fe4affb85a85051bf8dc7530c17a34dd85826c05d6bd8adec67b1ce22`.
- No CUDA launch, GPU allocation, recurrence dependency, or physical-node
  ownership is added.
- The added work is 16 masked int32 source loads and 17 masked int32 stores per
  request inside the already-required conv commit launch. The accepted length
  load is reused from the incumbent conv body.
- Physical node domain 32 and maximum 12 recurrence steps are unchanged.

## Qualification boundary

No Triton/CUDA compile, code generation, SASS/resource inspection, Docker/GPU
execution, byte gate, real SWE-Verified timing, hardware-floor measurement, or
synthetic probe was run for this checkpoint. Those gates remain pending until
the active B1 work is fully torn down.

Source commit: `bbfe2bfa6d30c1e3dfe48e6e00f68f793d28dd93`.
