# Experimental CFWD gate/norm overlap

The sources in this directory are unselected replacements for the qualified
native key-group CFWD kernel. They are based on commit
`be921b3dc21980915077d6ee02e15921603f5356` and are not referenced by the vLLM
patcher, the runtime selector, or a binary binding.

The experiment changes only the precompute schedule:

- Threads `0..steps-1` publish the clamped node table once.
- One CTA barrier publishes that table to both gate and normalization workers.
- Threads `0..steps*3-1` retain the incumbent gate arithmetic and write the
  recurrence scalars without a trailing CTA barrier.
- Other warps can enter the first K-normalization wave while those gate threads
  execute `softplus`, exponential, and sigmoid operations.
- The normalization barriers and the final precompute barrier publish the gate
  scalars before the recurrence loop consumes them.

`fr13_fixed32_cfwd_native_fullvalue_overlap.cu` isolates that overlap change.
`fr13_fixed32_cfwd_native_fullvalue_overlap_active_waves.cu` adds a second,
separable scheduling change: a CTA-uniform `ceil(steps / 4)` guard skips the two
normalization barriers in each inactive trailing wave. The fixed three-wave
loop bound remains available to the compiler, and `steps` comes from the
CTA-shared value published by the initial barrier, so every thread takes the
same guard. This combined variant must be measured separately so any effect is
not attributed to gate/norm overlap alone.

The canonical source and its frozen source SHA, resource contract, SASS
expectations, and binary binding remain unchanged. This experiment has no CUDA
compile, codegen, byte-correctness, real-task, or timing evidence. Selecting it
requires a pinned-toolchain compile followed by fresh resource/SASS checks and
the full all-depth byte gate before any identity or runtime wiring is changed.
