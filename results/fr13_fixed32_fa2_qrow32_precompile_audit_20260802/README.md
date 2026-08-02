# Fixed32 FA2 qrow32 precompile audit

Status: **source precompile audit passed; compile and runtime gates remain open**.

This audit regenerated the direct-page plus static-query qrow32 candidate from a
fresh `git archive` of pinned FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95`. The source patch changed the
expected files on its first pass and reported every file unchanged on its
second pass. The generated translation unit is byte-identical to the patcher's
checked-in constant.

No compiler-visible source blocker was found:

- Pinned FA2 uses C++17 and includes `flash_fwd_kernel.h` before the qrow32
  translation unit declares either explicit trait specialization.
- The `StaticPagedKVBlockSize` and `StaticQueryRows` specializations target the
  same exact BM32/N64/two-warp trait and precede the first kernel template use.
- The candidate supplies the pinned split-KV kernel's eight boolean template
  arguments in order: causal, local, alibi, even-MN, even-K, softcap, split,
  append-KV.
- The hidden launcher declaration and definition have the same C++ signature,
  namespace, and visibility. No C ABI is introduced.
- The generated filename matches pinned CMake's configure-time
  `csrc/flash_attn/src/flash_fwd_*.cu` source glob.
- The trait arithmetic is internally consistent: 64 threads, eight global-copy
  threads per row, eight K/V rows per thread, a 32-row MMA tile, and 80 KiB of
  dynamic shared memory.
- A 1024-row page contains exactly sixteen 64-row K blocks, so the specialized
  page quotient and remainder use the exact `n_block >> 4` and
  `(n_block & 15) << 6` mapping.

Two focused regression tests were added and pushed in commit `3b03a1460`.
The complete focused source set now reports 22 passing tests. No kernel-source
change was made because the audit found no concrete blocker.

This is not compile or runtime qualification. NVCC template instantiation,
the pinned GCC 11.4 link, hidden-symbol resolution, SASS/resource inspection,
canonical exact4 byte parity, and real SWE-Verified B4 timing remain required.
No compiler, linker, disassembler, container, GPU, synthetic probe, or timing
run was used for this artifact.
