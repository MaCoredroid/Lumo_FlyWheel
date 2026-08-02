# Fixed32 CFWD metadata uniform-writer source checkpoint

Status: source/static only. The candidate is default-off and has not been
compiled, run, byte-qualified, or timed.

The preceding metadata-fusion candidate correctly designated one writer
program per request, but expressed its 16 path elements as a statically
unrolled loop with masks on every load and store. Non-writer conv programs
could therefore retain all 32 predicated path memory operations plus the lens
store in emitted code even though their masks were false.

This checkpoint keeps the exact writer ownership
`pid_l == 0 && pid_c == 0`, adds a uniform scalar branch, and copies the 16
path elements with one `tl.arange(0, PATH_COLS)` vector. At source level:

- non-writer programs execute no metadata loads or stores inside the branch;
- the one writer program issues one vector path load, one vector path store,
  and one scalar length store;
- the existing accepted-length load is still shared with the conv-state body;
- the conv-state copy, exact one-shot lease, fallback, validation, recurrence,
  launch count, storage, and physical-32 domain are unchanged.

This specifically prevents the metadata transfer body from scaling with the
48-layer by channel-tile conv grid at source level. Triton code generation must
still prove that the uniform branch remains a branch and that non-writer
programs do not retain the unrolled memory operations.

Source commit: `c94943b4887eca8bd1ef1857e69a8d11ce21ac9a`.

Checks: 91 lifecycle/committer tests and 65 preseed/conv-wiring tests passed;
Python byte compilation, Ruff, and `git diff --check` passed. No GPU, Docker,
synthetic probe, real SWE-Verified task, or performance measurement was used.
