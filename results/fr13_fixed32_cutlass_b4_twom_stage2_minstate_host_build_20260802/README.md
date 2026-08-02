# Fixed32 B4 two-M Stage2 minimal-state host build

This reduced artifact records the CPU-only compile, link, import, resource,
and focused SASS audit for the `identity_twom_b4` source at `2d2314bff`.
The candidate uses the fixed two-M mapping and an explicit two-stage mainloop.

Both FP16 and BF16 candidate kernels use 168 registers per thread, zero stack
and local memory, and 1024 bytes of static shared memory. The focused SASS
audit found zero local loads and stores. This removes the prior candidate's
8-byte stack frame and its one `STL` plus three `LDL` instructions per dtype.

Against the stock-scheduler Stage2 control compiled into the same binary, the
candidate keeps the same 128 QMMA, 128 FFMA, 48 LDSM, and 16 STSM instructions
per dtype. Its total SASS instruction count is 1040 instead of 1688, and its
branch count is 39 instead of 89. These are static compiler facts, not a
runtime speed claim.

The two FP16/BF16 stock-scheduler Stage2 functions are SASS-identical to the
same functions in the prior pinned Stage2 binary. The complete extension hash
still changed because it now contains the revised two-M kernel, so the new
binary remains bound to a fresh real byte gate.

No GPU kernel or real task was run for this artifact. The binary, object, and
cubin remain unpublished host-build outputs. Real SWE-Verified B4 byte equality
must pass before any timing or hardware-floor claim.
