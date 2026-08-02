# Fixed32 B4 stock-shape Stage2 CUTLASS host build

This reduced artifact records the compile, link, import, resource, and focused
SASS audit for `identity_stockshape_stage2_b4`. The candidate preserves the
stock 64x128x128 tile and ping-pong schedule, uses the identity epilogue, and
fixes the mainloop depth at two stages. The direct selector remains blocked;
only the stock-serving byte diagnostic is installable.

Both BF16 and FP16 Stage2 kernels use 168 registers per thread, zero stack and
local memory, and 1024 bytes of static shared memory. The focused SASS audit
found no local load/store instructions. Import and the pinned binary verifier
passed.

No GPU kernel was run for this artifact. It makes no byte, task, timing, or
hardware-floor claim. The required next gate is the real SWE-Verified exact4
B4 Hydra27 K64/root byte comparison, followed by Tail23 if Hydra passes.
