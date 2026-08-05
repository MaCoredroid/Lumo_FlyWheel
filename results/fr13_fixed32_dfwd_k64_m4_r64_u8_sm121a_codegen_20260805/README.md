# Fixed32 DFWD K64 B4 M4 R64-U8 static codegen

Status: **static SM121a codegen pass; default off and runtime unwired**. No GPU
runtime, Docker workload, SWE-Verified task, byte comparison, timing, or
hardware-floor acceptance run was performed.

The exact-B4 candidate evaluates four independent BF16 `[1,5120]` inputs in
one head launch. Each lane keeps four scalar FP32 accumulators. For every
ordered K position it loads the BF16 `[65536,5120]` head weight once, reuses it
across the four requests, and performs four explicit round-to-nearest FMAs.
Each result retains the incumbent width-16 `8+4+2+1` shuffle reduction order.

The fixed geometry is 1,024 CTAs, 1,024 threads per CTA, 64 vocabulary rows per
CTA, and 16 lanes per row. Across the four requests, one U8 loop body has 40
loads, 40 BF16 conversions, and 32 FFMA instructions. The static comparison to
four independent B1 R64-U8 loop bodies is 4,760 versus 7,680 modeled loop
instructions, a 38.02% reduction; loads fall from 64 to 40. This is a scheduler
and instruction-count model, not measured speed.

CUDA 13.0.88 SM121a codegen reports 56 registers/thread and zero stack, local,
shared, spills, barriers, calls, or atomics. Two empty-output compilations with
the fixed `--frandom-seed=fr13_bf16_k64_m4_r64_u8` produced the same cubin
SHA-256, `c7b2a75dab16fa6a5c4f0038753b9da0f1f3d06ae6a22437797796a4ae9fd26b`.

The next step is a linked build in the pinned deployed Torch/CUDA ABI, then a
default-off real SWE-Verified B4 shadow gate comparing all 65,536 BF16 logits
at root and MTP depths 1-4 while always serving the incumbent. Only a clean
exact4 credential may admit full-step timing; exact16 remains required for
acceptance.
