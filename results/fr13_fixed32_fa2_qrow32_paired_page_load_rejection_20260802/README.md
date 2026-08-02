# Fixed32 FA2 qrow32 paired page-load review

Status: **source edit rejected due to register lifetime**.

For fixed qrow32 geometry, `virtual_page_idx = n_block >> 4`; it is therefore
CTA-uniform at every active paged-KV resolution site. The physical page ID and
row coordinate are also identical for K and V when they refer to the same
`n_block`.

The only adjacent K/V pair is initialization at
`flash_fwd_kernel.h:702-705`. The saved SM121a kernel already common-subexpression
eliminates this pair: the two source resolver calls generate one division
sequence and one `block_table` load at SASS `0x1bb0`. A source pair-return
helper would not remove another load and could inhibit the existing optimizer
result.

The other matching coordinates are pipeline-separated:

- masking-loop K prefetch for `n_block - 1` matches the first unmasked-loop V
  resolution after the loop decrement, but the saved page loads at `0x7db0`
  and `0xa230` are 584 instruction slots apart;
- steady-loop K prefetch for `n_block - 1` matches V at the next iteration,
  but retaining the page ID from `0xf060` across the `0x11b70 -> 0x9df0`
  back edge to `0xa230` spans about 757 instruction slots.

Those spans contain softmax, BF16 conversion, P-times-V MMA, loop control, and
asynchronous-copy scheduling. Sharing would require keeping at least one
32-bit physical page ID live, or two registers for a precomputed V address,
through that critical region. The saved baseline already reports 244
registers. Moving either resolver would instead alter the deliberate K
prefetch order. A CTA-shared page ID requires shared storage and a barrier;
a warp-lane load plus shuffle adds synchronization and still duplicates work
across the two warps.

No kernel source was changed. No compiler, linker, disassembler, GPU, byte
gate, or timing was run, and no raw SASS was published. The direct page-coordinate
candidate remains the current qrow32 source checkpoint.
