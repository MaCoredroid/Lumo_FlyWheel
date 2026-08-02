# Fixed32 FA2 qrow32 direct page-coordinate checkpoint

Status: **source candidate complete; SM121a instruction verification pending**.

The hidden qrow32 path now resolves the paged-KV coordinate directly from its
fixed geometry:

```text
virtual_page_idx = n_block >> 4
page_offset = ((n_block & 15) << 6) + block_row_offset
```

This is stronger than the preceding constant-divisor source candidate. That
version first constructed the 64-bit value `64 * n_block + block_row_offset`,
then shifted and masked it. The direct form keeps page-coordinate work in
32-bit block coordinates and widens only for the required page/row address
products. It therefore exposes a real additional opportunity to remove 64-bit
integer instructions, although no compiled instruction-count claim is made.

The equivalence proof is exhaustive over all 64 qrow32 threads, all 16 block
residues within a 1024-row page, every valid final partial-block clamp from
zero through 64 rows, and quotient representatives through the largest
nonnegative 32-bit `n_block`. Compile-time assertions bind block N=64, page
size=1024, two-warps/64-threads, eight copy threads per row, and eight rows per
thread. Stock and qrow16 retain the original dynamic divmod path.

Nineteen focused source tests pass, as do Python compilation and
`git diff --check`. No NVCC, C++, link, disassembly, GPU, byte gate, or timing
was run. CALL0, instruction/resource admission, pinned GCC11.4 ABI equality,
canonical real SWE-Verified exact4 Tail23/Hydra27 bytes, and real-task timing
remain pending after live B4 teardown.
