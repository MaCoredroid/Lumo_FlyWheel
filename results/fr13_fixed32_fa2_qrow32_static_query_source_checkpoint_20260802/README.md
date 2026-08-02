# Fixed32 FA2 qrow32 static-query checkpoint

Status: **source candidate complete; compile and real-task measurement pending**.

The hidden B4 qrow32 translation unit now carries a `StaticQueryRows=32`
trait. In the split-KV forward kernel, that exact trait makes the one query
tile statically full and fixes its tile index at zero. This removes source-level
runtime work for the query CTA early exit, Q and O row-copy predicates, two LSE
row predicates, and tree-bias query-row mapping and bounds.

The specialization is private and fail-closed. Admission still requires B4,
`total_q=128`, `seqlen_q=32`, the exact query prefix
`[0, 32, 64, 96, 128]`, dense contiguous 32x32 tree bias with zero offsets,
BM32/N64, two warps, 64 threads, and 1024-row paged KV. Q/O base addresses
continue to use `cu_seqlens_q`; only each admitted sequence's extent and sole
tile index become constants.

KV behavior is deliberately not specialized. `Is_even_MN` remains false, the
last-block K/V copy remains predicated, the score mask remains
`apply_mask<Is_causal, Is_even_MN>`, and tree-bias K-column bounds remain
dynamic. The patch does not change score math, softmax/reduction order,
async-copy order, shared/global storage, kernel launch count, or launch grid.
Stock and qrow16 paths retain the dynamic query behavior.

Twenty focused source tests pass, including exhaustive qrow32 fragment-row
coverage and a transformation of the pinned FA2 header with exact anchor
counts and idempotence. Python compilation and `git diff --check` also pass.
No NVCC, C++, link, disassembly, GPU, byte gate, or timing was run. Fresh
SM121a compilation, SASS verification, resource admission, canonical exact4
Tail23/Hydra27 byte gates, and real SWE-Verified B4 timing remain pending after
the live B4 campaign releases the GPU.
