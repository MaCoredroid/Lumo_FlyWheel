# Fixed32 FA2 qrow32 B1 split2 scratch v4

Status: **offline build and ABI audit pass; real SWE-Verified byte parity and
timing remain pending**.

Gate A attempt 7 proved that the exact private qlen-32 split2 call reached the
C++ launcher without split scratch. Stock FA2 invokes `set_params_splitkv`
only for its qlen-1 group-swapped decoding route, so `num_splits` and both
accumulation pointers remained zero/null for the physical32 call.

V4 reuses the stock allocator and tensor-lifetime path only when the exact
private sentinel is present, and it requires the caller's split count to equal
2 before allocation. Ordinary FA2 calls retain the stock predicate. The CUDA
attention and combine objects are byte-identical to v3; only `flash_api.cpp`
was recompiled and the library relinked.

The pinned SM121a library is
`ec36c5d26635fead8f626539ff98ab055a756af1e568dbadf88905a41f61862a`
(300,153,584 bytes). Its canonical source closure is
`3c559d80c65573932c5c7bfd5ef7081df6c3f1a3f6c888bc36a04ccc264d394b`.
The binary is stored locally with its build files and is not committed to Git.

The build used the pinned vLLM image with network disabled and no GPU device.
Dynamic defined symbols, undefined symbols, and dependency/runpath entries
match the qrow16 reference. Both private launchers remain local. The library
loads and registers `varlen_fwd_tree_bias` in the pinned image without a GPU.

No byte-parity, timing, TPS, acceptance, or hardware-floor claim is made here.
Admission still requires the real B1 SWE-Verified Gate A comparator, followed
by the production smoke and canonical exact4 full-step timing chain.
