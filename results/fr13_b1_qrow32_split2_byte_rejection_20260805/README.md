# FR13 B1 qrow32 split2 byte rejection

Status: **rejected for lossless production**.

Gate A attempt 8 reached the private qrow32 split2 attention and combine
kernels after v4 repaired scratch allocation. The authenticated B1
SWE-Verified event then failed the all-layer raw-byte comparison:

- BF16 output: 3,104,943 differing bytes across 16 layers;
- FP32 LSE: 9,551 differing bytes across 16 layers;
- per-layer BF16 range: 188,158 to 198,163 of 393,216 bytes;
- per-layer FP32 LSE range: 430 to 700 of 3,072 bytes.

The source establishes a reduction-topology mismatch. The captured reference
uses `VLLM_BATCH_INVARIANT=0`, hence `num_splits=0` and one ordered online
softmax reduction. The candidate forces `num_splits=2`, writes two independent
FP32 O/LSE partials, and invokes FA2's split-K combine. That reassociates the
softmax and output reductions. It cannot satisfy a raw-byte contract against
the incumbent one-part reduction in general, even when both paths implement
the same real-valued attention formula.

No scratch-layout defect was found. V4 allocates stock-shaped
`[split,batch,head,q,d]` O scratch and `[split,batch,head,q]` LSE scratch. The
private kernel's offsets and stock combine strides use the corresponding
split-major layout. The observed result is therefore a correctness-policy
rejection, not evidence that the repaired scratch pointers are mis-sized.

The live JSON's reference provenance is also inaccurate. It labels sentinel
`1179791667` as `qrow16 incumbent exact geometry`, but v4 `flash_api.cpp`
declares and dispatches only qrow32 sentinels `1179791668` and `1179791669`.
The alleged qrow16 tag falls through to generic stock FA2. The captured graph
still returned the baseline output, so candidate output was not served, but a
future result must not claim a qrow16 dispatch without a source-visible host
gate that actually calls the qrow16 launcher.

The selector now fails closed before replay or production selection when the
reference and candidate have different normalized reduction partition counts.
This does not alter CUDA code, qualify another arm, or claim performance.
No timing, TPS, acceptance, hardware-floor, task-payload, tensor, or candidate
production claim is included in this artifact.

## Identity

- frozen run commit: `49b1ecd37fa8a4618c6cfd3946069bddf865cdeb`;
- candidate SO SHA-256: `ec36c5d26635fead8f626539ff98ab055a756af1e568dbadf88905a41f61862a`;
- source closure SHA-256: `3c559d80c65573932c5c7bfd5ef7081df6c3f1a3f6c888bc36a04ccc264d394b`;
- focused verification: 51 tests passed.

No identity or digest of an excluded raw artifact is published.
