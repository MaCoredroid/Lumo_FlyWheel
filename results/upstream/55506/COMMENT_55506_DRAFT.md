@Karl0007 @izhuhaoran — you offered to add a kernel test for the K3 path if someone pointed at a harness. Here is one, model-free, run on a GB10: {{BRANCH_LINK}}

`tests/v1/worker/test_mamba_aligned_state_indices.py` drives `compute_aligned_state_indices` over synthetic per-request-slot tables whose block ids are unique per (slot, column): identity control, permuted mapping, `-1` sentinel, eager versus captured-and-replayed launch, and the binding order through the real `_ensure_align_ctx`. Eight cases pass on `a28e902`; the permuted case fails on the merge base and on `c99ba59` alone, so it discriminates.

Two findings:

1. Under `CUDAGraphMode.FULL`, `prepare_attn` passes `num_reqs_after_padding` while `idx_mapping` is `num_reqs` long, so the new load (mamba_utils.py:66) reads past the tensor for padding rows — compute-sanitizer reports an invalid 8-byte global read. Those rows previously resolved to block 0.
2. `initialize_from_forward_context` is first-writer-wins, and capture (cudagraph_utils.py:824) binds the gathered tables before any real batch reaches `preprocess_state`.

Take, adapt or ignore. AI-assisted; details in the branch.
