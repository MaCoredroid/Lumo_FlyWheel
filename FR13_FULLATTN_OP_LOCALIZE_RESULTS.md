# FR13 full_attention op-localization result

Date: 2026-06-07

Input artifact:
`output/fr13_fullattn_op_20260607T011853Z/tree3/logs/tree_attn_op_l3.pt`

Boot-free FA2 reference:
`scripts/fr13_fa2_tree_path_ref.py` feeds captured TREE_ATTN layer-3 tensors into
`vllm.vllm_flash_attn.flash_attn_varlen_func` FA2, without a server or model load.

Result:
- FA2 version: 2
- Context length: 18
- Spine rows: `[0, 1, 2, 4, 6]`
- TREE_ATTN vs FA2, all tree rows including branches: max_abs `0.0009765625`, mean_abs `1.5894572769070692e-08`, nonzero `1`
- TREE_ATTN vs FA2, spine: max_abs `0.0009765625`, mean_abs `3.1789145538141383e-08`, nonzero `1`
- Per row: rows `0,1,3,4,5,6,7,8,9` are exactly `0`; row `2` has one bf16-ULP difference.

Dense proxy check:
- FA2 spine vs dense fp32 replay: max_abs `0.007683992385864258`
- FA2 LSE vs dense logsumexp: max_abs `7.62939453125e-06`

Verdict:
The earlier dense `0.0077` is not TREE_ATTN-vs-FA2 divergence. On the same
captured q/k/v and the correct `prefix + ancestors + self` rows, TREE_ATTN is
FA2-equivalent to bf16 precision on spine and branches. The dense mismatch
appears after qk/softmax normalization, in `P@V`/output accumulation relative to
FA2/TREE_ATTN, so dense fp32 replay is not a valid FA2 proxy for this kernel.
