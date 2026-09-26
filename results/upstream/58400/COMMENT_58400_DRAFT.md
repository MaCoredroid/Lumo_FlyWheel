Ran a model-free check of the state invariant behind this: after a padded one-token
tail rides the FULL graph, do the recurrent-state bytes the next step reads match the
unpadded eager run? Real `gather_batch_req_state` / `prepare_attn` (FULL and NONE) /
`GDNAttentionMetadataBuilder.build` / `postprocess_state` over synthetic conv+SSM
pools, exact byte comparison, plus a negative control that routes the tail through the
prefill write-back and must fail.

They match. Diffing the produced metadata against the merge base, the only fields that
change are `decode_graph_eligible`, the uniform token count, and the tail row's
`is_prefilling` — every state index and accepted count is identical, and the GDN spec
branch never reads `is_prefilling`.

Two questions rather than findings:

- The committed column is `num_accepted - 1`, and nothing forces `num_accepted == 1`
  for a padded tail. `rejection_sample_method="synthetic"` accepts by rate without
  looking at the `-1`s — out of contract?
- All of the above is `mamba_cache_mode="none"`. I found no test reaching
  `run_fused_postprocess_align`; is align covered somewhere I missed?

Branch, if useful: {{BRANCH_LINK}}

Used Claude Code to help with this analysis and test.
