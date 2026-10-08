# DRAFT comment for vllm-project/vllm#57908 — NOT POSTED

Thanks for this. I tried to reproduce the aliasing on CPU.

Your two new tests discriminate: both fail at the merge base (`0ff0477`) and pass at head.
But at the base both stop on the first assertion, the `get_num_blocks_to_allocate` budget
(1 vs 2, 2 vs 3), so they show the extra reservation rather than an aliasing that existed
without it.

Tracing the served path I could not reach the write. Both runners migrate the state out of the
aligned source column before the forward: `preprocess_mamba` plans `src_col != dst_col`
whenever the hit is block-aligned (`mamba_utils.py:1481`, `:1498`, `:1503`; the V2 kernel uses
the same rule at `:543`/`:607`). `src_col == dst_col` happens only for a sub-block hit — which
is exactly what `_has_partial_local_hit` already arms
(`single_type_kv_cache_manager.py:168-178`). Is there a configuration where the pre-copy does
not run?

Separately, seven existing `cpu_test`s that pass at the base fail at head, including
`test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot`, which asserts the aligned Mamba
snapshot is *not* a copy source.

Repro + logs: {{BRANCH_LINK}}

*Investigated with AI assistance; every number is from a CPU run I can share.*
