# FR13 native + OUR cache control arm (user 2026-07-07: "native+cache means native + OUR cache",
# NOT stock vLLM's native APC which crashes on GDN@B=4). The apples cache-on bar for cat8+cache:
# nativemtp5_exseed = native MTP-5 decode (FLASH_ATTN, native qwen3_5_mtp spec, NO tree/tree-attn)
# + OUR forked lossless deployment cache (FR13_ENABLE_APC=1 + FR13_APC_EXACT_SEED=1), run via the
# forked launcher so the patcher applies. Isolates the tree DECODE superset (cat8 vs native MTP-5),
# both with our cache. GRAPH mode; clean speed config from the driver (FR10_METRICS=0, BATCH_INVARIANT=0).
# NOTE: unknown if native-decode + our-cache clears B=4 (the initial_state[~has_initial_state] restore is
# native-metadata-driven; the forked patcher MAY keep it consistent where stock doesn't). Observe on run.
run_variant native_ourcache_${TAG}   nativemtp5_exseed   5  1
