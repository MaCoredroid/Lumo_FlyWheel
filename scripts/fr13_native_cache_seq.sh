# FR13 native+CACHE control arm (user 2026-07-07): the apples cache-on bar for
# cat8+cache. nativemtp5apc = native MTP-5 + NATIVE_ENABLE_APC=1 + MAMBA/APC block
# flags (1024/float32, same sizing as the tree cache => apples). NO tree/stateless
# flags (native path). CAVEAT: native+APC hit a stock-vLLM CUDA device-assert
# (gdn_linear_attn.py:986) after 6/16 tasks last run — may not complete 16; partial
# numbers are still apples vs cat8+cache on the OVERLAPPING completed tasks.
# GRAPH mode, clean speed config inherited from the driver (FR10_METRICS=0,
# BATCH_INVARIANT=0). Sourced by fr13_b4_campaign_driver.sh (run_variant in scope).
run_variant nativemtp5apc_${TAG}   nativemtp5apc   5  1
