# FR13 native MTP-5 + NO cache + B=4 control arm (user 2026-07-08). KIND flash_ns5_nocache:
# forked launcher (=> KEEPS the s_per_fwd_gpu timer, apples to native+cache=nativemtp5_exseed) +
# clean FLASH_ATTN native MTP-5 (naive_mtp, no tree), CACHE OFF (no APC/EXACT_SEED). Tree env
# un-leaked (FR13_FA2_TREE_BIAS=0 etc). Isolates EXACTLY what our forked cache contributes:
#   native+cache (DONE) vs native+nocache (this) -> give-ups + derived_tps_gpu delta = the cache effect.
# Same test config as the rest of the matrix: qwen-code nudge-OFF temp 0.6 NO-wall subset_b4_sixteen
# B=4/CONC=4. Driver env: BSIZE=4 CONC=4 TAG=qc4 RUNROOT=output/fr13_qwencode_cachefirst.
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant native_nocache_qc4   flash_ns5_nocache   5  1
