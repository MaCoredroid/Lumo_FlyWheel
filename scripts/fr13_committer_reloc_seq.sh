# COMMITTER RE-LOCALIZATION: whole-committer (FR13_COMMIT_FULL) + inner walk (FR13_MULTIDRAFT, baked) +
# GDN replay (FR13_REPLAY_GPU_TIMER wraps launch_tree_gdn_replay). Splits the ~80ms surrounding into
# replay(GDN state advance) vs publish-python(idx dict + accepted-path list comps + DtoH + globals).
# whole = walk(4) + replay(?) + publish-rest. The dominant sub-component is the phase-3 target. Short run.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/reloc_cf2.json
export FR13_REPLAY_GPU_TIMER=1
export FR13_REPLAY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/reloc_replay.json
run_variant tail6_reloc  tail6_mt  21  1
