# FR13_PIGGYBACK V2 MECHANICAL pair (post-V0/V1 gates ONLY — ship rule: never arm before
# V0(b)+V0(d)+V1 green; the cat9_pb kind ARMS FR13_PIGGYBACK via its XFLAGS/sidecar).
# Split out of fr13_ladder_combined_seq.sh arms 3/4 so the async pair could run un-armed.
#
# GATES (MECHANICAL ONLY — S1 landed but slot-C V2.5 not yet green => NO accept/quality claims):
#   - engage needles: 'FR13_PIGGYBACK cat9_pb drafter engaged' + '[FR13_PIGGYBACK] committer
#     GDN replay DROPPED' (both fail-loud absent), drafts/draft == 17, 0 fatal events.
#   - CFWD collapse: cat9pb committer span vs cat9f same-session (~99 -> ~16ms target; the
#     native-committer bake makes cat9f's replay ~70ms — the collapse is vs the DEPLOYED baseline).
#   - graph+eager rows labeled (label_graph_vs_eager_every_row).
# cat9f = base cat9 on the FORKED launcher, piggyback OFF (valid same-session baseline).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
# ALL-FLAGS-ON (user directive): pb + native-committer(baked default) + async on BOTH arms.
# PEEL ORDER if cat9pb breaks: (1) drop async (unset FR13_SERVE_BATCH_FLAGS), (2) drop
# native committer (FR13_COMMITTER_NATIVE=0), (3) pb itself (cat9f = the unarmed baseline).
# PEEL-1 override: FR13_PB_NO_ASYNC=1 drops async (dbg6: embedding OOB suspect =
# next_token_ids -1 placeholders under async at propose time; sync probe passes,
# agentic multi-request steps hit it)
if [[ "${FR13_PB_NO_ASYNC:-0}" == "1" ]]; then unset FR13_SERVE_BATCH_FLAGS; else export FR13_SERVE_BATCH_FLAGS="--async-scheduling"; fi
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_pbmech/cat9pb_cfwd.json
run_variant cat9pb_${TAG}  cat9_pb 17  1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_pbmech/cat9f_cfwd.json
run_variant cat9f_${TAG}   cat9f    9  1
