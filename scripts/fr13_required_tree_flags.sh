# FR13_REQUIRED_TREE_FLAGS — SINGLE SOURCE OF TRUTH for env flags that MUST be ON
# for any branching-tree serving config (cat9/cat8/cat6/tail6/chain5/...).
#
# Why this file exists (2026-07-22): FR13_ATTN_KV_REMAP and FR13_SLOT_REORDER were
# both proven fixes ("BAKED" per project memory: project_fr13_garble_attn_kv_remap_fix.md,
# project_fr13_accept_mdep_fix_costgate.md) but were only ever hardcoded into
# fr13_launch_locked.sh and a dozen narrow one-off diagnostic scripts. The actual
# B4 agentic SWE-bench campaign path (fr13_launch_forked_fa2_tree_server.sh) never
# got them, so every tail6/cat8/cat6 campaign run through it -- weeks of runs --
# booted without the fixes. This file exists so that never happens again: update
# the list HERE ONLY, and every consumer (launcher default + assertion gate)
# picks it up automatically. Do not hardcode a copy of this list anywhere else.
#
# Consumers:
#   - fr13_launch_forked_fa2_tree_server.sh: sources this to set defaults
#   - fr13_bigdenom_swe_serve_variant.sh:    sources this to build its fail-loud
#                                             NEEDS assertion (tree-kind arms only)
#   - fr13_launch_locked.sh:                 sources this instead of its own
#                                             hardcoded `export FR13_ATTN_KV_REMAP=1`
#
# Format: "KEY=VALUE" strings, same shape docker -e / bash NEEDS arrays already use.
# Both flags are no-ops on non-branching configs (a linear chain's accepted path
# is already contiguous / M-independent), so defaulting them ON is behavior-
# preserving for every tree-launcher caller, not just branching kinds.
FR13_REQUIRED_TREE_FLAGS=(
  "FR13_ATTN_KV_REMAP=1"   # branching-tree foreign-KV garble fix (cat9 15/15->0/15)
  "FR13_SLOT_REORDER=1"    # FA2 accept M-dependence fix (superset +0.166 live-confirmed)
)
