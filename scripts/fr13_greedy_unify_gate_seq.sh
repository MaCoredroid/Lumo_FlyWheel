# GREEDY-UNIFY BYTE GATE (phase-1 cleanup, task #41): exercise the temp-0/all_greedy tree committer on
# REAL B=4 trees with FR13_GREEDY_UNIFY_GATE=1. Per greedy commit step the injected routing DUAL-RUNS
# both committers on the SAME logits: device-greedy (point-mass multidraft, _lumo_tree_canonical_multidraft_
# sample all_greedy=True) vs the old greedy path-LCP (_lumo_tree_path_lcp_max_greedy_sample), and records
# byte mismatches to the gate JSON. Served output stays the OLD committer (unchanged) so this is a pure
# diagnostic. temp-0 is forced via DEPLOY_FORCE_TEMP=0.0 (the ONLY way to reach all_greedy). PASS =
# mismatch_steps == 0 over the whole run => the temp-0 unification is byte-lossless on real trees (settles
# the duplicate-sibling tie the offline gate scripts/fr13_greedy_pointmass_byte_gate.py deferred). This is a
# committer byte-gate (single arm, no cross-arm accept compare) so subset size is not a config-drift axis.
# tail6 = the deployed geometry. run_variant is sourced from fr13_b4_campaign_driver.sh.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_gu1  tail6  21  1
