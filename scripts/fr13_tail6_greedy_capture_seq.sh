# PHASE-1 RIGOR: capture tail6 GREEDY committer paths (temp-0, FR10_METRICS=1) so the real-trace gate can
# test point-mass(top-down) == greedy(max-LCP) on the RIGHT geometry (tail6 21-node, not cat9 9-node).
# Settles whether the live 4.18-vs-4.81 gap is a via=1 bug / trajectory divergence, or a real top-down!=maxLCP
# difference on tail6's branched head + deep tail.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR10_METRICS=1
export LUMO_TREE_PATH_LCP_LOG=/workspace/output/fr13_tail6_gcap/tree_path_lcp_max.jsonl
run_variant tail6_gcap  tail6  21  1
