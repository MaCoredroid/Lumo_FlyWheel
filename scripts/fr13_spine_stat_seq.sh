# Spine-commit de-risk: measure P(spine-prefix accept) at temp 0.6 via the byte-neutral counter.
# Plain tail6 (deployed config). Counter -> /logs/fr13_spine_stats.json every 500 commits.
export GPU_UTIL=0.72
run_variant spine_stat tail6 21 1
