# DECISIVE deep-tail-degradation confirmation: depth-5 t33333 (the tail6 HEAD without the arctic tail,
# shipped lossless config) on the SAME 16-task set + same campaign as tail6_prewarm. Isolates the TAIL:
# t33333 = tail6 minus depths 6-11. If t33333 produces non-empty patches + resolves (like merged_cold 3/4)
# while tail6 is 13/15 empty -> the DEEP TAIL systematically degrades agentic coding (cost-gate CONFIRMED,
# not harness/temp-0.6 variance). If t33333 ALSO ~13/16 empty -> the empties are harness/task-set, NOT the
# tail -> my red-team was wrong + accept>5 may be recoverable. Also validates the depth-5 deliverable resolve.
export GPU_UTIL=0.78
unset FR13_PREWARM_TRIE
run_variant t33333_d5_${TAG} t33333 15 1
