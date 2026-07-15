# GATE 0.5 on the DELIVERABLE (design §6b): does the harness-aware PRE-WARM lift the accept>5 tail's
# accept_per_event above the established cold baseline (tail6 g4c = 4.277, committed 5.277) on the full
# 16-task fr9-matched set? Pre-warm is monotone-lossless (ADD only) -> can only raise or hold; magnitude
# is the question. Diagnosis so far (4-task merged): match RATE flat ~43%, prewarm adds LOW-conf rejected
# candidates -> ~0. This is the rigorous 16-task tail confirmation. corpus = the leakage-free 132-seg
# (--exclude-substr built; dumps are ~all astropy=the test repo so a bigger leakage-free corpus is supply-
# limited). If ~4.28 -> honest cost-gate on the windfall; if it jumps -> run the clean cold A/B.
export FR13_PREWARM_TRIE=/home/mark/shared/lumoFlyWheel/output/fr13_prewarm/corpus_harness.jsonl
run_variant tail6_prewarm_${TAG} tail6 21 1
