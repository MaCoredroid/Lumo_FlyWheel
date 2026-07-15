# accept>5 GATE 5 (16-task fr9-matched B=4): 3-way A/B on subset_b4_sixteen.
#   t33333   = MTP-only cat33333 (baseline, 15 nodes)
#   tail6    = MTP-head + arctic-tail (21 nodes, n_pad=32) -- the deliverable
#   suffonly = arctic-only cat33333 (Front-2 control, MTP deep-drafter disabled)
run_variant t33333_${TAG}   t33333   15 1
run_variant tail6_${TAG}    tail6    21 1
run_variant suffonly_${TAG} suffonly 15 1
