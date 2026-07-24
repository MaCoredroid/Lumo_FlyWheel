#!/usr/bin/env bash
# FR13 POST-REBOOT RE-GATE + NEW-LEVER QUEUE (2026-07-24, supersedes the
# step-2 sketch in post_reboot_protocol.sh). Serial; ONE lever per arm;
# every arm carries the MANDATORY legs:
#   - loop-watch: any task > 40 assistant turns = FAIL the arm
#   - accept-inflation check: accept above band (~4.3-4.5 on the 4-task set)
#     WITH degraded content = the degraded-accepts-deeper signature = FAIL
#   - report evals as "X pass, Y fail, X+Y finished"
# Gate discipline: same-boot in-process byte gates; graph-capture legs where
# the lever touches captured code (eager-only bit-identity does NOT cover
# deployment — the parent_gather lesson). Bake per the 4-task rule.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
echo "== QUEUE (blocked on bv1x cure-confirm PASS) =="

echo "-- 2a FR13_PARENT_GATHER regate UNDER GRAPH CAPTURE"
echo "   FR13_PARENT_GATHER=1 alone; 4-task; gate_live_graph pattern +"
echo "   loop-watch + accept-inflation. Clean -> bake candidate."

echo "-- 2b FR13_CONV_PREGATHER regate (row-id token NOW BUILT)"
echo "   Composite (req_ids, col0 page-ids) freshness token shipped"
echo "   (publish + trigger + consume); stage REFUSES without col0 publish."
echo "   FR13_CONV_PREGATHER=1 alone; verify [FR13_CPG_ROWID_TOKEN] misses"
echo "   needle stays ~0 AND staged-serve engagement is nonzero (vacuity"
echo "   audit) before trusting the arm. 4-task + legs."

echo "-- 2c FR13_FLAGS_INKERNEL first gate ever"
echo "   Offline byte gate (scan out bit-identity, flags values) -> 4-task."

echo "-- 2d FR13_HC_INTERNAL (NEW: scan h_cache -> internal-node rows only)"
echo "   (i) ENFORCE_EAGER=1 FR13_HC_INTERNAL=1 FR13_HC_INTERNAL_SELFCHECK=1"
echo "       boot: in-process byte A/B raises on any bit diff (out[:n])."
echo "   (ii) graph-capture leg: FR13_HC_INTERNAL=1 4-task + legs."
echo "   (iii) register/occupancy check: dump PTX/regcount, confirm the"
echo "         footprint drop is real (scout caveat: not guaranteed)."

echo "-- 2e FR13_CONV_NODEBANK (NEW: surgery piece 1+2)"
echo "   (i) offline route byte gate: python3 output/fr13_msr/gate_conv_nodebank_byte.py"
echo "       (CPU PASS 36/36 2026-07-24; re-run IN-CONTAINER on GPU)."
echo "   (ii) FR13_CONV_NODEBANK=1 4-task + legs; watch composition-change"
echo "        steps specifically (the ordinal-perm machinery is the new risk;"
echo "        multi-turn tasks exercise it at every turn boundary)."

echo "-- 2f FR13_SPEC_BLOCKS_CAP=12 (NEW: surgery piece 3; REQUIRES 2e baked)"
echo "   FR13_CONV_NODEBANK=1 FR13_SPEC_BLOCKS_CAP=12; boot-line pool"
echo "   capacity (expect ~+9 pages/request), cache hit-rate probe vs"
echo "   baseline (target closes 71%-vs-85% gap / prefill_frac toward 0.13),"
echo "   accept band, 4-task + legs."

echo "-- 2h FR13_CONV_WB_BATCHED (NEW B2c: committer host-gap attack)"
echo "   ONE batched conv writeback across requests replaces the per-b"
echo "   launch loop (B launches x 48 layers -> 1-2 x 48; pure data movement,"
echo "   disjoint dsts => byte-identical by construction). Composes with"
echo "   NODEBANK (bank + batched col0) and pool routes."
echo "   (i) in-container byte gate: per-b launches vs batched on same"
echo "       synthetic inputs, bit-compare all dst rows."
echo "   (ii) FR13_CONV_WB_BATCHED=1 4-task + legs; committer span delta"
echo "        (cfwd/event) is the win metric."

echo "-- 2g torchprof residual-naming arm (DIAGNOSTIC, no bake decision)"
echo "   bash output/fr13_msr/run_torchprof_070.sh ->"
echo "   reduce_torchprof_stacks.py names host-gap/index-soup/norms/sampler"
echo "   python sites for the NEXT build round."

echo "-- 3 clean-host bar confirm"
echo "   16-task tail6 full stack (whatever re-baked above) vs bar 50.99;"
echo "   bar16's 40.33 counts only after this repeat."
