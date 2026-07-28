#!/usr/bin/env bash
# FR13 POST-REBOOT RESUME PROTOCOL (2026-07-24 driver-degradation stop-line).
# Run AFTER the box reboots. Steps are serial; STOP at any FAIL.
#
# Context: 70-day uptime + heavy churn degraded the NVIDIA driver
# (1477x NV_ERR_NO_MEMORY memdesc failures) -> model behavior loops on ALL
# stacks incl. morning-clean code (bv1-exact replay looped). All verify
# levers (CONV_PREGATHER, PARENT_GATHER, FLAGS_INKERNEL) reverted as
# PRECAUTION — regate each here.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

echo "== step 0: driver health =="
sudo dmesg -T | grep -c NV_ERR_NO_MEMORY && echo "(should be 0 or only pre-reboot stamps)"
nvidia-smi | head -3

echo "== step 1: bv1-EXACT replay (worktree @04b4abdc8) — cure confirm =="
echo "   bash output/fr13_msr/launch_bv1exact.sh"
echo "   PASS = all 4 tasks in normal turn ranges (12907<=20, 13033<=30,"
echo "   13236<=35, 13398<=55) + accept ~4.3-4.5. LOOPS = deeper problem, stop."

echo "== step 2: per-lever regates UNDER GRAPH CAPTURE, one at a time =="
echo "   2a: FR13_PARENT_GATHER=1 alone (4-task; loop-watch + accept-inflation legs)"
echo "   2b: FR13_CONV_PREGATHER=1 alone (+ fix the col0-row token hole FIRST"
echo "       if convicting evidence reappears: row-id-aware freshness token)"
echo "   2c: FR13_FLAGS_INKERNEL=1 alone"
echo "   Each clean -> re-bake per the 4-task rule."

echo "== step 3: clean-host bar confirm =="
echo "   16-task tail6 full stack + re-confirm native5@0.70 (bar 50.99)."
echo "   Honest stack speed today: f70-class 37.37; bar16's 40.33 needs the"
echo "   clean-host repeat before it counts."

echo "== step 4: resume the ladder =="
echo "   Remaining measured attacks: nodebank patcher-half (kernel half done),"
echo "   spec-page cap (piece 3), attention-rows, B2c fold, sampler,"
echo "   profiler-named residuals. All designs/statuses in"
echo "   project_fr13_verify_kernel_first_numbers.md + FR13_SPECPAGE_SURGERY_DESIGN.md."
