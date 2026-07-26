#!/usr/bin/env bash
# COMMITTER_GRAPH combo re-gate (LOCAL): lean + R4 drafter graph + DVK gather-64k
# + COMMITTER_GRAPH. Probe accept band = dvkg64's own 2.802 floor (gather probe
# confound), NOT 3.13-3.26. Needles: CG ENGAGED + shim mode=gather + dg captured.
# Waits for dvkg64L (live arm) teardown.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "DVKG64L_DONE" output/fr13_msr/dvkg64L_console.log 2>/dev/null; do sleep 300; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30
env FR13_PARENT_GATHER=1 FR13_CONV_PREGATHER=1 FR13_FLAGS_INKERNEL=1 \
  FR13_HC_INTERNAL=0 FR13_SUBTREE_PARALLEL=1 FR13_SUBTREE_PARALLEL_SELFCHECK=0 \
  FR13_DRAFTER_GRAPH=1 FR13_DFWD_SPLIT_NEEDLE=1 \
  FR13_DRAFT_VOCAB_K=65536 \
  FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
  FR13_COMMITTER_GRAPH=1 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/cg_combo \
  GATE_CONTAINER=fr13-cg-combo \
  NEEDLE_PAT="FR13_COMMITTER_GRAPH ENGAGED" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
echo "CG_COMBO_GATE_DONE rc=$?"
