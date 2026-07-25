#!/usr/bin/env bash
# R4 gate leg-i (LOCAL): lean stack + FR13_DRAFTER_GRAPH=1, graph serve,
# local probes. Waits for the GPU to free (tree_b1 teardown), then boots.
# Checks: capture needle fires ("[FR13_DRAFTER_GRAPH] captured"), no crash,
# tok/draft=21 engagement, accept in probe band (draft corruption craters
# accept instantly — the sharp correctness signal for a replayed drafter).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30
env FR13_PARENT_GATHER=1 FR13_CONV_PREGATHER=1 FR13_FLAGS_INKERNEL=1 \
  FR13_HC_INTERNAL=0 FR13_SUBTREE_PARALLEL=1 FR13_SUBTREE_PARALLEL_SELFCHECK=0 \
  FR13_DRAFTER_GRAPH=1 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/r4_graph \
  GATE_CONTAINER=fr13-r4-graph \
  NEEDLE_PAT="FR13_DRAFTER_GRAPH] captured" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
echo "R4_GATE_DONE rc=$?"
