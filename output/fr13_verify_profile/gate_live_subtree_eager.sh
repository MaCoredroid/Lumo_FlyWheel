#!/usr/bin/env bash
# FR13_SUBTREE_PARALLEL eager byte gate (g2 leg i): selfcheck raises on any
# bit diff monolith-vs-path-route; derivation needle = preseed print.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
exec env FR13_HC_INTERNAL=0 FR13_PARENT_GATHER=0 \
  FR13_SUBTREE_PARALLEL=1 FR13_SUBTREE_PARALLEL_SELFCHECK=1 ENFORCE_EAGER=1 \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
