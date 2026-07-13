#!/usr/bin/env bash
# Controlled cat8/cat6/native A/B: greedy accept-bound (superset test) + temp-0.6 wall-TPS (the good metric).
# Sequential boots (1 GPU each), OFFLOAD_AGENT=0 (local probe), B via deploy default. Records results.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_accept_bound_exp
mkdir -p "$RUNROOT"
N=${N:-512}
run() {  # arm kind
  local arm=$1 kind=$2
  echo "===== ACCEPT/SPEED PROBE arm=$arm kind=$kind ====="
  ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=$N MAX_NUM_SEQS_OVR=${BSZ:-1} \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$arm" "$kind" subset_b4_four.json \
    > "$RUNROOT/$arm.log" 2>&1
  echo "[$arm] rc=$? ; containers after: $(docker ps -q | wc -l)"
}
run probe_cat8   cat8
run probe_cat6   cat6root
run probe_native nativemtp5_exseed
echo "===== EXPERIMENT DONE ====="
