#!/usr/bin/env bash
# Drafter-pass sweep (2026-10-02): T0 = deployed hydra27 (5 MTP passes); G = pre-registered
# suffix pass gate (3 passes on strong suffix match, else 5); A3 = same gate with a permissive
# predicate (~97% of steps at 3 passes; the 32-step runaway brake keeps 1 in 33 ungated).
# Tuning set = the 43-request corpus. One container at a time; strict idle check before each step.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
LOG=$ROOT/sweep.log
log(){ echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
A3ENV=$'FR14_SUFFIX_PASS_GATE_NGRAM=1\nFR14_SUFFIX_PASS_GATE_MIN_AGREE=0\nFR14_SUFFIX_PASS_GATE_MIN_HISTORY=1'
idle(){ for i in $(seq 1 360); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_all.sh|v2exp_pipeline|dist_sample.py|replay.py|run_swe_" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; }
log "queue start $(git -C /home/mark/shared/lumotree-v2exp-20260930 rev-parse --short HEAD)"
step "G1 gated (pre-registered)"   V2_ARM_KIND=G V2_TAG=-swG1 bash $V/run_tree_arm.sh sampled
step "T1 deployed hydra27"         V2_ARM_KIND=C V2_TAG=-swT1 bash $V/run_tree_arm.sh sampled
step "A1 always-3 (permissive)"    V2_ARM_KIND=G V2_TAG=-swA1 V2_EXTRA_ENV="$A3ENV" bash $V/run_tree_arm.sh sampled
step "capture (native mtp5, dumps on, 4 tasks)" bash $V/run_swe_capture.sh mtp5 $ROOT/capture/subset_capture4.json 1500
step "A2 always-3 (permissive)"    V2_ARM_KIND=G V2_TAG=-swA2 V2_EXTRA_ENV="$A3ENV" bash $V/run_tree_arm.sh sampled
step "G2 gated (pre-registered)"   V2_ARM_KIND=G V2_TAG=-swG2 bash $V/run_tree_arm.sh sampled
step "T2 deployed hydra27"         V2_ARM_KIND=C V2_TAG=-swT2 bash $V/run_tree_arm.sh sampled
step "M1 native mtp5 (drift control)" V2_TAG=-swM1 bash $V/run_native_arm.sh mtp5 sampled
log "queue complete"
