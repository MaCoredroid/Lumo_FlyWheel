#!/bin/bash
# Round-5 B=4 sweep: 5 rounds on the same 16-instance Verified subset, all at
# concurrency=4 / temp=0.6 / top_p=0.95, each with the per-agent spec-step trace.
#   1. config D (full T1+T2+T3+T4 suffix)
#   2. config E  MTP num_speculative_tokens=1
#   3. config E  MTP num_speculative_tokens=2
#   4. config E  MTP num_speculative_tokens=3
#   5. config E  MTP num_speculative_tokens=6
# Each round: the runner relaunches vLLM into the config (--apply-config), runs
# 16 tasks at B=4 with per-task auto join/commit/push, then exits; next round.
#
# Round 1 (D) assumes vLLM is ALREADY relaunched into config D (pre-loaded for
# trace validation) and so does NOT re-apply -- pass --r1-apply to force it.
set -u
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null; export LUMO_SUDO_PASSWORD
SUB=docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json
PY=.venv/bin/python
COMMON="--suite swe --subset $SUB --concurrency 4 --temp 0.6 --agent-wall-s 1800 --eval-timeout-s 1800 --nsight off"
R1APPLY=""; [ "${1:-}" = "--r1-apply" ] && R1APPLY="--apply-config"
LOG=/tmp/round5_b4_sweep.log
say(){ echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "ROUND 1/5: config D (B=4)"
$PY scripts/run_codex_experiment.py --exp-tag q36a_D_b4   --config D        $R1APPLY $COMMON >> "$LOG" 2>&1
say "ROUND 2/5: config E MTP=1 (B=4)"
$PY scripts/run_codex_experiment.py --exp-tag q36a_E1_b4 --config E --mtp 1 --apply-config $COMMON >> "$LOG" 2>&1
say "ROUND 3/5: config E MTP=2 (B=4)"
$PY scripts/run_codex_experiment.py --exp-tag q36a_E2_b4 --config E --mtp 2 --apply-config $COMMON >> "$LOG" 2>&1
say "ROUND 4/5: config E MTP=3 (B=4)"
$PY scripts/run_codex_experiment.py --exp-tag q36a_E3_b4 --config E --mtp 3 --apply-config $COMMON >> "$LOG" 2>&1
say "ROUND 5/5: config E MTP=6 (B=4)"
$PY scripts/run_codex_experiment.py --exp-tag q36a_E6_b4 --config E --mtp 6 --apply-config $COMMON >> "$LOG" 2>&1
say "SWEEP COMPLETE (all 5 rounds)"
