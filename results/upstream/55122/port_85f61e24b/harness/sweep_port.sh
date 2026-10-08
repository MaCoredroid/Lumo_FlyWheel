#!/bin/bash
# Full matrix, PRISTINE build. Protocol (Codex corrections):
#  - unfiltered stdout+stderr preserved verbatim in matrix_raw.log
#  - per-cell process exit status recorded (CELL_EXIT lines)
#  - abort on device fault (3), unexpected host rejection (4), fatal (96/97),
#    timeout (124) or any nonzero status
#  - every GPU invocation under flock + bounded timeout
H=/home/mark/shared/p4prime-55122/port/harness
LOG=/home/mark/shared/p4prime-55122/port/logs/matrix_raw.log
LOCK=/home/mark/shared/exp54928/gpu.lock
SEED=20260917
: > "$LOG"
for N in 355584 355588 400000 474112 474116; do
  for R in 1 4; do
    for K in 512 1024 2048; do
      for P in random tie equal; do
        echo "### CELL n=$N rows=$R k=$K pat=$P" >> "$LOG"
        timeout 900 flock "$LOCK" timeout 600 "$H/p4_harness_port" run "$R" "$K" "$N" "$P" 6 "$SEED" >> "$LOG" 2>&1
        rc=$?
        echo "CELL_EXIT n=$N rows=$R k=$K pat=$P rc=$rc" >> "$LOG"
        if [ "$rc" -ne 0 ]; then
          echo "ABORT: cell n=$N rows=$R k=$K pat=$P exited $rc" >> "$LOG"
          exit "$rc"
        fi
      done
    done
  done
done
echo "SWEEP_COMPLETE" >> "$LOG"
