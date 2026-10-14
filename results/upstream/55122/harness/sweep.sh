#!/bin/bash
cd /home/mark/shared/p4prime-55122/harness
LOCK=/home/mark/shared/exp54928/gpu.lock
SEED=20260911
for N in 355584 355588 400000 474112 474116; do
  for R in 1 4; do
    for K in 512 1024 2048; do
      for P in random tie equal; do
        timeout 600 flock $LOCK timeout 300 ./p4_harness run $R $K $N $P 6 $SEED 2>&1 \
          | grep -E '^DIAG|^RESULT|^SUMMARY|^STOP'
        rc=$?
        if [ $rc -eq 3 ]; then echo "ABORT: device fault"; exit 3; fi
      done
    done
  done
done
