#!/usr/bin/env bash
# Runs INSIDE the pinned vLLM container started by run_kernel_bench.sh. One fresh python process per method
# (isolated CUDA context, allocator peaks and fixed32 module flags), then the tree-size sweep, then aggregation.
set -uo pipefail
K=/workspace/scripts/v2exp/kernel
OUT=${KBENCH_OUT:-/out}
METHODS=${KBENCH_METHODS:-"lumo_fixed32 weaver_author_default weaver_aligned_local treewy_author_default naive_native_paths naive_torch_node"}
SWEEP=${KBENCH_SWEEP:-"lumo_generic authors naive"}
BENCH_ARGS=${KBENCH_ARGS:-}
SWEEP_ARGS=${KBENCH_SWEEP_ARGS:-}
mkdir -p "$OUT/methods" "$OUT/sweep" "$OUT/logs"
finish() {
  [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]] && chown -R "$HOST_UID:$HOST_GID" "$OUT" 2>/dev/null
  return 0
}
trap finish EXIT
{
  echo "utc=$(date -u +%FT%TZ) host=$(hostname)"
  python3 -c "import sys, torch, triton; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('triton', triton.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
  python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
  env | grep -E '^(FR13_|KBENCH_)' | sort
} > "$OUT/logs/versions.txt" 2>&1
nvidia-smi > "$OUT/logs/nvidia_smi_before.txt" 2>&1
echo "methods: $METHODS"; echo "sweep: $SWEEP"
RC=0
for m in $METHODS; do
  t0=$(date +%s)
  python3 "$K/kernel_bench.py" --method "$m" --out "$OUT/methods" $BENCH_ARGS > "$OUT/logs/$m.log" 2>&1
  rc=$?
  echo "$m rc=$rc elapsed_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/logs/rc.txt"
  [[ $rc -ne 0 ]] && RC=1
done
for f in $SWEEP; do
  t0=$(date +%s)
  python3 "$K/kernel_sweep.py" --family "$f" --out "$OUT/sweep" $SWEEP_ARGS > "$OUT/logs/sweep_$f.log" 2>&1
  rc=$?
  echo "sweep_$f rc=$rc elapsed_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/logs/rc.txt"
  [[ $rc -ne 0 ]] && RC=1
done
python3 "$K/aggregate.py" "$OUT" > "$OUT/logs/aggregate.log" 2>&1 || RC=1
nvidia-smi > "$OUT/logs/nvidia_smi_after.txt" 2>&1
echo "in_container rc=$RC" | tee -a "$OUT/logs/rc.txt"
exit $RC
