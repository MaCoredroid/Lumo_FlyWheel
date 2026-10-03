#!/usr/bin/env bash
# Same-build SWE-bench Verified agent study: native vLLM (ar|mtpK) or SGLang (sglang-s*k*d*) arm.
# Server: serve_native.sh (same launcher as the replay benchmark). Agent/proxy/evaluator: the
# deployment harness (offload proxy on alienware, run_swe_bench_q36_a.py), same subset and budgets
# as the ten-task LumoTree deployment (Cqc10). Usage: run_swe_native_arm.sh <arm>
set -uo pipefail
ARM=${1:?arm}
ROOT=/home/mark/shared/lumotree-v2exp-runs
WT=/home/mark/shared/lumotree-v2exp-20260930
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
SUBSET=config/fr13_fixed32/subset_b4_sixteen_qc_remainder_10.json
RUN=$ROOT/swe/${ARM}-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
echo "run=$RUN arm=$ARM wt=$(git -C $WT rev-parse HEAD) repo=$(git -C $REPO rev-parse HEAD) subset_sha=$(sha256sum $REPO/$SUBSET | cut -c1-64)"
PYTHONPATH=$WT/src $REPO/.venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" || echo "recover_host_memory failed"
free -g | tee "$RUN/free_before_boot.txt"
awk '/^MemFree:/{exit ($2/1048576 < 85)}' /proc/meminfo || { echo "MemFree < 85 GiB"; exit 1; }
bash "$WT/scripts/v2exp/serve_native.sh" "$ARM" "$RUN" || exit 1
NAME=v2exp-$ARM
cleanup() {
  curl -s http://127.0.0.1:9950/metrics > "$RUN/metrics_at_exit.txt" 2>/dev/null
  docker logs "$NAME" > "$RUN/engine.log" 2>&1; docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  (cd "$REPO" && LUMO_OFFLOAD_PROXY_PORT=8023 bash scripts/swe_x86_helpers/offload_codex_proxy.sh stop alienware >> "$RUN/offload_teardown.log" 2>&1)
  echo "stopped $NAME and offload proxy"
}
trap cleanup EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:9950/health >/dev/null; do
  sleep 10
  docker inspect -f '{{.State.Running}}' "$NAME" | grep -q true || { echo "container exited during boot"; exit 1; }
  (( $(date +%s) - t0 > 1800 )) && { echo "boot timeout"; exit 1; }
done
echo "healthy after $(( $(date +%s) - t0 ))s"
cd "$REPO"
set -a; source scripts/fr13_canonical_env.sh; set +a
unset FR13_FIXED32_INGRESS_SECRET_FILE FR13_FIXED32_INGRESS_TASK_IDS
export FR13_CAMPAIGN_TASK_BUDGET_S=9000 AGENT_WALL_S=9000 FR13_PROXY_RAW_DUMPS=off LUMO_OFFLOAD_PROXY_PORT=8023
ssh -o BatchMode=yes -o ConnectTimeout=15 alienware "curl -fsS -m 6 http://100.103.10.122:9950/health >/dev/null && echo ok" | grep -q ok \
  || { echo "alienware cannot reach GB10 :9950"; exit 5; }
bash scripts/swe_x86_helpers/offload_codex_proxy.sh sync alienware > "$RUN/offload_sync.log" 2>&1 || { echo "offload sync failed"; cat "$RUN/offload_sync.log"; exit 5; }
bash scripts/swe_x86_helpers/offload_codex_proxy.sh start alienware 100.103.10.122 "$RUN" > "$RUN/offload_start.log" 2>&1 || { echo "offload start failed"; cat "$RUN/offload_start.log"; exit 5; }
cat "$RUN/offload_start.log"
curl -s http://127.0.0.1:9950/metrics > "$RUN/metrics_before_swe.txt"
S0=$(date +%s)
.venv/bin/python scripts/run_swe_bench_q36_a.py --subset "$SUBSET" --out-root "$RUN/swe_out" \
  --concurrency 1 --eval-timeout-s 1800 --agent-wall-s 9000 \
  --agent-host alienware --agent-endpoint http://127.0.0.1:8023/v1 --eval-host alienware \
  --model qwen3.8-27b-nvfp4-radixark --model-name "qwen3.8-27b-nvfp4-radixark::qwen-code-0.19.4::v2exp-$ARM" \
  > "$RUN/swe_orchestrator.log" 2>&1
SWERC=$?
curl -s http://127.0.0.1:9950/metrics > "$RUN/metrics_after_swe.txt"
echo "swe orchestrator rc=$SWERC wall=$(( $(date +%s) - S0 ))s"; tail -5 "$RUN/swe_orchestrator.log"
bash scripts/swe_x86_helpers/offload_codex_proxy.sh fetch alienware "$RUN" > "$RUN/offload_fetch.log" 2>&1 || echo "offload fetch failed"
exit $SWERC
