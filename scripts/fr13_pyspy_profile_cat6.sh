#!/usr/bin/env bash
# FR13 Stage D boot-profile: localize the cat6-vs-E5 +28ms by py-spy'ing the LIVE
# cat6 worker during sustained B=1 decode. Non-invasive (no patch logic change):
# the only delta is --cap-add=SYS_PTRACE (gated by PROFILE_PTRACE_CAP) so py-spy can
# attach. Captures the WHOLE per-step host-wall breakdown (committer vs GDN-replay
# vs forward vs scheduler) in one flamegraph -> tests H1 (committer syncs) / H2
# (FR13_REPLAY_ROUTE GDN durable-state replay) / H4 (artifact).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
CONTAINER=fr13-prof-cat6
PORT=9951
OUT=output/fr13_pyspy_cat6
mkdir -p "$OUT"
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"

recover(){ PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
teardown(){
  echo "[teardown] docker rm -f $CONTAINER + recover"
  docker logs "$CONTAINER" > "$OUT/docker.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  recover
}
trap teardown EXIT

echo "[1/6] hygiene: recover + assert empty"
recover
[ -z "$(docker ps -q)" ] || { echo "FAIL: docker not empty"; docker ps; exit 2; }

echo "[2/6] boot cat6 (forked launcher, TREE=cat6, SYS_PTRACE cap)"
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  PROFILE_PTRACE_CAP=1 \
  LOG_DIR="$REPO/$OUT/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$OUT/launch.log" 2>&1
RC=$?; (( RC==0 )) || { echo "FAIL launcher rc=$RC"; tail -30 "$OUT/launch.log"; exit 2; }

echo "[3/6] wait /health"
T0=$(date +%s); HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
  [ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" = running ] || { echo FAIL-died; docker logs "$CONTAINER" 2>&1|tail -40; exit 2; }
  sleep 5
done
(( HEALTHY==1 )) || { echo FAIL-health; docker logs "$CONTAINER" 2>&1|tail -40; exit 2; }
echo "healthy after $(( $(date +%s)-T0 ))s"

echo "[4/6] install py-spy in container"
docker exec "$CONTAINER" bash -lc 'pip install -q py-spy 2>&1 | tail -2; which py-spy || python3 -m pip show py-spy 2>/dev/null | head -1' 2>&1 | tail -3

echo "[5/6] sustained B=1 decode loop (background) + py-spy record"
# decode loop: sequential 220-tok completions for ~80s
( for i in $(seq 1 12); do
    curl -fsS -m 60 "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.6-27b","prompt":"Write a long, detailed technical explanation of how modern CPU branch prediction works, step by step:","max_tokens":220,"temperature":0.6,"seed":1313,"stream":false}' \
      >/dev/null 2>>"$OUT/decode_curl.err" || true
  done ) &
DECODE_PID=$!
sleep 8   # let the first request warm + enter steady decode
# pick the busiest python PID (the EngineCore worker running forwards) inside the container
WPID=$(docker exec "$CONTAINER" bash -lc "ps -eo pid,pcpu,comm --sort=-pcpu | awk 'NR>1 && \$3 ~ /python|pt_main|VllmW/ {print \$1; exit}'")
echo "  worker pid (in-container) = $WPID"
docker exec "$CONTAINER" bash -lc "ps -eo pid,pcpu,rss,comm --sort=-pcpu | head -6" | sed 's/^/  /'
# flamegraph (40s) + speedscope + a few text dumps
docker exec "$CONTAINER" bash -lc "py-spy record -d 40 -r 200 --nonblocking -s -p $WPID -o /logs/cat6_pyspy.svg" 2>&1 | tail -3
docker exec "$CONTAINER" bash -lc "py-spy record -d 12 -r 200 --nonblocking -s -f speedscope -p $WPID -o /logs/cat6_pyspy.speedscope.json" 2>&1 | tail -2
for k in 1 2 3 4 5; do docker exec "$CONTAINER" bash -lc "py-spy dump --nonblocking -p $WPID" >> "$OUT/pyspy_dumps.txt" 2>&1 || true; sleep 1; done
wait $DECODE_PID 2>/dev/null || true

echo "[6/6] collect"
cp -f "$REPO/$OUT/logs/cat6_pyspy.svg" "$OUT/" 2>/dev/null || true
cp -f "$REPO/$OUT/logs/cat6_pyspy.speedscope.json" "$OUT/" 2>/dev/null || true
ls -la "$OUT" | sed 's/^/  /'
echo "DONE"
