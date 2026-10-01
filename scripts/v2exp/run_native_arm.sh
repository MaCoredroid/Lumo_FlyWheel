#!/usr/bin/env bash
# Boot one native arm, replay the request corpus, save logs, stop the server.
# Usage: run_native_arm.sh <ar|mtpK> <mode greedy|sampled> [max_tokens]
set -uo pipefail
ARM=$1; MODE=${2:-greedy}; MAXTOK=${3:-1024}
ROOT=/home/mark/shared/lumotree-v2exp-runs
WT=/home/mark/shared/lumotree-v2exp-20260930
RUN=$ROOT/replay/${ARM}-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"; exec > >(tee -a "$RUN/driver.log") 2>&1
echo "run=$RUN arm=$ARM mode=$MODE maxtok=$MAXTOK commit=$(git -C $WT rev-parse HEAD)"
bash "$WT/scripts/v2exp/serve_native.sh" "$ARM" "$RUN" || exit 1
NAME=v2exp-$ARM
cleanup() { docker logs "$NAME" > "$RUN/engine.log" 2>&1; docker stop -t 30 "$NAME" >/dev/null; docker rm "$NAME" >/dev/null; echo "stopped $NAME"; }
trap cleanup EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:9950/health >/dev/null; do
  sleep 10
  docker inspect -f '{{.State.Running}}' "$NAME" | grep -q true || { echo "container exited during boot"; exit 1; }
  (( $(date +%s) - t0 > 1500 )) && { echo "boot timeout"; exit 1; }
done
echo "healthy after $(( $(date +%s) - t0 ))s"
curl -s http://127.0.0.1:9950/metrics > "$RUN/metrics_boot.txt"
# warmup (not recorded)
python3 "$WT/scripts/v2exp/replay.py" --arm "$ARM-warmup" --requests "$ROOT/corpus/requests" \
  --out "$RUN/warmup.jsonl" --mode greedy --max-tokens 64 --limit 1
python3 "$WT/scripts/v2exp/replay.py" --arm "$ARM" --requests "$ROOT/corpus/requests" \
  --out "$RUN/replay.jsonl" --mode "$MODE" --max-tokens "$MAXTOK"
curl -s http://127.0.0.1:9950/metrics > "$RUN/metrics_end.txt"
echo "done $(date -u +%FT%TZ)"
