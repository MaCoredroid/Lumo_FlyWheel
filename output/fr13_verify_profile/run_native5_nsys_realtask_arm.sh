#!/usr/bin/env bash
# FR13 verify-profile arm 2: nsys capture during a REAL SWE-Verified agentic task
# (deployment-exact: offloaded qwen-code agent, forced temp 0.6, nudge-free,
# cache-ON, B=4 serving config with 1 in-flight task). Complements the arm-1
# static-probe capture (diagnostic-only per feedback_live_swe_verified_only):
# real 10-60k contexts exercise the attention/KV bucket the static probe
# understates, and temp-0.6 exercises the deployed sampler path.
# Task: astropy__astropy-14598 (the marathon decode-rich task, 10k+ drafts in
# kvr1) so the capture window is guaranteed mid-decode.
# NOTE: nsys --duration KILLS vllm at window end (observed arm-1 behavior:
# container exits) => the task dies mid-run. That is EXPECTED and fine -- we
# want the kernel capture, not the SWE verdict. The driver will report a task
# failure; ignore it.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_verify_profile
ARM=native5_nsysreal
ARMDIR="$RUNROOT/$ARM"
PORT=9950
DELAY=1200   # boot(~10min nsys) + variant warmup/engagement + task prefill margin
DUR=300
mkdir -p "$RUNROOT"
echo "=== NSYS REAL-TASK ARM native5 MTP-5 twin delay=${DELAY}s dur=${DUR}s ==="
date -u +%Y-%m-%dT%H:%M:%SZ
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty"; docker ps; exit 2; fi

cleanup() {
  docker ps -q --filter name=fr13-bigdenom | xargs -r docker rm -f >/dev/null 2>&1 || true
  pkill -f "run_swe_bench_q36_a.py.*nsysreal" 2>/dev/null || true
  sleep 3
  AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
  if (( AVAIL < 60 )); then
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# 1-task subset (the marathon task) -- same dict shape as the source subset
SUBSET="$RUNROOT/subset_14598.json"
python3 - "$SUBSET" <<'EOF'
import json, sys
src = json.load(open('output/fr13_b1_gold_swe/subset_b4_sixteen.json'))
sel = [i for i in src['instance_ids'] if '14598' in str(i)]
assert sel, '14598 not found in subset'
out = dict(src)
out['instance_ids'] = sel
out['note'] = 'nsys real-task profile subset: the marathon decode-rich task only'
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('subset:', sel)
EOF

# 1-arm sequence file: tail6 kind, campaign kvr1 env (cache-ON)
SEQ=/tmp/smoke_seq_nsysreal.sh
cat > "$SEQ" <<'EOF'
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
run_variant native5_nsysreal  nativemtp5_exseed  5  1
EOF

# campaign env + nsys env (LUMO_* auto-forwarded by the launcher's env scanner)
export LUMO_NSYS_WRAP_VLLM=1
export LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx
export LUMO_NSYS_DELAY_S=$DELAY
export LUMO_NSYS_DURATION_S=$DUR
export LUMO_NSYS_OUTPUT=/logs/nsys_native5_realtask
T_LAUNCH=$(date +%s)

RUNROOT="$RUNROOT" TAG=nsysreal SUBSET="$SUBSET" BSIZE=4 CONC=1 GPU_UTIL=0.60 \
  DEPLOY_FORCE_TEMP=0.6 SEQUENCE_FILE="$SEQ" \
  bash scripts/fr13_b4_campaign_driver.sh > "$RUNROOT/driver.nsysreal.log" 2>&1 &
DRIVER_PID=$!
echo "driver pid=$DRIVER_PID"

# watch for the container + banner to compute t0
T0=""
for i in $(seq 1 200); do
  BANNER_TS=$(docker logs -t fr13-bigdenom-native5_nsysreal 2>/dev/null | grep -m1 "version 0.19" | awk '{print $1}')
  if [[ -n "${BANNER_TS:-}" ]]; then T0=$(( $(date -d "${BANNER_TS}" +%s) - 8 )); break; fi
  kill -0 $DRIVER_PID 2>/dev/null || { echo "FAIL: driver died pre-banner"; tail -20 "$RUNROOT/driver.nsysreal.log"; exit 2; }
  sleep 5
done
[[ -n "$T0" ]] || { echo "FAIL: no banner"; exit 2; }
echo "banner t0=$T0 window=[$((T0+DELAY)),$((T0+DELAY+DUR)))"

# metrics bracket at window open/close (drafts-in-window for the reducer)
while (( $(date +%s) < T0 + DELAY )); do
  kill -0 $DRIVER_PID 2>/dev/null || { echo "FAIL: driver died pre-window"; exit 2; }
  sleep 5
done
W0=$(date +%s)
curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics" > "$RUNROOT/metrics_window_open.txt" 2>/dev/null || echo "WARN: metrics at open failed"
echo "window OPEN at t+$((W0-T0))s"
# FREEZE the variant/driver scripts NOW: if the task completes inside the window,
# the variant's teardown (docker rm -f) would kill nsys MID-REPORT-WRITE (this
# exact race destroyed the first attempt's capture at t+1449). SIGSTOP the bash
# processes that contain the teardown code; the offloaded agent keeps driving
# decode regardless. Thawed+killed after the report is stable.
pkill -STOP -f "fr13_bigdenom_swe_serve_variant" 2>/dev/null || true
pkill -STOP -f "fr13_b4_campaign_driver" 2>/dev/null || true
echo "variant/driver frozen for capture protection"
while (( $(date +%s) < T0 + DELAY + DUR - 45 )); do sleep 5; done
curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics" > "$RUNROOT/metrics_window_close.txt" 2>/dev/null || echo "WARN: metrics at close failed"
W1=$(date +%s)
echo "window CLOSE bracket at t+$((W1-T0))s"

ARMD="$RUNROOT/native5_nsysreal"
mkdir -p "$ARMD"
D0=$(awk '/^vllm:spec_decode_num_drafts_total/ {s+=$2} END {print s+0}' "$RUNROOT/metrics_window_open.txt" 2>/dev/null || echo 0)
D1=$(awk '/^vllm:spec_decode_num_drafts_total/ {s+=$2} END {print s+0}' "$RUNROOT/metrics_window_close.txt" 2>/dev/null || echo 0)
DRAFTS=$(python3 -c "print(max(float('$D1')-float('$D0'),0))")
if python3 -c "exit(0 if float('$DRAFTS') <= 0 else 1)"; then
  echo "WARN: metrics-bracket drafts=0; falling back to docker-log Drafted-token deltas"
  DRAFTS=$(docker logs fr13-bigdenom-native5_nsysreal 2>&1 | python3 -c "
import sys, re, calendar, time
tot = 0.0
for line in sys.stdin:
    if 'SpecDecoding metrics' not in line: continue
    m = re.search(r'(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*Drafted: (\d+) tokens', line)
    if not m: continue
    mo, dd, hh, mm, ss, tok = (int(x) for x in m.groups())
    ts = calendar.timegm((2026, mo, dd, hh, mm, ss, 0, 0, 0))
    if $W0 <= ts <= $W1:
        tot += tok
# per-10s-window Drafted token counts, 21 tokens per draft
print(tot / 5.0)")
fi
echo "{\"r\":1,\"wall_start\":$W0,\"wall_end\":$W1}" > "$ARMD/probe_walls.jsonl"
python3 -c "
import json
json.dump({'summary':{'spec_drafts': float('$DRAFTS')}}, open('$ARMD/window_probe_r1.json','w'))
print('drafts in window:', '$DRAFTS')"

# wait for the nsys report (container will be killed by nsys at duration end)
REP="$ARMD/logs/nsys_native5_realtask.nsys-rep"
PREV=-1; STABLE=0
for i in $(seq 1 150); do
  SZ=$(stat -c %s "$REP" 2>/dev/null || echo -1)
  if (( SZ > 0 && SZ == PREV )); then STABLE=$((STABLE+1)); else STABLE=0; fi
  PREV=$SZ
  (( STABLE >= 4 )) && { echo "REPORT_STABLE size=$SZ"; break; }
  sleep 5
done
SZ=$(stat -c %s "$REP" 2>/dev/null || echo -1)
(( SZ > 0 )) || { echo "FAIL: no nsys report at $REP"; ls -la "$ARMD/logs/" 2>/dev/null; exit 5; }
docker logs fr13-bigdenom-native5_nsysreal > "$ARMD/docker_full.log" 2>&1 || true
# thaw the frozen variant/driver, then stop them for good
pkill -CONT -f "fr13_bigdenom_swe_serve_variant" 2>/dev/null || true
pkill -CONT -f "fr13_b4_campaign_driver" 2>/dev/null || true
sleep 1
pkill -f "fr13_bigdenom_swe_serve_variant" 2>/dev/null || true
pkill -f "fr13_b4_campaign_driver" 2>/dev/null || true
kill $DRIVER_PID 2>/dev/null || true
echo "ARM_DONE report=$REP size=$SZ drafts=$DRAFTS"
date -u +%Y-%m-%dT%H:%M:%SZ