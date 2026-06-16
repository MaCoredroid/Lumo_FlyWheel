#!/bin/bash
# FR13 OFFLOAD VALIDATION — DURING-RUN prober (runs on the GB10, in parallel with
# an offloaded bigdenom run). Proves the contamination fix at runtime:
#   (A) waits for the alienware codex docker to be live (proxy + codex offloaded)
#   (B) snapshots the GB10 ps + docker WHILE codex is running -> must be vLLM-ONLY
#       (no swe-codex-* container, no inference_proxy python on the GB10)
#   (C) injects a SHORT network blip (iptables-drop alienware->GB10:9950) to prove
#       the proxy reconnects + the task SURVIVES; the blip is < the watchdog
#       OFFLOAD_LINK_DOWN_MAX_S so it must NOT trip DISCARD (run continues).
#       s/fwd is read GB10-LOCALLY so the wire-stall does not corrupt it.
# All artifacts land in $ARMDIR for the post-run verdict.
set -uo pipefail

ARMDIR=${1:?armdir}
GB10_TS_IP=${2:-100.103.10.122}
PORT=${3:-9950}
OFFLOAD_HOST=${4:-alienware}
BLIP_S=${BLIP_S:-25}                 # blip duration (< watchdog 300s => must survive)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
SNAP="$ARMDIR/during_run_gb10_snapshot.txt"
BLIPLOG="$ARMDIR/during_run_blip.log"
mkdir -p "$ARMDIR"

log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

# (A) wait up to 8 min for the alienware codex docker to appear (offloaded run live)
log "waiting for alienware codex docker (swe-codex-*) to go live..." | tee -a "$BLIPLOG"
CODEX_LIVE=0
W0=$(date +%s)
while (( $(date +%s) < W0 + 480 )); do
  CN=$(ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" \
        "docker ps --format '{{.Names}}' 2>/dev/null | grep -m1 '^swe-codex-'" 2>/dev/null)
  if [[ -n "$CN" ]]; then CODEX_LIVE=1; echo "alienware codex container live: $CN" | tee -a "$BLIPLOG"; break; fi
  sleep 5
done
if (( CODEX_LIVE == 0 )); then
  echo "WARN: alienware codex docker never observed live in 480s — snapshot anyway" | tee -a "$BLIPLOG"
fi

# (B) DURING-RUN GB10 snapshot — the proof the user wants. vLLM-ONLY?
{
  echo "===== GB10 DURING-RUN SNAPSHOT $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "--- GB10 docker ps (must be ONLY the vllm container) ---"
  docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
  echo "--- GB10 inference_proxy python procs (must be NONE on the GB10) ---"
  pgrep -af "lumo_flywheel_serving.inference_proxy" || echo "(none — good)"
  echo "--- GB10 codex-runner / swe-codex procs (must be NONE on the GB10) ---"
  pgrep -af "codex-runner|swe-codex|codex exec" || echo "(none — good)"
  echo "--- GB10 docker containers matching codex/swe (must be NONE) ---"
  docker ps --format '{{.Names}}' | grep -E 'codex|swe' || echo "(none — good)"
  echo "--- GB10 top CPU procs (vllm should dominate; no codex/proxy) ---"
  ps -eo pid,ppid,pcpu,pmem,comm --sort=-pcpu | head -12
  echo "--- nvidia-smi compute procs ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "(nvidia-smi compute query n/a)"
} > "$SNAP" 2>&1
log "GB10 during-run snapshot -> $SNAP" | tee -a "$BLIPLOG"

# also snapshot what IS on alienware (codex docker + proxy) to prove they moved there
{
  echo "===== ALIENWARE DURING-RUN SNAPSHOT $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "--- alienware docker ps (codex docker should be here) ---"
  ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'" 2>/dev/null
  echo "--- alienware inference_proxy python (the proxy should be HERE) ---"
  ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" "pgrep -af 'lumo_flywheel_serving.inference_proxy' || echo '(none)'" 2>/dev/null
} > "$ARMDIR/during_run_alienware_snapshot.txt" 2>&1
log "alienware during-run snapshot -> $ARMDIR/during_run_alienware_snapshot.txt"

# (C) NETWORK BLIP INJECTION — from alienware, DROP egress to GB10:9950 for BLIP_S.
# This breaks ONLY codex<->vLLM (port 9950); SSH (22)+ping stay alive so the
# watchdog records it honestly. The proxy must reconnect; the task must survive.
log "injecting ${BLIP_S}s network blip (alienware iptables-drop -> GB10:$PORT)..." | tee -a "$BLIPLOG"
ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" "sudo -n iptables -I OUTPUT 1 -d $GB10_TS_IP -p tcp --dport $PORT -j DROP && echo BLIP_ON" \
  2>&1 | tee -a "$BLIPLOG"
BLIP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "BLIP_START $BLIP_START dport=$PORT dur=${BLIP_S}s" >> "$BLIPLOG"
sleep "$BLIP_S"
ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" "sudo -n iptables -D OUTPUT -d $GB10_TS_IP -p tcp --dport $PORT -j DROP && echo BLIP_OFF" \
  2>&1 | tee -a "$BLIPLOG"
BLIP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "BLIP_END $BLIP_END" >> "$BLIPLOG"
# verify the rule is gone (don't leave alienware firewalling the GB10)
ssh "${SSH_OPTS[@]}" "$OFFLOAD_HOST" "sudo -n iptables -C OUTPUT -d $GB10_TS_IP -p tcp --dport $PORT -j DROP 2>/dev/null && echo 'WARN: blip rule STILL PRESENT' || echo 'blip rule cleared (good)'" \
  2>&1 | tee -a "$BLIPLOG"
log "blip window done ($BLIP_START -> $BLIP_END)" | tee -a "$BLIPLOG"

echo "DURING_RUN_VALIDATE_DONE armdir=$ARMDIR"
