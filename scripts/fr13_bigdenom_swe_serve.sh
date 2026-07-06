#!/usr/bin/env bash
# FR13 BIG-DENOMINATOR SWEServe arm (FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md sec 1, PHASE 2).
# ONE FULL UNCAPPED SWE-Verified task (astropy-12907) at B=1 with the proxy pair-dump ON
# (LUMO_PROXY_PAIR_DUMP_DIR) = the big-denominator served-stream capture vehicle.
#
#   ARM cat9   = DEPLOYED locked tree: scripts/fr13_launch_locked.sh
#                (num_spec=9 TREE_ATTN, FIX-1/2/3/A, REPLAY_ROUTE=1, FA2_TREE_BIAS=1,
#                 CONV_COMMITTED_PATH=1, LUMO_FB_KERNEL_ROWS=1 + LUMO_FB_PROJ_PAD_ROWS=16
#                 [the 2026-06-14 BAKED in_proj_ba pad, NOT in the 2026-06-13 gold gate],
#                 BATCH_INVARIANT=0; NO FR13_SCAN_ALIGN -> K1 NOT baked).
#   ARM native = E5 baseline: scripts/fr10_launch_speed_server.sh
#                (naive_mtp / FLASH_ATTN / num_speculative_tokens=5). The within-floor BAR.
#
# This is the gold-gate run_gold_arm.sh (output/fr13_b1_gold_swe/run_gold_arm.sh) adapted:
#   * cat9 boots the LOCKED launcher (so the baked pad fix is LIVE), and the class-#9
#     flag-live assert additionally requires LUMO_FB_KERNEL_ROWS=1 + LUMO_FB_PROJ_PAD_ROWS=16.
#   * native is byte-identical to the gold-gate native arm.
# Everything else (proxy+pair-dump, spec-engagement draft-shape assert, SWE window, health
# rule, teardown+recover) is the proven gold-gate path.
#
# Usage: fr13_bigdenom_swe_serve.sh <arm_name> <cat9|native> <subset.json> [rep_tag]
#   e.g. fr13_bigdenom_swe_serve.sh cat9_a   cat9   subset_astropy12907.json
#        fr13_bigdenom_swe_serve.sh native_a native subset_astropy12907.json
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

ARM=${1:?usage: fr13_bigdenom_swe_serve.sh <arm> <cat9|native> <subset.json> [rep_tag]}
KIND=${2:?cat9|native}
SUBSET=${3:?subset json (relative to RUNROOT or absolute)}

RUNROOT=output/fr13_bigdenom_swe
ARMDIR="$RUNROOT/$ARM"
[[ -f "$SUBSET" ]] || SUBSET="output/fr13_b1_gold_swe/$SUBSET"
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
CONTAINER="fr13-bigdenom-$ARM"
PORT=9950
PROXY_PORT=8022
EVAL_HOST=${EVAL_HOST:-alienware}
# --- FR13 AGENT-OFFLOAD (the unified-memory contamination fix) -----------------
# OFFLOAD_AGENT=1 (DEFAULT for clean deploy-speed): run the inference_proxy AND
# the SWE agent docker (qwen-code-runner:v1) on alienware (x86) so the GB10 runs
# ONLY vLLM during the agent loop. The GB10's 273 GB/s Grace+Blackwell unified memory is
# then uncontended -> the timing-sensitive deploy-speed (s/fwd) is clean. The
# lossless (argmax/distributional) numbers are timing-independent so unaffected.
# OFFLOAD_AGENT=0 = the legacy on-GB10 path (fallback if alienware is down).
# Flag-gated so nothing about the canonical measurement BASIS changes — only the
# codex+proxy LOCATION. s/fwd is ALWAYS read LOCALLY from the GB10 vLLM /metrics
# (never across the wire) so a Spark<->alienware blip can't corrupt the speed.
OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}
OFFLOAD_HOST=${OFFLOAD_HOST:-alienware}
OFFLOAD_PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}   # 8022 is taken on alienware
GB10_TS_IP=${GB10_TS_IP:-100.103.10.122}             # GB10 tailscale IP (gx10-edb9-1)
OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh
# Watchdog (req #5): max contiguous seconds the Spark<->alienware link may be
# down before we fail LOUD + teardown rather than hang.
OFFLOAD_LINK_DOWN_MAX_S=${OFFLOAD_LINK_DOWN_MAX_S:-300}
# B=4 depth-match knobs (defaults preserve byte-identity for existing callers):
#   SPEC_N           = native num_speculative_tokens (E3/E4/E5 -> 3/4/5). cat9 ignores.
#   MAX_NUM_SEQS_OVR = boot co-residency (B). 1 = legacy B=1; 4 = B=4 co-residency.
#   SWE_CONCURRENCY  = concurrent codex tasks driving the served stream.
SPEC_N=${SPEC_N:-5}
MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
mkdir -p "$ARMDIR/logs"

echo "=== BIGDENOM SWEServe ARM $ARM kind=$KIND subset=$SUBSET ==="
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD | tee "$ARMDIR/git_head.txt"

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# --- PRE-BOOT HYGIENE (every boot): recover + assert MemAvailable>=100GiB + docker ps empty ---
echo "[hygiene] recover_host_memory + assert free"
recover_host || true
.venv/bin/python - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=f.get("SwapTotal",0)-f.get("SwapFree",0)
if avail<100 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
(( $? == 0 )) || { echo "FAIL: pre-boot hygiene"; exit 2; }
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi
free -g | tee "$ARMDIR/free_before_boot.txt"

# stale proxy cleanup (host CPU process, not a GPU container)
OLDPID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
[[ -n "$OLDPID" ]] && kill "$OLDPID" 2>/dev/null
pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
sleep 1

# Teardown trap: docker rm -f + recover after the arm (even on early exit)
teardown(){
  echo "[teardown] kill proxy + docker rm -f $CONTAINER + recover_host_memory"
  kill "$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null)" 2>/dev/null
  pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
  # kill any link watchdog
  [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null
  # OFFLOAD: stop the remote proxy on alienware (clean teardown, req #5)
  if [[ "$OFFLOAD_AGENT" == "1" ]]; then
    LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" stop "$OFFLOAD_HOST" >> "$ARMDIR/offload_teardown.log" 2>&1 || true
  fi
  docker logs "$CONTAINER" > "$ARMDIR/docker_full.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  recover_host || true
  sleep 2
  free -g | tee "$ARMDIR/free_after_teardown.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_ended_at.txt"
}
trap teardown EXIT

# ---- boot server (class 11: everything pinned except the arm variable) ----
if [[ "$KIND" == "cat9" ]]; then
  # DEPLOYED cat9 via the LOCKED launcher (baked pad fix LIVE: LUMO_FB_KERNEL_ROWS=1).
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_locked.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
  PROBE_MODE=tree_mtp
  EXPECT_RATIO=9
else
  # COMPARATOR RULE: native = E{N} qwen3_5_mtp num_speculative_tokens=$SPEC_N via the SPEED launcher.
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  BATCH_INVARIANT=0 FR10_METRICS=0 \
  ATTENTION_BACKEND=FLASH_ATTN FR10_ENABLE_TREE_GDN=0 \
  FR10_DECODE_MODE_DEFAULT=naive_mtp \
  SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$SPEC_N}" \
  FR10_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr10_launch_speed_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
  PROBE_MODE=naive_mtp
  EXPECT_RATIO=$SPEC_N
fi
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -30 "$ARMDIR/launch.log"; exit 2; fi

T0=$(date +%s)
HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL: health not up in 1200s"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2; }
echo "healthy after $(( $(date +%s) - T0 ))s"

# ---- class 9: flag state in container env (deployed flags live in the worker) ----
docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
if [[ "$KIND" == "cat9" ]]; then
  NEEDS=("FR13_TREE_SAMPLE_ROW=1" "FR13_DRAFTER_SINGLE_LOGITS=1" "FR13_EAGER_PACK=1" \
         "FR13_TREE_CONV_FUSED=1" "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" \
         "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1" "FR13_CONV_COMMITTED_PATH=1" \
         "FR10_DECODE_MODE_DEFAULT=tree_mtp" \
         "LUMO_FB_KERNEL_ROWS=1" "LUMO_FB_PROJ_PAD_ROWS=16")
  # K1 must NOT be enabled (resolved DO NOT BAKE): assert FR13_SCAN_ALIGN is 0 / unset.
  if grep -q "^FR13_SCAN_ALIGN=1$" "$ARMDIR/container_env.txt"; then
    echo "FAIL: FR13_SCAN_ALIGN=1 present — K1 must NOT be baked (cat9 LOCKED)"; exit 3
  fi
else
  NEEDS=("FR10_ENABLE_TREE_GDN=0" "FR10_DECODE_MODE_DEFAULT=naive_mtp" \
         "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" "FR10_METRICS=0")
fi
for need in "${NEEDS[@]}"; do
  grep -q "^$need$" "$ARMDIR/container_env.txt" \
    || { echo "FAIL: env pin missing: $need"; exit 3; }
done
if [[ "$KIND" == "native" ]]; then
  grep -q "^SPEC_CONFIG={\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$SPEC_N}\$" "$ARMDIR/container_env.txt" \
    || { echo "FAIL: native SPEC_CONFIG pin missing (E$SPEC_N, no tree)"; exit 3; }
fi
echo "container env OK ($KIND)"

# ---- class 9: worker /proc/<pid>/environ bridge-needle (env actually reached EngineCore) ----
docker exec "$CONTAINER" bash -lc '
hit=""; pids=""
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  e="/proc/$p/environ"; [ -r "$e" ] || continue
  cmd="$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)"
  case "$cmd" in *vllm*|*VllmWorker*|*EngineCore*|*python*) ;; *) continue ;; esac
  dm="$(tr "\0" "\n" < "$e" | grep -E "^FR10_DECODE_MODE_DEFAULT=" || true)"
  if [ -n "$dm" ]; then
    pids="$pids $p"
    echo "PID $p cmd=[$cmd]"
    tr "\0" "\n" < "$e" | grep -E "^(FR10_DECODE_MODE_DEFAULT|LUMO_FB_KERNEL_ROWS|LUMO_FB_PROJ_PAD_ROWS|FR13_SCAN_ALIGN|SPEC_CONFIG|FR13_REPLAY_ROUTE)=" | sed "s/^/   /"
    hit="1"
  fi
done
echo "SUMMARY worker_env_seen=[$hit] pids=[$pids]"
' | tee "$ARMDIR/worker_environ_needle.txt"
grep -q "worker_env_seen=\[1\]" "$ARMDIR/worker_environ_needle.txt" \
  || { echo "FAIL: no vLLM worker with FR10_DECODE_MODE_DEFAULT in /proc environ (class #9/#10)"; exit 3; }
if [[ "$KIND" == "cat9" ]]; then
  grep -q "LUMO_FB_KERNEL_ROWS=1" "$ARMDIR/worker_environ_needle.txt" \
    || { echo "FAIL: LUMO_FB_KERNEL_ROWS=1 not live in worker environ (baked pad fix OFF)"; exit 3; }
fi
echo "worker environ needle OK ($KIND)"

# ---- class 9: FULL CUDA capture needle ----
docker logs "$CONTAINER" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  echo "[capture needle] skipped (ENFORCE_EAGER=1: eager mode, no graph capture — diagnostic/tap boot)"
else
  grep -m1 "Graph capturing finished" "$ARMDIR/boot_log_snapshot.txt" \
    || { echo "FAIL: no 'Graph capturing finished' in boot log (capture needle)"; exit 3; }
fi

# ---- identical warmup probe (fires engagement needles before codex) ----
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_warmup.txt"
.venv/bin/python scripts/fr10_quick_decode_tps_probe.py \
  --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
  --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
  --samples-per-prompt 1 --batch-size 1 --seed 1313 --top-p 1.0 \
  --wait-health 60 --request-timeout 900 --warmup-samples 0 \
  --modes "$PROBE_MODE" \
  --prompt-limit 1 --max-tokens 16 --temperature 0.0 \
  --out "$ARMDIR/warmup_probe.json" \
  --request-metrics-out "$ARMDIR/warmup_request_metrics.jsonl" \
  > "$ARMDIR/warmup_probe_stdout.log" 2>&1
RC=$?
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_warmup.txt"
if (( RC != 0 )); then
  echo "FAIL: warmup probe rc=$RC"
  tail -20 "$ARMDIR/warmup_probe_stdout.log"; docker logs "$CONTAINER" 2>&1 | tail -30; exit 4
fi
# class 9 + COMPARATOR RULE: spec engagement from RAW /metrics deltas
# (cat9: drafts>0 and draft_tokens/drafts == 9; native: == 5).
.venv/bin/python - "$ARMDIR" "$EXPECT_RATIO" <<'PY'
import sys
from pathlib import Path
armdir, expect = Path(sys.argv[1]), float(sys.argv[2])
def val(path, name):
    for line in (armdir / path).read_text().splitlines():
        if line.startswith(name):
            return float(line.rsplit(None, 1)[-1])
    return 0.0
d  = val("metrics_after_warmup.txt", "vllm:spec_decode_num_drafts_total") \
   - val("metrics_before_warmup.txt", "vllm:spec_decode_num_drafts_total")
dt = val("metrics_after_warmup.txt", "vllm:spec_decode_num_draft_tokens_total") \
   - val("metrics_before_warmup.txt", "vllm:spec_decode_num_draft_tokens_total")
assert d > 0, f"spec engagement FAIL (class 9): spec_drafts delta={d}"
ratio = dt / d
assert abs(ratio - expect) < 1e-9, \
    f"draft-shape FAIL (class 9): draft_tokens/drafts={ratio} expected {expect}"
print(f"spec engagement OK: drafts delta={d} draft_tokens/drafts={ratio}")
PY
(( $? == 0 )) || { echo "FAIL: spec engagement raw-counter assert"; exit 4; }

# reset prefix cache after warmup (identical on all arms; VLLM_SERVER_DEV_MODE=1)
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
  > "$ARMDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"

# ---- launch canonical proxy: temp 0.0 pin + served-stream pair dumps ON ----
# Two paths, flag-gated by OFFLOAD_AGENT:
#   OFFLOAD_AGENT=1 (default): proxy runs ON ALIENWARE (-> GB10:9950 over
#     tailscale); pair-dumps captured LOCALLY on alienware (rsynced back after
#     the run). The GB10 runs ONLY vLLM during the codex loop = clean s/fwd.
#   OFFLOAD_AGENT=0: legacy on-GB10 proxy (the prior co-located path).
mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
AGENT_ARGS=()
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  echo "[offload] OFFLOAD_AGENT=1 — proxy+agent on $OFFLOAD_HOST, GB10 stays vLLM-only"
  # (a) the GB10 vLLM must be reachable from alienware over tailscale before we
  #     start the remote proxy (req #6: measurement-side health is GB10-local,
  #     but the request-side path needs the link up to begin).
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$OFFLOAD_HOST" \
        "curl -fsS -m 6 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo ok" \
        2>/dev/null | grep -q ok; then
    echo "FAIL: alienware cannot reach GB10 vLLM at http://$GB10_TS_IP:$PORT/health"
    echo "      (network/firewall — set OFFLOAD_AGENT=0 to fall back to the on-GB10 path)"
    exit 5
  fi
  echo "[offload] alienware -> GB10 vLLM $GB10_TS_IP:$PORT/health OK"
  # (b) sync the proxy code to alienware + start it there (pins forced temp 0.0,
  #     pair-dumps ON, auto-continue ON — the canonical deploy pins, asserted by
  #     the helper). LUMO_OFFLOAD_PROXY_PORT=8023 (8022 taken on alienware).
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" sync "$OFFLOAD_HOST" > "$ARMDIR/offload_sync.log" 2>&1 \
    || { echo "FAIL: offload proxy sync to $OFFLOAD_HOST"; cat "$ARMDIR/offload_sync.log"; exit 5; }
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" start "$OFFLOAD_HOST" "$GB10_TS_IP" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_start.log" 2>&1 \
    || { echo "FAIL: offload proxy start on $OFFLOAD_HOST"; cat "$ARMDIR/offload_start.log"; exit 5; }
  cat "$ARMDIR/offload_start.log"
  # the offload helper already asserts the canonical pins on the REMOTE proxy
  # env (forced temp 0.0 / pair-dump / auto-continue) before returning OK.
  cp "$ARMDIR/offload_proxy_env.txt" "$ARMDIR/proxy_env.txt" 2>/dev/null || true
  # the codex orchestrator runs the codex docker on alienware against the
  # alienware-local proxy (127.0.0.1:8023). vLLM /metrics still read GB10-locally.
  AGENT_ARGS=(--agent-host "$OFFLOAD_HOST" \
              --agent-endpoint "http://127.0.0.1:$OFFLOAD_PROXY_PORT/v1")
  echo "proxy OK (OFFLOADED to $OFFLOAD_HOST:$OFFLOAD_PROXY_PORT; forced temp 0.0, pair dumps on, auto-continue on)"
else
  echo "[offload] OFFLOAD_AGENT=0 — legacy on-GB10 proxy+agent (co-located)"
  LUMO_PROXY_FORCE_TEMPERATURE=0.0 \
  LUMO_PROXY_REQUEST_DUMP_DIR="$PWD/$ARMDIR/proxy_request_dumps" \
  LUMO_PROXY_PAIR_DUMP_DIR="$PWD/$ARMDIR/proxy_pair_dumps" \
  LUMO_PROXY_LOG_PATH="$PWD/$ARMDIR/proxy.log" \
  LUMO_PROXY_NOHUP_PATH="$PWD/$ARMDIR/proxy.nohup" \
  LUMO_PROXY_STATE_ROOT="/tmp/fr13_bigdenom_proxy_state_${ARM}" \
  bash scripts/swe_x86_helpers/relaunch_proxy.sh > "$ARMDIR/proxy_launch.log" 2>&1
  sleep 3
  PROXY_PID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
  if [[ -n "$PROXY_PID" ]] && [[ -r "/proc/$PROXY_PID/environ" ]]; then
    tr '\0' '\n' < "/proc/$PROXY_PID/environ" | grep -E "^LUMO_(PROXY|TRACK_B)" | sort \
      > "$ARMDIR/proxy_env.txt"
  fi
  P0=$(date +%s); PROXY_OK=0
  while (( $(date +%s) < P0 + 60 )); do
    CODE=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PROXY_PORT/v1/models" 2>/dev/null)
    if [[ -n "$CODE" && "$CODE" != "000" ]]; then PROXY_OK=1; break; fi
    sleep 2
  done
  (( PROXY_OK == 1 )) || { echo "FAIL: proxy not healthy on :$PROXY_PORT"; tail -20 "$ARMDIR/proxy.nohup" 2>/dev/null; exit 5; }
  grep -q "LUMO_PROXY_FORCE_TEMPERATURE=0.0" "$ARMDIR/proxy_env.txt" \
    || { echo "FAIL: proxy temp pin missing (class 9)"; exit 5; }
  grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/proxy_env.txt" \
    || { echo "FAIL: proxy pair-dump pin missing (class 9)"; exit 5; }
  grep -q "LUMO_PROXY_AUTO_CONTINUE=1" "$ARMDIR/proxy_env.txt" \
    || { echo "FAIL: proxy auto-continue pin missing (early-exit class fe927e74)"; exit 5; }
  echo "proxy OK (forced temp 0.0, pair dumps on, auto-continue on)"
fi

# ---- eval offload pre-flight (attempt eval; do NOT fail the gate on infra) ----
EVAL_ARGS=()
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$EVAL_HOST" \
     "test -f ~/swe_eval_offload/swe_eval_x86_worker.py && echo ok" 2>/dev/null | grep -q ok; then
  EVAL_ARGS=(--eval-host "$EVAL_HOST")
  echo "eval offload: $EVAL_HOST reachable, worker present" | tee "$ARMDIR/eval_offload_preflight.txt"
else
  echo "WARN: eval offload $EVAL_HOST UNREACHABLE — eval attempted locally; " \
       "gate NOT failed on eval-infra unavailability (lossless verdict = served streams)" \
    | tee "$ARMDIR/eval_offload_preflight.txt"
fi

# ---- the SWE window: /metrics bracketing + the FULL UNCAPPED task ----
# AGENT_WALL_S (optional env): bound the codex agent wall for a DEV-iteration
# BOUNDED deployment run. Default UNSET = the proven full 25-min deployment wall
# (run_swe_bench_q36_a.py DEFAULT_AGENT_WALL_S = 1500s). When set, the /metrics
# brackets still wrap the real (bounded) codex trajectory -> a deployment-faithful
# deploy-speed reduction on a shorter window (the user-blessed bounded-turn path).
WALL_ARGS=()
[[ -n "${AGENT_WALL_S:-}" ]] && WALL_ARGS=(--agent-wall-s "$AGENT_WALL_S")

# ---- NETWORK-LINK WATCHDOG (OFFLOAD only; req #4/#5) -------------------------
# Throughout the SWE window, poll the Spark<->alienware tailscale link
# (alienware curls the GB10 vLLM /health). Each sample is logged with a
# timestamp so a request failure can be CLASSIFIED network-drop vs real (no #12
# fork mis-attribution). If the link is DOWN for > OFFLOAD_LINK_DOWN_MAX_S
# contiguous seconds, write a LINK_DEAD marker (the watchdog fails LOUD; the
# post-run classifier then flags the window for DISCARD rather than recording a
# wire-stalled s/fwd). s/fwd itself is read GB10-LOCALLY so it is stall-immune
# by construction — the watchdog only guards against a hung/contaminated window.
LINKLOG="$ARMDIR/offload_link_state.log"
LINK_DEAD_MARKER="$ARMDIR/offload_link_dead.flag"
WATCHDOG_PID=""
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  rm -f "$LINK_DEAD_MARKER"
  ( down_since=0
    while true; do
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      if ssh -o BatchMode=yes -o ConnectTimeout=8 "$OFFLOAD_HOST" \
           "curl -fsS -m 5 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo up" \
           2>/dev/null | grep -q up; then
        echo "[$ts] LINK up (alienware->GB10:$PORT/health)" >> "$LINKLOG"
        down_since=0
      else
        now=$(date +%s)
        [[ "$down_since" == "0" ]] && down_since=$now
        contig=$(( now - down_since ))
        echo "[$ts] LINK DOWN contig=${contig}s (CLASSIFIED network-drop, not a model fork)" >> "$LINKLOG"
        if (( contig > OFFLOAD_LINK_DOWN_MAX_S )); then
          echo "[$ts] LINK DEAD > ${OFFLOAD_LINK_DOWN_MAX_S}s — watchdog FAIL LOUD, marking window DISCARD" \
            | tee -a "$LINKLOG"
          echo "$ts contig=${contig}s" > "$LINK_DEAD_MARKER"
          break
        fi
      fi
      sleep 10
    done ) >> "$ARMDIR/offload_watchdog.log" 2>&1 &
  WATCHDOG_PID=$!
  echo "[offload] link watchdog pid=$WATCHDOG_PID (threshold ${OFFLOAD_LINK_DOWN_MAX_S}s)"
fi

curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_swe.txt"
S0=$(date +%s)
.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$SUBSET" \
  --out-root "$ARMDIR/swe_out" \
  --concurrency "$SWE_CONCURRENCY" \
  "${WALL_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${AGENT_ARGS[@]}" \
  > "$ARMDIR/swe_orchestrator.log" 2>&1
SWERC=$?
S1=$(date +%s)
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_swe.txt"
SWE_WALL=$((S1-S0))
echo "swe orchestrator rc=$SWERC wall=${SWE_WALL}s"
tail -5 "$ARMDIR/swe_orchestrator.log"

# stop the watchdog now the window is closed
[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null && WATCHDOG_PID=""

# ---- OFFLOAD: fetch the alienware pair-dumps back (resilient rsync, req #3) --
# the deploy-lossless recurrent-oracle rescore (a SEPARATE GB10 vLLM-only GPU
# phase) reads these. A dropped rsync must not lose the served stream.
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_fetch.log" 2>&1 || echo "WARN: offload pair-dump fetch had errors (see offload_fetch.log)"
  cat "$ARMDIR/offload_fetch.log"
  # NETWORK-STALL WINDOW DISCARD (req #4): if the watchdog saw the link dead,
  # FLAG this arm's deploy-speed window as not-recordable (do not silently
  # record a wire-stalled s/fwd). The lossless rescore can still proceed from
  # whatever pair-dumps were captured; only the SPEED window is discarded.
  if [[ -f "$LINK_DEAD_MARKER" ]]; then
    echo "OFFLOAD_LINK_DEAD: $(cat "$LINK_DEAD_MARKER") — deploy-speed window for $ARM is DISCARDED" \
      | tee "$ARMDIR/DEPLOY_SPEED_DISCARDED.flag"
    SWERC=12   # distinct rc: network-stall window, not a real/model failure
  fi
fi

# ---- FULL-RUN HEALTH RULE (class 9): record + flag wall-consumed ----
.venv/bin/python - "$ARMDIR" "$SWE_WALL" "$SWERC" <<'PY'
import json, sys
from pathlib import Path
armdir, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
metas = sorted(armdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
health = {"swe_orchestrator_rc": rc, "swe_window_wall_s": wall,
          "expected_codex_wall_s": 1500, "tasks": []}
flags = []
for m in metas:
    meta = json.loads(m.read_text())
    codex = meta.get("agent") or meta.get("codex") or {}
    t = {"instance_id": meta.get("instance_id"),
         "codex_elapsed_s": codex.get("elapsed_s"),
         "codex_timed_out": codex.get("timed_out"),
         "codex_exit_code": codex.get("exit_code"),
         "patch_bytes": meta.get("patch_bytes"),
         "empty_patch_retry": meta.get("empty_patch_retry"),
         "eval_exit_code": (meta.get("eval") or {}).get("exit_code"),
         "eval_offloaded": (meta.get("eval") or {}).get("offloaded"),
         "verdict": (meta.get("eval_report") or {}).get("verdict", "missing")}
    health["tasks"].append(t)
    el = codex.get("elapsed_s") or 0
    if el < 1400 and not codex.get("timed_out"):
        flags.append(f"{t['instance_id']}: codex consumed only {el:.0f}s of the 1500s wall "
                     "— FULL-RUN HEALTH RULE: early exit = OUR-side issue until proven "
                     "otherwise. INVESTIGATE + FIX + RE-RUN THIS ARM before reading a verdict.")
if not metas:
    flags.append("no runner_metadata.json produced — orchestrator-level failure")
health["health_flags"] = flags
(armdir / "health.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
sys.exit(1 if flags else 0)
PY
HEALTH_RC=$?
(( HEALTH_RC == 0 )) || echo "HEALTH FLAGGED (see $ARMDIR/health.json) — arm NOT verdict-grade until investigated"

# ---- NON-VACUITY: pair-dump engagement (zero pair dumps = vacuous arm) ----
NPAIR=$(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)
echo "pair dumps captured: $NPAIR"
.venv/bin/python - "$ARMDIR/proxy_pair_dumps" "$ARMDIR/pairdump_stats.json" <<'PY'
import json, sys, glob, os
ddir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(ddir, "pair_*.json")))
n_turns = 0; n_nonempty = 0; total_chars = 0; total_usage_out = 0
for fp in files:
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    n_turns += 1
    resp = r.get("response") or {}
    chars = 0
    for item in (resp.get("output") or []):
        t = item.get("type")
        if t in ("reasoning", "message"):
            for c in (item.get("content") or []):
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    chars += len(c["text"])
        elif t == "function_call":
            a = item.get("arguments")
            if isinstance(a, str):
                chars += len(a)
    total_chars += chars
    if chars > 0:
        n_nonempty += 1
    usage = resp.get("usage") or {}
    total_usage_out += int(usage.get("output_tokens") or 0)
stats = {"n_pair_files": len(files), "n_turns": n_turns,
         "n_turns_with_served_text": n_nonempty,
         "total_served_chars": total_chars,
         "total_usage_output_tokens": total_usage_out}
json.dump(stats, open(out, "w"), indent=2)
print(json.dumps(stats, indent=2))
PY
PAIR_NONEMPTY=$(.venv/bin/python -c 'import json,sys;d=json.load(open(sys.argv[1]));print("1" if d.get("n_turns_with_served_text",0)>0 and d.get("total_served_chars",0)>0 else "0")' "$ARMDIR/pairdump_stats.json")
if [[ "$PAIR_NONEMPTY" != "1" ]]; then echo "FAIL: pair-dump VACUOUS (no served text) — class #9"; SWERC=9; fi

echo "ARM_DONE $ARM kind=$KIND swerc=$SWERC health_rc=$HEALTH_RC pair_nonempty=$PAIR_NONEMPTY"
exit $SWERC
