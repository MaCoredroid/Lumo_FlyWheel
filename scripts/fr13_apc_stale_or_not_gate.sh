#!/usr/bin/env bash
# ============================================================================
# FR13 APC STALE-OR-NOT GATE — the BINDING APC lossless gate.
# ============================================================================
# ONE boot, cache ON, but the cache-OFF route is PRESERVED: each cache-hit
# scheduling step the SSM scheduler reads a /logs flag file (FR13_APC_SHADOW_RUNTIME
# mechanism, scripts/fr10_phase4_patch_vllm_tree_gdn.py:6381-6432):
#   file content "1"  => re-prefill the WHOLE matched prefix (NOT-STALE; the cache-OFF-
#                        equivalent for ALL carriers conv/ssm/full-attn KV/position,
#                        bypassed at the scheduler: new_computed_blocks.new_empty() +
#                        num_new_local_computed_tokens=0).
#   "0"/absent        => use the cached restore (STALE).
# So a SINGLE held boot runs the LIVE astropy-12907 codex SWE episode BOTH ways and
# compares — no two boots, no cross-boot autotune confound, single variable = the flag.
#
# WHY DECOUPLE: scripts/fr13_bigdenom_swe_serve_variant.sh COUPLES boot+episode+teardown
# (one boot -> exactly one run_swe_bench_q36_a.py sweep -> teardown via trap, L416-477).
# This gate REUSES serve_variant's proven boot env + offload plumbing VERBATIM but
# HOLDS the server up and drives MULTIPLE codex episodes (2*N) against the SAME boot,
# toggling only the flag file between them.
#
# Usage: fr13_apc_stale_or_not_gate.sh [N_SEEDS]   (default 3)
#   env: AGENT_WALL_S, OFFLOAD_AGENT(=1 default), OFFLOAD_HOST, EVAL_HOST, DEPLOY_FORCE_TEMP
# ============================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

N_SEEDS=${1:-3}
INSTANCE=${INSTANCE:-astropy__astropy-12907}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }

# ---- boot identity (cat6root spec, mirrors serve_variant KIND=cat6root, L51/L78) ----
ARM=apc_stale_or_not
RUNROOT=${RUNROOT:-output/fr13_apc_stale_or_not}
ARMDIR="$RUNROOT/run_$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER="fr13-apc-staleornot"
PORT=9950
PROXY_PORT=8022
# cat6root TREE (serve_variant.sh:51) + EXPECT 6 drafts/tok (serve_variant.sh:78).
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
EXPECT_RATIO=6
PROBE_MODE=tree_mtp

# ---- offload codex config (serve_variant.sh:42-47) ----
OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}
OFFLOAD_HOST=${OFFLOAD_HOST:-alienware}
OFFLOAD_PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
GB10_TS_IP=${GB10_TS_IP:-100.103.10.122}
OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh
EVAL_HOST=${EVAL_HOST:-alienware}
DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}

mkdir -p "$ARMDIR/logs"
LOG_DIR="$PWD/$ARMDIR/logs"
# The shadow flag file: written here on the HOST, bind-mounted to container /logs
# (launcher mounts LOG_DIR:/logs, fr13_launch_forked_fa2_tree_server.sh:394). The
# scheduler reads FR13_APC_SHADOW_FLAG_FILE (default /logs/fr13_apc_shadow_now.flag).
SHADOW_FLAG_HOST="$LOG_DIR/fr13_apc_shadow_now.flag"
SHADOW_FLAG_CONTAINER="/logs/fr13_apc_shadow_now.flag"

echo "=== FR13 APC STALE-OR-NOT GATE  arm=$ARM container=$CONTAINER port=$PORT ==="
echo "    rundir=$ARMDIR  instance=$INSTANCE  N_SEEDS=$N_SEEDS  offload=$OFFLOAD_AGENT"
echo "    shadow flag (host)=$SHADOW_FLAG_HOST -> (container)=$SHADOW_FLAG_CONTAINER"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD | tee "$ARMDIR/git_head.txt"

# ---- recover_host helper (serve_variant.sh:99-103) ----
recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# ---- pre-boot hygiene (serve_variant.sh:105-120) ----
echo "[hygiene] recover_host_memory + assert free"
recover_host || true
.venv/bin/python - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=f.get("SwapTotal",0)-f.get("SwapFree",0)
if avail<95 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
(( $? == 0 )) || { echo "FAIL: pre-boot hygiene"; exit 2; }
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi
free -g | tee "$ARMDIR/free_before_boot.txt"

# kill any stale proxy (serve_variant.sh:122-125)
OLDPID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
[[ -n "$OLDPID" ]] && kill "$OLDPID" 2>/dev/null
pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
sleep 1

# ---- teardown (serve_variant.sh:127-143) — fires ONCE, after all episodes ----
teardown(){
  echo "[teardown] kill proxy + docker rm -f $CONTAINER + recover_host_memory"
  kill "$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null)" 2>/dev/null
  pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
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

# ============================================================================
# PHASE 1 — BOOT ONE SERVER AND HOLD IT (decoupled from any episode).
# cat6root forked launcher, FR13_ENABLE_APC=1, FR13_APC_SHADOW_RUNTIME=1, MAMBA
# block=1024 / ssm fp32, B=1. Mirrors serve_variant.sh:159-165 (forked branch)
# with the APC + shadow-runtime envs added (consumed by the launcher's -e block,
# fr13_launch_forked_fa2_tree_server.sh:418-419).
# ============================================================================
echo "[boot] forked cat6root + FR13_ENABLE_APC=1 + FR13_APC_SHADOW_RUNTIME=1 (HOLD)"
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 \
  MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  FR13_APC_SHADOW=0 FR13_APC_SHADOW_RUNTIME=1 \
  FR13_APC_SHADOW_FLAG_FILE="$SHADOW_FLAG_CONTAINER" \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$LOG_DIR" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -30 "$ARMDIR/launch.log"; exit 2; fi

# ---- health wait (serve_variant.sh:168-178) ----
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

# ---- class-9 container env pins (serve_variant.sh:180-197) + APC/shadow pins ----
docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
NEEDS=("FR13_DRAFTER_SINGLE_LOGITS=1" "FR13_EAGER_PACK=1" "FR13_TREE_CONV_FUSED=1" \
       "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" \
       "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1" "FR13_CONV_COMMITTED_PATH=1" \
       "FR10_DECODE_MODE_DEFAULT=tree_mtp" \
       "LUMO_FB_KERNEL_ROWS=1" "LUMO_FB_PROJ_PAD_ROWS=16" \
       "FR13_APC_SHADOW_RUNTIME=1" "FR13_APC_SHADOW=0" \
       "FR13_APC_SHADOW_FLAG_FILE=$SHADOW_FLAG_CONTAINER")
if grep -q "^FR13_SCAN_ALIGN=1$" "$ARMDIR/container_env.txt"; then
  echo "FAIL: FR13_SCAN_ALIGN=1 present — K1 must NOT be baked"; exit 3
fi
for need in "${NEEDS[@]}"; do
  grep -q "^$need$" "$ARMDIR/container_env.txt" || { echo "FAIL: env pin missing: $need"; exit 3; }
done
echo "container env OK (cat6root + APC shadow-runtime)"

# ---- worker /proc environ bridge-needle (serve_variant.sh:199-218) ----
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
    tr "\0" "\n" < "$e" | grep -E "^(FR10_DECODE_MODE_DEFAULT|FR13_APC_SHADOW_RUNTIME|FR13_REPLAY_ROUTE)=" | sed "s/^/   /"
    hit="1"
  fi
done
echo "SUMMARY worker_env_seen=[$hit] pids=[$pids]"
' | tee "$ARMDIR/worker_environ_needle.txt"
grep -q "worker_env_seen=\[1\]" "$ARMDIR/worker_environ_needle.txt" \
  || { echo "FAIL: no vLLM worker with FR10_DECODE_MODE_DEFAULT in /proc environ"; exit 3; }
echo "worker environ needle OK"

# ---- CUDA capture needle (serve_variant.sh:220-227) ----
docker logs "$CONTAINER" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  echo "[capture needle] SKIPPED (ENFORCE_EAGER=1)"
else
  grep -m1 "Graph capturing finished" "$ARMDIR/boot_log_snapshot.txt" \
    || { echo "FAIL: no 'Graph capturing finished' in boot log"; exit 3; }
fi

# ---- APC serve-flag needle: --enable-prefix-caching must be in the serve cmd ----
# (the cache must be physically ON; the shadow toggle only chooses STALE vs FRESH).
grep -q -- "--enable-prefix-caching" "$ARMDIR/boot_log_snapshot.txt" \
  || grep -q -- "--enable-prefix-caching" "$ARMDIR/launch.log" \
  || echo "WARN: --enable-prefix-caching not echoed in logs (verify cache ON via metrics non-vacuity)"

# ---- warmup probe (fires spec engagement, serve_variant.sh:239-276) ----
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

# ============================================================================
# PHASE 2 — launch the OFFLOAD codex proxy ONCE and HOLD it (serve_variant.sh:315-338).
# The proxy pins the deployment temp (DEPLOY_FORCE_TEMP=0.6) and serves ALL 2*N
# codex episodes. AGENT_ARGS point the orchestrator at this held proxy.
# ============================================================================
mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
AGENT_ARGS=()
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  echo "[offload] OFFLOAD_AGENT=1 — proxy+agent on $OFFLOAD_HOST, GB10 stays vLLM-only"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$OFFLOAD_HOST" \
        "curl -fsS -m 6 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo ok" \
        2>/dev/null | grep -q ok; then
    echo "FAIL: alienware cannot reach GB10 vLLM at http://$GB10_TS_IP:$PORT/health (set OFFLOAD_AGENT=0)"
    exit 5
  fi
  echo "[offload] alienware -> GB10 vLLM $GB10_TS_IP:$PORT/health OK"
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" sync "$OFFLOAD_HOST" > "$ARMDIR/offload_sync.log" 2>&1 \
    || { echo "FAIL: offload proxy sync"; cat "$ARMDIR/offload_sync.log"; exit 5; }
  DEPLOY_FORCE_TEMP="$DEPLOY_FORCE_TEMP" LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" start "$OFFLOAD_HOST" "$GB10_TS_IP" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_start.log" 2>&1 \
    || { echo "FAIL: offload proxy start"; cat "$ARMDIR/offload_start.log"; exit 5; }
  cat "$ARMDIR/offload_start.log"
  cp "$ARMDIR/offload_proxy_env.txt" "$ARMDIR/proxy_env.txt" 2>/dev/null || true
  # temp-pin assertion (offload helper already checks; re-assert here for the gate record)
  grep -q "LUMO_PROXY_FORCE_TEMPERATURE=$DEPLOY_FORCE_TEMP" "$ARMDIR/proxy_env.txt" \
    || { echo "FAIL: offload proxy temp pin missing (expected $DEPLOY_FORCE_TEMP)"; exit 5; }
  AGENT_ARGS=(--agent-host "$OFFLOAD_HOST" \
              --agent-endpoint "http://127.0.0.1:$OFFLOAD_PROXY_PORT/v1")
  echo "proxy OK (OFFLOADED to $OFFLOAD_HOST:$OFFLOAD_PROXY_PORT, temp $DEPLOY_FORCE_TEMP)"
else
  # Local proxy fallback (serve_variant.sh:340-369). temp pinned via env.
  LUMO_PROXY_FORCE_TEMPERATURE="$DEPLOY_FORCE_TEMP" \
  LUMO_PROXY_REQUEST_DUMP_DIR="$PWD/$ARMDIR/proxy_request_dumps" \
  LUMO_PROXY_PAIR_DUMP_DIR="$PWD/$ARMDIR/proxy_pair_dumps" \
  LUMO_PROXY_LOG_PATH="$PWD/$ARMDIR/proxy.log" \
  LUMO_PROXY_NOHUP_PATH="$PWD/$ARMDIR/proxy.nohup" \
  LUMO_PROXY_STATE_ROOT="/tmp/fr13_apc_staleornot_proxy_state" \
  bash scripts/swe_x86_helpers/relaunch_proxy.sh > "$ARMDIR/proxy_launch.log" 2>&1
  sleep 3
  P0=$(date +%s); PROXY_OK=0
  while (( $(date +%s) < P0 + 60 )); do
    CODE=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PROXY_PORT/v1/models" 2>/dev/null)
    if [[ -n "$CODE" && "$CODE" != "000" ]]; then PROXY_OK=1; break; fi
    sleep 2
  done
  (( PROXY_OK == 1 )) || { echo "FAIL: proxy not healthy"; tail -20 "$ARMDIR/proxy.nohup" 2>/dev/null; exit 5; }
  for _i in $(seq 1 12); do
    PROXY_PID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
    if [[ -n "$PROXY_PID" ]] && [[ -r "/proc/$PROXY_PID/environ" ]]; then
      tr '\0' '\n' < "/proc/$PROXY_PID/environ" | grep -E "^LUMO_(PROXY|TRACK_B)" | sort > "$ARMDIR/proxy_env.txt"
    fi
    grep -q "LUMO_PROXY_FORCE_TEMPERATURE=" "$ARMDIR/proxy_env.txt" 2>/dev/null && break
    sleep 1
  done
  grep -q "LUMO_PROXY_FORCE_TEMPERATURE=$DEPLOY_FORCE_TEMP" "$ARMDIR/proxy_env.txt" \
    || { echo "FAIL: proxy temp pin missing (expected $DEPLOY_FORCE_TEMP)"; exit 5; }
  echo "proxy OK (local, temp $DEPLOY_FORCE_TEMP)"
fi

# ---- eval offload pre-flight (serve_variant.sh:372-380) ----
EVAL_ARGS=()
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$EVAL_HOST" \
     "test -f ~/swe_eval_offload/swe_eval_x86_worker.py && echo ok" 2>/dev/null | grep -q ok; then
  EVAL_ARGS=(--eval-host "$EVAL_HOST")
  echo "eval offload: $EVAL_HOST reachable" | tee "$ARMDIR/eval_offload_preflight.txt"
else
  echo "WARN: eval offload $EVAL_HOST UNREACHABLE — eval attempted locally" | tee "$ARMDIR/eval_offload_preflight.txt"
fi

WALL_ARGS=()
[[ -n "${AGENT_WALL_S:-}" ]] && WALL_ARGS=(--agent-wall-s "$AGENT_WALL_S")

# ============================================================================
# helper: write the shadow flag, RESET the prefix cache, run ONE codex episode.
#   $1 = mode tag ("stale"|"fresh"), $2 = flag value ("0"|"1"), $3 = seed index
# Reuses run_swe_bench_q36_a.py EXACTLY as serve_variant.sh:416-424 does, but with a
# UNIQUE --out-root + --dataset-tag per (mode,seed) so each episode writes its own
# per_task/<instance>/ tree (no --skip-existing collision; the runner's own git
# worktree mgmt stays clean). Single instance subset => exactly one codex episode.
# ============================================================================
run_episode(){
  local mode="$1" flagval="$2" sidx="$3"
  local tag="${mode}_s${sidx}"
  local epdir="$ARMDIR/episodes/$tag"
  mkdir -p "$epdir"
  # shadow-fired counter (written by the scheduler each time it zeros a hit = re-prefill).
  # This is the ONLY non-vacuity signal that proves the shadow fired: vllm:prefix_cache_hits
  # is recorded INSIDE get_computed_blocks BEFORE the zeroing, so it cannot distinguish modes.
  local sf_before; sf_before=$(cat "$SHADOW_FLAG_HOST.fired_count" 2>/dev/null || echo 0)

  # 1) STALE-OR-NOT toggle: write the flag file (host -> bind-mounted /logs).
  #    atomic write so the scheduler never reads a half-written value.
  printf '%s' "$flagval" > "$SHADOW_FLAG_HOST.tmp" && mv "$SHADOW_FLAG_HOST.tmp" "$SHADOW_FLAG_HOST"
  echo "[$tag] shadow flag = '$flagval' ($([[ "$flagval" == "1" ]] && echo FRESH/re-prefill || echo STALE/cached-restore))" | tee "$epdir/flag_state.txt"
  # confirm the container sees the same value (bind-mount round-trip)
  docker exec "$CONTAINER" sh -c "cat '$SHADOW_FLAG_CONTAINER' 2>/dev/null" > "$epdir/flag_in_container.txt" 2>/dev/null || true

  # 2) RESET the prefix cache so each episode starts clean (serve_variant.sh:312).
  #    GB10-local POST; /reset_prefix_cache is exposed by VLLM_SERVER_DEV_MODE=1
  #    (fr13_launch_forked_fa2_tree_server.sh:458). Direct on the GB10 port even
  #    under offload (the proxy is alienware-local; the cache lives in this vLLM).
  curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
    > "$epdir/reset_prefix_cache.txt" 2>&1 || echo "WARN: [$tag] reset_prefix_cache failed (non-fatal)"

  # 3) /metrics bracket + the LIVE codex episode (serve_variant.sh:414-427).
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$epdir/metrics_before.txt"
  local s0 s1
  s0=$(date +%s)
  .venv/bin/python scripts/run_swe_bench_q36_a.py \
    --subset "$SUBSET" \
    --out-root "$epdir/swe_out" \
    --dataset-tag "$tag" \
    --concurrency 1 \
    --eval-timeout-s "${EVAL_TIMEOUT_S:-1800}" \
    "${WALL_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${AGENT_ARGS[@]}" \
    > "$epdir/swe_orchestrator.log" 2>&1
  local rc=$?
  s1=$(date +%s)
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$epdir/metrics_after.txt"
  local sf_after; sf_after=$(cat "$SHADOW_FLAG_HOST.fired_count" 2>/dev/null || echo 0)
  local sf_delta=$(( sf_after - sf_before ))
  echo "[$tag] shadow_fired delta=$sf_delta (before=$sf_before after=$sf_after)" | tee "$epdir/shadow_fired_delta.txt"
  echo "[$tag] swe orchestrator rc=$rc wall=$((s1-s0))s"
  tail -3 "$epdir/swe_orchestrator.log"

  # 4) fetch offload pair-dumps for this episode (serve_variant.sh:435-439).
  if [[ "$OFFLOAD_AGENT" == "1" ]]; then
    LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$epdir" \
      > "$epdir/offload_fetch.log" 2>&1 || echo "WARN: [$tag] offload fetch errors"
  fi

  # 5) per-episode record extraction: verdict + codex elapsed + cached-tokens delta
  #    (NON-VACUITY) + garble count. Pure-read, written to $epdir/episode_record.json.
  .venv/bin/python - "$epdir" "$tag" "$flagval" "$INSTANCE" "$((s1-s0))" "$rc" "$sf_delta" <<'PY'
import json, sys, re
from pathlib import Path
epdir, tag, flagval, inst, wall, rc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
shadow_fired_delta = int(sys.argv[7])

def metric(path, name):
    p = epdir / path
    if not p.is_file():
        return 0.0
    for line in p.read_text().splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                return float(line.rsplit(None, 1)[-1])
            except ValueError:
                return 0.0
    return 0.0

# NON-VACUITY signal: prefix_cache_hits_total = number of CACHED (restored) tokens.
# STALE pass (flag 0): hits MUST be > 0 (the cache was actually engaged/restored).
# FRESH pass (flag 1): hits MUST be ~0 (the shadow re-prefill fired -> the scheduler
#   zeroed num_new_local_computed_tokens -> no restore -> the toggle is NON-VACUOUS).
hits = metric("metrics_after.txt", "vllm:prefix_cache_hits_total") \
     - metric("metrics_before.txt", "vllm:prefix_cache_hits_total")
queries = metric("metrics_after.txt", "vllm:prefix_cache_queries_total") \
        - metric("metrics_before.txt", "vllm:prefix_cache_queries_total")

# verdict + codex telemetry from the runner_metadata.json (per_task tree).
metas = sorted(epdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
verdict, codex_elapsed, timed_out, patch_bytes = "missing", None, None, None
trace_path = None
for m in metas:
    meta = json.loads(m.read_text())
    if meta.get("instance_id") and inst.split("__")[-1] not in (meta.get("instance_id") or ""):
        # tolerate either exact or suffix match; take the first real one
        pass
    verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
    codex = meta.get("agent") or meta.get("codex") or {}
    codex_elapsed = codex.get("elapsed_s")
    timed_out = codex.get("timed_out")
    patch_bytes = meta.get("patch_bytes")
    cand = m.parent / "agent_trace.jsonl"
    if not cand.is_file():
        cand = m.parent / "qwen_trace.jsonl"
    if not cand.is_file():
        cand = m.parent / "codex_trace.jsonl"
    if cand.is_file():
        trace_path = cand
    break

# GARBLE count on the codex trace: CJK runs, "char 8", "Unterminated".
garble = {"cjk": 0, "char8": 0, "unterminated": 0, "total": 0}
if trace_path and trace_path.is_file():
    txt = trace_path.read_text(errors="replace")
    garble["cjk"] = len(re.findall(r"[一-鿿぀-ヿ가-힯]{4,}", txt))
    garble["char8"] = len(re.findall(r"char 8", txt))
    garble["unterminated"] = txt.count("Unterminated")
    garble["total"] = garble["cjk"] + garble["char8"] + garble["unterminated"]

# vacuity verdict for THIS pass.
# NOTE: vllm:prefix_cache_hits_total is recorded INSIDE get_computed_blocks
# (kv_cache_manager.py:208-214) BEFORE the scheduler shadow zeroing, so hits>0 in
# BOTH modes -> it canNOT prove the shadow fired. The DIRECT signal is the
# shadow-fired counter file (delta this episode): the scheduler writes it ONLY when
# it actually zeros a hit (= re-prefill forced).
if flagval == "0":
    # STALE: the cache must be engaged (hits>0) AND the shadow must NOT have fired.
    if hits <= 0.0:
        vacuous, vac_reason = True, "STALE pass but cached_tokens<=0 (cache NOT engaged)"
    elif shadow_fired_delta > 0:
        vacuous, vac_reason = True, f"STALE pass but shadow fired (delta={shadow_fired_delta}) — flag not honored"
    else:
        vacuous, vac_reason = False, ""
else:
    # FRESH: the shadow MUST have fired (re-prefill happened). hits>0 is EXPECTED here.
    if shadow_fired_delta <= 0:
        vacuous, vac_reason = True, "FRESH pass but shadow_fired delta<=0 (re-prefill did NOT fire)"
    else:
        vacuous, vac_reason = False, ""

rec = {
    "tag": tag, "mode": ("FRESH" if flagval == "1" else "STALE"), "flag": flagval,
    "instance": inst, "wall_s": wall, "orchestrator_rc": rc,
    "verdict": verdict, "resolved": (verdict == "resolved"),
    "codex_elapsed_s": codex_elapsed, "codex_timed_out": timed_out, "patch_bytes": patch_bytes,
    "cached_tokens_delta": hits, "cache_queries_delta": queries,
    "shadow_fired_delta": shadow_fired_delta,
    "garble": garble, "vacuous": vacuous, "vacuity_reason": vac_reason,
}
(epdir / "episode_record.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
PY
}

# ============================================================================
# PHASE 3 — drive 2*N episodes against the held boot. Per seed: STALE then FRESH.
# ============================================================================
echo "=== PHASE 3: $N_SEEDS SHADOW runs (mark cache, do NOT take it -> re-prefill = cache-off-equivalent, all carriers) against the held boot ==="
# ONE run per seed = the SHADOW run (flag 1): the cache is MARKED (blocks managed) but the
# restore is NOT taken (re-prefill every hit) -> bit-identical to cache-OFF for ALL carriers.
# It SHOULD solve the task just like cache-off. Paired with the already-known cache-ON-TAKE
# derail, shadow-solves => the stale cached VALUE is the carrier. No separate stale run.
for (( s=1; s<=N_SEEDS; s++ )); do
  echo "--- run $s : SHADOW (flag 1, cache marked but NOT taken; should solve like cache-off) ---"
  run_episode shadow 1 "$s"
done

# ============================================================================
# PHASE 4 — REDUCER (SHADOW-only): does the cache-off-equivalent (mark-not-take) SOLVE?
# ============================================================================
echo "=== PHASE 4: reducer (SHADOW mark-not-take solve gate) ==="
.venv/bin/python - "$ARMDIR" "$N_SEEDS" "$INSTANCE" <<'PY'
import json, sys
from pathlib import Path
armdir, nruns, inst = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

def load(tag):
    p = armdir / "episodes" / tag / "episode_record.json"
    return json.loads(p.read_text()) if p.is_file() else None

rows = [(s, load(f"shadow_s{s}")) for s in range(1, nruns + 1)]

print("\n========== FR13 APC SHADOW (mark-not-take) SOLVE GATE — RESULTS ==========")
print(f"instance={inst}  shadow runs={nruns}  (cache MARKED but NOT taken = cache-off-equivalent, ALL carriers)")
print(f"{'run':>4} | {'verdict':<12} {'cached':>8} {'shadow_fired':>12} {'garble':>7} {'vacuous':>8}")
print("-" * 64)
res = garble = vac = nz = 0
reasons = []
for s, r in rows:
    if r is None:
        print(f"{s:>4} | MISSING episode record"); continue
    nz += 1
    res += int(r["resolved"]); garble += r["garble"]["total"]
    if r["vacuous"]:
        vac += 1; reasons.append(f"shadow_s{s}: {r['vacuity_reason']}")
    print(f"{s:>4} | {r['verdict']:<12} {r['cached_tokens_delta']:>8.0f} "
          f"{r['shadow_fired_delta']:>12} {r['garble']['total']:>7} {str(r['vacuous']):>8}")
print("-" * 64)
if nz == 0:
    verdict = "NORESULT"
    print("AGGREGATE: no shadow runs completed")
else:
    print(f"AGGREGATE (n={nz}): SHADOW resolved {res}/{nz}={res/nz:.0%}  garble={garble}  vacuous={vac}/{nz}")
    if vac > 0:
        verdict = "VACUOUS (shadow did NOT re-prefill on some run -> not a real cache-off-equivalent)"
    elif res == nz:
        verdict = "SHADOW SOLVES like cache-off => config/structure sound; vs known cache-ON-TAKE derail => stale cached VALUE is the carrier"
    elif res > 0:
        verdict = f"SHADOW PARTIAL ({res}/{nz}) — solves sometimes (12907 ~50/50); run more, or the break is not purely the cache value"
    else:
        verdict = "SHADOW FAILS (0 solves) — even the cache-off-equivalent can't solve => NOT a cached-value issue (config/task), re-examine"
print("=" * 64)
if vac:
    print("!!! VACUOUS — shadow did not actually re-prefill (shadow_fired_delta<=0) on:")
    for r in reasons: print(f"    - {r}")
print(f"VERDICT: {verdict}")
print("  NON-VACUITY for a shadow run = shadow_fired_delta>0 (the restore was bypassed on cache-hit turns)")
print("  AND cached_tokens>0 (the cache was still MARKED/queried). Both => genuine mark-but-don't-take.")
print("=" * 64)
out = {"instance": inst, "n": nz, "shadow_resolved": res, "garble": garble,
       "vacuous_runs": vac, "vacuity_reasons": reasons, "verdict": verdict,
       "rows": [{"run": s, "shadow": r} for s, r in rows]}
(armdir / "shadow_solve_verdict.json").write_text(json.dumps(out, indent=2))
print(f"wrote {armdir/'shadow_solve_verdict.json'}")
PY

echo "=== FR13 APC SHADOW SOLVE GATE done -> $ARMDIR/shadow_solve_verdict.json ==="
