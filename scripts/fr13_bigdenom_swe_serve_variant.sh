#!/usr/bin/env bash
# FR13 BIG-DENOMINATOR SWEServe VARIANT arm — generalizes fr13_bigdenom_swe_serve.sh
# to the speed-campaign candidate arms (the cat-shape swap + the two levers), while
# preserving the canonical hygiene / health / spec-engagement / pair-dump machinery.
#
# It is the EXACT deployment regime of fr13_bigdenom_swe_serve.sh (real SWE-Verified +
# codex agent loop, per-task /metrics brackets -> deploy-speed) with ONE thing varied:
# the boot config (which the deploy-speed reducer is agnostic to — it only needs the
# per-task vllm_metrics_pre/post.txt brackets + the correct --expected-tok-per-draft).
#
# ARMS (KIND):
#   cat9        = locked cat9 (== fr13_bigdenom_swe_serve.sh cat9; here for same-wall A/B)
#   cat9-opta   = locked cat9 + OPT-A (FR13_GB10_FP8_GEMV_CFG=1, lossless byte-identical)
#   cat9-opt1   = locked cat9 + OPT-1 (FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1)
#   cat6root    = R4 shape via the forked launcher TREE override (depth-5, 6 nodes, EXPECT 6)
#   cat10       = cat10 shape via the forked launcher TREE override (depth-5, 10 nodes, EXPECT 10)
#
# Each boots the LOCKED cat9 pipeline (FIX-1/2/3/A, REPLAY_ROUTE, FA2_TREE_BIAS,
# CONV_COMMITTED_PATH, LUMO_FB_KERNEL_ROWS=1 + PROJ_PAD_ROWS=16, BATCH_INVARIANT=0)
# except the one varied lever/shape. K1 (FR13_SCAN_ALIGN) NOT baked (asserted off).
#
# Usage: fr13_bigdenom_swe_serve_variant.sh <arm_name> <KIND> <subset.json>
#   AGENT_WALL_S (env): bound the codex wall for a DEV-iteration screen (e.g. 600).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

ARM=${1:?usage: fr13_bigdenom_swe_serve_variant.sh <arm> <KIND> <subset.json>}
KIND=${2:?cat9|cat9-opta|cat9-opt1|cat6root|cat10}
SUBSET=${3:?subset json}

RUNROOT=output/fr13_bigdenom_swe
ARMDIR="$RUNROOT/$ARM"
[[ -f "$SUBSET" ]] || SUBSET="output/fr13_b1_gold_swe/$SUBSET"
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
CONTAINER="fr13-bigdenom-$ARM"
PORT=9950
PROXY_PORT=8022
EVAL_HOST=${EVAL_HOST:-alienware}
mkdir -p "$ARMDIR/logs"

# Arm dispatch: launcher + TREE + extra flags + expected draft-token ratio.
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
CAT10_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]"
case "$KIND" in
  cat9)      LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=() ;;
  cat9-opta) LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=(FR13_GB10_FP8_GEMV_CFG=1) ;;
  cat9-opt1) LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=(FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1) ;;
  cat6root)  LAUNCHER=forked; TREEARG="$CAT6ROOT_TREE"; EXPECT_RATIO=6;  declare -a XFLAGS=() ;;
  cat10)     LAUNCHER=forked; TREEARG="$CAT10_TREE";    EXPECT_RATIO=10; declare -a XFLAGS=() ;;
  *) echo "FAIL: unknown KIND=$KIND"; exit 2 ;;
esac
PROBE_MODE=tree_mtp

echo "=== BIGDENOM-VARIANT SWEServe ARM $ARM kind=$KIND launcher=$LAUNCHER expect=$EXPECT_RATIO xflags=[${XFLAGS[*]:-none}] subset=$SUBSET ==="
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD | tee "$ARMDIR/git_head.txt"

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

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

OLDPID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
[[ -n "$OLDPID" ]] && kill "$OLDPID" 2>/dev/null
pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
sleep 1

teardown(){
  echo "[teardown] kill proxy + docker rm -f $CONTAINER + recover_host_memory"
  kill "$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null)" 2>/dev/null
  pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
  docker logs "$CONTAINER" > "$ARMDIR/docker_full.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  recover_host || true
  sleep 2
  free -g | tee "$ARMDIR/free_after_teardown.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_ended_at.txt"
}
trap teardown EXIT

# ---- boot server (class 11: everything pinned except the arm lever/shape) ----
# extra flags exported into THIS shell so the launcher's docker -e picks them up.
for kv in "${XFLAGS[@]:-}"; do [[ -n "$kv" ]] && export "$kv"; done
if [[ "$LAUNCHER" == "locked" ]]; then
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_locked.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
else
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  TREE="$TREEARG" FR10_METRICS=0 BATCH_INVARIANT=0 \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
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

# ---- class 9: flag state in container env ----
docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
NEEDS=("FR13_DRAFTER_SINGLE_LOGITS=1" "FR13_EAGER_PACK=1" "FR13_TREE_CONV_FUSED=1" \
       "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" \
       "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1" "FR13_CONV_COMMITTED_PATH=1" \
       "FR10_DECODE_MODE_DEFAULT=tree_mtp" \
       "LUMO_FB_KERNEL_ROWS=1" "LUMO_FB_PROJ_PAD_ROWS=16")
if grep -q "^FR13_SCAN_ALIGN=1$" "$ARMDIR/container_env.txt"; then
  echo "FAIL: FR13_SCAN_ALIGN=1 present — K1 must NOT be baked"; exit 3
fi
for need in "${NEEDS[@]}"; do
  grep -q "^$need$" "$ARMDIR/container_env.txt" || { echo "FAIL: env pin missing: $need"; exit 3; }
done
# arm-specific OPT flag must be LIVE in container env
for kv in "${XFLAGS[@]:-}"; do
  [[ -n "$kv" ]] && { grep -q "^$kv$" "$ARMDIR/container_env.txt" || { echo "FAIL: OPT flag not live: $kv"; exit 3; }; }
done
echo "container env OK ($KIND)"

# ---- worker /proc environ bridge-needle ----
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
    tr "\0" "\n" < "$e" | grep -E "^(FR10_DECODE_MODE_DEFAULT|LUMO_FB_KERNEL_ROWS|FR13_GB10_FP8_GEMV_CFG|FR13_GPU_COMMITTER|FR13_COMMITTER_SYNCKILL|SPEC_CONFIG|FR13_REPLAY_ROUTE)=" | sed "s/^/   /"
    hit="1"
  fi
done
echo "SUMMARY worker_env_seen=[$hit] pids=[$pids]"
' | tee "$ARMDIR/worker_environ_needle.txt"
grep -q "worker_env_seen=\[1\]" "$ARMDIR/worker_environ_needle.txt" \
  || { echo "FAIL: no vLLM worker with FR10_DECODE_MODE_DEFAULT in /proc environ"; exit 3; }
echo "worker environ needle OK ($KIND)"

# ---- FULL CUDA capture needle ----
docker logs "$CONTAINER" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
grep -m1 "Graph capturing finished" "$ARMDIR/boot_log_snapshot.txt" \
  || { echo "FAIL: no 'Graph capturing finished' in boot log"; exit 3; }

# ---- OPT engagement needle in boot/serve log (OPT-1 logs once; OPT-A patcher prints) ----
case "$KIND" in
  cat9-opt1)
    # the synckill needle fires on first committed event during warmup/codex; check post-warmup
    echo "[needle] OPT-1 synckill engagement checked after warmup (see docker_full.log)";;
  cat9-opta)
    grep -m1 "FR13_GB10_FP8_GEMV_CFG" "$ARMDIR/boot_log_snapshot.txt" >/dev/null 2>&1 \
      && echo "[needle] OPT-A patch present in boot log" || echo "[needle] OPT-A patch line not in boot log (override engages at forward time on GB10+small-M)";;
esac

# ---- warmup probe (fires spec engagement) ----
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

curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
  > "$ARMDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"

# ---- launch canonical proxy (forced temp 0.0 + pair dumps) ----
mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
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
(( PROXY_OK == 1 )) || { echo "FAIL: proxy not healthy"; tail -20 "$ARMDIR/proxy.nohup" 2>/dev/null; exit 5; }
grep -q "LUMO_PROXY_FORCE_TEMPERATURE=0.0" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy temp pin missing"; exit 5; }
grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy pair-dump pin missing"; exit 5; }
echo "proxy OK"

# ---- eval offload pre-flight ----
EVAL_ARGS=()
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$EVAL_HOST" \
     "test -f ~/swe_eval_offload/swe_eval_x86_worker.py && echo ok" 2>/dev/null | grep -q ok; then
  EVAL_ARGS=(--eval-host "$EVAL_HOST")
  echo "eval offload: $EVAL_HOST reachable" | tee "$ARMDIR/eval_offload_preflight.txt"
else
  echo "WARN: eval offload $EVAL_HOST UNREACHABLE — eval attempted locally" | tee "$ARMDIR/eval_offload_preflight.txt"
fi

# ---- SWE window: /metrics bracketing + codex loop ----
WALL_ARGS=()
[[ -n "${AGENT_WALL_S:-}" ]] && WALL_ARGS=(--agent-wall-s "$AGENT_WALL_S")
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_swe.txt"
S0=$(date +%s)
.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$SUBSET" \
  --out-root "$ARMDIR/swe_out" \
  --concurrency 1 \
  "${WALL_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  > "$ARMDIR/swe_orchestrator.log" 2>&1
SWERC=$?
S1=$(date +%s)
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_swe.txt"
SWE_WALL=$((S1-S0))
echo "swe orchestrator rc=$SWERC wall=${SWE_WALL}s"
tail -5 "$ARMDIR/swe_orchestrator.log"

# ---- OPT-1 post-run engagement needle ----
if [[ "$KIND" == "cat9-opt1" ]]; then
  docker logs "$CONTAINER" 2>&1 | grep -m1 "FR13_COMMITTER_SYNCKILL engaged" \
    | tee "$ARMDIR/opt1_synckill_needle.txt" \
    || { echo "FAIL: OPT-1 synckill never engaged (vacuous arm)"; SWERC=9; }
fi

# ---- health rule + pair-dump non-vacuity ----
.venv/bin/python - "$ARMDIR" "$SWE_WALL" "$SWERC" <<'PY'
import json, sys
from pathlib import Path
armdir, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
metas = sorted(armdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
health = {"swe_orchestrator_rc": rc, "swe_window_wall_s": wall, "tasks": []}
for m in metas:
    meta = json.loads(m.read_text())
    codex = meta.get("codex") or {}
    health["tasks"].append({"instance_id": meta.get("instance_id"),
        "codex_elapsed_s": codex.get("elapsed_s"),
        "codex_timed_out": codex.get("timed_out"),
        "patch_bytes": meta.get("patch_bytes"),
        "verdict": (meta.get("eval_report") or {}).get("verdict", "missing")})
(armdir / "health.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
PY

NPAIR=$(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)
echo "pair dumps captured: $NPAIR"

echo "ARM_DONE $ARM kind=$KIND swerc=$SWERC"
exit $SWERC
