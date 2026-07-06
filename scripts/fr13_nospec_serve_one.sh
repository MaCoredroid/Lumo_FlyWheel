#!/usr/bin/env bash
# FR13 NO-SPEC SWEServe single-arm boot+orchestrate — the per-arm worker called by
# scripts/fr13_nospec_cache_swe.sh. This is the NO-SPEC analog of
# scripts/fr13_bigdenom_swe_serve_variant.sh: it reuses that script's hygiene / health /
# proxy-launch / SWE-orchestrator / verdict-extraction scaffolding VERBATIM in structure,
# with exactly ONE thing changed — the BOOT:
#   spec arm:    scripts/fr13_launch_forked_fa2_tree_server.sh  (--speculative-config ALWAYS)
#   no-spec arm: scripts/fr10_launch_speed_server.sh + FR12_NO_SPECULATIVE_CONFIG=1
#                => that launcher OMITS --speculative-config (fr10_launch_speed_server.sh:300-302).
#
# Cache ON/OFF is selected by the caller via FR13_ENABLE_APC (1/0) + the matched cache env
# (MAMBA_BLOCK_SIZE=1024, MAMBA_SSM_CACHE_DTYPE=float32, CUDAGRAPH_MODE=PIECEWISE) exactly as
# the tree/spine arms set them.
#
# Usage: ARM=nospecon_r1 bash scripts/fr13_nospec_serve_one.sh <arm> <subset.json>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

ARM=${1:?usage: fr13_nospec_serve_one.sh <arm> <subset.json>}
SUBSET=${2:?subset json}

RUNROOT=${RUNROOT:-output/fr13_nospec_cache}
ARMDIR="$RUNROOT/$ARM"
[[ -f "$SUBSET" ]] || SUBSET="output/fr13_b1_gold_swe/$SUBSET"
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
CONTAINER="fr13-nospec-$ARM"
PORT=9950
PROXY_PORT=8022
EVAL_HOST=${EVAL_HOST:-alienware}

# --- codex offload (same machinery as fr13_bigdenom_swe_serve_variant.sh) ---
OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}
OFFLOAD_HOST=${OFFLOAD_HOST:-alienware}
OFFLOAD_PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
GB10_TS_IP=${GB10_TS_IP:-100.103.10.122}
OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh
OFFLOAD_LINK_DOWN_MAX_S=${OFFLOAD_LINK_DOWN_MAX_S:-300}
mkdir -p "$ARMDIR/logs"

MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
APC=${FR13_ENABLE_APC:-0}

echo "=== NO-SPEC SWEServe ARM $ARM cache=$([[ $APC == 1 ]] && echo ON || echo OFF) subset=$SUBSET ==="
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD 2>/dev/null | tee "$ARMDIR/git_head.txt"

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
  [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null
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

# ============================ BOOT (the ONLY divergence) ============================
# NO-SPEC base launcher. FR12_NO_SPECULATIVE_CONFIG=1 omits --speculative-config.
# FR10_DECODE_MODE_DEFAULT=naive_mtp keeps the tree GDN path OUT of the picture even
# though spec is off (defensive; with no spec config there is no draft path at all).
# Cache env (FR13_ENABLE_APC / MAMBA_* / CUDAGRAPH_MODE) is INHERITED from the caller's
# exported environment, matched to the tree/spine arms when cache-ON.
#
# CONFOUND-FLAG: the spec arms run with FULL CUDA-graph capture (no --enforce-eager) and
# ATTENTION_BACKEND=TREE_ATTN. This no-spec arm runs ATTENTION_BACKEND=FLASH_ATTN (the
# native decode backend; there is no tree to attend over) and KEEPS graph capture ON
# (ENFORCE_EAGER unset) so the cache path is the SAME captured-graph regime the spec
# cache-ON arm exercises (the launcher comments root-cause the cache garble to the
# captured FULL decode graph; CUDAGRAPH_MODE=PIECEWISE matches that mitigation). We do
# NOT force eager here: eager would change the cache path (no captured-graph GDN read)
# and confound the very thing under test. See RETURN notes for the full divergence list.
export FR12_NO_SPECULATIVE_CONFIG=1
export FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-naive_mtp}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASH_ATTN}
# BUGFIX: a ${VAR:+VAR=val} command-PREFIX does NOT work via expansion -- bash treats the
# expanded token (e.g. MAMBA_BLOCK_SIZE=1024) as a COMMAND, not an assignment prefix -> rc=127
# "command not found". MAMBA_*/CUDAGRAPH_MODE already flow via the caller's inherited exports;
# conditionally (re)export so the launcher's -e passthrough sees them when set, and they are
# absent (cache-OFF) when the caller unset them.
for _fr13_v in MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE CUDAGRAPH_MODE; do
  [ -n "${!_fr13_v:-}" ] && export "$_fr13_v"
done
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=${GPU_UTIL:-0.82} MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR10_METRICS=0 BATCH_INVARIANT=0 \
  FR13_ENABLE_APC="$APC" \
  FR10_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr10_launch_speed_server.sh > "$ARMDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -40 "$ARMDIR/launch.log"; exit 2; fi
# ===================================================================================

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

# ---- NO-SPEC engagement assert (the inverse of the variant's spec-engagement assert) ----
# The spec arms assert spec_decode_num_drafts_total INCREASES. Here we assert the engine
# is genuinely no-spec: spec metrics are ABSENT or never increment. This is the fail-loud
# anti-vacuity gate for THIS arm (a stray --speculative-config would invalidate the
# discriminator). The serve command in the launch log must NOT contain --speculative-config.
docker logs "$CONTAINER" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
if grep -q -- "--speculative-config" "$ARMDIR/boot_log_snapshot.txt" "$ARMDIR/launch.log" 2>/dev/null; then
  echo "FAIL: --speculative-config present in boot log — arm is NOT no-spec"; exit 3
fi
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_boot.txt" 2>/dev/null || true
if grep -qE "^vllm:spec_decode_num_drafts_total\{" "$ARMDIR/metrics_boot.txt" 2>/dev/null; then
  V=$(grep -E "^vllm:spec_decode_num_drafts_total\{" "$ARMDIR/metrics_boot.txt" | awk '{print $NF}' | tail -1)
  # presence with a NONZERO count would mean a spec path engaged -> fail loud.
  if [[ -n "$V" && "$V" != "0.0" && "$V" != "0" ]]; then
    echo "FAIL: spec_decode_num_drafts_total=$V nonzero at boot — spec engaged"; exit 3
  fi
fi
echo "no-spec engagement OK (no --speculative-config; spec drafts metric absent/zero)"

# ---- cache-config assert (only when cache-ON) ----
# BUGFIX: vLLM logs the PARSED config (underscores: enable_prefix_caching=True), NOT the raw
# --enable-prefix-caching flag, so the old dash-format grep was a FALSE NEGATIVE that killed
# healthy cache-ON boots. Assert the config-format strings vLLM actually emits. The critical
# anti-vacuity check is enable_prefix_caching=True (proves the cache is genuinely on); PIECEWISE
# proves the cudagraph mode matches the tree/spine arms. mamba_block_size/ssm_dtype are NOT
# logged greppably but ride in the same APC_FLAGS string -> guaranteed present iff prefix_caching
# is on; the "Prefix caching in Mamba cache" line confirms the mamba cache path engaged.
if [[ "$APC" == "1" ]]; then
  for need in "enable_prefix_caching=True" "enable_chunked_prefill=True" "PIECEWISE" "Prefix caching in Mamba cache"; do
    grep -qiF -- "$need" "$ARMDIR/boot_log_snapshot.txt" 2>/dev/null \
      || { echo "FAIL: cache-ON arm missing config: '$need' (vLLM config-format)"; exit 3; }
  done
  echo "cache-ON config OK (enable_prefix_caching=True + chunked_prefill=True + Mamba cache + PIECEWISE)"
fi

curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
  > "$ARMDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"

# ============================ PROXY + ORCHESTRATOR ============================
# IDENTICAL to fr13_bigdenom_swe_serve_variant.sh:311-426 (offload-gated proxy launch +
# eval pre-flight + /metrics-bracketed SWE window). The orchestrator + verdict path are
# byte-identical to the spec arms so the rate is apples-to-apples.
mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
AGENT_ARGS=()
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  echo "[offload] OFFLOAD_AGENT=1 — proxy+agent on $OFFLOAD_HOST, GB10 stays vLLM-only"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$OFFLOAD_HOST" \
        "curl -fsS -m 6 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo ok" \
        2>/dev/null | grep -q ok; then
    echo "FAIL: alienware cannot reach GB10 vLLM at http://$GB10_TS_IP:$PORT/health (set OFFLOAD_AGENT=0 to fall back)"
    exit 5
  fi
  echo "[offload] alienware -> GB10 vLLM $GB10_TS_IP:$PORT/health OK"
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" sync "$OFFLOAD_HOST" > "$ARMDIR/offload_sync.log" 2>&1 \
    || { echo "FAIL: offload proxy sync"; cat "$ARMDIR/offload_sync.log"; exit 5; }
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" start "$OFFLOAD_HOST" "$GB10_TS_IP" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_start.log" 2>&1 \
    || { echo "FAIL: offload proxy start"; cat "$ARMDIR/offload_start.log"; exit 5; }
  cat "$ARMDIR/offload_start.log"
  cp "$ARMDIR/offload_proxy_env.txt" "$ARMDIR/proxy_env.txt" 2>/dev/null || true
  AGENT_ARGS=(--agent-host "$OFFLOAD_HOST" \
              --agent-endpoint "http://127.0.0.1:$OFFLOAD_PROXY_PORT/v1")
  echo "proxy OK (OFFLOADED to $OFFLOAD_HOST:$OFFLOAD_PROXY_PORT)"
else
  LUMO_PROXY_FORCE_TEMPERATURE="${DEPLOY_FORCE_TEMP:-0.6}" \
  LUMO_PROXY_REQUEST_DUMP_DIR="$PWD/$ARMDIR/proxy_request_dumps" \
  LUMO_PROXY_PAIR_DUMP_DIR="$PWD/$ARMDIR/proxy_pair_dumps" \
  LUMO_PROXY_LOG_PATH="$PWD/$ARMDIR/proxy.log" \
  LUMO_PROXY_NOHUP_PATH="$PWD/$ARMDIR/proxy.nohup" \
  LUMO_PROXY_STATE_ROOT="/tmp/fr13_nospec_proxy_state_${ARM}" \
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
  grep -q "LUMO_PROXY_FORCE_TEMPERATURE=${DEPLOY_FORCE_TEMP:-0.6}" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy temp pin missing (expected ${DEPLOY_FORCE_TEMP:-0.6})"; exit 5; }
  grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy pair-dump pin missing"; exit 5; }
  echo "proxy OK"
fi

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
        echo "[$ts] LINK up" >> "$LINKLOG"; down_since=0
      else
        now=$(date +%s); [[ "$down_since" == "0" ]] && down_since=$now
        contig=$(( now - down_since ))
        echo "[$ts] LINK DOWN contig=${contig}s" >> "$LINKLOG"
        if (( contig > OFFLOAD_LINK_DOWN_MAX_S )); then
          echo "[$ts] LINK DEAD > ${OFFLOAD_LINK_DOWN_MAX_S}s — watchdog FAIL LOUD" | tee -a "$LINKLOG"
          echo "$ts contig=${contig}s" > "$LINK_DEAD_MARKER"; break
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
  --eval-timeout-s "${EVAL_TIMEOUT_S:-1800}" \
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

[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null && WATCHDOG_PID=""

if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_fetch.log" 2>&1 || echo "WARN: offload fetch errors (see offload_fetch.log)"
  cat "$ARMDIR/offload_fetch.log"
  if [[ -f "$LINK_DEAD_MARKER" ]]; then
    echo "OFFLOAD_LINK_DEAD: $(cat "$LINK_DEAD_MARKER") — window for $ARM DISCARDED" \
      | tee "$ARMDIR/SWE_DISCARDED.flag"
    SWERC=12
  fi
fi

# ---- health rule + verdict roll-up (same shape as the variant) ----
.venv/bin/python - "$ARMDIR" "$SWE_WALL" "$SWERC" <<'PY'
import json, sys
from pathlib import Path
armdir, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
metas = sorted(armdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
health = {"swe_orchestrator_rc": rc, "swe_window_wall_s": wall, "tasks": []}
for m in metas:
    meta = json.loads(m.read_text())
    codex = meta.get("agent") or meta.get("codex") or {}
    health["tasks"].append({"instance_id": meta.get("instance_id"),
        "codex_elapsed_s": codex.get("elapsed_s"),
        "codex_timed_out": codex.get("timed_out"),
        "patch_bytes": meta.get("patch_bytes"),
        "verdict": (meta.get("eval_report") or {}).get("verdict", "missing")})
(armdir / "health.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
PY

echo "ARM_DONE $ARM cache=$([[ $APC == 1 ]] && echo ON || echo OFF) swerc=$SWERC"
exit $SWERC
