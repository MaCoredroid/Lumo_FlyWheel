#!/usr/bin/env bash
# E7a step-2 serving drivers (fresh-prefix pilot). Hardened per review 05:
#   * fail-closed GPU inventory query (query failure != empty inventory), dedicated-port check, health must belong
#     to OUR container (docker port mapping), RAM-vs-used-swap guard BEFORE the launcher's own swapoff/swapon;
#   * EXIT/ERR trap saves the owned container's logs/identity and removes ONLY that container;
#   * derived launchers with READ-ONLY /models mounts (e7a_native_launch.sh, e7a_capture_launch.sh; diffs recorded);
#   * no credential is sourced or copied anywhere (passwordless `sudo -n` serves the launchers' recover_host_memory);
#   * capture provenance: the one-shot payload must NOT exist after health (else it was consumed by boot/warmup ->
#     INVALID), must exist after the single real request, and is bound to request id, prompt token count, prefix
#     hash, layer, topology, and the per-request logs written by the server.
# Usage:
#   serve_drivers.sh native-score <run_dir> <pool.json>
#   serve_drivers.sh capture <run_dir> <prefix_id> <pool.json> <layer_prefix>
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
C=""; RUN=""

precheck() {  # $1 report path ; fails closed
  local d=$1 apps rc
  { echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g; swapon --show || true
    echo "--- containers ---"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' || true
    echo "--- vllm/EngineCore procs ---"; { ps -eo pid,etime,rss,args | grep -i "vllm\|EngineCore" | grep -v grep | cut -c1-200; } || echo "(none)"
  } > "$d" 2>&1
  set +e; apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1); rc=$?; set -e
  echo "--- compute apps (nvidia-smi rc=$rc) ---" >> "$d"; echo "$apps" >> "$d"
  if (( rc != 0 )); then echo "REFUSING: nvidia-smi compute-apps query FAILED (rc=$rc): $apps" >&2; return 3; fi
  if [[ -n "$(echo "$apps" | tr -d '[:space:]')" ]]; then echo "REFUSING: a compute process is present on the GPU (see $d)" >&2; return 3; fi
  local avail_kib swap_used_kib
  avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); swap_used_kib=$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {print t-f}' /proc/meminfo)
  echo "MemAvailable_GiB=$((avail_kib/1048576)) SwapUsed_GiB=$((swap_used_kib/1048576))" >> "$d"
  if (( swap_used_kib > 0 && avail_kib < swap_used_kib + 8388608 )); then echo "REFUSING: not enough RAM to swap in (avail=$((avail_kib/1048576))GiB, swap used=$((swap_used_kib/1048576))GiB)" >&2; return 3; fi
}

port_free() { ! (ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"); }

wait_health() {  # $1 port, $2 container, $3 timeout s ; health must belong to $2
  local port=$1 c=$2 t=$3 i=0 mapped
  while (( i < t )); do
    if [[ -z "$(docker ps -q -f name="^${c}$")" ]]; then echo "container $c exited"; docker logs "$c" 2>&1 | tail -40; return 4; fi
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      mapped=$(docker port "$c" 9950/tcp 2>/dev/null | head -1 || true)
      if [[ "$mapped" != *":$port" ]]; then echo "health answered on :$port but container $c maps '$mapped'"; return 7; fi
      echo "healthy after ${i}s (container $c maps $mapped)"; return 0
    fi
    sleep 5; i=$((i+5))
  done
  echo "health timeout"; docker logs "$c" 2>&1 | tail -40; return 5
}

record_boot() {  # $1 run_dir $2 container $3 launcher
  local d=$1 c=$2 l=$3
  docker inspect "$c" > "$d/docker_inspect.json" 2>&1 || true
  sha256sum "$l" > "$d/launcher.sha256"
  git -C "$WT" rev-parse HEAD > "$d/worktree_head.txt"
  { echo "image_digest=$IMAGE"; docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}'; } > "$d/image_identity.txt" 2>&1 || true
  docker inspect "$c" --format '{{range .Mounts}}{{.Source}}->{{.Destination}} rw={{.RW}}{{"\n"}}{{end}}' > "$d/mounts.txt" 2>&1 || true
  if grep -q "^/models->/models rw=true" "$d/mounts.txt"; then echo "REFUSING: /models mounted read-write" >&2; return 8; fi
}

cleanup() {  # EXIT/ERR: save logs/identity of OUR container only, then remove it; never touch other containers
  local rc=$?
  if [[ -n "$C" ]] && docker ps -a -q -f name="^${C}$" | grep -q .; then
    docker logs "$C" > "$RUN/docker_logs.txt" 2>&1 || true
    docker inspect "$C" > "$RUN/docker_inspect_final.json" 2>&1 || true
    docker stop -t 30 "$C" >/dev/null 2>&1 || true
    docker rm -f "$C" >/dev/null 2>&1 || true
    echo "cleanup: removed owned container $C (rc=$rc)" >> "$RUN/driver_trace.txt"
  fi
  [[ -n "$RUN" ]] && { echo "exit_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" >> "$RUN/driver_trace.txt"; precheck "$RUN/precheck_after.txt" >/dev/null 2>&1 || true; }
  exit $rc
}
trap cleanup EXIT

arm_oom_guard() {  # protects the login session from a unified-memory OOM; scoped to OUR container name only
  GPU_GUARD_NAME_GLOB="$1" setsid bash "$WT/scripts/gpu_oom_guard.sh" >/dev/null 2>&1 </dev/null &
  disown 2>/dev/null || true
}

cmd=$1; shift
case "$cmd" in
  native-score)
    RUN=$1; POOL=$2; mkdir -p "$RUN/logs"; PORT=9951
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cmd=native-score" > "$RUN/driver_trace.txt"
    precheck "$RUN/precheck_before.txt"
    port_free "$PORT" || { echo "REFUSING: port $PORT in use" >&2; exit 9; }
    C="e7a-native-$(date -u +%H%M%S)"
    cp "$E/e7a_native_launch.diff" "$RUN/launcher.diff"; sha256sum "$WT/scripts/fr13_launch_native_mtp_server.sh" > "$RUN/launcher.sha256.orig"
    env REPO="$WT" CONTAINER="$C" PORT="$PORT" GPU_UTIL="${GPU_UTIL:-0.6}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
        MAX_NUM_SEQS=1 GPU_OOM_GUARD=1 LOG_DIR="$RUN/logs" SEED=20260921 \
        bash "$E/e7a_native_launch.sh" > "$RUN/launcher_stdout.txt" 2>&1
    record_boot "$RUN" "$C" "$E/e7a_native_launch.sh"
    wait_health "$PORT" "$C" 1500 | tee "$RUN/health.txt"
    curl -s "http://127.0.0.1:$PORT/v1/models" > "$RUN/models.json" || true
    python3 "$E/prefix_pool.py" score --pool "$POOL" --server "http://127.0.0.1:$PORT" --model qwen3.6-27b --out "$RUN/prefix_scores.json" | tee "$RUN/score_stdout.txt"
    python3 "$E/prefix_pool.py" select --scores "$RUN/prefix_scores.json" --pool "$POOL" --out "$RUN/frozen_prefixes.json" ${SELECT_EXTRA:-} | tee "$RUN/select_stdout.txt"
    echo "native-score done: $RUN"
    ;;
  capture)
    RUN=$1; PID=$2; POOL=$3; LAYER=$4; mkdir -p "$RUN/logs"; PORT=9950
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cmd=capture prefix=$PID layer=$LAYER" > "$RUN/driver_trace.txt"
    precheck "$RUN/precheck_before.txt"
    port_free "$PORT" || { echo "REFUSING: port $PORT in use" >&2; exit 9; }
    C="e7a-capture-$(date -u +%H%M%S)"
    cp "$E/e7a_capture_launch.diff" "$RUN/launcher.diff"; sha256sum "$WT/scripts/fr10_launch_speed_server.sh" > "$RUN/launcher.sha256.orig"
    sha256sum "$E/e7a_capture_shim.py" > "$RUN/shim.sha256"
    env REPO="$WT" CONTAINER="$C" PORT="$PORT" GPU_UTIL="${GPU_UTIL:-0.6}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" MAX_NUM_SEQS=1 \
        ATTENTION_BACKEND=TREE_ATTN FR13_REPLAY_ROUTE=1 E7A_CAPTURE_SHIM=1 FR10_METRICS=1 \
        FR10_TREE_GDN_CAPTURE_PAYLOAD=/logs/tree_gdn_capture_payload.pt \
        FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX="$LAYER" FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS="${CAPTURE_NUM_TOKENS:-10}" \
        FR13_FINAL_LOGIT_CAPTURE=/logs/tree_final_logits.pt FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS=10 FR13_FINAL_LOGIT_CAPTURE_LIMIT=4 \
        LUMO_TREE_PATH_LCP_LOG=/logs/tree_path_lcp.jsonl LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl \
        LUMO_MTP_DRAFT_TRACE_FILE=/logs/fr10_mtp_draft_trace.jsonl \
        LOG_DIR="$RUN/logs" \
        bash "$E/e7a_capture_launch.sh" > "$RUN/launcher_stdout.txt" 2>&1
    record_boot "$RUN" "$C" "$E/e7a_capture_launch.sh"
    arm_oom_guard "$C"
    wait_health "$PORT" "$C" 1500 | tee "$RUN/health.txt"
    sleep 5
    if [[ -e "$RUN/logs/tree_gdn_capture_payload.pt" ]]; then
      echo "INVALID: one-shot capture consumed before the real request (boot/warmup)" | tee "$RUN/CAPTURE_INVALID_warmup.txt" >&2
      mv "$RUN/logs/tree_gdn_capture_payload.pt" "$RUN/logs/tree_gdn_capture_payload.CONSUMED_BY_WARMUP.pt"
      exit 6
    fi
    ls -la "$RUN/logs" > "$RUN/logs_listing_before_request.txt"
    for f in tree_path_lcp.jsonl tree_sampler_debug.jsonl fr10_mtp_draft_trace.jsonl fr10_tree_depth_positions.jsonl; do
      if [[ -f "$RUN/logs/$f" ]]; then wc -l < "$RUN/logs/$f" > "$RUN/logs/$f.lines_before_request"; else echo 0 > "$RUN/logs/$f.lines_before_request"; fi
    done
    echo "request_start_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$RUN/driver_trace.txt"
    python3 - "$POOL" "$PID" "$PORT" "$RUN" <<'PY'
import json, sys, urllib.request, time, hashlib
pool, pid, port, run = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pfx = next(p for p in json.load(open(pool))["prefixes"] if p["id"] == pid)
body = {"model": "qwen3.6-27b", "prompt": pfx["text"], "max_tokens": 32, "temperature": 0.0, "seed": 20260921, "logprobs": 5}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    out = json.load(r)
json.dump({"prefix_id": pid, "prefix_sha256": pfx["prefix_sha256"], "prefix_tokens_pool": pfx["tokens"], "source": pfx["source"],
           "request": {k: v for k, v in body.items() if k != "prompt"}, "prompt_sha256": hashlib.sha256(pfx["text"].encode()).hexdigest(),
           "response_id": out.get("id"), "usage": out.get("usage"), "response_text": out["choices"][0].get("text"),
           "response_logprobs_first": (out["choices"][0].get("logprobs") or {}).get("top_logprobs", [None])[0],
           "latency_s": time.time() - t0}, open(f"{run}/capture_request.json", "w"), indent=1)
print("completion ok; id:", out.get("id"), "usage:", out.get("usage"))
PY
    echo "request_end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$RUN/driver_trace.txt"
    for i in $(seq 1 60); do [[ -s "$RUN/logs/tree_gdn_capture_payload.pt" ]] && break; sleep 2; done
    stat -c 'payload_mtime_utc=%y size=%s' "$RUN/logs/tree_gdn_capture_payload.pt" >> "$RUN/driver_trace.txt" 2>/dev/null || echo "payload_missing" >> "$RUN/driver_trace.txt"
    cp "$RUN/logs/e7a_capture_shim.json" "$RUN/e7a_capture_shim.json" 2>/dev/null || echo "(shim report missing)" > "$RUN/e7a_capture_shim.MISSING"
    ls -la "$RUN/logs" > "$RUN/logs_listing_after_request.txt"
    curl -s "http://127.0.0.1:$PORT/metrics" > "$RUN/metrics_after.txt" 2>/dev/null || true
    [[ -s "$RUN/logs/tree_gdn_capture_payload.pt" ]] && echo "capture done: $RUN" || { echo "capture MISSING: $RUN" >&2; exit 6; }
    ;;
  *) echo "unknown cmd $cmd" >&2; exit 2 ;;
esac
