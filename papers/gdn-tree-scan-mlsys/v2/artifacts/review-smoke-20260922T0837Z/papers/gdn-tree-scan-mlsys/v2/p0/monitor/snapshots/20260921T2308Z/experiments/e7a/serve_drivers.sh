#!/usr/bin/env bash
# E7a step-2 serving drivers (fresh-prefix pilot). Every boot uses a UNIQUE owned container name, its own LOG_DIR
# under this worktree, the read-only /models mount the launchers already use, and records: served_model.json,
# repo_identity.json, launcher sha256 (+ diff for the derived launcher), docker inspect (image digest, env, cmd),
# host resources before/after, and `docker logs`. Nothing here touches other containers or processes.
#
# Usage:
#   serve_drivers.sh native-score <run_dir> <pool.json>           boot native stock MTP-5 reference, score pool, select, stop
#   serve_drivers.sh capture <run_dir> <prefix_id> <pool.json> <layer_prefix>   one capture boot for one frozen prefix
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"

precheck() {  # resources + no foreign compute + memory gates (read-only report; the launchers enforce their own)
  local d=$1
  { echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g; swapon --show || true
    echo "--- compute apps ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    echo "--- containers ---"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' || true
    echo "--- vllm/EngineCore procs ---"; { ps -eo pid,etime,rss,args | grep -i "vllm\|EngineCore" | grep -v grep | cut -c1-200; } || echo "(none)"
  } > "$d" 2>&1
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING: a compute process is present on the GPU (see $d)" >&2; return 3
  fi
}

wait_health() {  # $1 port, $2 container, $3 timeout s
  local port=$1 c=$2 t=$3 i=0
  while (( i < t )); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then echo "healthy after ${i}s"; return 0; fi
    if [[ -z "$(docker ps -q -f name="^${c}$")" ]]; then echo "container $c exited"; docker logs "$c" 2>&1 | tail -40; return 4; fi
    sleep 5; i=$((i+5))
  done
  echo "health timeout"; docker logs "$c" 2>&1 | tail -40; return 5
}

record_boot() {  # $1 run_dir $2 container $3 launcher
  local d=$1 c=$2 l=$3
  docker inspect "$c" > "$d/docker_inspect.json" 2>&1 || true
  sha256sum "$l" > "$d/launcher.sha256"
  git -C "$WT" rev-parse HEAD > "$d/worktree_head.txt"
  { echo "image_digest=$IMAGE"; docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}' ; } > "$d/image_identity.txt" 2>&1 || true
}

stop_container() {  # only OUR container
  local c=$1 d=$2
  docker logs "$c" > "$d/docker_logs.txt" 2>&1 || true
  docker stop -t 30 "$c" >/dev/null 2>&1 || true
  docker rm -f "$c" >/dev/null 2>&1 || true
}

# The launchers' recover_host_memory() needs sudo; the project's documented credential file is sourced if present
# (never echoed). Without it the routine silently skips and the launcher's own memory gate still decides.
[[ -f /home/mark/shared/lumoFlyWheel/.lumo.local.env ]] && { set +u; . /home/mark/shared/lumoFlyWheel/.lumo.local.env; set -u; export LUMO_SUDO_PASSWORD="${LUMO_SUDO_PASSWORD:-}"; }

arm_oom_guard() {  # protects the login session from a unified-memory OOM; scoped to OUR container name only
  local c=$1
  GPU_GUARD_NAME_GLOB="$c" setsid bash "$WT/scripts/gpu_oom_guard.sh" >/dev/null 2>&1 </dev/null &
  disown 2>/dev/null || true
}

cmd=$1; shift
case "$cmd" in
  native-score)
    RUN=$1; POOL=$2; mkdir -p "$RUN/logs"
    C="e7a-native-$(date -u +%H%M%S)"; PORT=9951
    precheck "$RUN/precheck_before.txt"
    # launcher-owned recovery (drop caches + swapoff/swapon) happens INSIDE the launcher; recorded here as such
    env REPO="$WT" CONTAINER="$C" PORT="$PORT" GPU_UTIL="${GPU_UTIL:-0.6}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
        MAX_NUM_SEQS=1 GPU_OOM_GUARD=1 LOG_DIR="$RUN/logs" SEED=20260921 \
        bash "$WT/scripts/fr13_launch_native_mtp_server.sh" > "$RUN/launcher_stdout.txt" 2>&1
    record_boot "$RUN" "$C" "$WT/scripts/fr13_launch_native_mtp_server.sh"
    wait_health "$PORT" "$C" 1500 | tee "$RUN/health.txt"
    curl -s "http://127.0.0.1:$PORT/v1/models" > "$RUN/models.json" || true
    # score inside the SAME image environment? No GPU needed for HTTP; use host python3 (stdlib only)
    python3 "$E/prefix_pool.py" score --pool "$POOL" --server "http://127.0.0.1:$PORT" --model qwen3.6-27b --out "$RUN/prefix_scores.json" | tee "$RUN/score_stdout.txt"
    python3 "$E/prefix_pool.py" select --scores "$RUN/prefix_scores.json" --out "$RUN/frozen_prefixes.json" | tee "$RUN/select_stdout.txt"
    stop_container "$C" "$RUN"
    precheck "$RUN/precheck_after.txt" || true
    echo "native-score done: $RUN"
    ;;
  capture)
    RUN=$1; PID=$2; POOL=$3; LAYER=$4; mkdir -p "$RUN/logs"
    C="e7a-capture-$(date -u +%H%M%S)"; PORT=9950
    precheck "$RUN/precheck_before.txt"
    cp "$E/e7a_capture_launch.diff" "$RUN/launcher.diff"; sha256sum "$WT/scripts/fr10_launch_speed_server.sh" >> "$RUN/launcher.sha256.orig"
    env REPO="$WT" CONTAINER="$C" PORT="$PORT" GPU_UTIL="${GPU_UTIL:-0.6}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" MAX_NUM_SEQS=1 \
        ATTENTION_BACKEND=TREE_ATTN FR13_REPLAY_ROUTE=0 FR10_METRICS=1 \
        FR10_TREE_GDN_CAPTURE_PAYLOAD=/logs/tree_gdn_capture_payload.pt \
        FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX="$LAYER" \
        FR13_FINAL_LOGIT_CAPTURE=/logs/tree_final_logits.pt FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS=10 FR13_FINAL_LOGIT_CAPTURE_LIMIT=4 \
        LUMO_TREE_PATH_LCP_LOG=/logs/tree_path_lcp.jsonl LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl \
        LOG_DIR="$RUN/logs" \
        bash "$E/e7a_capture_launch.sh" > "$RUN/launcher_stdout.txt" 2>&1
    record_boot "$RUN" "$C" "$E/e7a_capture_launch.sh"
    arm_oom_guard "$C"
    wait_health "$PORT" "$C" 1500 | tee "$RUN/health.txt"
    python3 - "$POOL" "$PID" "$PORT" "$RUN" <<'PY'
import json, sys, urllib.request, time, hashlib
pool, pid, port, run = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pfx = next(p for p in json.load(open(pool))["prefixes"] if p["id"] == pid)
body = {"model": "qwen3.6-27b", "prompt": pfx["text"], "max_tokens": 32, "temperature": 0.0, "seed": 20260921, "logprobs": 5}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    out = json.load(r)
json.dump({"prefix_id": pid, "prefix_sha256": pfx["prefix_sha256"], "request": {k: v for k, v in body.items() if k != "prompt"},
           "prompt_sha256": hashlib.sha256(pfx["text"].encode()).hexdigest(), "response": out, "latency_s": time.time() - t0},
          open(f"{run}/capture_request.json", "w"), indent=1)
print("completion ok; tokens:", out.get("usage"))
PY
    for i in $(seq 1 60); do [[ -s "$RUN/logs/tree_gdn_capture_payload.pt" ]] && break; sleep 2; done
    ls -la "$RUN/logs" > "$RUN/logs_listing.txt"
    curl -s "http://127.0.0.1:$PORT/metrics" > "$RUN/metrics_after.txt" 2>/dev/null || true
    stop_container "$C" "$RUN"
    precheck "$RUN/precheck_after.txt" || true
    [[ -s "$RUN/logs/tree_gdn_capture_payload.pt" ]] && echo "capture done: $RUN" || { echo "capture MISSING: $RUN" >&2; exit 6; }
    ;;
  *) echo "unknown cmd $cmd" >&2; exit 2 ;;
esac
