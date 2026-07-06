#!/bin/bash
# FR13 CODEX-OFFLOAD orchestration (runs on the GB10). Subcommands:
#
#   sync   <host>                  rsync proxy code (src/ + registry + remote launcher) to alienware
#   start  <host> <gb10_ip> <armdir>   launch the proxy on alienware -> GB10:9950, health-check it
#   fetch  <host> <armdir>        rsync the alienware pair/request dumps back into <armdir>
#   stop   <host>                 kill the remote proxy on alienware
#
# WHY: the deployment harness puts (2) the inference_proxy and (3) the codex-runner
# docker on the GB10, where they steal unified-memory bandwidth (273 GB/s Grace+
# Blackwell) + CPU from vLLM and CONTAMINATE the timing-sensitive deploy-speed
# numbers. Moving (2)+(3) to alienware leaves the GB10 running ONLY vLLM. The
# lossless (argmax/distributional) numbers are timing-independent so unaffected;
# only speed is. The canonical measurement BASIS is unchanged — only WHERE the
# codex+proxy compute runs. vLLM stays on the GB10 (the GPU).
#
# Reuses the eval-offload SSH/rsync plumbing pattern (run_swe_bench_q36_a.py:392-432).
set -uo pipefail

REPO_ROOT=/home/mark/shared/lumoFlyWheel
REMOTE_ROOT='~/lumo_proxy_offload'                 # repo mirror on alienware
REMOTE_REPO="$REMOTE_ROOT/repo"
REMOTE_VENV='~/swe_eval_offload/venv/bin/python'   # reuse the eval-offload venv (has requests+yaml)
REMOTE_PAIR_DUMPS="$REMOTE_ROOT/pair_dumps"
REMOTE_REQ_DUMPS="$REMOTE_ROOT/request_dumps"
# alienware-local proxy port. 8022 is OCCUPIED on alienware by an unrelated
# root-owned host service (the lumo-alpha-dev --network=host container env), so
# the offload proxy listens on 8023 there. This is alienware-LOCAL: the GB10
# side (vLLM :9950) and the legacy on-GB10 proxy (:8022) are unaffected.
PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 \
          -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new)

CMD=${1:?usage: offload_codex_proxy.sh <sync|start|fetch|stop> ...}
shift

case "$CMD" in
  sync)
    HOST=${1:?host}
    # create the remote tree, then rsync only what the proxy needs.
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE_REPO/src $REMOTE_PAIR_DUMPS $REMOTE_REQ_DUMPS && echo ok" >/dev/null
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
      "$REPO_ROOT/src/lumo_flywheel_serving/" \
      "$HOST:$REMOTE_REPO/src/lumo_flywheel_serving/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$REPO_ROOT/model_registry.yaml" \
      "$REPO_ROOT/scripts/swe_x86_helpers/relaunch_proxy_remote.sh" \
      "$HOST:$REMOTE_REPO/"
    ssh "${SSH_OPTS[@]}" "$HOST" "chmod +x $REMOTE_REPO/relaunch_proxy_remote.sh && \
      $REMOTE_VENV -c 'import requests,yaml;print(\"proxy-deps-ok\")'" \
      || { echo "FAIL: alienware proxy deps missing (requests/yaml)"; exit 3; }
    echo "OFFLOAD_SYNC_OK host=$HOST repo=$REMOTE_REPO"
    ;;

  start)
    HOST=${1:?host}
    GB10_IP=${2:?gb10 tailscale ip}
    ARMDIR=${3:?armdir (for logging)}
    UPSTREAM="http://$GB10_IP:9950"
    # sanity: alienware MUST be able to reach the GB10 vLLM before we launch.
    if ! ssh "${SSH_OPTS[@]}" "$HOST" \
         "curl -fsS -m 6 $UPSTREAM/health >/dev/null 2>&1 && echo ok" 2>/dev/null | grep -q ok; then
      echo "FAIL: alienware cannot reach GB10 vLLM at $UPSTREAM/health (network/firewall)"; exit 4
    fi
    # launch the remote proxy with the canonical pins (forced deployment temp, pair dumps ON).
    # DEPLOY_FORCE_TEMP defaults to 0.6 = the REAL deployment temp (codex/q36-a, run_round5_b4_sweep
    # temp=0.6/top_p=0.95). temp 0.0 was WRONG for the speed/accept measurement (user 2026-06-16): the
    # tree tax is TEMP-DEPENDENT (~1.03x greedy -> ~1.4x t0.6) so greedy understates it, and at greedy a
    # lossless arm must byte-match native. Use DEPLOY_FORCE_TEMP=0.0 ONLY for the deterministic temp-0
    # argmax-flip lossless gate, never for the deployment speed/accept run.
    FORCE_TEMP="${DEPLOY_FORCE_TEMP:-0.6}"
    # FRESH per-rollout dumps: clear OUR remote pair/request dump dirs before relaunching the
    # proxy so each rollout's fetch pulls ONLY this rollout's dumps. The dirs otherwise
    # accumulate across runs and every arm's rsync pulls the whole history -> the
    # "byte-identical copied dumps" that confounded prior per-arm runaway/cache analysis.
    ssh "${SSH_OPTS[@]}" "$HOST" "rm -rf $REMOTE_PAIR_DUMPS/* $REMOTE_REQ_DUMPS/* 2>/dev/null; mkdir -p $REMOTE_PAIR_DUMPS $REMOTE_REQ_DUMPS" >/dev/null 2>&1 || true
    # FR13 §59 offload stream-stall fix-stack (default OFF; ENABLE at a LATER relaunch,
    # not now). The three LUMO_PROXY_SSE_* envs below are wired-through with OFF defaults
    # (empty LOG/CAPTURE_DIR + HEARTBEAT_S=0 => byte-identical relay). To turn them on at
    # the next relaunch, export before invoking this script (do NOT add a '#'-comment
    # INSIDE the quoted ssh env block below — it would comment out the launch command):
    #   export LUMO_PROXY_SSE_LOG=1                                  # per-request terminal-reason -> offload_proxy.log
    #   export LUMO_PROXY_SSE_CAPTURE_DIR="$REMOTE_ROOT/sse_capture" # per-chunk-timestamped chat-path capture jsonl
    #   export LUMO_PROXY_SSE_HEARTBEAT_S=15                         # empty-delta heartbeat on upstream idle (chat only)
    ssh "${SSH_OPTS[@]}" "$HOST" "\
      LUMO_PROXY_OFFLOAD_REPO=$REMOTE_REPO \
      LUMO_PROXY_OFFLOAD_VENV=$REMOTE_VENV \
      LUMO_PROXY_UPSTREAM_BASE_URL=$UPSTREAM \
      LUMO_PROXY_LISTEN_HOST=127.0.0.1 LUMO_PROXY_LISTEN_PORT=$PROXY_PORT \
      LUMO_PROXY_FORCE_TEMPERATURE=$FORCE_TEMP \
      LUMO_PROXY_FORCE_TOP_P=${LUMO_PROXY_FORCE_TOP_P:-} \
      LUMO_PROXY_FORCE_TOP_K=${LUMO_PROXY_FORCE_TOP_K:-} \
      LUMO_PROXY_FORCE_PRESENCE_PENALTY=${LUMO_PROXY_FORCE_PRESENCE_PENALTY:-} \
      LUMO_PROXY_FORCE_MIN_P=${LUMO_PROXY_FORCE_MIN_P:-} \
      LUMO_PROXY_QWEN_SAMPLING=${LUMO_PROXY_QWEN_SAMPLING:-1} \
      LUMO_PROXY_PAIR_DUMP_DIR=$REMOTE_PAIR_DUMPS \
      LUMO_PROXY_REQUEST_DUMP_DIR=$REMOTE_REQ_DUMPS \
      LUMO_PROXY_SSE_LOG=${LUMO_PROXY_SSE_LOG:-} \
      LUMO_PROXY_SSE_CAPTURE_DIR=${LUMO_PROXY_SSE_CAPTURE_DIR:-} \
      LUMO_PROXY_SSE_HEARTBEAT_S=${LUMO_PROXY_SSE_HEARTBEAT_S:-0} \
      LUMO_PROXY_THINK_BUDGET=${LUMO_PROXY_THINK_BUDGET:-} \
      LUMO_PROXY_THINK_CUTOFF=${LUMO_PROXY_THINK_CUTOFF:-} \
      LUMO_PROXY_ALLOW_MODELS_GET=${LUMO_PROXY_ALLOW_MODELS_GET:-} \
      bash $REMOTE_REPO/relaunch_proxy_remote.sh" \
      > "$ARMDIR/offload_proxy_launch.log" 2>&1 \
      || { echo "FAIL: remote proxy launch rc=$?"; cat "$ARMDIR/offload_proxy_launch.log"; exit 5; }
    # health-check the remote proxy over SSH (it listens on alienware 127.0.0.1).
    P0=$(date +%s); OK=0
    while (( $(date +%s) < P0 + 60 )); do
      CODE=$(ssh "${SSH_OPTS[@]}" "$HOST" \
        "curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PROXY_PORT/v1/models 2>/dev/null" 2>/dev/null)
      if [[ -n "$CODE" && "$CODE" != "000" ]]; then OK=1; break; fi
      sleep 2
    done
    (( OK == 1 )) || { echo "FAIL: remote proxy not healthy on $HOST:$PROXY_PORT"; exit 5; }
    # capture the remote proxy env for the class-9 pin assertions (mirrors on-GB10).
    ssh "${SSH_OPTS[@]}" "$HOST" "p=\$(cat /tmp/lumo_offload_proxy_${PROXY_PORT}.pid 2>/dev/null); \
      [ -r /proc/\$p/environ ] && tr '\0' '\n' < /proc/\$p/environ | grep -E '^LUMO_(PROXY|TRACK_B)' | sort" \
      > "$ARMDIR/offload_proxy_env.txt" 2>/dev/null || true
    grep -q "LUMO_PROXY_FORCE_TEMPERATURE=$FORCE_TEMP" "$ARMDIR/offload_proxy_env.txt" \
      || { echo "FAIL: remote proxy temp pin missing/mismatch (class 9; expected $FORCE_TEMP)"; exit 5; }
    grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/offload_proxy_env.txt" \
      || { echo "FAIL: remote proxy pair-dump pin missing (class 9)"; exit 5; }
    grep -q "LUMO_PROXY_AUTO_CONTINUE=1" "$ARMDIR/offload_proxy_env.txt" \
      || { echo "FAIL: remote proxy auto-continue pin missing"; exit 5; }
    echo "OFFLOAD_PROXY_OK host=$HOST upstream=$UPSTREAM (forced temp $FORCE_TEMP, pair dumps on, auto-continue on)"
    ;;

  fetch)
    HOST=${1:?host}
    ARMDIR=${2:?armdir}
    mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
    # rsync the alienware pair/request dumps back into the GB10 armdir for the
    # SEPARATE deploy-lossless recurrent-oracle rescore (vLLM-only GPU phase).
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:$REMOTE_PAIR_DUMPS/" "$ARMDIR/proxy_pair_dumps/" 2>/dev/null || true
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:$REMOTE_REQ_DUMPS/" "$ARMDIR/proxy_request_dumps/" 2>/dev/null || true
    # also pull the remote proxy log + the request-metrics capture (deploy-speed
    # basis is GB10 /metrics, but the proxy capture has the per-request rows).
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:/tmp/lumo_offload_proxy_${PROXY_PORT}.log" "$ARMDIR/offload_proxy.log" 2>/dev/null || true
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl" \
      "$ARMDIR/offload_request_metrics.jsonl" 2>/dev/null || true
    NPAIR=$(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)
    echo "OFFLOAD_FETCH_OK host=$HOST pair_dumps_back=$NPAIR"
    ;;

  stop)
    HOST=${1:?host}
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "p=\$(cat /tmp/lumo_offload_proxy_${PROXY_PORT}.pid 2>/dev/null); [ -n \"\$p\" ] && kill \$p 2>/dev/null; \
       pkill -f lumo_flywheel_serving.inference_proxy 2>/dev/null; echo stopped" 2>/dev/null || true
    echo "OFFLOAD_PROXY_STOPPED host=$HOST"
    ;;

  *)
    echo "unknown subcommand: $CMD"; exit 2;;
esac
