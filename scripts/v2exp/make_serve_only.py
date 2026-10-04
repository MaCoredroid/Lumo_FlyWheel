import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
anchor = '  _fixed32_proxy_begin_tmp="$ARMDIR/fixed32_proxy_ingress_begin.json.tmp"'
idx = [i for i, l in enumerate(lines) if l == anchor]
assert len(idx) == 1, idx
hook = r'''  # ---- v2exp serve-only hook (claude/v2-experiments-20260930) ----
  # After the engine ingress campaign has begun on a fully booted, warmed fixed32
  # server, hold the server for an external fixed-input replay instead of starting
  # the SWE agent. Exiting here runs the normal teardown trap.
  if [[ "${V2EXP_SERVE_ONLY:-0}" == "1" ]]; then
    printf '{"secret_file":"%s","task_ids":"%s","port":"%s","armdir":"%s","container":"%s"}\n' \
      "$FIXED32_INGRESS_SECRET_FILE" "$FR13_FIXED32_INGRESS_TASK_IDS" "$PORT" "$ARMDIR_ABS" \
      "$CONTAINER_RUNTIME_REF" > "${V2EXP_READY_FILE:?}"
    echo "[v2exp] serve-only: engine campaign begun; waiting for ${V2EXP_STOP_FILE:?}"
    while [[ ! -e "$V2EXP_STOP_FILE" ]]; do
      sleep 15
      docker inspect -f '{{.State.Running}}' "$CONTAINER_RUNTIME_REF" 2>/dev/null | grep -q true \
        || { echo "[v2exp] container no longer running"; break; }
    done
    echo "[v2exp] stop requested $(date -u +%FT%TZ)"
    exit 0
  fi'''
lines[idx[0]:idx[0]] = hook.split("\n")
open(dst, "w").write("\n".join(lines))
print("inserted at line", idx[0] + 1)
