#!/usr/bin/env bash
# Engine cleanup contract for the native-smoke launcher v2.1 (sourced; reviewer finding on v2 stop_engine lines 112-119):
#   stop_engine  -> returns 0 ONLY if the owned container never existed / was already verifiably stopped+removed, or is now VERIFIED stopped
#                   (docker stop rc 0 AND inspect State.Running == false AND State.Status in exited|dead|created) AND removed.  engine.log and
#                   engine_inspect.json are preserved BEFORE the verdict; engine_stopped_utc.txt is written ONLY after the verified stopped state;
#                   failures are written to engine_cleanup_state.txt + engine_cleanup.err and returned as rc 8 (not stopped / still running) or
#                   rc 9 (stopped but not removed).  Never masks: it does not call exit.
#   finish_run   -> terminal status: the driver's non-zero rc always wins; otherwise a cleanup failure makes the run terminal status non-zero.
# Requires: $OUT (run dir), $CONTAINER (owned container name), write_receipt (defined by the launcher; takes a status string).
stop_engine() {
  local name="$CONTAINER" st="$OUT/engine_cleanup_state.txt"
  if [[ -f "$st" ]] && grep -qx "stopped_and_removed" "$st"; then return 0; fi              # idempotent after a verified cleanup
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    [[ -f "$st" ]] || echo "no_owned_container" > "$st"; return 0
  fi
  docker stop -t 60 "$name" > "$OUT/docker_stop.out" 2>&1; local stop_rc=$?
  docker logs "$name" > "$OUT/engine.log" 2>&1 || echo "docker logs failed rc=$?" >> "$OUT/engine_cleanup.err"
  docker inspect "$name" > "$OUT/engine_inspect.json" 2>> "$OUT/engine_cleanup.err" || echo "docker inspect failed" >> "$OUT/engine_cleanup.err"
  local status running
  status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo "inspect-failed"); running=$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || echo "inspect-failed")
  docker inspect -f '{{.State.ExitCode}}' "$name" > "$OUT/engine_exit.txt" 2>/dev/null || echo "unknown" > "$OUT/engine_exit.txt"
  if [[ $stop_rc -ne 0 || "$running" != "false" || ( "$status" != "exited" && "$status" != "dead" && "$status" != "created" ) ]]; then
    echo "not_stopped stop_rc=$stop_rc status=$status running=$running" > "$st"
    echo "ENGINE CLEANUP FAILURE: owned container $name not verifiably stopped (stop rc=$stop_rc, status=$status, running=$running); logs/inspect preserved" | tee -a "$OUT/engine_cleanup.err"
    return 8
  fi
  date -u +%FT%TZ > "$OUT/engine_stopped_utc.txt"
  if ! docker rm "$name" > "$OUT/docker_rm.out" 2>&1; then
    echo "stopped_but_not_removed stop_rc=$stop_rc status=$status" > "$st"
    echo "ENGINE CLEANUP FAILURE: owned container $name verified stopped but could not be removed" | tee -a "$OUT/engine_cleanup.err"
    return 9
  fi
  echo "stopped_and_removed" > "$st"; return 0
}
finish_run() {
  local drv_rc="$1"; stop_engine; local crc=$?
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_contention_after.txt" 2>&1 || true
  RECEIPT_WRITTEN=1; write_receipt "COMPLETED_driver_rc=${drv_rc}_cleanup_rc=${crc}"
  if [[ "$drv_rc" -ne 0 ]]; then return "$drv_rc"; fi
  return "$crc"
}
