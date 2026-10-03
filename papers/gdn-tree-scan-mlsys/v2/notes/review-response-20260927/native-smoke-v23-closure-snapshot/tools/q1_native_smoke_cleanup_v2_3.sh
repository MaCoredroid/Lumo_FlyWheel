#!/usr/bin/env bash
# Engine cleanup + finalization contract for the native-smoke launcher v2.3 (sourced AFTER the gate has validated this file's hash).
# v2.3 = v2.2 + tiny ordering fix: FINALIZED.txt is written BEFORE write_receipt (so the sealed receipt records the actual final marker and NO artifact is
# written after the seal) and a write_receipt failure makes the terminal status non-zero (rc 10) instead of claiming success without a receipt.
# v2.2 = v2.1 + (1) ONE owner for cleanup/finalization: finalize() runs stop_engine + receipt exactly once and sets FINALIZED=1; on_exit_trap()
# does nothing once finalized (no second stop, no receipt rewrite) and otherwise finalizes an unexpected abort; (2) a Docker enumeration
# failure is distinguished from a successful empty result: owned_container_state() returns 1 on query failure and stop_engine then refuses
# with rc 8 ("docker_query_failed") instead of treating the engine as absent.
#   stop_engine -> 0 only if the owned container is absent (query succeeded) / already verifiably stopped+removed, or VERIFIED stopped
#                  (stop rc 0, State.Running false, State.Status exited|dead|created) AND removed; logs/inspect preserved before the verdict;
#                  engine_stopped_utc.txt only after the verified stopped state; rc 8 = not verified stopped / query failed, rc 9 = stopped but not removed.
#   finalize STATUS PRIMARY_RC -> stop_engine once, receipt "STATUS_cleanup_rc=<c>", FINALIZED=1; returns PRIMARY_RC if non-zero else the cleanup rc.
#   on_exit_trap RC -> returns RC unchanged when already finalized; else finalize "ABORTED_OR_FAILED_rc=RC" RC.
# Requires: $OUT, $CONTAINER, write_receipt (launcher-defined; takes a status string). Never calls exit.
FINALIZED=${FINALIZED:-0}
owned_container_state() {
  local out rc
  out=$(docker ps -a --format '{{.Names}}' 2>>"$OUT/engine_cleanup.err"); rc=$?
  if [[ $rc -ne 0 ]]; then echo "query_failed"; return 1; fi
  if grep -qx "$CONTAINER" <<<"$out"; then echo "present"; else echo "absent"; fi
  return 0
}
stop_engine() {
  local name="$CONTAINER" st="$OUT/engine_cleanup_state.txt" q qrc
  if [[ -f "$st" ]] && grep -qx "stopped_and_removed" "$st"; then return 0; fi              # idempotent after a verified cleanup
  q=$(owned_container_state); qrc=$?
  if [[ $qrc -ne 0 ]]; then
    echo "not_verified docker_query_failed" > "$st"
    echo "ENGINE CLEANUP FAILURE: docker ps enumeration failed; owned container $name state UNKNOWN (not treated as absent)" | tee -a "$OUT/engine_cleanup.err"
    return 8
  fi
  if [[ "$q" == "absent" ]]; then [[ -f "$st" ]] || echo "no_owned_container" > "$st"; return 0; fi
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
finalize() {
  local status_prefix="$1" primary_rc="$2"
  if [[ "${FINALIZED:-0}" -eq 1 ]]; then return "$primary_rc"; fi                          # single owner: never a second cleanup / receipt
  FINALIZED=1
  stop_engine; local crc=$?
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_contention_after.txt" 2>&1 || true
  echo "${status_prefix}_cleanup_rc=${crc}" > "$OUT/FINALIZED.txt"                       # final marker BEFORE the seal: the receipt records it; nothing is written after
  RECEIPT_WRITTEN=1
  if ! write_receipt "${status_prefix}_cleanup_rc=${crc}"; then
    echo "RECEIPT WRITE FAILED for ${status_prefix}_cleanup_rc=${crc}" >&2
    if [[ "$primary_rc" -ne 0 ]]; then return "$primary_rc"; fi
    return 10
  fi
  if [[ "$primary_rc" -ne 0 ]]; then return "$primary_rc"; fi
  return "$crc"
}
on_exit_trap() {
  local rc="$1"
  if [[ "${FINALIZED:-0}" -eq 1 ]]; then return "$rc"; fi                                    # already finalized: touch nothing
  finalize "ABORTED_OR_FAILED_rc=${rc}" "$rc"
}
