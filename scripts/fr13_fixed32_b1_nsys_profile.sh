#!/usr/bin/env bash
# Attribution-only fixed32 profile on canonical real SWE-Verified B1 traffic.
# Nsight terminates the wrapped server when the bounded capture ends, so this
# script must never be used as acceptance evidence.
set -uo pipefail

declare -a NSYS_STOPPED_PIDS=()
declare -a NSYS_STOPPED_START_TICKS=()
DRIVER_PID=""
DRIVER_START_TICKS=""
PROFILE_CONTAINER_ID=""
PROFILE_CONTAINER_CIDFILE=""
PROFILE_CONTAINER_STARTED_AT=""
PROFILE_CONTAINER_INIT_PID=""
PROFILE_CONTAINER_RESTART_COUNT=""
PROFILE_CONTAINER_RUNNING=0
PROFILE_CONTAINER_STATUS=""
PROFILE_CONTAINER_EXIT_CODE=""
PROCESS_IDENTITY=""
PROCESS_IDENTITY_STAT=""
ENGINE_CORE_PID=""
ENGINE_CORE_START_TICKS=""
ENGINE_CORE_CURRENT_START_TICKS=""
ENGINE_CORE_LIVENESS_OK=0
NSYS_SESSION_ID=""
NSYS_SESSION_NAME=""
NSYS_SESSION_STATE=""
NSYS_SESSION_PRESENT=0
NSYS_SESSION_QUERY_OK=0
NSYS_INJECTED_SESSION_ID=""
NSYS_EXPECTED_SESSION_NAME=""
NSYS_LIFECYCLE_ERROR=""
NSYS_COLLECTION_OBSERVED=0
NSYS_POST_COLLECTION_OBSERVED=0
NSYS_EXACT_STOP_COMPLETED=0
NSYS_CONTAINER_TERMINAL_OK=0
NSYS_PROVEN_REPORT_IDENTITY=""
NSYS_PROVEN_REPORT_SHA256=""
ENGINE_LEDGER_SNAPSHOT=""
PRESERVE_RECOVERABLE_STATE=0
PRESERVED_CONTAINER=""
NSYS_EXPECTED_DRIVER_SCRIPT=scripts/fr13_b4_campaign_driver.sh
NSYS_EXPECTED_VARIANT_SCRIPT=scripts/fr13_bigdenom_swe_serve_variant.sh
NSYS_EXPECTED_VARIANT_KIND=tail6_fixed32
NSYS_CONTAINER_BIN=/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys
NSYS_CONTAINER_TIMEOUT_BIN=/usr/bin/timeout
NSYS_CONTAINER_BASH_BIN=/bin/bash
LUMO_NSYS_QUERY_TIMEOUT_S=${LUMO_NSYS_QUERY_TIMEOUT_S:-10}
LUMO_NSYS_QUERY_KILL_AFTER_S=${LUMO_NSYS_QUERY_KILL_AFTER_S:-2}
LUMO_NSYS_QUERY_OUTER_TIMEOUT_S=${LUMO_NSYS_QUERY_OUTER_TIMEOUT_S:-15}
LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S=${LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S:-2}
LUMO_NSYS_STOP_TIMEOUT_S=${LUMO_NSYS_STOP_TIMEOUT_S:-60}
LUMO_NSYS_STOP_KILL_AFTER_S=${LUMO_NSYS_STOP_KILL_AFTER_S:-5}
LUMO_NSYS_STOP_OUTER_TIMEOUT_S=${LUMO_NSYS_STOP_OUTER_TIMEOUT_S:-70}
LUMO_NSYS_STOP_OUTER_KILL_AFTER_S=${LUMO_NSYS_STOP_OUTER_KILL_AFTER_S:-5}
LUMO_NSYS_EXPORT_TIMEOUT_S=${LUMO_NSYS_EXPORT_TIMEOUT_S:-300}
LUMO_NSYS_EXPORT_KILL_AFTER_S=${LUMO_NSYS_EXPORT_KILL_AFTER_S:-5}
LUMO_NSYS_HASH_TIMEOUT_S=${LUMO_NSYS_HASH_TIMEOUT_S:-300}
LUMO_NSYS_HASH_KILL_AFTER_S=${LUMO_NSYS_HASH_KILL_AFTER_S:-5}

_lifecycle_log() {
  local message=$1
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message" \
    >> "${NSYS_LIFECYCLE_LOG:-/dev/stderr}"
}

_process_stat_field() {
  local pid=$1
  local offset=$2
  local stat_line
  local stat_tail
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_tail=${stat_line##*) }
  set -- $stat_tail
  (( $# >= offset )) || return 1
  printf '%s\n' "${!offset}"
}

_process_start_ticks() {
  # /proc/PID/stat field 22 is word 20 after the parenthesized comm field.
  _process_stat_field "$1" 20
}

_process_parent_pid() {
  # /proc/PID/stat field 4 is word 2 after the parenthesized comm field.
  _process_stat_field "$1" 2
}

_process_state() {
  # /proc/PID/stat field 3 is word 1 after the parenthesized comm field.
  _process_stat_field "$1" 1
}

_process_identity_is_live() {
  local pid=$1
  local expected_start=$2
  local actual_start
  actual_start=$(_process_start_ticks "$pid") || return 1
  [[ "$actual_start" == "$expected_start" ]]
}

_process_argv_matches_script() {
  local pid=$1
  local expected_script=$2
  shift 2
  local -a argv=()
  local expected_arg
  local index=2
  mapfile -d '' -t argv < "/proc/$pid/cmdline" || return 1
  (( ${#argv[@]} == 2 + $# )) || return 1
  [[ "${argv[0]##*/}" == "bash" ]] || return 1
  [[ "${argv[1]}" == "$expected_script" ]] || return 1
  for expected_arg in "$@"; do
    [[ "${argv[$index]}" == "$expected_arg" ]] || return 1
    (( index += 1 ))
  done
}

_record_stopped_pid() {
  local pid=$1
  local start_ticks=$2
  local recorded
  for recorded in "${NSYS_STOPPED_PIDS[@]}"; do
    [[ "$recorded" == "$pid" ]] && return 0
  done
  NSYS_STOPPED_PIDS+=("$pid")
  NSYS_STOPPED_START_TICKS+=("$start_ticks")
}

_stop_exact_pid() {
  local pid=$1
  local expected_start=$2
  local label=$3
  local state=""
  local attempt

  _process_identity_is_live "$pid" "$expected_start" || {
    _lifecycle_log "FAIL: $label PID $pid identity changed before SIGSTOP"
    return 1
  }
  kill -STOP "$pid" || {
    _lifecycle_log "FAIL: SIGSTOP failed for $label PID $pid"
    return 1
  }
  # Record immediately: even a task in uninterruptible sleep can have a pending
  # stop signal, and every signalled identity must receive the matching thaw.
  _record_stopped_pid "$pid" "$expected_start"
  for (( attempt=0; attempt < 100; attempt++ )); do
    _process_identity_is_live "$pid" "$expected_start" || {
      _lifecycle_log "FAIL: $label PID $pid exited after SIGSTOP"
      return 1
    }
    state=$(_process_state "$pid") || return 1
    [[ "$state" == "T" || "$state" == "t" ]] && break
    sleep 0.01
  done
  if [[ "$state" != "T" && "$state" != "t" ]]; then
    _lifecycle_log "FAIL: $label PID $pid did not enter a stopped state"
    return 1
  fi
  _lifecycle_log "stopped $label pid=$pid start_ticks=$expected_start"
}

freeze_exact_control_ancestors() {
  local driver_pid=$1
  local driver_start=$2
  local driver_script=$3
  local variant_script=$4
  local arm=$5
  local kind=$6
  local subset=$7
  local children_text=""
  local child
  local child_parent
  local variant_pid=""
  local variant_start=""
  local matches=0

  _process_identity_is_live "$driver_pid" "$driver_start" || {
    _lifecycle_log "FAIL: async driver PID $driver_pid is no longer the launched process"
    return 1
  }
  _process_argv_matches_script "$driver_pid" "$driver_script" || {
    _lifecycle_log "FAIL: async driver PID $driver_pid command identity mismatch"
    return 1
  }
  if [[ -r "/proc/$driver_pid/task/$driver_pid/children" ]]; then
    IFS= read -r children_text \
      < "/proc/$driver_pid/task/$driver_pid/children" || true
  fi
  for child in $children_text; do
    child_parent=$(_process_parent_pid "$child") || continue
    [[ "$child_parent" == "$driver_pid" ]] || continue
    if _process_argv_matches_script \
        "$child" "$variant_script" "$arm" "$kind" "$subset"; then
      variant_pid=$child
      variant_start=$(_process_start_ticks "$child") || continue
      (( matches += 1 ))
    fi
  done
  if (( matches != 1 )); then
    _lifecycle_log \
      "FAIL: expected one exact variant-shell child of driver $driver_pid, found $matches"
    return 1
  fi

  # Only these two waiting shells are frozen. Their runner/watchdog descendants,
  # the remote agent, and the profiled container continue real SWE execution.
  _stop_exact_pid "$driver_pid" "$driver_start" "campaign driver" || return 1
  _process_identity_is_live "$variant_pid" "$variant_start" \
    && [[ "$(_process_parent_pid "$variant_pid")" == "$driver_pid" ]] || {
      _lifecycle_log "FAIL: variant-shell ancestry changed while freezing driver"
      return 1
    }
  _stop_exact_pid "$variant_pid" "$variant_start" "variant shell" || return 1
  _process_identity_is_live "$driver_pid" "$driver_start" \
    && _process_identity_is_live "$variant_pid" "$variant_start" \
    && [[ "$(_process_parent_pid "$variant_pid")" == "$driver_pid" ]] || {
      _lifecycle_log "FAIL: frozen driver/variant identities failed final validation"
      return 1
    }
  _lifecycle_log \
    "control ancestors frozen driver=$driver_pid variant=$variant_pid; descendants remain runnable"
}

thaw_exact_control_ancestors() {
  local index
  local pid
  local start_ticks
  for (( index=${#NSYS_STOPPED_PIDS[@]} - 1; index >= 0; index-- )); do
    pid=${NSYS_STOPPED_PIDS[$index]}
    start_ticks=${NSYS_STOPPED_START_TICKS[$index]}
    if _process_identity_is_live "$pid" "$start_ticks"; then
      kill -CONT "$pid" 2>/dev/null || true
      _lifecycle_log "thawed exact pid=$pid start_ticks=$start_ticks"
    else
      _lifecycle_log \
        "thaw skipped pid=$pid because its recorded process identity is no longer live"
    fi
  done
  NSYS_STOPPED_PIDS=()
  NSYS_STOPPED_START_TICKS=()
}

_pin_running_profile_container() {
  local candidate_id=$1
  local expected_name=$2
  local identity=""
  local actual_id=""
  local actual_name=""
  local actual_status=""
  local actual_running=""
  local actual_started_at=""
  local actual_pid=""
  local actual_restart_count=""
  local actual_exit_code=""
  local extra=""

  [[ "$candidate_id" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -n "$expected_name" ]] || return 1
  identity=$(
    docker inspect --format \
      '{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}} {{.State.ExitCode}}' \
      "$candidate_id" \
      2>> "$NSYS_LIFECYCLE_LOG"
  ) || return 1
  read -r \
    actual_id actual_name actual_status actual_running actual_started_at \
    actual_pid actual_restart_count actual_exit_code extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$candidate_id" ]] \
    && [[ "$actual_name" == "/$expected_name" ]] \
    && [[ "$actual_status" == "running" ]] \
    && [[ "$actual_running" == "true" ]] \
    && [[ "$actual_started_at" != 0001-01-01T00:00:00* ]] \
    && [[ "$actual_pid" =~ ^[1-9][0-9]*$ ]] \
    && [[ "$actual_restart_count" =~ ^[0-9]+$ ]] \
    && [[ "$actual_exit_code" =~ ^-?[0-9]+$ ]] || return 1

  PROFILE_CONTAINER_ID=$actual_id
  PROFILE_CONTAINER_STARTED_AT=$actual_started_at
  PROFILE_CONTAINER_INIT_PID=$actual_pid
  PROFILE_CONTAINER_RESTART_COUNT=$actual_restart_count
  PROFILE_CONTAINER_RUNNING=1
  PROFILE_CONTAINER_STATUS=$actual_status
  PROFILE_CONTAINER_EXIT_CODE=$actual_exit_code
}

_reattest_profile_container() {
  local expected_name=${1:-${CONTAINER:-}}
  local identity=""
  local actual_id=""
  local actual_name=""
  local actual_status=""
  local actual_running=""
  local actual_started_at=""
  local actual_pid=""
  local actual_restart_count=""
  local actual_exit_code=""
  local extra=""

  [[ "$PROFILE_CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -n "$expected_name" ]] || return 1
  [[ -n "$PROFILE_CONTAINER_STARTED_AT" ]] || return 1
  [[ "$PROFILE_CONTAINER_INIT_PID" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$PROFILE_CONTAINER_RESTART_COUNT" =~ ^[0-9]+$ ]] || return 1
  identity=$(
    docker inspect --format \
      '{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}} {{.State.ExitCode}}' \
      "$PROFILE_CONTAINER_ID" \
      2>> "$NSYS_LIFECYCLE_LOG"
  ) || return 1
  read -r \
    actual_id actual_name actual_status actual_running actual_started_at \
    actual_pid actual_restart_count actual_exit_code extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$PROFILE_CONTAINER_ID" ]] \
    && [[ "$actual_name" == "/$expected_name" ]] \
    && [[ "$actual_started_at" == "$PROFILE_CONTAINER_STARTED_AT" ]] \
    && [[ "$actual_restart_count" == "$PROFILE_CONTAINER_RESTART_COUNT" ]] \
    && [[ "$actual_exit_code" =~ ^-?[0-9]+$ ]] || return 1

  case "$actual_status:$actual_running" in
    running:true)
      [[ "$actual_pid" == "$PROFILE_CONTAINER_INIT_PID" ]] || return 1
      PROFILE_CONTAINER_RUNNING=1
      ;;
    exited:false)
      [[ "$actual_pid" == "0" ]] || return 1
      PROFILE_CONTAINER_RUNNING=0
      ;;
    *)
      return 1
      ;;
  esac
  PROFILE_CONTAINER_STATUS=$actual_status
  PROFILE_CONTAINER_EXIT_CODE=$actual_exit_code
}

_reattest_running_profile_container() {
  local expected_name=${1:-${CONTAINER:-}}
  _reattest_profile_container "$expected_name" \
    && (( PROFILE_CONTAINER_RUNNING == 1 ))
}

refresh_engine_core_start_ticks() {
  local query_rc=0

  ENGINE_CORE_CURRENT_START_TICKS=""
  [[ "$ENGINE_CORE_PID" =~ ^[1-9][0-9]*$ ]] || return 1
  _reattest_profile_container "$CONTAINER" || return 2
  (( PROFILE_CONTAINER_RUNNING == 1 )) || return 1
  ENGINE_CORE_CURRENT_START_TICKS=$(
    timeout --signal=TERM \
      --kill-after="${LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S}s" \
      "${LUMO_NSYS_QUERY_OUTER_TIMEOUT_S}s" \
      docker exec "$PROFILE_CONTAINER_ID" \
      "$NSYS_CONTAINER_TIMEOUT_BIN" --signal=TERM \
      --kill-after="${LUMO_NSYS_QUERY_KILL_AFTER_S}s" \
      "${LUMO_NSYS_QUERY_TIMEOUT_S}s" \
      "$NSYS_CONTAINER_BASH_BIN" -c '
        pid=$1
        [[ -r "/proc/$pid/stat" ]] || exit 1
        IFS= read -r stat_line < "/proc/$pid/stat" || exit 1
        stat_tail=${stat_line##*) }
        set -- $stat_tail
        (( $# >= 20 )) || exit 1
        printf "%s\n" "${20}"
      ' bash "$ENGINE_CORE_PID" \
      2>> "$NSYS_LIFECYCLE_LOG"
  )
  query_rc=$?
  _reattest_profile_container "$CONTAINER" || return 2
  (( query_rc == 0 )) \
    && [[ "$ENGINE_CORE_CURRENT_START_TICKS" =~ ^[1-9][0-9]*$ ]]
}

refresh_run_identity_evidence() {
  local -a container_ids=()
  local engine_core_pid=""
  local process_identity_before=""
  local process_identity_after=""
  local injected_session_id=""
  local liveness_rc=0

  if [[ -z "$PROFILE_CONTAINER_ID" ]]; then
    if [[ ! -e "$PROFILE_CONTAINER_CIDFILE" \
       && ! -L "$PROFILE_CONTAINER_CIDFILE" ]]; then
      return 0
    fi
    if [[ ! -f "$PROFILE_CONTAINER_CIDFILE" \
       || -L "$PROFILE_CONTAINER_CIDFILE" \
       || ! "$PROFILE_CONTAINER_CIDFILE" -nt "$RUN_BOUNDARY" ]]; then
      NSYS_LIFECYCLE_ERROR=\
"run-bound fixed32 Docker cidfile is malformed, aliased, or stale"
      return 2
    fi
    mapfile -t container_ids < "$PROFILE_CONTAINER_CIDFILE"
    if (( ${#container_ids[@]} != 1 )) \
        || [[ ! "${container_ids[0]}" =~ ^[0-9a-f]{64}$ ]]; then
      NSYS_LIFECYCLE_ERROR=\
"run-bound fixed32 Docker cidfile does not contain one full container ID"
      return 2
    fi
    if ! _pin_running_profile_container "${container_ids[0]}" "$CONTAINER"; then
      NSYS_LIFECYCLE_ERROR=\
"fixed32 Docker cidfile does not attest the expected running container incarnation"
      return 2
    fi
    _lifecycle_log \
      "attested run container id=$PROFILE_CONTAINER_ID name=$CONTAINER started_at=$PROFILE_CONTAINER_STARTED_AT init_pid=$PROFILE_CONTAINER_INIT_PID restart_count=$PROFILE_CONTAINER_RESTART_COUNT"
  elif ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container immutable identity/incarnation attestation no longer matches"
    return 2
  fi

  if [[ -n "$NSYS_INJECTED_SESSION_ID" ]]; then
    if [[ ! -f "$PROCESS_IDENTITY" \
       || -L "$PROCESS_IDENTITY" \
       || ! -s "$PROCESS_IDENTITY" \
       || ! "$PROCESS_IDENTITY" -nt "$RUN_BOUNDARY" ]]; then
      NSYS_LIFECYCLE_ERROR=\
"pinned fixed32 process identity became malformed, aliased, empty, or stale"
      return 2
    fi
    process_identity_before=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY") \
      || {
        NSYS_LIFECYCLE_ERROR="unable to restat pinned fixed32 process identity"
        return 2
      }
    if [[ "$process_identity_before" != "$PROCESS_IDENTITY_STAT" ]]; then
      NSYS_LIFECYCLE_ERROR=\
"run-bound fixed32 process identity changed after session binding"
      return 2
    fi
    if (( PROFILE_CONTAINER_RUNNING == 1 )); then
      refresh_engine_core_start_ticks
      liveness_rc=$?
      if (( liveness_rc == 2 )); then
        NSYS_LIFECYCLE_ERROR=\
"run container identity or incarnation changed during EngineCore liveness validation"
        return 2
      fi
      ENGINE_CORE_LIVENESS_OK=0
      if (( PROFILE_CONTAINER_RUNNING == 1 && liveness_rc == 0 )) \
          && [[ "$ENGINE_CORE_CURRENT_START_TICKS" \
            == "$ENGINE_CORE_START_TICKS" ]]; then
        ENGINE_CORE_LIVENESS_OK=1
      fi
    else
      ENGINE_CORE_LIVENESS_OK=0
    fi
    return 0
  fi
  if [[ ! -e "$PROCESS_IDENTITY" && ! -L "$PROCESS_IDENTITY" ]]; then
    return 0
  fi
  if [[ ! -f "$PROCESS_IDENTITY" \
     || -L "$PROCESS_IDENTITY" \
     || ! -s "$PROCESS_IDENTITY" \
     || ! "$PROCESS_IDENTITY" -nt "$RUN_BOUNDARY" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"run-bound fixed32 process identity is malformed, aliased, empty, or stale"
    return 2
  fi
  process_identity_before=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY") \
    || {
      NSYS_LIFECYCLE_ERROR="unable to stat run-bound fixed32 process identity"
      return 2
    }
  if ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container identity changed before process-identity validation"
    return 2
  fi
  injected_session_id=$(
    "$JQ_BIN" -er '
      if (
        type == "object"
        and .schema == "fr13-fixed32-process-identity-v1"
        and (.engine_core | type == "object")
        and (.engine_core.pid | type == "number")
        and (.engine_core.pid > 1)
        and .engine_core.argv == ["VLLM::EngineCore"]
        and (.engine_core.environ | type == "array")
        and all(.engine_core.environ[]; type == "string")
        and (
          [
            .engine_core.environ[]
            | select(startswith("NSYS_PROFILING_SESSION_ID="))
          ]
          | length == 1
        )
      ) then
        (
          .engine_core.environ[]
          | select(startswith("NSYS_PROFILING_SESSION_ID="))
          | sub("^NSYS_PROFILING_SESSION_ID="; "")
          | select(test("^[1-9][0-9]*$"))
        )
      else
        empty
      end
    ' "$PROCESS_IDENTITY" 2>> "$NSYS_LIFECYCLE_LOG"
  ) || {
    NSYS_LIFECYCLE_ERROR=\
"EngineCore process identity has no unique numeric Nsight session ID"
    return 2
  }
  engine_core_pid=$(
    "$JQ_BIN" -er '.engine_core.pid' "$PROCESS_IDENTITY" \
      2>> "$NSYS_LIFECYCLE_LOG"
  ) || {
    NSYS_LIFECYCLE_ERROR="unable to read EngineCore PID from process identity"
    return 2
  }
  process_identity_after=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY") \
    || {
      NSYS_LIFECYCLE_ERROR="unable to restat run-bound fixed32 process identity"
      return 2
    }
  if [[ "$process_identity_after" != "$process_identity_before" ]] \
      || ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"process or container identity changed during Nsight identity validation"
    return 2
  fi
  if (( PROFILE_CONTAINER_RUNNING == 0 )); then
    NSYS_LIFECYCLE_ERROR=\
"exact run container exited during initial EngineCore identity validation"
    return 2
  fi
  ENGINE_CORE_PID=$engine_core_pid
  refresh_engine_core_start_ticks
  liveness_rc=$?
  if (( liveness_rc != 0 || PROFILE_CONTAINER_RUNNING == 0 )); then
    ENGINE_CORE_PID=""
    NSYS_LIFECYCLE_ERROR=\
"unable to pin a live EngineCore PID/start identity"
    return 2
  fi
  PROCESS_IDENTITY_STAT=$process_identity_after
  ENGINE_CORE_START_TICKS=$ENGINE_CORE_CURRENT_START_TICKS
  ENGINE_CORE_LIVENESS_OK=1
  NSYS_INJECTED_SESSION_ID=$injected_session_id
  _lifecycle_log \
    "attested EngineCore-injected Nsight session id=$NSYS_INJECTED_SESSION_ID engine_pid=$ENGINE_CORE_PID engine_start_ticks=$ENGINE_CORE_START_TICKS container_id=$PROFILE_CONTAINER_ID"
}

accept_terminal_profile_container() {
  (( PROFILE_CONTAINER_RUNNING == 0 )) || return 1
  if [[ "$PROFILE_CONTAINER_STATUS" != "exited" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"run container entered an unsupported non-running state"
    return 2
  fi
  if [[ "$PROFILE_CONTAINER_EXIT_CODE" != "0" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"exact run container exited nonzero with code $PROFILE_CONTAINER_EXIT_CODE"
    return 2
  fi
  if [[ -z "$NSYS_SESSION_ID" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"exact run container exited before Nsight session binding"
    return 2
  fi
  if (( NSYS_POST_COLLECTION_OBSERVED == 0 \
      && NSYS_EXACT_STOP_COMPLETED == 0 )); then
    if [[ "$NSYS_SESSION_STATE" == "Collection" ]]; then
      NSYS_LIFECYCLE_ERROR=\
"exact run container exited during Nsight Collection"
    else
      NSYS_LIFECYCLE_ERROR=\
"exact run container exited before a proven post-Collection transition"
    fi
    return 2
  fi
  if (( NSYS_CONTAINER_TERMINAL_OK == 0 )); then
    _lifecycle_log \
      "accepted exact exited0 container incarnation after proven Nsight terminal transition"
  fi
  NSYS_CONTAINER_TERMINAL_OK=1
  NSYS_SESSION_QUERY_OK=1
  NSYS_SESSION_PRESENT=0
  NSYS_SESSION_STATE="ContainerExited"
}

snapshot_terminal_engine_ledger() {
  local destination=$1
  local source_path=/logs/fr13_fixed32_engine_ingress.jsonl
  local snapshot_tmp="${destination}.tmp.$$"
  local expected_owner=""
  local snapshot_identity=""

  if (( NSYS_CONTAINER_TERMINAL_OK != 1 )); then
    NSYS_LIFECYCLE_ERROR=\
"refusing engine-ledger snapshot without exited0 container terminal proof"
    return 1
  fi
  if [[ ! "$NSYS_PROVEN_REPORT_IDENTITY" =~ \
      ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] \
      || [[ ! "$NSYS_PROVEN_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    NSYS_LIFECYCLE_ERROR=\
"refusing engine-ledger snapshot without the latched report proof"
    return 1
  fi
  if [[ -e "$destination" || -L "$destination" \
     || -e "$snapshot_tmp" || -L "$snapshot_tmp" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"fresh host engine-ledger snapshot path already exists"
    return 1
  fi
  expected_owner=$(id -u) || {
    NSYS_LIFECYCLE_ERROR="unable to identify the host snapshot owner"
    return 1
  }
  if ! _reattest_profile_container "$CONTAINER" \
      || (( PROFILE_CONTAINER_RUNNING != 0 )) \
      || [[ "$PROFILE_CONTAINER_STATUS" != "exited" ]] \
      || [[ "$PROFILE_CONTAINER_EXIT_CODE" != "0" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"exact exited0 container identity drifted before engine-ledger snapshot"
    return 1
  fi
  if ! docker cp \
      "${PROFILE_CONTAINER_ID}:${source_path}" "$snapshot_tmp" \
      >> "$NSYS_LIFECYCLE_LOG" 2>&1; then
    NSYS_LIFECYCLE_ERROR=\
"docker cp failed for the exact terminal container engine ledger"
    if [[ -f "$snapshot_tmp" && ! -L "$snapshot_tmp" \
       && "$(stat -c '%u' "$snapshot_tmp" 2>/dev/null)" == "$expected_owner" ]]; then
      rm -f -- "$snapshot_tmp"
    fi
    return 1
  fi
  if ! _reattest_profile_container "$CONTAINER" \
      || (( PROFILE_CONTAINER_RUNNING != 0 )) \
      || [[ "$PROFILE_CONTAINER_STATUS" != "exited" ]] \
      || [[ "$PROFILE_CONTAINER_EXIT_CODE" != "0" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"exact exited0 container identity drifted during engine-ledger snapshot"
    if [[ -f "$snapshot_tmp" && ! -L "$snapshot_tmp" \
       && "$(stat -c '%u' "$snapshot_tmp" 2>/dev/null)" == "$expected_owner" ]]; then
      rm -f -- "$snapshot_tmp"
    fi
    return 1
  fi
  if [[ ! -f "$snapshot_tmp" || -L "$snapshot_tmp" \
     || ! -s "$snapshot_tmp" || ! -r "$snapshot_tmp" ]] \
      || [[ "$(stat -c '%u' "$snapshot_tmp" 2>/dev/null)" != "$expected_owner" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"copied engine ledger is not a host-owned readable nonempty regular file"
    if [[ -f "$snapshot_tmp" && ! -L "$snapshot_tmp" \
       && "$(stat -c '%u' "$snapshot_tmp" 2>/dev/null)" == "$expected_owner" ]]; then
      rm -f -- "$snapshot_tmp"
    fi
    return 1
  fi
  snapshot_identity=$(stat -c '%d:%i:%h:%s' "$snapshot_tmp") || {
    NSYS_LIFECYCLE_ERROR="unable to stat copied engine-ledger snapshot"
    return 1
  }
  if ! mv -Tn -- "$snapshot_tmp" "$destination" \
      || [[ ! -f "$destination" || -L "$destination" \
         || ! -s "$destination" || ! -r "$destination" ]] \
      || [[ "$(stat -c '%u' "$destination" 2>/dev/null)" != "$expected_owner" ]] \
      || [[ "$(stat -c '%d:%i:%h:%s' "$destination" 2>/dev/null)" \
        != "$snapshot_identity" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"unable to publish a stable host-owned engine-ledger snapshot"
    return 1
  fi
  ENGINE_LEDGER_SNAPSHOT=$destination
  _lifecycle_log \
    "snapshotted exact terminal container engine ledger to $ENGINE_LEDGER_SNAPSHOT"
}

validate_engine_core_liveness_after_session_query() {
  if (( PROFILE_CONTAINER_RUNNING == 1 \
      && ENGINE_CORE_LIVENESS_OK == 0 \
      && NSYS_POST_COLLECTION_OBSERVED == 0 )); then
    NSYS_LIFECYCLE_ERROR=\
"pinned EngineCore process identity is no longer live before the proven Nsight terminal transition"
    return 2
  fi
}

refresh_run_nsys_session() {
  local snapshot
  local row=""
  local row_count=0
  local name_row_count=0
  local previous_session_state=$NSYS_SESSION_STATE
  local query_rc=0

  NSYS_SESSION_QUERY_OK=0
  NSYS_SESSION_PRESENT=0
  refresh_run_identity_evidence
  case $? in
    0) ;;
    2) return 2 ;;
    *) return 1 ;;
  esac
  [[ -n "$PROFILE_CONTAINER_ID" ]] \
    && [[ -n "$NSYS_INJECTED_SESSION_ID" ]] || return 0
  if (( PROFILE_CONTAINER_RUNNING == 0 )); then
    accept_terminal_profile_container
    return $?
  fi
  NSYS_SESSION_STATE=""
  if ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container identity or incarnation changed before Nsight session query"
    return 2
  fi
  if (( PROFILE_CONTAINER_RUNNING == 0 )); then
    accept_terminal_profile_container
    return $?
  fi
  snapshot=$(
    timeout --signal=TERM \
      --kill-after="${LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S}s" \
      "${LUMO_NSYS_QUERY_OUTER_TIMEOUT_S}s" \
      docker exec "$PROFILE_CONTAINER_ID" \
      "$NSYS_CONTAINER_TIMEOUT_BIN" --signal=TERM \
      --kill-after="${LUMO_NSYS_QUERY_KILL_AFTER_S}s" \
      "${LUMO_NSYS_QUERY_TIMEOUT_S}s" \
      "$NSYS_CONTAINER_BIN" sessions list --output-format=json \
      2>> "$NSYS_LIFECYCLE_LOG"
  )
  query_rc=$?
  if ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container identity or incarnation changed during Nsight session query"
    return 2
  fi
  if (( query_rc != 0 )); then
    if (( PROFILE_CONTAINER_RUNNING == 0 )); then
      accept_terminal_profile_container
      return $?
    fi
    NSYS_LIFECYCLE_ERROR=\
"container-scoped Nsight session query failed for the exact run container"
    return 2
  fi
  "$JQ_BIN" -e '
    type == "array"
    and all(.[];
      (.id | type == "string")
      and (.name | type == "string")
      and (.state | type == "string")
      and (.accessible | type == "boolean")
    )
  ' <<< "$snapshot" >/dev/null || {
    NSYS_LIFECYCLE_ERROR=\
"container-scoped Nsight session query returned malformed JSON"
    return 2
  }
  NSYS_SESSION_QUERY_OK=1
  name_row_count=$(
    "$JQ_BIN" -r --arg name "$NSYS_EXPECTED_SESSION_NAME" \
      '[.[] | select(.name == $name)] | length' <<< "$snapshot"
  ) || {
    NSYS_LIFECYCLE_ERROR=\
"unable to validate the run-unique Nsight session name"
    return 2
  }
  if (( name_row_count > 1 )); then
    NSYS_LIFECYCLE_ERROR=\
"container Nsight namespace has duplicate rows for the run-unique session name"
    return 2
  fi

  if [[ -z "$NSYS_SESSION_ID" ]]; then
    [[ -n "$NSYS_INJECTED_SESSION_ID" ]] || return 0
    row_count=$(
      "$JQ_BIN" -r --arg id "$NSYS_INJECTED_SESSION_ID" \
        '[.[] | select(.id == $id)] | length' <<< "$snapshot"
    ) || {
      NSYS_LIFECYCLE_ERROR=\
"unable to validate the EngineCore-injected Nsight session ID"
      return 2
    }
    if (( row_count > 1 )); then
      NSYS_LIFECYCLE_ERROR=\
"Nsight sessions list has duplicate rows for the injected session ID"
      return 2
    fi
    if (( row_count == 0 )); then
      if (( name_row_count == 1 )); then
        NSYS_LIFECYCLE_ERROR=\
"run-unique Nsight session name is attached to a different session ID"
        return 2
      fi
      if (( PROFILE_CONTAINER_RUNNING == 0 )); then
        accept_terminal_profile_container
        return $?
      fi
      validate_engine_core_liveness_after_session_query || return 2
      return 0
    fi
    row=$(
      "$JQ_BIN" -c --arg id "$NSYS_INJECTED_SESSION_ID" \
        '.[] | select(.id == $id)' <<< "$snapshot"
    ) || {
      NSYS_LIFECYCLE_ERROR=\
"unable to read the EngineCore-injected Nsight session row"
      return 2
    }
    if [[ ! "$NSYS_INJECTED_SESSION_ID" =~ ^[1-9][0-9]*$ ]] \
        || [[ "$("$JQ_BIN" -r '.name' <<< "$row")" \
          != "$NSYS_EXPECTED_SESSION_NAME" ]] \
        || [[ "$("$JQ_BIN" -r '.accessible' <<< "$row")" != "true" ]]; then
      NSYS_LIFECYCLE_ERROR=\
"injected Nsight session row does not match the pinned accessible run session"
      return 2
    fi
    NSYS_SESSION_ID=$NSYS_INJECTED_SESSION_ID
    NSYS_SESSION_NAME=$NSYS_EXPECTED_SESSION_NAME
    _lifecycle_log \
      "bound exact run Nsight session id=$NSYS_SESSION_ID name=$NSYS_SESSION_NAME"
  fi

  NSYS_SESSION_PRESENT=0
  NSYS_SESSION_STATE=""
  if [[ "$NSYS_SESSION_ID" != "$NSYS_INJECTED_SESSION_ID" ]] \
      || [[ "$NSYS_SESSION_NAME" != "$NSYS_EXPECTED_SESSION_NAME" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"bound Nsight session identity drifted from run-bound evidence"
    return 2
  fi
  row_count=$(
    "$JQ_BIN" -r --arg id "$NSYS_SESSION_ID" \
      '[.[] | select(.id == $id)] | length' <<< "$snapshot"
  ) || {
    NSYS_LIFECYCLE_ERROR="unable to revalidate the bound Nsight session ID"
    return 2
  }
  if (( row_count > 1 )); then
    NSYS_LIFECYCLE_ERROR=\
"Nsight sessions list has duplicate rows for the bound session ID"
    return 2
  fi
  if (( row_count == 0 )); then
    if (( name_row_count == 1 )); then
      NSYS_LIFECYCLE_ERROR=\
"bound Nsight session name moved to a different session ID"
      return 2
    fi
    if [[ "$previous_session_state" == "Collection" \
       || "$previous_session_state" == "Generation" ]] \
        || (( NSYS_COLLECTION_OBSERVED == 1 )); then
      NSYS_POST_COLLECTION_OBSERVED=1
    fi
    if (( PROFILE_CONTAINER_RUNNING == 0 )); then
      accept_terminal_profile_container
      return $?
    fi
    validate_engine_core_liveness_after_session_query || return 2
    return 0
  fi
  row=$(
    "$JQ_BIN" -c --arg id "$NSYS_SESSION_ID" \
      '.[] | select(.id == $id)' <<< "$snapshot"
  ) || {
    NSYS_LIFECYCLE_ERROR="unable to read the bound Nsight session row"
    return 2
  }
  if [[ "$("$JQ_BIN" -r '.name' <<< "$row")" \
        != "$NSYS_EXPECTED_SESSION_NAME" ]] \
      || [[ "$("$JQ_BIN" -r '.accessible' <<< "$row")" != "true" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"bound Nsight session row lost its pinned name or accessibility"
    return 2
  fi
  NSYS_SESSION_PRESENT=1
  NSYS_SESSION_STATE=$("$JQ_BIN" -r '.state' <<< "$row")
  case "$NSYS_SESSION_STATE" in
    Collection)
      if (( NSYS_POST_COLLECTION_OBSERVED == 1 )); then
        NSYS_LIFECYCLE_ERROR=\
"exact Nsight session regressed to Collection after its terminal transition"
        return 2
      fi
      NSYS_COLLECTION_OBSERVED=1
      ;;
    Generation)
      NSYS_POST_COLLECTION_OBSERVED=1
      ;;
    *)
      if (( NSYS_COLLECTION_OBSERVED == 1 )); then
        NSYS_POST_COLLECTION_OBSERVED=1
      fi
      ;;
  esac
  if (( PROFILE_CONTAINER_RUNNING == 0 )); then
    accept_terminal_profile_container
    return $?
  fi
  validate_engine_core_liveness_after_session_query || return 2
}

verify_report_readable() {
  local report=$1
  local readability_output=$2
  timeout --signal=TERM \
    --kill-after="${LUMO_NSYS_EXPORT_KILL_AFTER_S}s" \
    "${LUMO_NSYS_EXPORT_TIMEOUT_S}s" \
    "$LUMO_NSYS_BIN" export \
    --type=info \
    --force-overwrite=true \
    --output="$readability_output" \
    "$report" \
    >> "$NSYS_LIFECYCLE_LOG" 2>&1 \
    && [[ -s "$readability_output" ]]
}

stop_exact_nsys_session() {
  local refresh_rc=0
  local stop_rc=0
  [[ "$NSYS_SESSION_ID" =~ ^[1-9][0-9]*$ ]] \
    && [[ "$NSYS_SESSION_ID" == "$NSYS_INJECTED_SESSION_ID" ]] \
    && [[ "$NSYS_SESSION_NAME" == "$NSYS_EXPECTED_SESSION_NAME" ]] \
    || return 1
  refresh_run_nsys_session
  refresh_rc=$?
  (( refresh_rc == 0 )) \
    && (( NSYS_SESSION_QUERY_OK == 1 )) \
    && (( NSYS_SESSION_PRESENT == 1 )) \
    && [[ "$NSYS_SESSION_STATE" == "Collection" ]] \
    && [[ "$NSYS_SESSION_ID" == "$NSYS_INJECTED_SESSION_ID" ]] \
    && [[ "$NSYS_SESSION_NAME" == "$NSYS_EXPECTED_SESSION_NAME" ]] \
    || {
      _lifecycle_log \
        "WARN: refusing Nsight stop without a fresh exact Collection row"
      return 1
    }
  refresh_engine_core_start_ticks
  refresh_rc=$?
  if (( refresh_rc != 0 || PROFILE_CONTAINER_RUNNING == 0 )) \
      || [[ "$ENGINE_CORE_CURRENT_START_TICKS" \
        != "$ENGINE_CORE_START_TICKS" ]]; then
    NSYS_LIFECYCLE_ERROR=\
"refusing Nsight stop without a fresh live EngineCore PID/start identity"
    return 1
  fi
  if ! _reattest_running_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container is not the exact attested running target before Nsight stop"
    return 1
  fi
  _lifecycle_log "requesting stop for exact Nsight session id=$NSYS_SESSION_ID"
  timeout --signal=TERM \
    --kill-after="${LUMO_NSYS_STOP_OUTER_KILL_AFTER_S}s" \
    "${LUMO_NSYS_STOP_OUTER_TIMEOUT_S}s" \
    docker exec "$PROFILE_CONTAINER_ID" \
      "$NSYS_CONTAINER_TIMEOUT_BIN" --signal=TERM \
      --kill-after="${LUMO_NSYS_STOP_KILL_AFTER_S}s" \
      "${LUMO_NSYS_STOP_TIMEOUT_S}s" \
      "$NSYS_CONTAINER_BIN" stop --session="$NSYS_SESSION_ID" \
    >> "$NSYS_LIFECYCLE_LOG" 2>&1
  stop_rc=$?
  if ! _reattest_profile_container "$CONTAINER"; then
    NSYS_LIFECYCLE_ERROR=\
"run container identity or incarnation changed during Nsight stop"
    return 1
  fi
  if (( stop_rc != 0 )); then
    NSYS_LIFECYCLE_ERROR=\
"container-scoped Nsight stop failed for the exact run session"
    return 1
  fi
  NSYS_EXACT_STOP_COMPLETED=1
  NSYS_POST_COLLECTION_OBSERVED=1
  if (( PROFILE_CONTAINER_RUNNING == 0 )); then
    accept_terminal_profile_container
    return $?
  fi
}

preserve_recoverable_container() {
  local reason=$1
  PRESERVE_RECOVERABLE_STATE=1
  if [[ -z "$PROFILE_CONTAINER_ID" ]] \
      || ! _reattest_profile_container "$CONTAINER"; then
    _lifecycle_log "FAIL: $reason; exact container identity is unavailable"
    return 1
  fi
  PRESERVED_CONTAINER="${CONTAINER}-preserved-${STAMP:-unknown-run}"
  if docker inspect "$PRESERVED_CONTAINER" >/dev/null 2>&1; then
    _lifecycle_log \
      "FAIL: preservation target already exists: $PRESERVED_CONTAINER"
    return 1
  fi
  if ! docker rename "$PROFILE_CONTAINER_ID" "$PRESERVED_CONTAINER" \
      >> "$NSYS_LIFECYCLE_LOG" 2>&1; then
    _lifecycle_log \
      "FAIL: unable to rename immutable container $PROFILE_CONTAINER_ID for recovery; refusing rm-f"
    return 1
  fi
  if ! _reattest_profile_container "$PRESERVED_CONTAINER"; then
    _lifecycle_log \
      "FAIL: renamed container failed immutable ID/name re-attestation; refusing rm-f"
    return 1
  fi
  _lifecycle_log \
    "FAIL: $reason; preserved recoverable container as $PRESERVED_CONTAINER"
  printf 'CONTAINER_ID=%s\nPRESERVED_CONTAINER=%s\nREPORT_PATH=%s\nREASON=%s\n' \
    "$PROFILE_CONTAINER_ID" "$PRESERVED_CONTAINER" "$REPORT" "$reason" \
    > "$RUNROOT_ABS/nsys_recovery_state.txt"
}

fail_nsys_lifecycle() {
  local reason=$1
  NSYS_LIFECYCLE_ERROR=$reason
  if (( ${#NSYS_STOPPED_PIDS[@]} == 0 )) \
      && [[ -n "$DRIVER_PID" ]] \
      && _process_identity_is_live "$DRIVER_PID" "$DRIVER_START_TICKS"; then
    freeze_exact_control_ancestors \
      "$DRIVER_PID" "$DRIVER_START_TICKS" \
      "$NSYS_EXPECTED_DRIVER_SCRIPT" \
      "$NSYS_EXPECTED_VARIANT_SCRIPT" \
      "$ARM" "$NSYS_EXPECTED_VARIANT_KIND" "$SUBSET" || true
  fi
  preserve_recoverable_container "$reason" || true
  return 1
}

report_stability_is_eligible() {
  local control_frozen=$1
  (( NSYS_SESSION_QUERY_OK == 1 )) \
    && [[ -n "$NSYS_SESSION_ID" ]] \
    && (( NSYS_POST_COLLECTION_OBSERVED == 1 )) \
    && (( NSYS_CONTAINER_TERMINAL_OK == 1 )) \
    && (( control_frozen == 1 )) \
    && [[ "$NSYS_SESSION_STATE" != "Collection" ]] \
    && [[ "$NSYS_SESSION_STATE" != "Generation" ]]
}

wait_for_fresh_stable_report() {
  local driver_pid=$1
  local driver_start=$2
  local report=$3
  local readability_output=$4
  local run_boundary=$5
  local identity_wait_started_at=$SECONDS
  local session_started_at=-1
  local collection_started_at=-1
  local generation_started_at=-1
  local forced_stop=0
  local control_frozen=0
  local stable_polls=0
  local report_tuple=""
  local previous_tuple=""
  local report_sha256=""
  local refresh_after_stop_rc=0

  while true; do
    refresh_run_nsys_session
    case $? in
      0) ;;
      1)
        _lifecycle_log "WARN: transient Nsight session query failure"
        ;;
      2)
        fail_nsys_lifecycle "$NSYS_LIFECYCLE_ERROR"
        return 1
        ;;
    esac

    if (( session_started_at < 0 )) \
        && [[ -n "$NSYS_INJECTED_SESSION_ID" ]]; then
      session_started_at=$SECONDS
      _lifecycle_log \
        "starting delayed-session deadlines from pinned EngineCore identity"
    fi

    if (( NSYS_SESSION_QUERY_OK == 1 )) \
        && [[ "$NSYS_SESSION_STATE" == "Collection" ]] \
        && (( control_frozen == 0 )); then
      if ! freeze_exact_control_ancestors \
          "$driver_pid" "$driver_start" \
          "$NSYS_EXPECTED_DRIVER_SCRIPT" \
          "$NSYS_EXPECTED_VARIANT_SCRIPT" \
          "$ARM" "$NSYS_EXPECTED_VARIANT_KIND" "$SUBSET"; then
        fail_nsys_lifecycle \
          "could not identify and freeze the exact driver/variant ancestry at collection entry"
        return 1
      fi
      control_frozen=1
      collection_started_at=$SECONDS
      _lifecycle_log "Nsight collection window opened with teardown control frozen"
    elif (( NSYS_SESSION_QUERY_OK == 1 )) \
        && [[ "$NSYS_SESSION_STATE" == "Generation" ]] \
        && (( control_frozen == 0 )); then
      # This should be unreachable with normal polling, but preserving a late
      # report is safer than allowing an unguarded teardown.
      if ! freeze_exact_control_ancestors \
          "$driver_pid" "$driver_start" \
          "$NSYS_EXPECTED_DRIVER_SCRIPT" \
          "$NSYS_EXPECTED_VARIANT_SCRIPT" \
          "$ARM" "$NSYS_EXPECTED_VARIANT_KIND" "$SUBSET"; then
        fail_nsys_lifecycle \
          "missed Collection and could not freeze exact controls during Generation"
        return 1
      fi
      control_frozen=1
      generation_started_at=$SECONDS
      _lifecycle_log "WARN: first guarded Nsight state was Generation"
    fi

    if (( NSYS_SESSION_QUERY_OK == 1 )) \
        && [[ "$NSYS_SESSION_STATE" == "Generation" ]] \
        && (( generation_started_at < 0 )); then
      generation_started_at=$SECONDS
      _lifecycle_log "Nsight report generation began"
    fi

    if report_stability_is_eligible "$control_frozen"; then
      if [[ -f "$report" && ! -L "$report" \
         && -s "$report" && "$report" -nt "$run_boundary" ]]; then
        report_tuple=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$report") \
          || report_tuple=""
        if [[ -n "$report_tuple" && "$report_tuple" == "$previous_tuple" ]]; then
          (( stable_polls += 1 ))
        else
          stable_polls=1
          previous_tuple=$report_tuple
        fi
      else
        stable_polls=0
        previous_tuple=""
      fi
    else
      stable_polls=0
      previous_tuple=""
    fi

    if (( stable_polls >= LUMO_NSYS_REPORT_STABLE_POLLS )); then
      if verify_report_readable "$report" "$readability_output" \
          && [[ -f "$report" && ! -L "$report" ]] \
          && [[ "$(stat -c '%d:%i:%h:%s:%Y:%Z' "$report" 2>/dev/null)" \
            == "$report_tuple" ]]; then
        report_sha256=$(
          timeout --signal=TERM \
            --kill-after="${LUMO_NSYS_HASH_KILL_AFTER_S}s" \
            "${LUMO_NSYS_HASH_TIMEOUT_S}s" \
            sha256sum -- "$report" \
            | awk 'NR == 1 {print $1}'
        ) || report_sha256=""
        if [[ "$report_sha256" =~ ^[0-9a-f]{64}$ ]] \
            && [[ "$(stat -c '%d:%i:%h:%s:%Y:%Z' "$report" 2>/dev/null)" \
              == "$report_tuple" ]]; then
          NSYS_PROVEN_REPORT_IDENTITY=$report_tuple
          NSYS_PROVEN_REPORT_SHA256=$report_sha256
        else
          _lifecycle_log \
            "WARN: stable report changed or could not be hashed after readability proof"
          stable_polls=0
          previous_tuple=""
          continue
        fi
        _lifecycle_log \
          "fresh report is stable, readable, and cryptographically latched"
        return 0
      fi
      _lifecycle_log "WARN: stable report is not yet readable via pinned nsys"
      stable_polls=0
      previous_tuple=""
    fi

    if (( session_started_at < 0 )) \
        && (( SECONDS - identity_wait_started_at \
          >= LUMO_NSYS_BOOT_TIMEOUT_S )); then
      fail_nsys_lifecycle \
        "timed out waiting for the run-bound EngineCore Nsight identity"
      return 1
    fi
    if (( session_started_at >= 0 )) \
        && [[ -z "$NSYS_SESSION_ID" ]] \
        && (( SECONDS - session_started_at >= LUMO_NSYS_SESSION_TIMEOUT_S )); then
      fail_nsys_lifecycle "timed out waiting for the fresh run Nsight session"
      return 1
    fi
    if (( session_started_at >= 0 )) \
        && [[ -n "$NSYS_SESSION_ID" ]] \
        && (( control_frozen == 0 )) \
        && (( SECONDS - session_started_at >= LUMO_NSYS_COLLECTION_TIMEOUT_S )); then
      fail_nsys_lifecycle "timed out waiting for Nsight Collection entry"
      return 1
    fi
    if (( collection_started_at >= 0 && generation_started_at < 0 )) \
        && (( SECONDS - collection_started_at >= LUMO_NSYS_COLLECTION_MAX_S )); then
      if (( NSYS_SESSION_QUERY_OK == 1 )) \
          && (( NSYS_SESSION_PRESENT == 1 )) \
          && [[ "$NSYS_SESSION_STATE" == "Collection" ]] \
          && (( forced_stop == 0 )); then
        forced_stop=1
        if ! stop_exact_nsys_session; then
          refresh_run_nsys_session
          refresh_after_stop_rc=$?
          if (( refresh_after_stop_rc != 0 \
              || NSYS_SESSION_QUERY_OK != 1 )); then
            fail_nsys_lifecycle \
              "could not revalidate the exact Nsight session after stop failure"
            return 1
          fi
          if (( NSYS_SESSION_PRESENT == 1 )) \
              && [[ "$NSYS_SESSION_STATE" == "Collection" ]]; then
            fail_nsys_lifecycle \
              "Nsight collection exceeded its bound and exact session stop failed"
            return 1
          fi
        fi
        generation_started_at=$SECONDS
      elif (( NSYS_SESSION_QUERY_OK == 1 )); then
        generation_started_at=$SECONDS
        _lifecycle_log \
          "exact Nsight session left Collection before a forced stop was needed"
      elif (( SECONDS - collection_started_at \
          >= LUMO_NSYS_COLLECTION_MAX_S + LUMO_NSYS_STOP_TIMEOUT_S )); then
        fail_nsys_lifecycle \
          "could not revalidate the exact Nsight session at the collection bound"
        return 1
      fi
    fi
    if (( generation_started_at >= 0 )) \
        && (( SECONDS - generation_started_at >= LUMO_NSYS_REPORT_TIMEOUT_S )); then
      fail_nsys_lifecycle \
        "timed out waiting for a stable readable Nsight report"
      return 1
    fi
    if ! _process_identity_is_live "$driver_pid" "$driver_start" \
        && (( control_frozen == 0 )); then
      fail_nsys_lifecycle \
        "campaign driver exited before teardown controls were frozen"
      return 1
    fi
    sleep "$LUMO_NSYS_POLL_S"
  done
}

validate_nsys_delayed_collection_timeouts() {
  local minimum_timeout_s=$((LUMO_NSYS_DELAY_S + LUMO_NSYS_DURATION_S))

  (( LUMO_NSYS_BOOT_TIMEOUT_S >= ${HEALTH_TIMEOUT_S:-1200} )) || {
    echo \
      "FAIL: EngineCore identity timeout must cover the server health timeout" \
      >&2
    return 1
  }
  (( LUMO_NSYS_SESSION_TIMEOUT_S >= minimum_timeout_s )) || {
    echo \
      "FAIL: Nsight session discovery timeout must cover delay plus capture duration" \
      >&2
    return 1
  }
  (( LUMO_NSYS_COLLECTION_TIMEOUT_S >= minimum_timeout_s )) || {
    echo \
      "FAIL: Nsight Collection-entry timeout must cover delay plus capture duration" \
      >&2
    return 1
  }
}

validate_nsys_attribution_task_wall() {
  [[ "$LUMO_NSYS_SWE_AGENT_WALL_S" =~ ^[1-9][0-9]*$ ]] || {
    echo \
      "FAIL: attribution SWE task wall must be a strict positive integer" \
      >&2
    return 1
  }
  [[ "$LUMO_NSYS_DURATION_S" =~ ^[1-9][0-9]*$ ]] || {
    echo "FAIL: Nsight duration must be a strict positive integer" >&2
    return 1
  }
  (( 10#$LUMO_NSYS_SWE_AGENT_WALL_S >= 10#$LUMO_NSYS_DURATION_S )) || {
    echo \
      "FAIL: attribution SWE task wall must cover the Nsight capture duration" \
      >&2
    return 1
  }
}

profile_cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  thaw_exact_control_ancestors
  if (( PRESERVE_RECOVERABLE_STATE == 0 )); then
    if [[ -n "$PROFILE_CONTAINER_ID" ]] \
        && _reattest_profile_container "$CONTAINER"; then
      docker rm -f "$PROFILE_CONTAINER_ID" >/dev/null 2>&1 || true
    elif [[ -n "$PROFILE_CONTAINER_ID" ]]; then
      _lifecycle_log \
        "container cleanup skipped because immutable ID/name re-attestation failed"
    fi
  else
    _lifecycle_log \
      "skipping container removal; recovery state is intentionally preserved"
  fi
  exit "$rc"
}

# Tests source the lifecycle functions without executing a real GPU campaign.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO=$(cd "$REPO" && pwd)
cd "$REPO"

STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "FAIL: profiler STAMP must be a UTC basic timestamp" >&2
  exit 2
}
TAG=${TAG:-b1_nsys_f32_${STAMP}}
RUNROOT=${RUNROOT:-output/fr13_fixed32_b1_nsys_${STAMP}}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
ARM=tail6_fixed32_${TAG}
CONTAINER=fr13-bigdenom-${ARM}
NSYS_EXPECTED_SESSION_NAME="fr13-fixed32-${STAMP}-p$$"
REPORT="$RUNROOT/$ARM/logs/fr13_fixed32_b1_real_swe.nsys-rep"
READABILITY="$RUNROOT/$ARM/logs/fr13_fixed32_b1_real_swe.readability.info"
REDUCED="$RUNROOT/$ARM/logs/fr13_fixed32_b1_nsys_attribution.json"

running_containers=$(docker ps -q) || {
  echo "FAIL: unable to enumerate running Docker containers" >&2
  exit 2
}
if [[ -n "$running_containers" ]]; then
  echo "FAIL: GPU campaign containers are already running" >&2
  exit 2
fi
same_name_containers=$(
  docker ps -aq --filter "name=^/${CONTAINER}$"
) || {
  echo "FAIL: unable to check the exact fixed32 Docker container name" >&2
  exit 2
}
if [[ -n "$same_name_containers" ]]; then
  echo "FAIL: fixed32 exact container name already exists: $CONTAINER" >&2
  exit 2
fi
unset running_containers same_name_containers

export LUMO_SWE_AUTOCOMMIT=0
export FR13_FIXED32_ATTRIBUTION_ONLY=1
export FR13_FIXED32_NVTX_PROFILE=1
export LUMO_NSYS_WRAP_VLLM=1
export LUMO_NSYS_SESSION_NAME="$NSYS_EXPECTED_SESSION_NAME"
export LUMO_NSYS_BIN=${LUMO_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}
export LUMO_NSYS_TRACE=${LUMO_NSYS_TRACE:-cuda,cuda-sw,nvtx}
export LUMO_NSYS_DELAY_S=${LUMO_NSYS_DELAY_S:-1200}
export LUMO_NSYS_DURATION_S=${LUMO_NSYS_DURATION_S:-300}
export LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}
export LUMO_NSYS_CONFIG_DIRECTIVES="${LUMO_NSYS_CONFIG_DIRECTIVES:-CuptiUseRawGpuTimestamps=false}"
export LUMO_NSYS_OUTPUT=/logs/fr13_fixed32_b1_real_swe
export HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1200}
LUMO_NSYS_BOOT_TIMEOUT_S=${LUMO_NSYS_BOOT_TIMEOUT_S:-1800}
LUMO_NSYS_SESSION_TIMEOUT_S=${LUMO_NSYS_SESSION_TIMEOUT_S:-1500}
LUMO_NSYS_COLLECTION_TIMEOUT_S=${LUMO_NSYS_COLLECTION_TIMEOUT_S:-1500}
LUMO_NSYS_COLLECTION_MAX_S=${LUMO_NSYS_COLLECTION_MAX_S:-420}
LUMO_NSYS_REPORT_TIMEOUT_S=${LUMO_NSYS_REPORT_TIMEOUT_S:-900}
LUMO_NSYS_QUERY_TIMEOUT_S=${LUMO_NSYS_QUERY_TIMEOUT_S:-10}
LUMO_NSYS_QUERY_KILL_AFTER_S=${LUMO_NSYS_QUERY_KILL_AFTER_S:-2}
LUMO_NSYS_QUERY_OUTER_TIMEOUT_S=${LUMO_NSYS_QUERY_OUTER_TIMEOUT_S:-15}
LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S=${LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S:-2}
LUMO_NSYS_STOP_TIMEOUT_S=${LUMO_NSYS_STOP_TIMEOUT_S:-60}
LUMO_NSYS_STOP_KILL_AFTER_S=${LUMO_NSYS_STOP_KILL_AFTER_S:-5}
LUMO_NSYS_STOP_OUTER_TIMEOUT_S=${LUMO_NSYS_STOP_OUTER_TIMEOUT_S:-70}
LUMO_NSYS_STOP_OUTER_KILL_AFTER_S=${LUMO_NSYS_STOP_OUTER_KILL_AFTER_S:-5}
LUMO_NSYS_EXPORT_TIMEOUT_S=${LUMO_NSYS_EXPORT_TIMEOUT_S:-300}
LUMO_NSYS_EXPORT_KILL_AFTER_S=${LUMO_NSYS_EXPORT_KILL_AFTER_S:-5}
LUMO_NSYS_HASH_TIMEOUT_S=${LUMO_NSYS_HASH_TIMEOUT_S:-300}
LUMO_NSYS_HASH_KILL_AFTER_S=${LUMO_NSYS_HASH_KILL_AFTER_S:-5}
LUMO_NSYS_REPORT_STABLE_POLLS=${LUMO_NSYS_REPORT_STABLE_POLLS:-3}
LUMO_NSYS_POLL_S=${LUMO_NSYS_POLL_S:-2}
LUMO_NSYS_SWE_AGENT_WALL_S=${LUMO_NSYS_SWE_AGENT_WALL_S:-900}
JQ_BIN=${JQ_BIN:-$(command -v jq)}

OUTPUT_ROOT=$(realpath -m "$REPO/output")
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ENGINE_LEDGER_SNAPSHOT=\
"$RUNROOT_ABS/fr13_fixed32_engine_ingress.container_snapshot.jsonl"
PROFILE_CONTAINER_CIDFILE=\
"$RUNROOT_ABS/$ARM/logs/fr13_fixed32_container.cid"
PROCESS_IDENTITY="$RUNROOT_ABS/$ARM/fixed32_process_identity.json"
case "$RUNROOT_ABS" in
  "$OUTPUT_ROOT"/*) ;;
  *)
    echo "FAIL: raw profiler artifacts must remain below ignored output/" >&2
    exit 2
    ;;
esac
if [[ -e "$RUNROOT_ABS" ]]; then
  echo "FAIL: profiler RUNROOT must be new (stale evidence is forbidden)" >&2
  exit 2
fi
git check-ignore -q "$RUNROOT_ABS" || {
  echo "FAIL: raw profiler RUNROOT is not ignored by Git" >&2
  exit 2
}
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] || {
  echo "FAIL: canonical exact4 SWE-Verified subset hash drift" >&2
  exit 2
}
[[ "$LUMO_NSYS_BIN" == "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys" ]] || {
  echo "FAIL: fixed32 attribution requires the pinned Nsight Systems binary" >&2
  exit 2
}
[[ -x "$LUMO_NSYS_BIN" ]] || {
  echo "FAIL: Nsight Systems executable is unavailable" >&2
  exit 2
}
[[ "$LUMO_NSYS_TRACE" == "cuda,cuda-sw,nvtx" ]] || {
  echo "FAIL: fixed32 attribution requires cuda,cuda-sw,nvtx tracing" >&2
  exit 2
}
[[ "$LUMO_NSYS_SESSION_NAME" == "$NSYS_EXPECTED_SESSION_NAME" ]] \
  && [[ "$LUMO_NSYS_SESSION_NAME" =~ \
    ^fr13-fixed32-[0-9]{8}T[0-9]{6}Z-p[1-9][0-9]*$ ]] || {
  echo "FAIL: fixed32 attribution requires a pinned run-unique Nsight session name" >&2
  exit 2
}
for _positive_nsys_value in \
  "$LUMO_NSYS_DELAY_S" "$LUMO_NSYS_DURATION_S" "$LUMO_NSYS_FLUSH_MS"; do
  [[ "$_positive_nsys_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FAIL: Nsight delay, duration, and flush interval must be positive integers" >&2
    exit 2
  }
done
unset _positive_nsys_value
[[ "$LUMO_NSYS_DELAY_S" == "1200" && "$LUMO_NSYS_DURATION_S" == "300" ]] || {
  echo "FAIL: attribution requires the canonical 1200s delay/300s capture" >&2
  exit 2
}
[[ "$LUMO_NSYS_FLUSH_MS" == "100" ]] || {
  echo "FAIL: attribution requires a 100ms CUDA flush interval" >&2
  exit 2
}
[[ "$LUMO_NSYS_CONFIG_DIRECTIVES" == "CuptiUseRawGpuTimestamps=false" ]] || {
  echo "FAIL: fixed32 attribution Nsight config directive drift" >&2
  exit 2
}
[[ -x "$JQ_BIN" ]] || {
  echo "FAIL: jq is required for structured Nsight session tracking" >&2
  exit 2
}
command -v timeout >/dev/null || {
  echo "FAIL: timeout is required for bounded Nsight session stop" >&2
  exit 2
}
for _positive_lifecycle_value in \
  "$HEALTH_TIMEOUT_S" \
  "$LUMO_NSYS_BOOT_TIMEOUT_S" \
  "$LUMO_NSYS_SESSION_TIMEOUT_S" \
  "$LUMO_NSYS_COLLECTION_TIMEOUT_S" \
  "$LUMO_NSYS_COLLECTION_MAX_S" \
  "$LUMO_NSYS_REPORT_TIMEOUT_S" \
  "$LUMO_NSYS_QUERY_TIMEOUT_S" \
  "$LUMO_NSYS_QUERY_KILL_AFTER_S" \
  "$LUMO_NSYS_QUERY_OUTER_TIMEOUT_S" \
  "$LUMO_NSYS_QUERY_OUTER_KILL_AFTER_S" \
  "$LUMO_NSYS_STOP_TIMEOUT_S" \
  "$LUMO_NSYS_STOP_KILL_AFTER_S" \
  "$LUMO_NSYS_STOP_OUTER_TIMEOUT_S" \
  "$LUMO_NSYS_STOP_OUTER_KILL_AFTER_S" \
  "$LUMO_NSYS_EXPORT_TIMEOUT_S" \
  "$LUMO_NSYS_EXPORT_KILL_AFTER_S" \
  "$LUMO_NSYS_HASH_TIMEOUT_S" \
  "$LUMO_NSYS_HASH_KILL_AFTER_S" \
  "$LUMO_NSYS_POLL_S"; do
  [[ "$_positive_lifecycle_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FAIL: Nsight lifecycle timeouts and poll interval must be positive integers" >&2
    exit 2
  }
done
unset _positive_lifecycle_value
(( LUMO_NSYS_QUERY_OUTER_TIMEOUT_S \
    > LUMO_NSYS_QUERY_TIMEOUT_S + LUMO_NSYS_QUERY_KILL_AFTER_S )) || {
  echo "FAIL: host Nsight query bound must outlive its in-container bound" >&2
  exit 2
}
(( LUMO_NSYS_STOP_OUTER_TIMEOUT_S \
    > LUMO_NSYS_STOP_TIMEOUT_S + LUMO_NSYS_STOP_KILL_AFTER_S )) || {
  echo "FAIL: host Nsight stop bound must outlive its in-container bound" >&2
  exit 2
}
validate_nsys_delayed_collection_timeouts || exit 2
validate_nsys_attribution_task_wall || exit 2
[[ "$LUMO_NSYS_REPORT_STABLE_POLLS" =~ ^[1-9][0-9]*$ ]] \
  && (( LUMO_NSYS_REPORT_STABLE_POLLS >= 3 )) || {
  echo "FAIL: Nsight report stability requires at least three polls" >&2
  exit 2
}
mkdir -m 700 "$RUNROOT_ABS"
NSYS_LIFECYCLE_LOG="$RUNROOT_ABS/nsys_lifecycle.log"
RUN_BOUNDARY="$RUNROOT_ABS/.nsys_report_run_boundary"
touch "$RUN_BOUNDARY"
[[ ! -e "$REPORT" && ! -e "$READABILITY" ]] || {
  echo "FAIL: fresh run-bound profiler outputs already exist" >&2
  exit 2
}
trap profile_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
RUNROOT="$RUNROOT" \
TAG="$TAG" \
SUBSET="$SUBSET" \
BSIZE=1 \
CONC=1 \
WALL=0 \
SWE_AGENT_WALL_S="$LUMO_NSYS_SWE_AGENT_WALL_S" \
DEPLOY_FORCE_TEMP=0.6 \
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
  bash scripts/fr13_b4_campaign_driver.sh \
  >"$RUNROOT/driver.log" 2>&1 &
DRIVER_PID=$!
DRIVER_START_TICKS=$(_process_start_ticks "$DRIVER_PID")
if [[ -z "$DRIVER_START_TICKS" ]]; then
  echo "FAIL: unable to attest asynchronous campaign driver PID" >&2
  exit 3
fi
_lifecycle_log \
  "launched async campaign driver pid=$DRIVER_PID start_ticks=$DRIVER_START_TICKS"

if ! wait_for_fresh_stable_report \
    "$DRIVER_PID" "$DRIVER_START_TICKS" \
    "$REPORT" "$READABILITY" "$RUN_BOUNDARY"; then
  echo "FAIL: $NSYS_LIFECYCLE_ERROR" >&2
  [[ -n "$PRESERVED_CONTAINER" ]] \
    && echo "RECOVERY: container preserved as $PRESERVED_CONTAINER" >&2
  exit 3
fi
if [[ ! -s "$REPORT" ]]; then
  echo "FAIL: bounded real-SWE Nsight report was not produced" >&2
  exit 3
fi
if [[ ! "$NSYS_PROVEN_REPORT_IDENTITY" =~ \
    ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] \
    || [[ ! "$NSYS_PROVEN_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "FAIL: lifecycle-proven report identity was not retained" >&2
  exit 3
fi
if ! snapshot_terminal_engine_ledger "$ENGINE_LEDGER_SNAPSHOT"; then
  snapshot_error=$NSYS_LIFECYCLE_ERROR
  fail_nsys_lifecycle "$snapshot_error" || true
  echo "FAIL: $snapshot_error" >&2
  [[ -n "$PRESERVED_CONTAINER" ]] \
    && echo "RECOVERY: container preserved as $PRESERVED_CONTAINER" >&2
  exit 3
fi
thaw_exact_control_ancestors
wait "$DRIVER_PID"
driver_rc=$?
set -e

if ! find "$RUNROOT/$ARM/swe_out/verified/per_task" -mindepth 1 -maxdepth 1 \
  -type d -name 'astropy__astropy-*' -print -quit 2>/dev/null \
  | grep -q .; then
  echo "FAIL: profiler window has no real SWE-Verified task evidence" >&2
  exit 4
fi

if ! .venv/bin/python scripts/fr13_fixed32_nsys_reduce.py \
  "$REPORT" \
  --output "$REDUCED" \
  --nsys-bin "$LUMO_NSYS_BIN" \
  --expected-report-identity "$NSYS_PROVEN_REPORT_IDENTITY" \
  --expected-report-sha256 "$NSYS_PROVEN_REPORT_SHA256" \
  --subset "$SUBSET" \
  --runtime-manifest-launch "$RUNROOT/runtime_manifest.at_launch.json" \
  --runtime-manifest-end "$RUNROOT/runtime_manifest.at_end.json" \
  --external-manifest-launch "$RUNROOT/external_manifest.at_launch.json" \
  --external-manifest-end "$RUNROOT/external_manifest.at_end.json" \
  --process-identity "$RUNROOT/$ARM/fixed32_process_identity.json" \
  --container-identity "$RUNROOT/$ARM/fixed32_container_identity.json" \
  --runtime-attestation \
    "$RUNROOT/$ARM/logs/fr13_fixed32_runtime_attestation.json" \
  --pretask-zero-traffic "$RUNROOT/$ARM/fixed32_pretask_zero_traffic.json" \
  --proxy-ledger "$RUNROOT/$ARM/logs/fr13_fixed32_proxy_ingress.jsonl" \
  --engine-ledger "$ENGINE_LEDGER_SNAPSHOT" \
  --mode tail6_fixed32 \
  --batch-size 1 \
  --concurrency 1 \
  --driver-rc "$driver_rc" \
  --nsys-delay-s "$LUMO_NSYS_DELAY_S" \
  --nsys-duration-s "$LUMO_NSYS_DURATION_S" \
  --nsys-flush-ms "$LUMO_NSYS_FLUSH_MS" \
  --nsys-trace "$LUMO_NSYS_TRACE" \
  --nsys-config-directives "$LUMO_NSYS_CONFIG_DIRECTIVES" \
  --nsys-discard-environment true; then
  echo "FAIL: privacy-safe Nsight attribution reduction failed" >&2
  exit 5
fi

printf '%s\n' \
  "attribution_only=true" \
  "acceptance_valid=false" \
  "driver_rc=$driver_rc" \
  "report_bytes=$(stat -c %s "$REPORT")" \
  "reduced_sha256=$(sha256sum "$REDUCED" | awk '{print $1}')"
