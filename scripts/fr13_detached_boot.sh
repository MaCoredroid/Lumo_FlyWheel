#!/usr/bin/env bash
# FR13 detached-boot wrapper: run a serve-variant arm in its OWN session so a
# SIGTERM to the launching (background) shell CANNOT propagate to the container.
# The serve script's `trap teardown EXIT` binds the container's life to its
# process; when the launcher shell is stopped by the harness, the EXIT trap
# fires -> docker rm -f. setsid puts the serve in a new session/pgroup that the
# harness's task-stop (which signals the launcher's pgroup) never reaches.
#
# Also installs a signal logger: if the serve IS still signaled, we record the
# signal + timestamp + parent chain to $ARMDIR/death.log to catch the killer.
#
#   usage: fr13_detached_boot.sh <arm> <cmd...>
#   the caller pre-exports all FR13_*/PROBE_* env; RUNROOT must be set.
set -uo pipefail
ARM="$1"; shift
RR="${RUNROOT:?RUNROOT required}"
mkdir -p "$RR/$ARM"
DL="$RR/$ARM/death.log"
PIDF="$RR/$ARM/detached.pid"

# The serve command runs inside a subshell that traps signals (logs who/when)
# and is launched with setsid + nohup + stdin</dev/null so it is fully
# detached from the launcher's controlling terminal and process group.
setsid bash -c '
  trap "echo \"SIG=TERM t=$(date -u +%H:%M:%S) ppid=$(ps -o ppid= -p $$ 2>/dev/null) pgrp=$(ps -o pgid= -p $$ 2>/dev/null)\" >> '"$DL"'" TERM
  trap "echo \"SIG=HUP  t=$(date -u +%H:%M:%S) ppid=$(ps -o ppid= -p $$ 2>/dev/null)\" >> '"$DL"'" HUP
  trap "echo \"SIG=INT  t=$(date -u +%H:%M:%S)\" >> '"$DL"'" INT
  echo "detached-serve start t=$(date -u +%H:%M:%S) pid=$$ session=$(ps -o sid= -p $$ 2>/dev/null)" >> '"$DL"'
  '"$*"'
  echo "detached-serve exit rc=$? t=$(date -u +%H:%M:%S)" >> '"$DL"'
' >/dev/null 2>&1 < /dev/null &
DPID=$!
echo "$DPID" > "$PIDF"
disown "$DPID" 2>/dev/null || true
echo "detached arm=$ARM launcher_pid=$DPID (session-detached); poll $RR/$ARM + death.log"
