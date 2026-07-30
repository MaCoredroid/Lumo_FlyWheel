#!/usr/bin/env bash
# GPU/UNIFIED-MEMORY OOM GUARD for the GB10 (117 GiB unified host+GPU).
#
# WHY THIS EXISTS (2026-06-24, after ~8 session OOMs): on unified memory a vLLM
# diagnostic boot can drive *GPU* allocation to the wall -> NVRM_NO_MEMORY -> the
# kernel's GLOBAL oom-killer reclaims by nuking every positive-oom_score_adj
# process, which on this box is the WHOLE systemd --user session stack
# (systemd --user, sd-pam, pipewire, dbus, xdg-*). Killing sd-pam/systemd --user
# tears down the login session via SIGTERM cascade -- and that cascade is NOT
# governed by oom_score_adj. So the -1000 on claude/tmux CANNOT save the session
# from a GPU OOM (proven in dmesg: claude was never a direct victim, yet died
# with the session). The docker --memory cap also can't help: GPU allocations via
# the NVIDIA runtime bypass the container cgroup.
#
# THE ONLY robust lever is to never reach the wall. This guard converts an
# *incipient* GPU OOM into a CONTAINER KILL (relaunchable, loud) well before the
# kernel decides to free memory by killing the session. It polls unified-mem
# `available` (which == GPU-free on this box) and `docker kill`s the fr13
# container if it drops below the floor for 2 consecutive reads.
#
# Detached + self-protected. Exits cleanly once no fr13 container is alive.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$REPO" 2>/dev/null || true

FLOOR_MIB=${GPU_GUARD_FLOOR_MIB:-9000}   # kill container if avail < this (MiB)
POLL_S=${GPU_GUARD_POLL_S:-2}
NAME_GLOB=${GPU_GUARD_NAME_GLOB:-fr13}    # container name substring to protect against
CONTAINER_ID=${GPU_GUARD_CONTAINER_ID:-}
EXPECTED_NAME=${GPU_GUARD_EXPECTED_NAME:-}
LOG=${GPU_GUARD_LOG:-output/gpu_oom_guard.log}
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

# self-protect: this guard must NOT be the victim
echo $$ > /proc/self/oom_score_adj 2>/dev/null || true
echo -1000 > /proc/self/oom_score_adj 2>/dev/null || true

_log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

if [[ -n "$CONTAINER_ID" || -n "$EXPECTED_NAME" ]]; then
  [[ "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ && -n "$EXPECTED_NAME" ]] || {
    _log "invalid immutable container guard identity"
    exit 2
  }
  _log \
    "gpu_oom_guard START floor=${FLOOR_MIB}MiB poll=${POLL_S}s id=$CONTAINER_ID expected_name=$EXPECTED_NAME pid=$$"
else
  _log "gpu_oom_guard START floor=${FLOOR_MIB}MiB poll=${POLL_S}s glob=$NAME_GLOB pid=$$"
fi

_immutable_container_is_live() {
  local identity=""
  local actual_id=""
  local actual_name=""
  local running=""
  local extra=""
  identity=$(
    docker inspect --format '{{.Id}} {{.Name}} {{.State.Running}}' \
      "$CONTAINER_ID" 2>/dev/null
  ) || return 1
  read -r actual_id actual_name running extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$CONTAINER_ID" ]] \
    && [[ "$actual_name" == "/$EXPECTED_NAME" ]] \
    && [[ "$running" == "true" ]]
}

low=0
idle=0
while :; do
  avail=$(free -m | awk '/Mem/{print $7}')
  if [[ -n "$CONTAINER_ID" ]]; then
    if ! _immutable_container_is_live; then
      _log "immutable container disappeared, stopped, or changed name; guard EXIT"
      exit 0
    fi
    cont=$CONTAINER_ID
  else
    cont=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i "$NAME_GLOB" | head -1)
  fi
  if [ -z "$cont" ]; then
    idle=$((idle+1))
    # no protected container for ~30s -> nothing to guard, exit so we don't linger
    [ "$idle" -ge $((30/POLL_S)) ] && { _log "no '$NAME_GLOB' container for ~30s; guard EXIT"; exit 0; }
    sleep "$POLL_S"; continue
  fi
  idle=0
  if [ "${avail:-99999}" -lt "$FLOOR_MIB" ]; then
    low=$((low+1))
    _log "WARN avail=${avail}MiB < floor=${FLOOR_MIB} (strike $low/2) container=$cont"
    if [ "$low" -ge 2 ]; then
      if [[ -n "$CONTAINER_ID" ]] \
          && ! _immutable_container_is_live; then
        _log "immutable identity changed before kill; guard EXIT without action"
        exit 0
      fi
      _log "!!! GPU-OOM IMMINENT (avail=${avail}MiB) -> docker kill $cont to save the session"
      if ! docker kill "$cont" >/dev/null 2>&1; then
        if [[ -z "$CONTAINER_ID" ]] || _immutable_container_is_live; then
          docker rm -f "$cont" >/dev/null 2>&1 || true
        fi
      fi
      _log "killed $cont; guard continues watching"
      low=0
    fi
  else
    low=0
  fi
  sleep "$POLL_S"
done
