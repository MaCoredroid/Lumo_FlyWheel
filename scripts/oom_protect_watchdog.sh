#!/usr/bin/env bash
# Persistent OOM-protection watchdog for the GB10 (unified memory) tmux/Claude session.
# Re-applies oom_score_adj=-1000 every few seconds to the tmux server + every claude
# process tree, so a Claude OOM-RESTART (which spawns a fresh, unprotected process at
# oom_score_adj=0) is re-protected within one tick. Detached + self-protected.
# Needs LUMO_SUDO_PASSWORD (.lumo.local.env). Stop with: pkill -f oom_protect_watchdog
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
# shellcheck disable=SC1091
source .lumo.local.env 2>/dev/null || true
PW="${LUMO_SUDO_PASSWORD:-}"
[ -z "$PW" ] && { echo "FAIL: LUMO_SUDO_PASSWORD not set"; exit 1; }
INTERVAL="${OOM_WATCHDOG_INTERVAL:-3}"

# protect self first
echo -1000 > "/proc/$$/oom_score_adj" 2>/dev/null || true

while true; do
  PIDS=$( { pgrep -x tmux; pgrep -f 'claude'; pgrep -f 'oom_protect_watchdog'; } 2>/dev/null \
          | sort -u | tr '\n' ' ' )
  # only those not already -1000 (cheap check avoids sudo churn)
  NEED=""
  for p in $PIDS; do
    cur=$(cat "/proc/$p/oom_score_adj" 2>/dev/null || echo 0)
    [ "$cur" != "-1000" ] && NEED="$NEED $p"
  done
  if [ -n "${NEED// }" ]; then
    printf '%s\n' "$PW" | sudo -S bash -lc "for p in $NEED; do echo -1000 > /proc/\$p/oom_score_adj 2>/dev/null; done" 2>/dev/null
  fi
  sleep "$INTERVAL"
done
