#!/usr/bin/env bash
# OOM-protect the tmux / Claude-Code monitoring session on the GB10 (unified memory).
# Sets oom_score_adj=-1000 (exempt) on the tmux server + every tmux client + the
# claude process tree (claude + ancestors up to the tmux pane + all descendants),
# AND on the tmux server so panes/shells forked LATER inherit the exemption.
# Effect: when host memory is exhausted the OOM killer can no longer pick the
# monitoring stack, so it takes the relaunchable vLLM docker container instead.
# Idempotent + re-runnable. Needs LUMO_SUDO_PASSWORD (from .lumo.local.env).
#   usage: bash scripts/oom_protect_session.sh   (run once per session, or via cron tick)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
# shellcheck disable=SC1091
source .lumo.local.env 2>/dev/null || true
PW="${LUMO_SUDO_PASSWORD:-}"
if [ -z "$PW" ]; then echo "FAIL: LUMO_SUDO_PASSWORD not set (source .lumo.local.env)"; exit 1; fi

# Build the PID set: tmux (server+clients) + the claude tree (ancestors+descendants).
PIDS=$(.venv/bin/python - <<'PY'
import os, glob
def comm(p):
    try: return open(f"/proc/{p}/comm").read().strip()
    except Exception: return ""
def ppid(p):
    try:
        # field 4 of stat is ppid; handle comm with spaces via the ')' split
        s=open(f"/proc/{p}/stat").read(); after=s[s.rindex(')')+2:].split()
        return int(after[1])
    except Exception: return 0
def cmdline(p):
    try: return open(f"/proc/{p}/cmdline").read().replace("\0"," ")
    except Exception: return ""
allp=[int(os.path.basename(d)) for d in glob.glob("/proc/[0-9]*")]
keep=set()
# tmux server + clients
for p in allp:
    if comm(p).startswith("tmux") or "tmux" in cmdline(p).split(" ")[0:1]:
        keep.add(p)
# claude roots
roots=[p for p in allp if comm(p)=="claude" or " claude " in (" "+cmdline(p)+" ") or cmdline(p).startswith("claude ")]
# ancestors of each claude root (stop at init/0)
for r in roots:
    keep.add(r); a=ppid(r)
    while a and a>1: keep.add(a); a=ppid(a)
# descendants (transitive) of tmux+claude roots
children={}
for p in allp:
    children.setdefault(ppid(p),[]).append(p)
stack=list(keep)
while stack:
    p=stack.pop()
    for c in children.get(p,[]):
        if c not in keep: keep.add(c); stack.append(c)
# never touch pid 1 / kthreads
keep.discard(1)
print(" ".join(str(p) for p in sorted(keep)))
PY
)
if [ -z "${PIDS// }" ]; then echo "FAIL: no tmux/claude pids found"; exit 1; fi
echo "[oom-protect] pids: $PIDS"

# Write -1000 via a single sudo bash (one prompt). -1000 = exempt from OOM killer.
CMD="for p in $PIDS; do echo -1000 > /proc/\$p/oom_score_adj 2>/dev/null && echo \"  set \$p -> \$(cat /proc/\$p/oom_score_adj) (\$(cat /proc/\$p/comm 2>/dev/null))\"; done"
printf '%s\n' "$PW" | sudo -S bash -lc "$CMD"
echo "[oom-protect] done. (re-run after spawning new long-lived shells; the tmux server is set so most children inherit it.)"
