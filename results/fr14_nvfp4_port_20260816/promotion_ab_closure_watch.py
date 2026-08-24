#!/usr/bin/env python3
"""EXECUTION-CLOSURE WATCH — interim protocol after site 24.

Site 24: the 24k-ceiling landing edited fr13_bigdenom_swe_serve_variant.sh fourteen
minutes into a live drain. bash reads a script by INCREMENTAL BYTE OFFSET, so when the
orchestrator returned after 13398 the shell resumed at a stale offset into rewritten
bytes and executed fragment text ("cho", the tail of an echo). The QC died at 3/15 and
the cause was invisible until the corpse was read.

I cannot stop another lane from landing. I CAN make the collision announce itself:
snapshot the closure at boot, and compare on every watch tick. An alarm at minute
fifteen costs one boot; discovery at hour four costs the drain.

    snapshot  -> writes closure.json into the runroot at boot
    check     -> compares live state to the snapshot; exit 1 and ALARM on any drift

The closure is the set of files a fixed32 boot actually executes from the repo, plus
HEAD. Bash-sourced files are listed first because they carry the byte-offset hazard;
Python files are re-imported per process and merely need to be consistent.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816")

# BASH — the byte-offset hazard lives here. A running shell re-reads these by offset.
BASH_CLOSURE = [
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_canonical_env.sh",
    "scripts/fr13_fixed32_floor_timers_seq.sh",
    "scripts/fr13_required_tree_flags.sh",
    "scripts/swe_x86_helpers/offload_codex_proxy.sh",
]
# PYTHON — imported per process; drift still invalidates provenance.
PY_CLOSURE = [
    "scripts/fr13_patch_fa2_tree_bias.py",
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_device_multidraft_kernel.py",
    "scripts/fr13_fixed32_topology.py",
    "scripts/fr13_fixed32_contract.py",
    "scripts/fr13_floor_gate.py",
    "scripts/fr13_qrow32_b1_pass_sidecar.py",
    "scripts/run_swe_bench_q36_a.py",
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
]


def _stat(rel: str) -> dict | None:
    p = REPO / rel
    if not p.is_file():
        return None
    b = p.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b),
            "mtime": int(p.stat().st_mtime)}


def _live_executors(rel: str) -> list[int]:
    """PIDs whose command line names this exact SOURCE path.

    WHY THIS EXISTS. Before site 24 a changed bash file was always a byte-offset
    hazard: a running shell re-reads its source by offset, so an edit landing
    mid-run could make it resume inside a different file. Site 24 removed that for
    the RESIDENT set -- every resident script now execs a content-addressed snapshot
    under tmp-scratch/fr13_snapshots and never reads its source again -- and the
    non-resident set (the launcher) returns within minutes of boot.

    So "this file changed" no longer implies "a shell is mis-reading it". The hazard
    needs a LIVE process executing THAT PATH. A snapshot execution names the snapshot,
    a different string, so it correctly does not match here: editing a source cannot
    disturb a snapshot taken from it.

    Read /proc directly rather than shelling out to pgrep, and match WHOLE ARGV TOKENS
    rather than substrings. Both of those are scar tissue from real false positives:

      * pgrep puts the pattern into its own command line and matches itself. That
        happened while diagnosing the 2026-08-24 firing and briefly looked like a live
        executor of a script that had exited two hours earlier.
      * substring matching reproduces the bug through a different door. Any shell
        invoked as `bash -c '<script text mentioning this path>'` carries the path
        inside ONE argv token, so a substring test reports the observer's own shell as
        an executor. The mutation proof for this function caught exactly that.

    A token equal to the path -- or whose realpath is -- is an actual executed file.
    """
    target = (REPO / rel).resolve()
    me = {os.getpid(), os.getppid()}
    hits: list[int] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in me:
            continue
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue  # exited between listdir and open; cannot be a live executor
        for tok in raw.decode("utf-8", "replace").split("\0"):
            if not tok:
                continue
            if tok == rel or tok == str(target):
                hits.append(int(pid))
                break
            try:
                if Path(tok).resolve() == target:
                    hits.append(int(pid))
                    break
            except (OSError, ValueError):
                continue
    return hits


def snapshot() -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=no"],
        capture_output=True, text=True).stdout.strip()
    return {
        "schema": "fr14.closure_watch.v1",
        "head": head,
        "dirty_at_boot": [l.strip() for l in dirty.splitlines() if l.strip()],
        "bash": {r: _stat(r) for r in BASH_CLOSURE},
        "python": {r: _stat(r) for r in PY_CLOSURE},
    }


def check(snap_path: Path) -> int:
    snap = json.loads(snap_path.read_text())
    now = snapshot()
    alarms = []
    if now["head"] != snap["head"]:
        alarms.append(("HEAD", snap["head"][:12], now["head"][:12], "moved"))
    for kind in ("bash", "python"):
        for rel, was in snap[kind].items():
            is_ = now[kind].get(rel)
            if was is None and is_ is None:
                continue
            if was is None or is_ is None or was["sha256"] != is_["sha256"]:
                if kind == "bash":
                    pids = _live_executors(rel)
                    sev = (f"CRITICAL (bash byte-offset hazard; live executors {pids})"
                           if pids else
                           "changed-after-use (bash, but NO live process executes this "
                           "path: resident scripts run from their site-24 snapshot and "
                           "non-resident ones have already exited)")
                else:
                    sev = "provenance"
                alarms.append((rel, (was or {}).get("sha256", "absent")[:12],
                               (is_ or {}).get("sha256", "absent")[:12], sev))
    # SEVERITY SPLIT, added after the watch's FIRST live firing. HEAD moved for a
    # docs-only ledger commit while every closure file stayed byte-identical, and the
    # verdict read "treat the drain as compromised". That is a false alarm, and a watch
    # that cries wolf on every ledger commit is a watch nobody reads -- the same
    # detector decay this campaign keeps finding. A HEAD move with an unchanged closure
    # is a PROVENANCE NOTE; only a changed FILE is the site-24 hazard.
    # SECOND NARROWING, after the 2026-08-24 firing. The first split (below) fixed
    # HEAD-moved-but-files-clean. This one fixes the other half: a changed bash file
    # was still hard-coded CRITICAL, which was right BEFORE site 24 and wrong after
    # it. On 2026-08-24 the watch called a live drain "compromised" because a lane
    # edited fr13_launch_forked_fa2_tree_server.sh -- a script site 24 classified as
    # NON-RESIDENT, which had exited 14 minutes before the edit landed, while the
    # resident script was demonstrably running from its snapshot (three digests
    # agreeing). The alarm was true about the bytes and wrong about the consequence.
    # A watch that says "compromised" when nothing is compromised gets ignored, which
    # is the detector decay this campaign keeps finding in its own instruments.
    hazards = [a for a in alarms if str(a[3]).startswith("CRITICAL")]
    changed_after_use = [a for a in alarms
                         if a[0] != "HEAD" and str(a[3]).startswith("changed-after-use")]
    file_alarms = hazards
    head_moved = any(a[0] == "HEAD" for a in alarms)
    out = {"schema": "fr14.closure_watch.check.v2",
           "snapshot_head": snap["head"][:12], "live_head": now["head"][:12],
           "head_moved": head_moved,
           "alarms": [{"path": a[0], "at_boot": a[1], "now": a[2], "severity": a[3]}
                      for a in alarms],
           "ALARM": bool(file_alarms)}
    if file_alarms:
        out["VERDICT"] = ("EXECUTION CLOSURE FILE CHANGED MID-DRAIN. Any bash entry is a "
                          "site-24 repeat in progress: the running shell will resume at a "
                          "stale byte offset. Report and treat the drain as compromised.")
    elif changed_after_use:
        out["VERDICT"] = (
            "Closure files changed mid-drain, but NO live process executes any of them: "
            "the resident set runs from its site-24 snapshot and the non-resident set has "
            "already exited. Recorded for provenance -- the boot's bytes are not what "
            "changed. NOT a site-24 hazard; the drain is not compromised.")
    elif head_moved:
        out["VERDICT"] = ("HEAD moved but EVERY closure file is byte-identical -- the "
                          "executed code did not change. Provenance note only: the boot "
                          "binds its snapshot head, not live HEAD. Not a site-24 hazard.")
    else:
        out["VERDICT"] = "closure frozen; drain provenance intact"
    print(json.dumps(out, indent=1))
    return 1 if file_alarms else 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: promotion_ab_closure_watch.py {snapshot|check} <closure.json>",
              file=sys.stderr)
        return 2
    mode, path = argv[1], Path(argv[2])
    if mode == "snapshot":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
        print(f"closure snapshot -> {path}")
        return 0
    if mode == "check":
        return check(path)
    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
