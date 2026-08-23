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
                sev = "CRITICAL (bash byte-offset hazard)" if kind == "bash" else "provenance"
                alarms.append((rel, (was or {}).get("sha256", "absent")[:12],
                               (is_ or {}).get("sha256", "absent")[:12], sev))
    out = {"schema": "fr14.closure_watch.check.v1",
           "snapshot_head": snap["head"][:12], "live_head": now["head"][:12],
           "alarms": [{"path": a[0], "at_boot": a[1], "now": a[2], "severity": a[3]}
                      for a in alarms],
           "ALARM": bool(alarms)}
    if alarms:
        out["VERDICT"] = ("EXECUTION CLOSURE MOVED MID-DRAIN. Any bash entry is a site-24 "
                          "repeat in progress: the running shell will resume at a stale "
                          "byte offset. Report and treat the drain as compromised.")
    else:
        out["VERDICT"] = "closure frozen; drain provenance intact"
    print(json.dumps(out, indent=1))
    return 1 if alarms else 0


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
