#!/usr/bin/env python3
"""Idempotent STATUS tick for the confirmation captures: recount verified runs across the given roots and rewrite the tick
line + header stamp. Usage: tick_status.py <STATUS.md> <root>... (single writer; never run concurrently with another STATUS edit)."""
import json, sys, glob, os, re, datetime as dt
status, roots = sys.argv[1], sys.argv[2:]
done = []
for ROOT in roots:
    for d in sorted(p for p in glob.glob(f"{ROOT}/capture_*_p*") if os.path.isdir(p)):
        f = f"{d}/capture_provenance.json"
        if os.path.exists(f):
            r = json.load(open(f)); done.append((os.path.basename(d).split('_')[-1], r["all_pass"]))
n_ok = sum(1 for x in done if x[1]); ids = " ".join(x[0] for x in done if x[1]); now = dt.datetime.now(dt.timezone.utc).strftime("%H:%M")
fails = [x[0] for x in done if not x[1]]
s = open(status).read()
s = re.sub(r'## Current state \(tick 2026-09-22T\d\d:\d\dZ', f'## Current state (tick 2026-09-22T{now}Z', s, count=1)
s = re.sub(r'\| E7a step 2 — verified fresh confirmation captures \| \*\*\d+ / 32\*\*[^\n]*',
           f'| E7a step 2 — verified fresh confirmation captures | **{n_ok} / 32** ({ids} — inspector 59c5b2b9… 51/51 each; {"all verdicts PASS" if not fails else "FAILED verdicts: " + " ".join(fails)}) | part 2 running since 02:45:43Z; {32-n_ok} boots remain (tick {now}Z); ETA ≈ 05:15Z |', s, count=1)
open(status, "w").write(s); print("tick", n_ok, "/ 32 at", now, "fails:", fails)
