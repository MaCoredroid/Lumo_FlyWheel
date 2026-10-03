#!/usr/bin/env python3
"""Freeze / verify the q1v3 protocol, case set and every executed source (sha256).

  freeze.py --write    record FREEZE.json (do this once, after review, before the first GPU run; commit it)
  freeze.py --verify   run_all.sh calls this first and refuses to launch on any difference
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = ["PROTOCOL.md", "cases.json", "q1v3_common.py", "q1v3_hooks.py", "cases.py", "client.py", "patch_runner.py",
          "make_cand_launch.py", "reduce.py", "serve_native_q1v3.sh", "run_all.sh"]


def digest():
    files = FROZEN + sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(HERE, "fixtures", "requests", "*.json")))
    return {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true"); g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    path = os.path.join(HERE, "FREEZE.json")
    cur = digest()
    if a.write:
        json.dump({"schema": "q1v3.freeze.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": cur}, open(path, "w"), indent=1, sort_keys=True)
        print(f"wrote {path} ({len(cur)} files)"); return 0
    if not os.path.exists(path):
        print("FREEZE.json missing: run freeze.py --write after review"); return 2
    want = json.load(open(path))["files"]
    diff = sorted(set(want) ^ set(cur)) + sorted(f for f in set(want) & set(cur) if want[f] != cur[f])
    if diff:
        print("FROZEN INPUTS CHANGED:", diff); return 2
    print(f"freeze verified ({len(cur)} files)"); return 0


if __name__ == "__main__":
    sys.exit(main())
