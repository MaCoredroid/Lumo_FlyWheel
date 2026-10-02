#!/usr/bin/env python3
"""Freeze the disjoint confirmation request set from the capture run.
Rule (fixed before looking at any replay on it): drop dumps whose messages hash matches the tuning corpus
or that lack messages; group by agent conversation (hash of the first user message); allocate ~TARGET requests across
tasks proportionally to their request counts (largest remainder); within a task take evenly spaced
requests in capture order (filename timestamp). Writes copies + manifest with sha256 per file.
Usage: build_confirm_set.py CAPTURE_DUMP_DIR TUNING_DIR OUT_DIR [TARGET=43]"""
import glob, hashlib, json, os, re, shutil, sys

src, tune, out = sys.argv[1], sys.argv[2], sys.argv[3]
target = int(sys.argv[4]) if len(sys.argv) > 4 else 43


def mhash(d):
    return hashlib.sha256(json.dumps(d.get("messages"), sort_keys=True).encode()).hexdigest()


def task(d):
    """Conversation key: hash of the first user message (one per agent session), labelled with the
    first SWE instance id found anywhere in the request when present."""
    u = [m for m in d.get("messages", []) if m.get("role") == "user"]
    key = hashlib.sha256(json.dumps(u[0].get("content", "") if u else "", sort_keys=True).encode()).hexdigest()[:10]
    ids = re.findall(r"astropy__astropy-\d+|[a-z_]+__[a-z_]+-\d+", json.dumps(d.get("messages", [])))
    return key + (":" + ids[0] if ids else "")


tuning = {mhash(json.load(open(f))) for f in glob.glob(os.path.join(tune, "*.json"))}
by_task, dropped = {}, []
for f in sorted(glob.glob(os.path.join(src, "*.json"))):
    d = json.load(open(f))
    if not d.get("messages"):
        dropped.append((os.path.basename(f), "no messages")); continue
    if mhash(d) in tuning:
        dropped.append((os.path.basename(f), "in tuning corpus")); continue
    by_task.setdefault(task(d), []).append(f)
n = sum(len(v) for v in by_task.values()); target = min(target, n)
quota = {t: target * len(v) / n for t, v in by_task.items()}
alloc = {t: int(q) for t, q in quota.items()}
for t in sorted(quota, key=lambda t: quota[t] - alloc[t], reverse=True)[: target - sum(alloc.values())]:
    alloc[t] += 1
os.makedirs(out, exist_ok=True)
manifest = {"schema": "v2exp.confirm_set.v1", "source": src, "tuning": tune, "target": target, "rule": __doc__.split("Usage")[0].strip(),
            "dropped": dropped, "tasks": {}, "files": []}
for t, files in sorted(by_task.items()):
    k = alloc[t]; idx = sorted({round(i * (len(files) - 1) / max(1, k - 1)) for i in range(k)}) if k > 1 else ([len(files) // 2] if k else [])
    manifest["tasks"][t] = {"available": len(files), "selected": len(idx)}
    for i in idx:
        f = files[i]; dst = os.path.join(out, os.path.basename(f)); shutil.copy2(f, dst)
        manifest["files"].append({"name": os.path.basename(f), "task": t, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest()})
manifest["n"] = len(manifest["files"])
json.dump(manifest, open(os.path.join(os.path.dirname(out.rstrip("/")), "MANIFEST.json"), "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("n", "tasks", "dropped")}, indent=1))
