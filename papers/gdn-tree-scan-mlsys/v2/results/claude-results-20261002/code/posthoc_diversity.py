import json, glob, collections, os, sys
os.chdir(sys.argv[1])
for d in sorted(glob.glob("*/dist.jsonl")):
    by = collections.defaultdict(list)
    for l in open(d):
        r = json.loads(l)
        by[r["request"]].append((r.get("reasoning") or "") + "|" + (r.get("content") or "") + "|" + (r.get("tool_calls") or ""))
    uniq = [len(set(v)) / len(v) for v in by.values()]
    first = [len(set(s[:12] for s in v)) for v in by.values()]
    ln = [sum(len(s) for s in v) / len(v) for v in by.values()]
    rc = sum(1 for v in by.values() for s in v if s.startswith("|"))
    print(f"{d.split('/')[0]:40s} unique_frac={sum(uniq)/len(uniq):.3f} distinct_first12={sum(first)/len(first):.1f} mean_chars={sum(ln)/len(ln):.0f} no_reasoning={rc}")
