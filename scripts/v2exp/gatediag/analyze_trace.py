#!/usr/bin/env python3
"""Analyze a passive gate trace: per tree step, the 32 inputs, the gated flag of the proposal that fed
it, and the TAW accepted path. Usage: analyze_trace.py TRACE.jsonl TOKENIZER.json [--show 6]"""
import collections, json, sys
from tokenizers import Tokenizer

PARENT = (-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9, 9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29, 30)
SPINE = [0, 1, 4, 9, 14, 19, 24, 26, 28, 29, 30, 31]


def depth(r):
    d = 0
    while r:
        r = PARENT[r]; d += 1
    return d


def main():
    path, tokp = sys.argv[1], sys.argv[2]
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 6
    tok = Tokenizer.from_file(tokp)
    recs = [json.loads(l) for l in open(path)]
    last_drafts = None
    steps = []
    cur = None
    for r in recs:
        if r["what"] == "drafts":
            last_drafts = r
        elif r["what"] == "tree":
            cur = {"tree": r, "fed_by": last_drafts}
            steps.append(cur)
        elif r["what"] == "taw" and cur is not None and "taw" not in cur:
            cur["taw"] = r
    stats = collections.defaultdict(lambda: collections.Counter())
    shown = 0
    for s in steps:
        t, fd, taw = s["tree"], s.get("fed_by"), s.get("taw")
        if not taw or not fd:
            continue
        gated = bool((fd.get("work") or {}).get("gated"))
        k = "gated" if gated else "ungated"
        ids = t["ids"]
        stats[k]["steps"] += 1
        stats[k]["inputs_match_drafts"] += int(ids[1:] == fd["tokens"])
        acc = taw["acc"]; rows = taw["rows"][:acc]
        stats[k]["acc_sum"] += acc
        for d in range(1, 12):
            stats[k][f"reach{d}"] += int(acc >= d)
        stats[k]["accepted_all_spine"] += int(all(r in SPINE for r in rows))
        if gated and shown < show:
            shown += 1
            sp = [ids[r] for r in SPINE]
            print(f"--- gated step {t['step']} acc={acc} rows={rows} emitted={taw['tokens'][:taw['len']]}")
            print("  spine d0..11:", sp)
            print("  spine text  :", repr(tok.decode(sp[:12])))
            for d in (3, 4, 5):
                kids = [r for r in range(32) if depth(r) == d]
                print(f"  depth{d} rows {kids}: {[ids[r] for r in kids]}")
            print("  emitted text:", repr(tok.decode(taw["tokens"][:taw["len"]])))
    for k, c in stats.items():
        n = c["steps"] or 1
        print(k, "steps", c["steps"], "inputs==drafts", c["inputs_match_drafts"], "acc/step %.3f" % (c["acc_sum"] / n),
              "survival", [round(c[f"reach{d}"] / n, 3) for d in range(1, 12)], "spine-only paths", c["accepted_all_spine"])


if __name__ == "__main__":
    main()
