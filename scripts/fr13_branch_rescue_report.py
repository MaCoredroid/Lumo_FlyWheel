#!/usr/bin/env python3
"""Branch-rescue report: from the FR13_BRANCH_ACCEPT_DIAG per-flat-verify-row accept
histogram, compute what % of committed tokens were accepted at a BRANCH node
("spine failed, branch caught it") vs on the pure spine.

The instrument (in the model-runner attn-KV-remap apply site, gated behind
FR13_BRANCH_ACCEPT_DIAG=1) dumps a json: {"rows": {flat_row: count}, "committed": N,
"events": E, "steps": S}. flat_row is the +1-anchored verify row (node i -> row i+1,
nodes sorted by (len, path)). A row is SPINE iff its path is all-zeros (the greedy
child-0 chain); any other row is a BRANCH. branch-rescue rate = branch-row accepts /
total committed.

Usage:
  fr13_branch_rescue_report.py --hist HIST.json --tree-name cat8
  fr13_branch_rescue_report.py --hist HIST.json --spec-config '<SPEC_CONFIG json>'
  fr13_branch_rescue_report.py --selftest
"""
import argparse, json, sys, ast

# Canonical trees (path tuples). Must match scripts/fr13_launch_locked + the committer.
TREES = {
    "cat8": [(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0, 0)],
    "cat6": [(0,), (1,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)],  # cat6root
    "cat6root": [(0,), (1,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)],
    "cat9": [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1),
             (0, 0, 0, 1), (0, 0, 0, 0, 1)],
}


def flat_row_classes(paths):
    """Map flat verify row (1-indexed, +1 anchor) -> ('spine'|'branch', path).
    Nodes are laid out in sorted((len, path)) order; node index i -> flat row i+1."""
    ordered = sorted(paths, key=lambda p: (len(p), p))
    out = {}
    for i, p in enumerate(ordered):
        out[i + 1] = ("spine" if all(c == 0 for c in p) else "branch", p)
    return out


def parse_tree(args):
    if args.tree_name:
        key = args.tree_name.strip()
        if key not in TREES:
            sys.exit(f"unknown --tree-name {key!r}; known: {sorted(TREES)}")
        return TREES[key], key
    raw = args.tree or args.spec_config
    if not raw:
        sys.exit("need --tree-name, --tree, or --spec-config")
    if args.spec_config:
        cfg = json.loads(raw)
        raw = cfg.get("speculative_token_tree")
        if isinstance(raw, str):
            raw = raw
    # tree may be a json/py list of lists
    try:
        lst = json.loads(raw)
    except Exception:
        lst = ast.literal_eval(raw)
    return [tuple(p) for p in lst], "custom"


def report(hist, paths, label):
    cls = flat_row_classes(paths)
    rows = {int(k): int(v) for k, v in hist.get("rows", {}).items()}
    spine = branch = unknown = 0
    per_branch = {}
    for row, cnt in rows.items():
        c = cls.get(row)
        if c is None:
            unknown += cnt
        elif c[0] == "spine":
            spine += cnt
        else:
            branch += cnt
            per_branch[c[1]] = per_branch.get(c[1], 0) + cnt
    total = spine + branch + unknown
    print(f"=== branch-rescue report ({label}) ===")
    print(f"  tree nodes={len(paths)}  spine rows={sorted(r for r,c in cls.items() if c[0]=='spine')}"
          f"  branch rows={sorted(r for r,c in cls.items() if c[0]=='branch')}")
    print(f"  committed tokens (hist rows sum) = {total}  "
          f"(diag counters: committed={hist.get('committed')} events={hist.get('events')} steps={hist.get('steps')})")
    if total == 0:
        print("  NO DATA — histogram empty (diag not armed / vacuous run).")
        return
    print(f"  SPINE  accepts = {spine:>8}  ({100*spine/total:.2f}%)")
    print(f"  BRANCH accepts = {branch:>8}  ({100*branch/total:.2f}%)  <-- 'spine failed, branch caught it'")
    if unknown:
        print(f"  UNKNOWN rows   = {unknown:>8}  ({100*unknown/total:.2f}%)  (rows not in tree — CHECK tree match!)")
    if per_branch:
        print("  per-branch-node accepts:")
        for p, c in sorted(per_branch.items(), key=lambda kv: -kv[1]):
            print(f"    {str(p):<18} {c:>8}  ({100*c/total:.2f}%)")


def selftest():
    # cat8: spine rows {1,3,5,7,8}, branch rows {2,4,6}
    cls = flat_row_classes(TREES["cat8"])
    spine_rows = sorted(r for r, c in cls.items() if c[0] == "spine")
    branch_rows = sorted(r for r, c in cls.items() if c[0] == "branch")
    assert spine_rows == [1, 3, 5, 7, 8], spine_rows
    assert branch_rows == [2, 4, 6], branch_rows
    # synthetic hist: 90 spine accepts (row1), 10 branch accepts (row2) -> 10% rescue
    hist = {"rows": {"1": 90, "2": 10}, "committed": 100, "events": 100, "steps": 100}
    cls_ = flat_row_classes(TREES["cat8"])
    rows = {int(k): int(v) for k, v in hist["rows"].items()}
    branch = sum(v for r, v in rows.items() if cls_[r][0] == "branch")
    total = sum(rows.values())
    assert branch == 10 and total == 100 and 100 * branch / total == 10.0
    # cat6 sanity: spine {1,3,4,5,6}, branch {2}
    c6 = flat_row_classes(TREES["cat6"])
    assert sorted(r for r, c in c6.items() if c[0] == "branch") == [2], c6
    print("selftest OK: cat8 spine={1,3,5,7,8} branch={2,4,6}; cat6 branch={2}; rate math OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist")
    ap.add_argument("--tree-name")
    ap.add_argument("--tree")
    ap.add_argument("--spec-config")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.hist:
        sys.exit("need --hist HIST.json (or --selftest)")
    hist = json.load(open(a.hist))
    paths, label = parse_tree(a)
    report(hist, paths, label)


if __name__ == "__main__":
    main()
