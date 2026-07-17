#!/usr/bin/env python3
"""RIGOROUS offline gate: validate the point-mass (all_greedy) committer against
the ACTUAL old greedy path-LCP committer on REAL captured traces.

The live dual-run gate is architecturally impossible (both committers do the
FR13_EAGER_PACK GDN state-advance, which consumes per-step staged scan flags
exactly once). Instead we validate token-selection equivalence offline on real
`tree_path_lcp_max.jsonl` captures (the greedy committer's own per-step log).

Each row carries, per node: draft_token_ids, parent_target_ids (= target argmax
used to decide that node's acceptance), self_target_ids (= bonus argmax at that
node), plus path_scores (root-to-leaf paths => the parent structure) and the
committer's OWN output: accepted_node_ids + emitted_tokens.

We reconstruct the point-mass committer's walk (accept child iff draft ==
parent_target_ids[child]; bonus = self_target_ids[leaf]) and require it to
reproduce the logged accepted_node_ids + emitted_tokens BYTE-FOR-BYTE. We also
count duplicate-sibling occurrences (the only case the synthetic gate deferred:
two children of one node carrying the SAME argmax draft, where the point-mass
source pick would be ambiguous). Zero duplicate siblings => the synthetic
distinct-sibling proof (0/4000) covers every real case.

Run: PYTHONPATH=src:scripts .venv/bin/python scripts/fr13_greedy_unify_real_trace_gate.py [glob...]
"""
import glob
import json
import sys


def parents_from_path_scores(path_scores, node_count):
    """Recover parent[node] from the root-to-leaf paths (parent = predecessor)."""
    parent = {}
    for ps in path_scores:
        path = ps["path"]
        for i, node in enumerate(path):
            par = path[i - 1] if i > 0 else -1
            if node in parent and parent[node] != par:
                # inconsistent parent across paths => malformed capture
                return None
            parent[node] = par
    return parent


def pointmass_greedy_walk(parent, draft, parent_tgt, self_tgt, max_len):
    """Point-mass committer == greedy longest-prefix. Returns (accepted_path, emitted)."""
    # children map
    children = {}
    for node, par in parent.items():
        children.setdefault(par, []).append(node)
    for par in children:
        children[par].sort()  # source order = node order
    cur = -1
    accepted = []
    emitted = []
    dup_sibling_hits = 0
    for _ in range(max_len + 2):
        kids = children.get(cur, [])
        if not kids:
            if cur >= 0:
                emitted.append(int(self_tgt[cur]))  # bonus = self argmax at leaf
            break
        # greedy token at this level = the target argmax the children were scored
        # against. All siblings share the parent's next-token argmax; the capture
        # stores it per-child as parent_target_ids[child] (identical across sibs).
        g = int(parent_tgt[kids[0]])
        matches = [k for k in kids if int(draft[k]) == g]
        if len(matches) > 1:
            dup_sibling_hits += 1
        emitted.append(g)
        if not matches:
            break
        acc = matches[0]  # first-match (source order) tie-break
        if g != int(draft[acc]):
            break
        accepted.append(acc)
        cur = acc
    return accepted, emitted, dup_sibling_hits


def main():
    pats = sys.argv[1:] or [
        "output/**/tree_path_lcp_max.jsonl",
        "output/**/tree_path_lcp.jsonl",
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        print("no capture files found"); return 1

    total = mism = dup_total = parsed = skipped = 0
    mism_examples = []
    for fp in files:
        try:
            fh = open(fp)
        except Exception:
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") != "tree_path_lcp_max":
                continue
            if not all(k in d for k in ("draft_token_ids", "parent_target_ids",
                                        "self_target_ids", "path_scores",
                                        "accepted_node_ids", "emitted_tokens")):
                skipped += 1
                continue
            nc = int(d["node_count"])
            parent = parents_from_path_scores(d["path_scores"], nc)
            if parent is None:
                skipped += 1
                continue
            total += 1
            acc, emit, dup = pointmass_greedy_walk(
                parent, d["draft_token_ids"], d["parent_target_ids"],
                d["self_target_ids"], max_len=nc + 2,
            )
            dup_total += dup
            if acc != list(d["accepted_node_ids"]) or emit != list(d["emitted_tokens"]):
                mism += 1
                if len(mism_examples) < 8:
                    mism_examples.append({
                        "file": fp.split("/")[-2],
                        "recon_path": acc, "log_path": d["accepted_node_ids"],
                        "recon_emit": emit, "log_emit": d["emitted_tokens"],
                        "dup": dup,
                    })
            parsed += 1
        fh.close()

    print(f"files={len(files)} rows_validated={total} mismatches={mism} "
          f"duplicate_sibling_rows={dup_total} skipped={skipped}")
    for ex in mism_examples:
        print("  MISMATCH", ex)
    if mism == 0 and dup_total == 0:
        print("PASS: point-mass == actual old greedy committer on all real traces; "
              "ZERO duplicate siblings => synthetic distinct-sibling proof is complete")
    elif mism == 0 and dup_total > 0:
        print(f"PASS on output BUT {dup_total} duplicate-sibling rows exist => "
              "point-mass source pick must be made deterministic (first-match) to stay byte-exact")
    else:
        print("FAIL: point-mass diverges from logged greedy — investigate")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
