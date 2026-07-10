#!/usr/bin/env python3
"""FR13 garble BINDING analysis — read the committer accept-trace (.commit) and
measure, AT the actual commit node, the tree-verify prob the committer committed
each token at. Answers: is the misspell a drift-INFLATED wrong-accept?

The committer is PROVEN distribution-lossless (commits token t at exactly its
post-constraint tree-verify prob `committed_prob`). So a NON-argmax commit at a
CONFIDENT position (target argmax prob high) where the correct token WAS drafted
is the committer faithfully emitting a token the tree-verify target gave
`committed_prob` mass to. If no-spec masks that token (~1e-6, from the localizer),
then `committed_prob` IS the forward-drift inflation, measured at the commit node
with NO gather->commit alignment trap.

Usage:
  python3 scripts/fr13_commit_trace_analyze.py --trace <path>.commit [--endpoint http://127.0.0.1:9950 --model qwen3.6-27b]
  (--endpoint enables /detokenize to check committed vs argmax are near-neighbors)
"""
import argparse, json, collections, statistics, sys, urllib.request


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("event") == "commit_trace":
                recs.append(o)
    return recs


def detok(endpoint, model, ids):
    """best-effort id->text via vLLM /detokenize; returns {id: text} or {}."""
    if not endpoint:
        return {}
    out = {}
    for tid in sorted(set(int(x) for x in ids)):
        try:
            req = urllib.request.Request(
                endpoint.rstrip("/") + "/detokenize",
                data=json.dumps({"model": model, "tokens": [tid]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                out[tid] = json.loads(r.read()).get("prompt", "")
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--confident", type=float, default=0.9,
                    help="argmax_prob threshold to call a position 'confident'")
    ap.add_argument("--max-examples", type=int, default=40)
    args = ap.parse_args()

    recs = load(args.trace)
    if not recs:
        print(f"NO commit_trace records in {args.trace} (propagation off? flag unset? EAGER off?)")
        sys.exit(2)

    n = len(recs)
    non_argmax = [r for r in recs if not r.get("committed_is_argmax")]
    accepted = [r for r in recs if r.get("accepted")]
    # the drift-inflated wrong-accept signature: committed a NON-argmax token at a
    # CONFIDENT position (target strongly prefers argmax) where argmax WAS drafted.
    wa = [r for r in non_argmax
          if (r.get("argmax_prob") or 0) >= args.confident and r.get("argmax_drafted")]

    print(f"=== commit_trace: {n} commits ===")
    print(f"  argmax commits:     {n - len(non_argmax)} ({100*(n-len(non_argmax))/n:.1f}%)")
    print(f"  NON-argmax commits: {len(non_argmax)} ({100*len(non_argmax)/n:.1f}%)")
    print(f"  accepted / residual: {len(accepted)} / {n - len(accepted)}")
    print()
    print(f"=== drift-inflated WRONG-ACCEPT candidates ===")
    print(f"  (committed != argmax, argmax_prob>={args.confident}, correct token WAS drafted)")
    print(f"  count: {len(wa)} ({100*len(wa)/n:.2f}% of all commits)")
    if wa:
        cp = [r["committed_prob"] for r in wa]
        ap_ = [r["argmax_prob"] for r in wa]
        cp_s = sorted(cp, reverse=True)
        print(f"  committed_prob (tree-verify p the committer USED for the wrong token):")
        print(f"     max={max(cp):.4g} mean={statistics.mean(cp):.4g} median={statistics.median(cp):.4g} min={min(cp):.4g}")
        print(f"     top-15: {[round(x,4) for x in cp_s[:15]]}")
        print(f"  argmax_prob (the correct token's tree-verify p at those nodes):")
        print(f"     mean={statistics.mean(ap_):.4g} median={statistics.median(ap_):.4g}")
        # buckets of committed_prob
        b = collections.Counter()
        for x in cp:
            if x >= 0.1: b[">=0.1"] += 1
            elif x >= 0.03: b["0.03-0.1"] += 1
            elif x >= 0.01: b["0.01-0.03"] += 1
            elif x >= 1e-3: b["1e-3-0.01"] += 1
            else: b["<1e-3"] += 1
        print(f"  committed_prob buckets: {dict(b)}")

        # detokenize a sample to check near-neighbor spelling
        ids = []
        for r in wa[:args.max_examples]:
            ids += [r["committed_token"], r["argmax_token"]]
        texts = detok(args.endpoint, args.model, ids)
        print()
        print(f"  === examples (committed[p] vs argmax[p]) ===")
        for r in wa[:args.max_examples]:
            ct, at = int(r["committed_token"]), int(r["argmax_token"])
            cs = repr(texts.get(ct, f"<{ct}>")) if texts else f"<{ct}>"
            as_ = repr(texts.get(at, f"<{at}>")) if texts else f"<{at}>"
            print(f"    committed {cs}[{r['committed_prob']:.4g}]  vs argmax {as_}[{r['argmax_prob']:.4g}]  accepted={r['accepted']}")
    print()
    print("INTERPRETATION: wrong-accept committed_prob is the ACCEPT-TIME tree-verify prob "
          "(post temp+top-p) the correct committer acted on. Join to no-spec (localizer) at the "
          "same positions: committed_prob >> no-spec prob (~1e-6) => forward-drift INFLATION proven "
          "at the commit node. committed_prob ~ no-spec => genuine model tail (not drift).")


if __name__ == "__main__":
    main()
