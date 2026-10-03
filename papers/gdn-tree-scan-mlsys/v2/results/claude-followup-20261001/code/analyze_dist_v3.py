#!/usr/bin/env python3
"""Distribution diagnostic v3 (multi-boot; rules in DIST-PROTOCOL-v3.md, frozen before collection).
Usage: analyze_dist_v3.py BOOTS_JSON TOKENIZER OUT_JSON [--perms 300]
BOOTS_JSON: {"label": {"arm": "AR|TREE|MTP5|NEG", "path": ".../dist.jsonl"}, ...}"""
import argparse, itertools, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizers import Tokenizer
import analyze_dist_v2 as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("boots"); ap.add_argument("tokenizer"); ap.add_argument("out")
    ap.add_argument("--positions", type=int, default=16); ap.add_argument("--perms", type=int, default=300)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer); rng = random.Random(20261002)
    boots = json.load(open(a.boots))
    data = {k: A.load(v["path"], tok, a.positions) for k, v in boots.items()}
    arm = {k: v["arm"] for k, v in boots.items()}
    ar = sorted(k for k in boots if arm[k] == "AR")
    pairs = [(x, y) for x, y in itertools.combinations(ar, 2)]
    for k in sorted(boots):
        if arm[k] in ("TREE", "MTP5", "NEG"):
            pairs += [(k, y) for y in ar]
    res = {}
    for x, y in pairs:
        r = A.compare(data[x], data[y], a.positions, a.perms, rng)
        res[f"{x}|{y}"] = r
        print(x, y, r, flush=True)
    ex = lambda x, y: res[f"{x}|{y}"]["excess"]
    aa = [ex(x, y) for x, y in itertools.combinations(ar, 2)]
    E_AA = max(aa)
    def mean_vs_ar(k): return sum(ex(k, y) for y in ar) / len(ar)
    neg = [k for k in boots if arm[k] == "NEG"]; tree = [k for k in boots if arm[k] == "TREE"]; mtp = [k for k in boots if arm[k] == "MTP5"]
    neg_m = sum(mean_vs_ar(k) for k in neg) / len(neg)
    sens = neg_m > E_AA
    def judge(ks):
        ms = {k: round(mean_vs_ar(k), 3) for k in ks}
        if not sens:
            return ms, "uninformative (sensitivity rule failed)"
        if all(m <= E_AA for m in ms.values()):
            return ms, "consistent at plain-decoding boot-to-boot resolution"
        if any(m > neg_m for m in ms.values()):
            return ms, "inconsistent (deviation exceeds the temperature-0.65 control)"
        return ms, "inconclusive (between AR boot-to-boot spread and the control)"
    mtp_ms, mtp_v = judge(mtp); tree_ms, tree_v = judge(tree)
    if sens and not mtp_v.startswith("consistent"):
        tree_v = "inconclusive (positive control MTP-5 not consistent: rule too strict); raw: " + tree_v
    verdict = {"E_AA_max_ar_ar_excess": E_AA, "ar_ar_excess": aa, "neg_mean_excess_vs_ar": round(neg_m, 3),
               "sensitivity_pass": sens, "mtp5_mean_excess_vs_ar": mtp_ms, "mtp5_verdict": mtp_v,
               "lumotree_mean_excess_vs_ar": tree_ms, "lumotree_verdict": tree_v}
    print(json.dumps(verdict, indent=1))
    json.dump({"positions": a.positions, "perms": a.perms, "pairs": res, "verdict": verdict}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
