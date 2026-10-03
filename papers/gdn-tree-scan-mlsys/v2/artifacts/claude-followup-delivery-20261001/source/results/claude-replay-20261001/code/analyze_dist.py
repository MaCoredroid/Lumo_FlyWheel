#!/usr/bin/env python3
"""Distributional-correctness analysis.

For each request and token position j (1..L), compare the empirical distribution of the
j-th output token between an arm and the plain-decoding reference with a permutation test
on total-variation distance. Outputs are tokenized identically for every arm (same tokenizer,
same text reconstruction), so any deterministic function of the output is a valid statistic.

Calibration: the same analysis between two independent plain-decoding runs (AR vs AR') gives
the null behaviour (p-values ~ uniform, TV at the sampling-noise floor).

Usage: analyze_dist.py DIST_ROOT TOKENIZER_JSON [--positions 16] [--perms 2000]
"""
import argparse, collections, glob, json, math, os, random, sys
from tokenizers import Tokenizer


def load(path, tok, L):
    by_req = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        text = (r.get("reasoning") or "") + "\x00" + (r.get("content") or "") + "\x00" + (r.get("tool_calls") or "")
        ids = tok.encode(text, add_special_tokens=False).ids[:L]
        by_req[r["request"]].append(ids)
    return by_req


def tv(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = len(a), len(b)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def perm_p(a, b, perms, rng):
    obs = tv(a, b)
    pool = a + b; na = len(a); ge = 0
    for _ in range(perms):
        rng.shuffle(pool)
        if tv(pool[:na], pool[na:]) >= obs - 1e-12:
            ge += 1
    return obs, (ge + 1) / (perms + 1)


def compare(x, y, L, perms, rng):
    tvs, ps = [], []
    for req in sorted(set(x) & set(y)):
        for j in range(L):
            a = [s[j] for s in x[req] if len(s) > j]
            b = [s[j] for s in y[req] if len(s) > j]
            if len(a) < 10 or len(b) < 10:
                continue
            t, p = perm_p(a, b, perms, rng)
            tvs.append(t); ps.append(p)
    n = len(ps)
    fisher = -2 * sum(math.log(p) for p in ps)  # ~ chi2(2n) under H0 (conservative with discrete p)
    # Wilson-Hilferty normal approximation for chi2 upper tail
    k = 2 * n
    z = ((fisher / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return {"tests": n, "mean_tv": round(sum(tvs) / n, 4), "frac_p_lt_0.05": round(sum(p < 0.05 for p in ps) / n, 3),
            "frac_p_lt_0.01": round(sum(p < 0.01 for p in ps) / n, 3), "fisher_z": round(z, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("tokenizer")
    ap.add_argument("--positions", type=int, default=16); ap.add_argument("--perms", type=int, default=2000)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    rng = random.Random(20261001)
    runs = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*", "dist.jsonl"))):
        runs[os.path.basename(os.path.dirname(d))] = load(d, tok, a.positions)
    ar = [k for k in runs if k.startswith("ar-")]
    if not ar:
        sys.exit("no AR reference run")
    ref = ar[0]
    print(f"reference: {ref}; runs: {list(runs)}")
    out = {}
    for name, data in runs.items():
        if name == ref:
            continue
        res = compare(data, runs[ref], a.positions, a.perms, rng)
        out[name] = res
        print(f"{name:40s} vs {ref}: {res}")
    json.dump({"reference": ref, "positions": a.positions, "perms": a.perms, "results": out},
              open(os.path.join(a.root, "dist_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
