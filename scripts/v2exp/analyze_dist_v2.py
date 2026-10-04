#!/usr/bin/env python3
"""Distribution diagnostic v2 (frozen with DIST-PROTOCOL.md before candidate results were read).

Observable: the output text stream of a vLLM arm (reasoning, content, tool-call text joined with
U+001F), tokenized with the model tokenizer, first L positions, padded with an END category when the
output is shorter (no conditioning on length). For arms that share the vLLM reasoning/tool parsers this
is a deterministic function of the generated token sequence.

Statistic: T = sum over requests r and positions j of TV(P_x[r,j], P_y[r,j]).
Null: permute whole continuations between the two arms within each request (keeps within-continuation
dependence), recompute T. Report T_obs, null mean/sd, excess = T_obs - null mean, and permutation p.
Usage: analyze_dist_v2.py PAIRS_JSON TOKENIZER_JSON OUT_JSON [--positions 16] [--perms 1000]
PAIRS_JSON: {"label": ["candidate_dist.jsonl", "reference_dist.jsonl"], ...}
"""
import argparse, collections, json, random, sys
from tokenizers import Tokenizer
END = -1


def load(path, tok, L):
    by_req = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            sys.exit(f"error record in {path}")
        text = "\x1f".join([r.get("reasoning") or "", r.get("content") or "", r.get("tool_calls") or ""])
        ids = tok.encode(text, add_special_tokens=False).ids[:L]
        by_req[r["request"]].append(tuple(ids + [END] * (L - len(ids))))
    return by_req


def stat(groups, L):
    total = 0.0
    for a, b in groups:
        na, nb = len(a), len(b)
        for j in range(L):
            ca = collections.Counter(s[j] for s in a); cb = collections.Counter(s[j] for s in b)
            total += 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))
    return total


def compare(x, y, L, perms, rng):
    reqs = sorted(set(x) & set(y))
    groups = [(x[r], y[r]) for r in reqs]
    t_obs = stat(groups, L)
    null = []
    for _ in range(perms):
        pg = []
        for a, b in groups:
            pool = list(a) + list(b); rng.shuffle(pool); pg.append((pool[:len(a)], pool[len(a):]))
        null.append(stat(pg, L))
    m = sum(null) / len(null); sd = (sum((v - m) ** 2 for v in null) / (len(null) - 1)) ** 0.5
    p = (1 + sum(v >= t_obs for v in null)) / (1 + len(null))
    return {"requests": len(reqs), "samples_x": sum(len(a) for a, _ in groups), "samples_y": sum(len(b) for _, b in groups),
            "T_obs": round(t_obs, 3), "null_mean": round(m, 3), "null_sd": round(sd, 3),
            "excess": round(t_obs - m, 3), "z": round((t_obs - m) / sd, 2) if sd else None, "p_perm": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("tokenizer"); ap.add_argument("out")
    ap.add_argument("--positions", type=int, default=16); ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer); rng = random.Random(20261001)
    res = {}
    for label, (cand, ref) in json.load(open(a.pairs)).items():
        res[label] = compare(load(cand, tok, a.positions), load(ref, tok, a.positions), a.positions, a.perms, rng)
        print(label, res[label], flush=True)
    json.dump({"positions": a.positions, "perms": a.perms, "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
