#!/usr/bin/env python3
"""Sampling-correctness analysis of TreeHost's deployed tree sampler from a pendiag capture.

1. History check: for every captured tree row, the token set whose logit the processors changed is compared with
   (a) the flattened-chain history vLLM's stock code builds (output + spec[:j]) and (b) the correct per-path history
   (output + the tokens on the row's own root->node path).
2. Distortion: per tree node, total variation between the row the sampler used (observed penalties, /T, exact-k
   top-k as in the Triton path, top-p) and the correct autoregressive row (per-path penalties, /T, tie-keeping top-k
   as in the single-row PyTorch path, top-p), on the top-256 pre-processing support.
3. Walk test (taw_mc): the deployed walk on the rows it actually received must reproduce them exactly (losslessness
   of the walk); the same walk judged against the correct rows measures the end-to-end deviation. Planted faults
   (double temperature, top-p dropped, correlated uniforms, target rows shifted by one) must be detected first.
Usage: analyze_pen.py PEN_TRACE.jsonl OUT.json [--walks 400000] [--steps 6]"""
import json, math, sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taw_mc as M  # noqa: E402

T_, TOPK, TOPP = 0.6, 20, 0.95
DP = [int(p) for p in M.T.DRAFT_PARENT]
INACTIVE = {17, 22, 24, 26}


def path_tokens(u, spec):
    out = []
    while u >= 0:
        out.append(spec[u]); u = DP[u]
    return set(out)


def topk_topp(logits, exact_k):
    x = logits.clone()
    if exact_k:                                    # Triton path: exactly k, ties by lower index
        order = sorted(range(len(x)), key=lambda i: (-float(x[i]), i))
        keep = set(order[:TOPK]); mask = torch.tensor([i not in keep for i in range(len(x))])
    else:                                          # PyTorch path: keep everything >= the k-th value
        kth = torch.topk(x, TOPK).values[-1]; mask = x < kth
    x[mask] = float("-inf")
    p = torch.softmax(x.double(), -1)
    srt, idx = torch.sort(p)                       # ascending, as vLLM's apply_top_p
    cum = srt.cumsum(0)
    drop = cum <= 1 - TOPP; drop[-1] = False
    x[idx[drop]] = float("-inf")
    return x


def load_steps(path):
    calls = [json.loads(l) for l in open(path)]
    calls = [c for c in calls if "rows" in c and c["nrows"] == 31]
    return [(calls[i], calls[i + 1]) for i in range(0, len(calls) - 1, 2)]


def build(step):
    tgt, slf = step
    spec = tgt["spec"][0]; assert len(spec) == 31
    base = set(tgt["rows"][0]["pen_ids"])
    support = sorted({t for r in tgt["rows"] + slf["rows"] for t in r["top_ids"]} | set(spec))
    pos = {t: i for i, t in enumerate(support)}
    V = len(support)
    hist = {"flat_match": 0, "correct_match": 0, "rows": 0}
    rows = {}
    for kind, call in (("target", tgt), ("self", slf)):
        dep, cor = torch.full((31, V), float("-inf")), torch.full((31, V), float("-inf"))
        for j, r in enumerate(call["rows"]):
            pen = set(r["pen_ids"])
            flat = base | set(spec[:j])
            node = DP[j] if kind == "target" else j          # target row j = distribution at parent(j); self row j = at j
            correct = base | (path_tokens(node, spec) if node >= 0 else set())
            hist["rows"] += 1; hist["flat_match"] += pen == flat; hist["correct_match"] += pen == correct
            pre = torch.full((V,), float("-inf"))
            for t, v in zip(r["top_ids"], r["top_vals"]):
                pre[pos[t]] = v
            pv = sorted(r["pen_vals"]); pres = -pv[0] if pv else 1.0
            d_obs = torch.zeros(V); c_cor = torch.zeros(V)
            for t in pen:
                if t in pos: d_obs[pos[t]] = -pres
            for t in correct:
                if t in pos: c_cor[pos[t]] = -pres
            dep[j] = topk_topp((pre + d_obs) / T_, exact_k=True)
            cor[j] = topk_topp((pre + c_cor) / T_, exact_k=False)
        rows[kind] = (dep, cor)
    # compress to the columns that are finite in any processed row, plus the draft tokens (order preserved)
    keep = torch.zeros(V, dtype=torch.bool)
    for dep, cor in rows.values():
        keep |= torch.isfinite(dep).any(0) | torch.isfinite(cor).any(0)
    for t in spec:
        keep[pos[t]] = True
    idx = keep.nonzero().flatten()
    remap = {int(c): i for i, c in enumerate(idx.tolist())}
    rows = {k: (d[:, idx].contiguous(), c[:, idx].contiguous()) for k, (d, c) in rows.items()}
    drafts = torch.tensor([remap[pos[t]] for t in spec])
    return rows, drafts, hist, int(idx.numel())


def tv(a, b):
    return 0.5 * float((torch.softmax(a.double(), -1) - torch.softmax(b.double(), -1)).abs().sum())


def main():
    path, out = sys.argv[1], sys.argv[2]
    walks = int(sys.argv[sys.argv.index("--walks") + 1]) if "--walks" in sys.argv else 400000
    nsteps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 6
    steps = load_steps(path)
    report = {"steps_captured": len(steps), "history": {"rows": 0, "flat_match": 0, "correct_match": 0}, "per_step": []}
    ch = M.active_children()
    for si, st in enumerate(steps):
        rows, drafts, hist, V = build(st)
        for k in hist:
            report["history"][k] += hist[k]
        (td, tc), (sd, sc) = rows["target"], rows["self"]
        node_tv = {}
        for v in range(-1, 31):
            kids = ch.get(v, [])
            if kids:
                node_tv[v] = tv(td[kids[0]], tc[kids[0]])
            elif v >= 0 and v not in INACTIVE:
                node_tv[v] = tv(sd[v], sc[v])
        rec = {"step": si, "V": V, "node_tv": {str(k): round(x, 4) for k, x in node_tv.items()}}
        if si < nsteps:
            outs = M.run_walks(td, sd, drafts, walks)
            cnt = M.node_counts(outs, V)
            n0 = int(cnt[-1].sum())
            visits = {v: int(c.sum()) / n0 for v, c in cnt.items()}
            rec["visit_prob"] = {str(k): round(x, 4) for k, x in visits.items()}
            rec["expected_tv_per_step"] = round(sum(visits.get(v, 0) * node_tv.get(v, 0) for v in node_tv), 4)
            rec["walk_vs_used_rows"] = M.test(td, sd, drafts, td, sd, walks)
            rec["walk_vs_correct_rows"] = M.test(td, sd, drafts, tc, sc, walks)
            faults = {"double_temperature": (td / T_, sd / T_, None),
                      "target_rows_shifted": (torch.roll(td, 1, 0), sd, None)}
            u = torch.rand(walks, int(M.T.WALK_CAP), 3); u[:, :, 1] = u[:, :, 0]
            faults["correlated_uniforms"] = (td, sd, u)
            u2 = torch.rand(walks, int(M.T.WALK_CAP), 3); u2[:, :, 1] = 0.0
            faults["always_accept"] = (td, sd, u2)
            u3 = torch.rand(walks, int(M.T.WALK_CAP), 3); u3[:, :, 2] = 0.5
            faults["fixed_residual_draw"] = (td, sd, u3)
            nt = td.clone(); ns = sd.clone()
            rec["faults"] = {}
            for name, (a, b, uu) in faults.items():
                if uu is not None:
                    r = M.test(td, sd, drafts, td, sd, walks, uniforms=uu)
                else:
                    r = M.test(a, b, drafts, td, sd, walks)
                rec["faults"][name] = {"detected": bool(r["rejected"]), "nodes_rejected": len(r["rejected"]), "nodes_tested": r["nodes_tested"]}
            for k in ("walk_vs_used_rows", "walk_vs_correct_rows"):
                rec[k] = {"nodes_tested": rec[k]["nodes_tested"], "nodes_rejected": len(rec[k]["rejected"]),
                          "rejected": [(r["node"], r["visits"], round(r["tv_hat"], 4)) for r in rec[k]["rejected"]]}
        report["per_step"].append(rec)
        print(json.dumps({k: rec[k] for k in rec if k not in ("node_tv", "visit_prob")})[:900], flush=True)
    json.dump(report, open(out, "w"), indent=1)
    print("history:", report["history"])


if __name__ == "__main__":
    main()
