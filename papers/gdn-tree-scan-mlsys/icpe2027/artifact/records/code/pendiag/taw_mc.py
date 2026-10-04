#!/usr/bin/env python3
"""Monte Carlo test of the DEPLOYED fixed32 tree-accept walk (_fr13_fixed32_taw_execute_torch, the exact float
oracle the CUDA path matches byte-for-byte) on fixed rows, against exact per-node target distributions.

Losslessness statement tested: given the walk reached node v, the token it emits there is distributed as p_v, where
p_v = softmax(target row of v's first child) if v has active children, else softmax(self row of v).
The vocabulary is compressed to the union of row supports and draft tokens, preserving id order (the walk's
inverse-CDF order); every other token has probability 0 in every row, so the walk's distribution is unchanged.
Per node: G-test of emitted-token counts vs expected probabilities (bins with expected < 5 pooled), Bonferroni over
tested nodes."""
import importlib.util, math, os, sys
import torch

SCRIPTS = "/home/user/shared/treehost-v2exp-20260930/scripts"
sys.path.insert(0, SCRIPTS)
import fr13_fixed32_topology as T  # noqa: E402

_spec = importlib.util.spec_from_file_location("fr13k", os.path.join(SCRIPTS, "fr13_device_multidraft_kernel.py"))
K = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(K)
MODE = "hydra27_fixed32"
PARENT = [-1] + [int(p) + 1 if int(p) >= 0 else 0 for p in T.DRAFT_PARENT]   # physical-row parents (row 0 = root)


def entry_for(n):
    K._fr13_fixed32_test_set_env(T, MODE)
    K.fr13_fixed32_taw_set_work_callback(lambda _p: None)
    vm = int(T.VALID_MASK_BY_MODE[MODE]); dev = torch.device("cpu")
    K.fr13_fixed32_taw_preseed(dev, mode=MODE, valid_mask=vm)
    e1 = K._FR13_FIXED32_TAW_CACHE[K.fr13_fixed32_taw_cache_key(MODE, vm, 1, dev)]
    e = {}
    for k, v in e1.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            e[k] = v.repeat(n, *([1] * (v.dim() - 1))).clone()
        else:
            e[k] = v
    e["batch_size"] = n
    e["starts"] = torch.zeros(n, dtype=e1["starts"].dtype)
    return e, e1


def active_children():
    ch = {}
    for parent, kids in T.active_child_lists(MODE).items():
        ch[int(parent)] = [int(k) for k in kids]
    return ch            # draft-index space: root = -1, drafts 0..30 (accepted_path_rows hold draft + 1)


def run_walks(target, selfr, drafts, n, uniforms=None, chunk=50000):
    """target/selfr: [31, Vc] processed logits (compressed vocab); drafts: [31] compressed ids. Returns list of
    (path rows, emitted tokens)."""
    outs = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        e, _ = entry_for(m)
        u = torch.rand(m, int(T.WALK_CAP), 3) if uniforms is None else uniforms[s:s + m]
        d = drafts.view(1, -1).repeat(m, 1).to(torch.int64)
        bonus = torch.zeros(m, dtype=torch.int64)
        r = K._fr13_fixed32_taw_execute_torch(T, e, d, target, selfr, bonus, u, walk_cap=int(T.WALK_CAP))
        tok, tlen, rows, alen = r[0], r[1], r[2], r[3]
        outs.append((tok.clone(), tlen.clone(), rows.clone(), alen.clone()))
    return outs


def node_counts(outs, Vc):
    """counts[node] = histogram (Vc) of the token emitted at that node, over walks that reached it."""
    counts = {}
    for tok, tlen, rows, alen in outs:
        for i in range(tok.shape[0]):
            L = int(alen[i]); path = [-1] + [int(x) - 1 for x in rows[i, :L]]
            ntok = int(tlen[i])
            for j, v in enumerate(path):
                if j < ntok:
                    counts.setdefault(v, torch.zeros(Vc, dtype=torch.long))[int(tok[i, j])] += 1
    return counts


def expected(target, selfr, ch):
    exp = {}
    for v in range(-1, 31):
        kids = ch.get(v, [])
        if kids:
            exp[v] = torch.softmax(target[kids[0]].double(), -1)           # target row of the first child = distribution at v
        elif v >= 0:
            exp[v] = torch.softmax(selfr[v].double(), -1)                  # leaf: its own (self) row
    return exp


def g_test(cnt, p):
    n = int(cnt.sum())
    if bool((cnt[p <= 0] > 0).any()):      # a token with zero target probability was emitted: impossible event
        return float("inf"), 0.0, n
    e = p * n
    big = e >= 5
    obs = torch.cat([cnt[big].double(), cnt[~big].double().sum().view(1)])
    ex = torch.cat([e[big], e[~big].sum().view(1)])
    keep = ex > 0
    obs, ex = obs[keep], ex[keep]
    if obs.numel() < 2:                       # single support bin: only the zero-probability check applies
        return 0.0, 1.0, n
    g = 2 * float((obs[obs > 0] * (obs[obs > 0] / ex[obs > 0]).log()).sum())
    df = max(1, int(keep.sum()) - 1)
    # Wilson-Hilferty chi-square tail
    z = ((g / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    return g, pval, n


def test(target, selfr, drafts, ref_target, ref_self, n, alpha=1e-3, uniforms=None, min_visits=2000):
    ch = active_children()
    outs = run_walks(target, selfr, drafts, n, uniforms)
    cnt = node_counts(outs, target.shape[1])
    exp = expected(ref_target, ref_self, ch)
    res = []
    tested = [v for v in cnt if v in exp and int(cnt[v].sum()) >= min_visits]
    for v in sorted(tested):
        g, p, nv = g_test(cnt[v], exp[v])
        tv = 0.5 * float((cnt[v].double() / nv - exp[v]).abs().sum())
        res.append({"node": v, "visits": nv, "G": round(g, 1), "p": p, "tv_hat": round(tv, 4)})
    bonf = alpha / max(1, len(res))
    return {"nodes_tested": len(res), "rejected": [r for r in res if r["p"] < bonf], "results": res, "alpha_bonferroni": bonf}


def _selfcheck():
    torch.manual_seed(0)
    Vc = 40
    target = torch.randn(31, Vc) * 2; selfr = torch.randn(31, Vc) * 2
    drafts = torch.randint(0, Vc, (31,))
    ok = test(target, selfr, drafts, target, selfr, 100000)
    bad = test(target, selfr, drafts, target / 0.6, selfr / 0.6, 100000)
    print("selfcheck clean: tested", ok["nodes_tested"], "rejected", len(ok["rejected"]), "| double-temp fault: rejected", len(bad["rejected"]))


if __name__ == "__main__":
    _selfcheck()
