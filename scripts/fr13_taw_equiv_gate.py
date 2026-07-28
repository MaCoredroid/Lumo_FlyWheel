#!/usr/bin/env python3
"""FR13_TAW distribution-equivalence gate vs the legacy device-multidraft walk.

Regime: logits sharpened toward the draft tokens so the walk ACCEPTS deep
(mean accepted len ~2-4), exercising every branch (source draw, accept,
residual, leaf bonus). Compares per-position accept rates + mean lens over N
seeds; TAW is distribution-equal (NOT byte-equal: different rng consumption),
the same standard used when device-multidraft replaced the numpy host walk.

PASS bar: |Δ per-position accept rate| < 3*binomial_sigma at every position
AND |Δ mean len| < 0.05.
"""
import sys, math
import torch
sys.path.insert(0, "scripts")
import fr13_device_multidraft_kernel as dm

torch.manual_seed(7)
V, B, N = 96, 4, 8000
# tail6-like topology: root->(2 kids), spine chain + 1 branch per depth
parents = [-1, -1, 0, 0, 2, 2, 4]
NC = len(parents)
counts = [NC] * B
parents_flat = torch.tensor(parents * B)
MAXSPEC = 5

def make_case(seed):
    g = torch.Generator().manual_seed(seed)
    drafts = torch.randint(0, V, (NC * B,), generator=g)
    tl = torch.randn(NC * B, V, generator=g)
    # sharpen: put extra mass on each node's CHILDREN's draft tokens so accepts happen
    for r in range(B):
        st = r * NC
        for node in range(NC):
            par = parents[node]
            row = st + (0 if par < 0 else par)  # parent's target row feeds child accept
            tl[st + node] = tl[st + node] * 0.3
        for node in range(NC):
            tl[st + node, drafts[st + node]] += 3.0
    sl = torch.randn(NC * B, V, generator=g) * 0.5
    bonus = torch.randint(0, V, (B, 1), generator=g)
    return drafts, tl, sl, bonus

def stats(fn):
    pos_lens = torch.zeros(MAXSPEC + 2)
    total = 0
    for t in range(N):
        drafts, tl, sl, bonus = make_case(50_000 + t)
        gens = {i: torch.Generator().manual_seed(90_000 + t * B + i) for i in range(B)}
        out = fn(drafts, tl, sl, bonus, gens)
        rows, _, alens, paths, _ = out
        for al in (alens if isinstance(alens, list) else list(alens)):
            pos_lens[: int(al) + 1] += 1  # survival counts per depth
            total += 1
    return pos_lens / total, (pos_lens.sum() / total)

def legacy(drafts, tl, sl, bonus, gens):
    return dm.fr13_device_multidraft_commit(
        counts, drafts, parents_flat, tl, sl, None, bonus, MAXSPEC, generators=gens)

def taw(drafts, tl, sl, bonus, gens):
    return dm.fr13_taw_commit(
        counts, drafts, parents_flat, tl, sl, bonus, MAXSPEC, generators=gens)

surv_l, len_l = stats(legacy)
surv_t, len_t = stats(taw)
n = N * B
print("depth-survival legacy:", " ".join(f"{x:.4f}" for x in surv_l.tolist()))
print("depth-survival taw:   ", " ".join(f"{x:.4f}" for x in surv_t.tolist()))
print(f"mean accepted len: legacy={len_l:.4f} taw={len_t:.4f}")
fails = []
for d in range(MAXSPEC + 2):
    p = float(surv_l[d]); q = float(surv_t[d])
    sigma = math.sqrt(max(p * (1 - p), 1e-9) / n)
    if abs(p - q) > 3 * sigma + 1e-9:
        fails.append((d, p, q, 3 * sigma))
if abs(float(len_l - len_t)) > 0.05:
    fails.append(("meanlen", float(len_l), float(len_t), 0.05))
if fails:
    print(">>> FAIL:", fails)
    sys.exit(1)
print(">>> PASS — TAW distribution-equal to legacy walk (per-depth survival + mean len)")
