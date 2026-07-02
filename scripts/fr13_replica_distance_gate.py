#!/usr/bin/env python3
"""FR13 REPLICA DISTANCE GATE — the self-noise floor vs cross-distance PERMUTATION test (temp 0.6).

Implements the exact 4-step lossless verdict:
  1. Replicate: run A (ref) and B (test) each K times at temp 0.6, different seeds, SAME prompts.
  2. Self-noise floor: the distribution of A-vs-A' distances (A's replicas among themselves).
  3. Cross distance: A-vs-B.
  4. Test: is the A-vs-B distance distribution significantly ABOVE the A-vs-A' floor?
       within floor  -> LOSSLESS (the resolve gap was seed)
       above  floor  -> CARRIER

No teacher-forcing, no argmax surrogate: the "distance" is computed on the STOCHASTIC deployment
outcome itself. Default distance = normalized Hamming over the per-task resolve vector (fraction of
shared tasks where two replicas disagree on resolved/not). Two replicas that resolve the same set of
tasks have distance 0; opposite outcomes on every shared task = 1.

Statistic (one-sided): delta = mean(cross A-B distances) - mean(within-A distances).
Null: config label is exchangeable (B is just another A-replica). Pool the n_a+n_b replica resolve
vectors, relabel into groups of the original sizes over ALL C(n_a+n_b, n_a) splits (or a random sample
if too many), recompute delta*. p = (#{delta* >= delta_obs} + 1) / (n_perm + 1).

Usage:
  fr13_replica_distance_gate.py <runroot> --ref native --test chain5 [--alpha 0.05] [--json]
Read-only. Reuses fr13_replica_selfnoise_gate._task_verdicts / _replicas.
"""
import argparse, itertools, json, os, sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_replica_selfnoise_gate import _task_verdicts, _replicas


def _resolve_vectors(arms, tasks):
    """[(arm_dir, {task: bool})] restricted to `tasks`; drop arms with no overlap."""
    out = []
    for d in arms:
        v = _task_verdicts(d)
        vv = {t: v[t] for t in tasks if t in v}
        if vv:
            out.append((d, vv))
    return out


def _hamming(va, vb):
    """Normalized Hamming over SHARED tasks (fraction of disagreements). None if no shared task."""
    shared = set(va) & set(vb)
    if not shared:
        return None
    dis = sum(1 for t in shared if va[t] != vb[t])
    return dis / len(shared)


def _within(group):
    """Distances over all unordered pairs inside `group` (the self-noise floor)."""
    ds = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            d = _hamming(group[i], group[j])
            if d is not None:
                ds.append(d)
    return ds


def _cross(group_a, group_b):
    """Distances over all ordered cross pairs a x b."""
    ds = []
    for a in group_a:
        for b in group_b:
            d = _hamming(a, b)
            if d is not None:
                ds.append(d)
    return ds


def _delta(vecs, is_test):
    """delta = mean(cross) - mean(within-ref) for a boolean test-mask over pooled vecs."""
    ref = [v for v, t in zip(vecs, is_test) if not t]
    test = [v for v, t in zip(vecs, is_test) if t]
    if len(ref) < 2 or not test:
        return None
    within = _within(ref)
    cross = _cross(ref, test)
    if not within or not cross:
        return None
    return (sum(cross) / len(cross)) - (sum(within) / len(within)), within, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runroot")
    ap.add_argument("--ref", default="native")
    ap.add_argument("--test", default="chain5")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-perms", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ref_arms, test_arms = _replicas(a.runroot, a.ref), _replicas(a.runroot, a.test)
    if len(ref_arms) < 2 or not test_arms:
        print(f"need >=2 ref replicas (self-noise floor) and >=1 test (ref='{a.ref}':{len(ref_arms)}, "
              f"test='{a.test}':{len(test_arms)})", file=sys.stderr); sys.exit(2)

    # common task universe = union of all tasks seen in any replica
    all_tasks = set()
    for d in ref_arms + test_arms:
        all_tasks |= set(_task_verdicts(d))
    all_tasks = sorted(all_tasks)
    ref_v = _resolve_vectors(ref_arms, all_tasks)
    test_v = _resolve_vectors(test_arms, all_tasks)
    ref_vecs = [v for _, v in ref_v]
    test_vecs = [v for _, v in test_v]
    n_a, n_b = len(ref_vecs), len(test_vecs)
    if n_a < 2 or n_b < 1:
        print("insufficient replicas with resolve data", file=sys.stderr); sys.exit(2)

    obs = _delta(ref_vecs + test_vecs, [False] * n_a + [True] * n_b)
    if obs is None:
        print("no computable distances (no shared tasks across replicas)", file=sys.stderr); sys.exit(2)
    delta_obs, within, cross = obs
    floor_mean = sum(within) / len(within)
    cross_mean = sum(cross) / len(cross)

    # permutation null: relabel n_b of the pooled (n_a+n_b) vectors as "test"
    pooled = ref_vecs + test_vecs
    N = n_a + n_b
    total_splits = comb(N, n_b)
    exhaustive = total_splits <= a.max_perms
    ge = 0
    n_perm = 0
    if exhaustive:
        idxs = range(N)
        for test_idx in itertools.combinations(idxs, n_b):
            mask = [i in test_idx for i in range(N)]
            r = _delta(pooled, mask)
            if r is None:
                continue
            n_perm += 1
            if r[0] >= delta_obs - 1e-12:
                ge += 1
    else:
        import random
        rng = random.Random(0)
        base = list(range(N))
        for _ in range(a.max_perms):
            rng.shuffle(base)
            mask = [False] * N
            for i in base[:n_b]:
                mask[i] = True
            r = _delta(pooled, mask)
            if r is None:
                continue
            n_perm += 1
            if r[0] >= delta_obs - 1e-12:
                ge += 1
    p = (ge + 1) / (n_perm + 1) if n_perm else 1.0
    verdict = ("FAIL(CARRIER: A-B distance above A's self-noise floor)"
               if p < a.alpha else "PASS(LOSSLESS: within self-noise floor; resolve gap = seed)")

    out = {
        "ref": a.ref, "test": a.test, "K_ref": n_a, "K_test": n_b, "tasks": len(all_tasks),
        "self_noise_floor_mean": round(floor_mean, 4),
        "self_noise_floor_dist": sorted(round(x, 3) for x in within),
        "cross_distance_mean": round(cross_mean, 4),
        "cross_distance_dist": sorted(round(x, 3) for x in cross),
        "delta_obs": round(delta_obs, 4),
        "perm": {"n_perm": n_perm, "exhaustive": exhaustive, "total_splits": total_splits},
        "p": p, "alpha": a.alpha, "verdict": verdict,
    }
    if a.json:
        print(json.dumps(out, indent=2)); return
    print(f"==== FR13 REPLICA DISTANCE GATE (temp 0.6, permutation)  "
          f"ref='{a.ref}'(K={n_a}) vs test='{a.test}'(K={n_b})  tasks={len(all_tasks)} ====")
    print(f"  self-noise floor (A-vs-A'):  mean={floor_mean:.4f}  dist={out['self_noise_floor_dist']}")
    print(f"  cross distance   (A-vs-B) :  mean={cross_mean:.4f}  dist={out['cross_distance_dist']}")
    print(f"  delta = cross - floor     :  {delta_obs:+.4f}")
    print(f"  permutation p (1-sided, {'exhaustive' if exhaustive else 'sampled'} "
          f"n={n_perm}/{total_splits}): {p:.4f}")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
