#!/usr/bin/env python3
"""FR13 REPLICA SELF-NOISE GATE — statistically adjudicate carrier-vs-seed AT temp 0.6, no teacher-forcing.

The problem: at temp 0.6 a single run per config (N=1) confounds a real config effect (kernel/cache
"carrier") with sampling luck (seed). A trace diff describes but cannot attribute.

The fix: run each config K times at temp 0.6 with DIFFERENT seeds. The config's own replica-to-replica
variation IS the seed-noise floor. A config is a CARRIER only if the test config's outcome distribution
sits beyond the reference config's self-noise floor — tested statistically, on the stochastic
distribution itself (no argmax / teacher-forcing surrogate).

Primary metric = resolve outcome (binary), K replicas per (task, config):
  self-noise    : ref replicas among themselves — a task with 0<k<K for the REF is seed-sensitive
                  (a single-run flip there is pure seed); k in {0,K} = deterministic for the ref.
  cross test    : Cochran-Mantel-Haenszel (stratified by task) ref-vs-test on resolved/not — pools
                  power across tasks and folds the binomial seed-noise into the null. Reports MH odds
                  ratio + p. FAIL(carrier) if p<alpha AND OR>1 (ref resolves more); else PASS(within seed).
Secondary (higher-frequency, from the char-8 gate): pooled char-8/turn + degeneration/turn per config
  with the same one-sided Poisson rate test — a per-turn signal that needs far fewer replicas.

Usage:
  fr13_replica_selfnoise_gate.py <runroot> --ref native --test chain5 [--alpha 0.05] [--json]
    groups arm dirs in <runroot> by substring: dirs containing 'native' = ref replicas, 'chain5' = test.
Read-only. Reuses fr13_char8_degeneration_gate.arm_stats for the per-turn signals.
"""
import argparse, glob, json, os, sys
from math import sqrt
from scipy.stats import chi2, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_char8_degeneration_gate import arm_stats


def _task_verdicts(arm_dir):
    """{task: resolved_bool} for one replica arm dir."""
    out = {}
    PT = glob.glob(os.path.join(arm_dir, "**", "per_task", "*"), recursive=True)
    for d in [x for x in PT if os.path.isdir(x)]:
        t = os.path.basename(d).replace("astropy__astropy-", "")
        v = None
        for f in glob.glob(os.path.join(d, "**", "eval_report.json"), recursive=True):
            try:
                j = json.load(open(f)); er = j.get("eval_report") or j
                v = er.get("verdict") or ("resolved" if er.get("resolved") else er.get("failure_mode"))
            except Exception:
                pass
        if v is not None:
            out[t] = (v == "resolved")
    return out


def _replicas(runroot, sub):
    return sorted(d for d in glob.glob(os.path.join(runroot, "m_*"))
                  if os.path.isdir(d) and sub in os.path.basename(d))


def cmh(strata):
    """Cochran-Mantel-Haenszel over 2x2 strata [(a,b,c,d)]: a=ref-resolved, b=ref-fail, c=test-res, d=test-fail.
    Returns (X2, p_two_sided, MH_odds_ratio). Ref-resolves-more => OR>1."""
    sum_a = sum_E = sum_V = num = den = 0.0
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        if c1 == 0 or c2 == 0 or r1 == 0 or r2 == 0:
            continue  # non-informative stratum (all same outcome or one arm empty)
        used += 1
        sum_a += a
        sum_E += r1 * c1 / n
        sum_V += r1 * r2 * c1 * c2 / (n * n * (n - 1))
        num += a * d / n
        den += b * c / n
    if sum_V <= 0 or used == 0:
        return (0.0, 1.0, None)
    X2 = (abs(sum_a - sum_E) - 0.5) ** 2 / sum_V
    p = chi2.sf(X2, 1)
    orr = (num / den) if den > 0 else (float("inf") if num > 0 else None)
    return (X2, p, orr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runroot")
    ap.add_argument("--ref", default="native", help="substring identifying REFERENCE replica arms")
    ap.add_argument("--test", default="chain5", help="substring identifying TEST replica arms")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ref_arms = _replicas(a.runroot, a.ref)
    test_arms = _replicas(a.runroot, a.test)
    if not ref_arms or not test_arms:
        print(f"need >=1 replica arm per config (ref='{a.ref}':{len(ref_arms)}, test='{a.test}':{len(test_arms)})",
              file=sys.stderr); sys.exit(2)

    # per-task resolve counts k/K per config
    def counts(arms):
        tasks = {}
        for d in arms:
            for t, ok in _task_verdicts(d).items():
                r, k = tasks.get(t, (0, 0))
                tasks[t] = (r + (1 if ok else 0), k + 1)
        return tasks
    ref_c, test_c = counts(ref_arms), counts(test_arms)
    all_tasks = sorted(set(ref_c) | set(test_c))

    strata = []
    rows = []
    seed_sensitive_ref = 0
    for t in all_tasks:
        ra, rk = ref_c.get(t, (0, 0))
        ta, tk = test_c.get(t, (0, 0))
        strata.append((ra, rk - ra, ta, tk - ta))
        if rk and 0 < ra < rk:
            seed_sensitive_ref += 1
        rows.append({"task": t, "ref": f"{ra}/{rk}", "test": f"{ta}/{tk}",
                     "ref_seed_sensitive": bool(rk and 0 < ra < rk)})
    X2, p, orr = cmh(strata)
    resolve_verdict = ("FAIL(carrier: ref resolves more, beyond seed)"
                       if (p < a.alpha and (orr or 0) > 1) else "PASS(within-seed-noise)")

    # secondary per-turn signals (pooled over replicas)
    def pooled_turn(arms):
        T = C = D = 0
        for d in arms:
            s = arm_stats(d); T += s["turns"]; C += s["char8"]; D += s["degen_turns"]
        return T, C, D
    rT, rC, rD = pooled_turn(ref_arms)
    tT, tC, tD = pooled_turn(test_arms)
    def rate_fail(cr, er, ct, et):
        if not er or not et:
            return (1.0, None, "NA")
        n = cr + ct
        if n == 0:
            return (1.0, 1.0, "PASS(char8-lossless)")
        pv = binomtest(ct, n, et / (er + et), alternative="greater").pvalue
        rr = (ct / et) / (cr / er) if cr / er > 0 else (float("inf") if ct / et > 0 else 1.0)
        return (pv, rr, "FAIL(char8-carrier)" if (pv < a.alpha and rr > 1) else "PASS(char8-lossless)")
    c8_p, c8_rr, c8_v = rate_fail(rC, rT, tC, tT)
    dg_p, dg_rr, dg_v = rate_fail(rD, rT, tD, tT)

    out = {
        "ref": a.ref, "test": a.test, "K_ref": len(ref_arms), "K_test": len(test_arms),
        "per_task": rows, "ref_seed_sensitive_tasks": seed_sensitive_ref,
        "resolve_gate": {"cmh_X2": round(X2, 3), "p": p, "mh_odds_ratio": orr, "verdict": resolve_verdict},
        "char8_gate": {"ref_per1k": round(1000*rC/rT,2) if rT else None,
                       "test_per1k": round(1000*tC/tT,2) if tT else None,
                       "rate_ratio": c8_rr, "p": c8_p, "verdict": c8_v},
        "degen_gate": {"rate_ratio": dg_rr, "p": dg_p, "verdict": dg_v},
    }
    if a.json:
        print(json.dumps(out, indent=2)); return
    print(f"==== FR13 REPLICA SELF-NOISE GATE (temp 0.6)  ref='{a.ref}'(K={len(ref_arms)}) vs test='{a.test}'(K={len(test_arms)}) ====")
    print(f"{'task':<10}{'ref k/K':>10}{'test k/K':>10}{'ref-seed-sensitive':>22}")
    for r in rows:
        print(f"{r['task']:<10}{r['ref']:>10}{r['test']:>10}{str(r['ref_seed_sensitive']):>22}")
    print(f"\n  ref seed-sensitive tasks (0<k<K): {seed_sensitive_ref}/{len(rows)}  "
          f"(a single-run flip on these = SEED, not carrier)")
    print(f"\n  RESOLVE gate (CMH stratified by task): X2={X2:.3f}  p={p:.4f}  MH-OR={orr}  -> {resolve_verdict}")
    print(f"  char-8 gate (per-turn Poisson):  ref={out['char8_gate']['ref_per1k']}/1k  "
          f"test={out['char8_gate']['test_per1k']}/1k  p={c8_p:.4f} -> {c8_v}")
    print(f"  degeneration gate:  p={dg_p:.4f} -> {dg_v}")


if __name__ == "__main__":
    main()
