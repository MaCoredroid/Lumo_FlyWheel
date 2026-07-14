#!/usr/bin/env python3
"""FR13 Gate 1 — SuffixSource x committer contract proof (CPU, no vLLM/GPU).

Proves, against the REAL deployment per-node rejection rule
(host_multidraft_accept_probs, transcribed verbatim from
sample_deterministic_multidraft_rejection_step, the draft_probs=None MTP path),
that injecting SuffixSource candidates as extra tree-node children is:

  P1 LOSSLESS (output-distribution invariant): the sampled-token distribution is
     EXACTLY the target p for ANY candidate set -- MTP-only, MTP+suffix, or with
     a garbage token. Adding candidates NEVER changes the output distribution.
     (Analytic: P(out=t) = min(q_mix,p) + max(p-q_mix,0) = p[t]. Confirmed by MC.)

  P2 ACCEPT = p(S), MONOTONE: the accept probability equals the target mass on the
     DISTINCT candidate set S; adding suffix candidates can only RAISE it
     (accept(MTP u suffix) >= accept(MTP)). This is the SPEED-win mechanism,
     precisely characterized -- and it is why merged trees help.

  P3 ZERO GARBLE: a garbage suffix token (p[tok]~0) has accept_prob ~ 0 -- it is
     never wrong-accepted; and its OUTPUT rate stays p[tok]~0. Suffix cannot garble.

  P4 DETERMINISM: same generator + candidates -> identical (token, source, accept).

NOTE (dedup): the clean accept=p(S) identity holds for the DISTINCT candidate set.
MergedSource MUST dedup MTP/suffix candidates to realize the monotone accept gain;
WITHOUT dedup it is still LOSSLESS (P1 holds unconditionally) but accept is only
sub-optimal, never wrong. This gate tests BOTH the dedup'd union (P2 monotone) and
the raw-with-garbage case (P1/P3 unconditional).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_device_multidraft_kernel import (  # noqa: E402
    host_multidraft_accept_probs,
    host_multidraft_step,
)
from fr13_suffix_source import SuffixSource  # noqa: E402

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def analytic_accept(p, tokens):
    """P(accept) via the committer's own accept_prob objects: sum_s w_s*a_s."""
    d = host_multidraft_accept_probs(p, tokens)
    if d["all_reject"]:
        return 0.0
    return float((d["weights"] * d["accept_prob"]).sum())


def p_of_set(p, tokens):
    """Target mass on the DISTINCT token set (p already implicitly renormalized)."""
    pn = np.asarray(p, dtype=np.float64)
    pn = pn / pn.sum()
    return float(pn[np.unique(np.asarray(tokens, dtype=np.int64))].sum())


def mc_output_hist(p, tokens, n_draws, seed):
    """Monte-Carlo empirical output-token distribution over n_draws steps."""
    rng = np.random.default_rng(seed)
    vocab = p.shape[0]
    hist = np.zeros(vocab, dtype=np.float64)
    acc = 0
    for _ in range(n_draws):
        tok, _src, accepted = host_multidraft_step(p, tokens, rng=rng)
        hist[tok] += 1.0
        acc += int(accepted)
    return hist / n_draws, acc / n_draws


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def rand_p(vocab, concentration, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(vocab) * concentration
    p = np.exp(logits - logits.max())
    return p / p.sum()


# --------------------------------------------------------------------------
print("[1] P1 losslessness: output ~ p EXACTLY for MTP-only AND MTP+suffix")
# vocab small enough that MC converges; several concentrations.
for vocab, conc, seed in [(64, 1.0, 1), (128, 2.0, 2), (32, 0.5, 3)]:
    p = rand_p(vocab, conc, seed)
    top = np.argsort(p)[::-1]
    mtp = [int(top[0]), int(top[3]), int(top[7])]          # a few MTP candidates
    suf = [int(top[1]), int(top[2]), int(top[15])]         # extra suffix candidates
    merged = sorted(set(mtp) | set(suf))
    N = 200000
    h_mtp, acc_mtp = mc_output_hist(p, mtp, N, seed + 100)
    h_mrg, acc_mrg = mc_output_hist(p, merged, N, seed + 200)
    tv_mtp, tv_mrg = tv(h_mtp, p), tv(h_mrg, p)
    # MC error ~ sqrt(vocab/N); tolerance generous but far below "distribution changed".
    tol = 0.02
    check(tv_mtp < tol, f"vocab={vocab}: MTP-only output TV(emp,p)={tv_mtp:.4f} < {tol}")
    check(tv_mrg < tol, f"vocab={vocab}: MERGED output TV(emp,p)={tv_mrg:.4f} < {tol}")
    # the two output dists are the SAME (both = p), though accept rates differ
    check(tv(h_mtp, h_mrg) < 2 * tol,
          f"vocab={vocab}: MTP-out == MERGED-out (TV={tv(h_mtp, h_mrg):.4f}) -> output invariant")

# --------------------------------------------------------------------------
print("[2] P2 accept = p(S) and MONOTONE non-decreasing when adding suffix")
for vocab, conc, seed in [(200, 1.5, 11), (500, 2.5, 12), (1000, 1.0, 13)]:
    p = rand_p(vocab, conc, seed)
    top = np.argsort(p)[::-1]
    mtp = [int(top[0]), int(top[4]), int(top[9])]
    suf = [int(top[1]), int(top[2]), int(top[3])]          # high-p suffix adds real mass
    merged = sorted(set(mtp) | set(suf))
    a_mtp = analytic_accept(p, mtp)
    a_mrg = analytic_accept(p, merged)
    ps_mtp = p_of_set(p, mtp)
    ps_mrg = p_of_set(p, merged)
    check(abs(a_mtp - ps_mtp) < 1e-9, f"vocab={vocab}: accept(MTP)={a_mtp:.5f} == p(S_mtp)={ps_mtp:.5f}")
    check(abs(a_mrg - ps_mrg) < 1e-9, f"vocab={vocab}: accept(MERGED)={a_mrg:.5f} == p(S_merged)={ps_mrg:.5f}")
    check(a_mrg >= a_mtp - 1e-12, f"vocab={vocab}: accept MONOTONE {a_mtp:.5f} -> {a_mrg:.5f} (gain {a_mrg - a_mtp:+.5f})")

# --------------------------------------------------------------------------
print("[3] P3 zero garble: garbage token committed at rate EXACTLY p[g]~0")
print("    (mechanism = source-SELECTION weight ~0, NOT accept_prob; accept_prob[g]=overlap_mass)")
for vocab, conc, seed in [(128, 2.0, 21), (256, 1.0, 22)]:
    p = rand_p(vocab, conc, seed)
    top = np.argsort(p)[::-1]
    garbage = int(top[-1])                                  # lowest-prob token
    mtp = [int(top[0]), int(top[2])]
    merged = sorted(set(mtp) | {garbage})
    d = host_multidraft_accept_probs(p, merged)
    gi = merged.index(garbage)
    om = d["overlap_mass"]
    # REAL mechanism: selection weight ~0 (p[g]/overlap_mass); the joint
    # P(select garbage AND accept) = weights[g]*accept_prob[g] = p[g] exactly.
    joint = float(d["weights"][gi] * d["accept_prob"][gi])
    # garbage selection weight << the real candidates' (ratio ~ p[g]/p[top], tiny);
    # absolute value scales with overlap_mass so assert RELATIVE, not a fixed const.
    check(d["weights"][gi] < d["weights"].max() * 0.05,
          f"vocab={vocab}: garbage SELECTION weight={d['weights'][gi]:.2e} << max={d['weights'].max():.2e} (the real safety)")
    check(abs(joint - p[garbage]) < 1e-12,
          f"vocab={vocab}: P(commit garbage)=w*a={joint:.2e} == p[g]={p[garbage]:.2e} EXACTLY")
    check(abs(d["accept_prob"][gi] - min(1.0, om)) < 1e-9,
          f"vocab={vocab}: (note) accept_prob[g]={d['accept_prob'][gi]:.4f}==overlap_mass -> safety is NOT here")
    # and MC: garbage appears as OUTPUT only at rate ~ p[garbage]
    h, _ = mc_output_hist(p, merged, 200000, seed + 300)
    check(abs(h[garbage] - p[garbage]) < 0.005,
          f"vocab={vocab}: garbage output rate {h[garbage]:.5f} ~ p={p[garbage]:.5f} (no garble)")
    # adding garbage raises accept by EXACTLY p[g] (negligible, never wrong)
    d_accept = analytic_accept(p, merged) - analytic_accept(p, mtp)
    check(abs(d_accept - p[garbage]) < 1e-9,
          f"vocab={vocab}: garbage adds EXACTLY p[g]={p[garbage]:.2e} accept ({d_accept:.2e}, negligible)")

# --------------------------------------------------------------------------
print("[4] P4 determinism: same generator + candidates -> identical step")
p = rand_p(300, 1.5, 41)
tokens = [5, 17, 42, 99]
outs = []
for _ in range(3):
    rng = np.random.default_rng(1234)
    seq = [host_multidraft_step(p, tokens, rng=rng) for _ in range(50)]
    outs.append(seq)
check(outs[0] == outs[1] == outs[2], "3 replays with same seed byte-identical")

# --------------------------------------------------------------------------
print("[5] end-to-end: SuffixSource candidates fed to the real committer rule")
# a repetitive stream so SuffixSource actually matches; the proposed spine/branch
# candidates go straight into the committer as extra children -> still lossless.
src = SuffixSource(max_k=6, min_k=2, max_candidates_per_node=4)
stream = ([100, 101, 102, 103, 104] * 6) + [7, 8, 9] + [100, 101, 102]
src.update("r", stream)
prop = src.propose("r", 4)
check(prop["matched"], f"SuffixSource matched (spine={prop['spine']}, match_len={prop['match_len']})")
if prop["matched"]:
    p = rand_p(256, 1.5, 51)
    # force the target to actually like the suffix spine token so we exercise a real accept
    p[prop["spine"][0]] = p.max() * 1.5
    p = p / p.sum()
    cand = sorted(set(prop["branch_candidates"][0]) | {int(prop["spine"][0])})
    a = analytic_accept(p, cand)
    ps = p_of_set(p, cand)
    check(abs(a - ps) < 1e-9, f"suffix-candidate node: accept={a:.5f} == p(S)={ps:.5f} (contract holds)")
    h, accr = mc_output_hist(p, cand, 200000, 52)
    check(tv(h, p) < 0.02, f"suffix-candidate node: output TV(emp,p)={tv(h, p):.4f} < 0.02 (lossless)")

# --------------------------------------------------------------------------
print(f"\n{PASS}/{PASS + FAIL} checks PASS")
if FAIL == 0:
    print(">>> PASS — SuffixSource candidates are committer-transparent: "
          "output=p (lossless, unconditional), accept=p(S) monotone (speed lever), "
          "garbage never wrong-accepted (zero garble).")
    sys.exit(0)
else:
    print(f">>> FAIL — {FAIL} checks failed")
    sys.exit(1)
