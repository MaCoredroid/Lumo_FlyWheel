#!/usr/bin/env python3
"""FR13 DEVICE MULTIDRAFT offline distribution-equivalence gate.

CPU-ONLY, boot-free. Proves the device-side multidraft committer
(``scripts/fr13_device_multidraft_kernel.py``) produces the SAME accept
DISTRIBUTION as the host reference for identical logits/tree -- the offline
distributional-losslessness proof the task requires. RNG samples differ
(numpy vs torch device generators); the DISTRIBUTION must match.

The host reference is the deployment accept rule:
  sample_deterministic_multidraft_rejection_step
  (src/lumo_flywheel_serving/fr10_tree_rejection_sampler.py :170-218)
  == the per-node rule the patcher's _lumo_tree_canonical_multidraft_sample
  walks (fr10_phase4_patch_vllm_tree_gdn.py :8306-8396, draft_probs=None).

Three checks (all per a stress matrix of (target_probs_row, child_draft_tokens)
cases covering: distinct drafts, duplicate drafts, a dominant target token, a
near-uniform target, a draft set with ZERO overlap mass -> all-reject, and many
random rows):

  A. PER-NODE ANALYTIC accept probs MATCH within float:
       host_multidraft_accept_probs (numpy) source weights + per-source accept
       prob + residual distribution  ==  the device kernel's analytic objects
       (recomputed via the device math on CPU tensors)  to <= 1e-9 abs.
     This is the *distribution* equality independent of any sampling.

  B. MARGINAL OUTPUT distribution MATCHES analytically:
       the exact closed-form output distribution of the node (accept the
       selected draft with prob accept_prob, else residual) computed two ways
       (host-objects vs device-objects) agrees <= 1e-9. This is the SpecInfer /
       multi-draft canonical output the committer commits.

  C. SAMPLED FREQUENCIES match within sampling noise:
       N draws from host_multidraft_step (numpy rng) vs N draws from the device
       kernel device_multidraft_node_step (torch cpu generator); the token /
       accept-rate frequencies agree within a 6-sigma binomial band. Confirms
       the device draws really follow the matched distribution (not just that
       the analytic objects line up).

Exit 0 = PASS. Nonzero = FAIL (and which check / case).

Usage:  python3 scripts/fr13_device_multidraft_offline_gate.py [--draws N]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "scripts" / "fr13_device_multidraft_kernel.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KM = _load("_fr13_device_multidraft_kernel", KERNEL)

try:
    import torch
except Exception:  # pragma: no cover
    print("FAIL: torch unavailable -- the device kernel needs torch even on CPU")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Device analytic objects on CPU (the math the device kernel runs, exposed as
# distributions so they can be A/B'd against the host objects without sampling).
# This recomputes EXACTLY what device_multidraft_node_step uses internally.
# ---------------------------------------------------------------------------
def device_analytic_objects(p_np: np.ndarray, child_draft_tokens):
    p = torch.tensor(p_np, dtype=torch.float64)
    p = p / p.sum()
    tokens = torch.tensor(list(child_draft_tokens), dtype=torch.long)
    overlaps = p[tokens]
    overlap_mass = overlaps.sum()
    vocab = int(p.shape[0])
    if float(overlap_mass.item()) <= 0.0:
        return {
            "overlap_mass": 0.0,
            "weights": None,
            "accept_prob": None,
            "residual": p.numpy().copy(),
            "all_reject": True,
        }
    weights = overlaps / overlap_mass
    accept_prob = torch.empty(tokens.shape[0], dtype=torch.float64)
    for src in range(tokens.shape[0]):
        same = tokens == tokens[src]
        q_mix_token = weights[same].sum()
        accept_prob[src] = torch.clamp(p[int(tokens[src])] / q_mix_token, max=1.0)
    q_mix_vocab = torch.zeros(vocab, dtype=torch.float64)
    q_mix_vocab.scatter_add_(0, tokens, weights)
    residual = torch.clamp(p - q_mix_vocab, min=0.0)
    mass = residual.sum()
    if float(mass.item()) == 0.0:
        residual = p
    else:
        residual = residual / mass
    return {
        "overlap_mass": float(overlap_mass.item()),
        "weights": weights.numpy(),
        "accept_prob": accept_prob.numpy(),
        "residual": residual.numpy(),
        "all_reject": False,
    }


def marginal_output_distribution(p_np, child_draft_tokens, objs):
    """Exact closed-form node output distribution from the {weights, accept_prob,
    residual} objects. P(emit token t) =
        sum_s w[s] * accept_prob[s] * [draft[s]==t]    (accepted branch)
      + (sum_s w[s] * (1-accept_prob[s])) * residual[t]  (rejected branch)
    On all-reject: residual (== p) directly."""
    vocab = int(np.asarray(p_np).shape[0])
    out = np.zeros(vocab, dtype=np.float64)
    if objs["all_reject"]:
        return np.asarray(objs["residual"], dtype=np.float64).copy()
    tokens = np.asarray(list(child_draft_tokens), dtype=np.int64)
    w = np.asarray(objs["weights"], dtype=np.float64)
    ap = np.asarray(objs["accept_prob"], dtype=np.float64)
    reject_mass = 0.0
    for s in range(tokens.shape[0]):
        out[int(tokens[s])] += w[s] * ap[s]
        reject_mass += w[s] * (1.0 - ap[s])
    out += reject_mass * np.asarray(objs["residual"], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Stress matrix of (target_probs_row, child_draft_tokens) cases.
# ---------------------------------------------------------------------------
def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def stress_cases(seed=1234):
    rng = np.random.default_rng(seed)
    cases = []

    # 1. distinct drafts, moderate vocab, random logits.
    for _ in range(8):
        vocab = int(rng.integers(16, 64))
        p = _softmax(rng.normal(0, 2.0, size=vocab))
        nch = int(rng.integers(1, 5))
        toks = rng.choice(vocab, size=nch, replace=False).tolist()
        cases.append(("distinct", p, toks))

    # 2. DUPLICATE drafts (two children propose the same token id) -> exercises
    #    the q_mix_token sum-over-duplicates branch.
    for _ in range(6):
        vocab = int(rng.integers(16, 48))
        p = _softmax(rng.normal(0, 1.5, size=vocab))
        t = int(rng.integers(0, vocab))
        toks = [t, t, int(rng.integers(0, vocab))]
        cases.append(("duplicate", p, toks))

    # 3. dominant target token (one token ~0.95): high accept on that draft.
    for _ in range(4):
        vocab = int(rng.integers(16, 40))
        logits = rng.normal(0, 0.5, size=vocab)
        logits[int(rng.integers(0, vocab))] += 8.0
        p = _softmax(logits)
        toks = rng.choice(vocab, size=int(rng.integers(1, 4)), replace=False).tolist()
        cases.append(("dominant", p, toks))

    # 4. near-uniform target.
    for _ in range(3):
        vocab = int(rng.integers(20, 50))
        p = _softmax(rng.normal(0, 0.05, size=vocab))
        toks = rng.choice(vocab, size=int(rng.integers(1, 5)), replace=False).tolist()
        cases.append(("uniform", p, toks))

    # 5. ZERO overlap mass: draft tokens land where p == 0 (force via a vocab
    #    with hard zeros). all-reject -> residual == p.
    vocab = 32
    logits = np.full(vocab, -1e9)
    live = [3, 7, 11, 19]
    for i in live:
        logits[i] = rng.normal(0, 1.0)
    p = _softmax(logits)
    dead = [t for t in range(vocab) if t not in live][:3]
    cases.append(("zero_overlap", p, dead))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200_000,
                    help="samples for the frequency check (per case)")
    args = ap.parse_args()

    cases = stress_cases()
    fail = 0
    n_a = n_b = n_c = 0

    for idx, (label, p, toks) in enumerate(cases):
        host = KM.host_multidraft_accept_probs(p, toks)
        dev = device_analytic_objects(p, toks)

        # --- A: per-node analytic objects match within float ---
        if host["all_reject"] != dev["all_reject"]:
            print(f"FAIL A [{idx}/{label}]: all_reject mismatch "
                  f"{host['all_reject']} vs {dev['all_reject']}")
            fail += 1
            continue
        if not host["all_reject"]:
            dw = np.abs(np.asarray(host["weights"]) - np.asarray(dev["weights"])).max()
            da = np.abs(np.asarray(host["accept_prob"]) - np.asarray(dev["accept_prob"])).max()
            if dw > 1e-9 or da > 1e-9:
                print(f"FAIL A [{idx}/{label}]: weights d={dw:.2e} accept d={da:.2e}")
                fail += 1
                continue
        dr = np.abs(np.asarray(host["residual"]) - np.asarray(dev["residual"])).max()
        if dr > 1e-9:
            print(f"FAIL A [{idx}/{label}]: residual d={dr:.2e}")
            fail += 1
            continue
        n_a += 1

        # --- B: closed-form output distribution matches ---
        out_host = marginal_output_distribution(p, toks, host)
        out_dev = marginal_output_distribution(p, toks, dev)
        db = np.abs(out_host - out_dev).max()
        # sanity: it is a normalized distribution.
        if abs(out_host.sum() - 1.0) > 1e-6:
            print(f"FAIL B [{idx}/{label}]: host output not normalized "
                  f"sum={out_host.sum():.6f}")
            fail += 1
            continue
        if db > 1e-9:
            print(f"FAIL B [{idx}/{label}]: output dist d={db:.2e}")
            fail += 1
            continue
        n_b += 1

        # --- C: sampled frequencies match within 6-sigma binomial band ---
        N = int(args.draws)
        # host draws (numpy rng).
        hrng = np.random.default_rng(7000 + idx)
        h_tokens = np.empty(N, dtype=np.int64)
        h_accept = np.empty(N, dtype=bool)
        for i in range(N):
            tok, _src, acc = KM.host_multidraft_step(p, toks, rng=hrng)
            h_tokens[i] = tok
            h_accept[i] = acc
        # device draws (torch cpu generator).
        dgen = torch.Generator(device="cpu")
        dgen.manual_seed(9000 + idx)
        p_row = torch.tensor(p, dtype=torch.float32)
        child_t = torch.tensor(toks, dtype=torch.long)
        d_tokens = np.empty(N, dtype=np.int64)
        d_accept = np.empty(N, dtype=bool)
        for i in range(N):
            tok, _src, acc = KM.device_multidraft_node_step(
                p_row, child_t, generator=dgen
            )
            d_tokens[i] = tok
            d_accept[i] = acc

        # accept rate band (only meaningful when not all-reject).
        if not host["all_reject"]:
            exp_accept = float(np.dot(host["weights"], host["accept_prob"]))
            band = 6.0 * np.sqrt(max(exp_accept * (1 - exp_accept), 1e-12) / N)
            hr = h_accept.mean()
            dr2 = d_accept.mean()
            if abs(hr - exp_accept) > band + 1e-6 or abs(dr2 - exp_accept) > band + 1e-6:
                print(f"FAIL C [{idx}/{label}]: accept-rate host={hr:.4f} "
                      f"dev={dr2:.4f} exp={exp_accept:.4f} band={band:.4f}")
                fail += 1
                continue

        # token-frequency band vs the closed-form output distribution.
        vocab = int(p.shape[0])
        h_freq = np.bincount(h_tokens, minlength=vocab) / N
        d_freq = np.bincount(d_tokens, minlength=vocab) / N
        worst = 0.0
        worst_tok = -1
        for t in range(vocab):
            pt = out_host[t]
            if pt <= 0:
                # tokens with zero output prob must not appear (beyond noise).
                if h_freq[t] > 5.0 / N or d_freq[t] > 5.0 / N:
                    print(f"FAIL C [{idx}/{label}]: zero-prob token {t} sampled "
                          f"host={h_freq[t]:.5f} dev={d_freq[t]:.5f}")
                    fail += 1
                    worst_tok = -2
                    break
                continue
            band = 6.0 * np.sqrt(max(pt * (1 - pt), 1e-12) / N)
            dh = abs(h_freq[t] - pt)
            dd = abs(d_freq[t] - pt)
            if max(dh, dd) > band + 1.0 / N:
                if max(dh, dd) > worst:
                    worst = max(dh, dd)
                    worst_tok = t
        if worst_tok == -2:
            continue
        if worst_tok >= 0:
            pt = out_host[worst_tok]
            band = 6.0 * np.sqrt(max(pt * (1 - pt), 1e-12) / N)
            print(f"FAIL C [{idx}/{label}]: token {worst_tok} freq off "
                  f"by {worst:.5f} (6sig band {band:.5f}, p={pt:.5f})")
            fail += 1
            continue
        n_c += 1

    print()
    print(f"A (per-node analytic objects match):   {n_a}/{len(cases)}")
    print(f"B (closed-form output dist match):     {n_b}/{len(cases)}")
    print(f"C (sampled freq within 6-sigma band):  {n_c}/{len(cases)}  "
          f"(N={args.draws} draws/case)")
    if fail:
        print(f"\nGATE FAIL: {fail} case(s) failed")
        return 1
    print("\nGATE PASS: device multidraft committer is distribution-equivalent "
          "to the host reference (per-node accept probs match within float; "
          "sampled frequencies match within sampling noise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
