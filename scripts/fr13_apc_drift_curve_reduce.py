#!/usr/bin/env python3
"""FR13 APC DRIFT CURVE reducer (offline, no GPU).

Reads the captures produced by scripts/fr13_apc_drift_curve.sh and emits the
drift curve: mamba_block_size -> {state drift, logit drift}, where

  REFERENCE = the single-pass / single-chunk no-cache boot (cfg, REF_BLOCK >=
              prefix length): ground-truth per-GDN-layer recurrent final_state
              + ground-truth decode logits on the fixed seq49 input.
  TEST b    = the cache-ON align boot at mamba_block_size=b: the seq49 hit-turn
              prefill final_state per layer + the decode logits on the same input.

DRIFT per block_size, vs the reference:
  (a) per-GDN-layer recurrent STATE drift: max_abs + mean_abs of (test-ref)
      final_state. We report the first-divergent layer (smallest layer index whose
      max_abs > --state-thresh) and the overall max across layers. Reuses the
      _max_abs metric shape of scripts/fr13_apc_cacherow_diff.py:67 (leading-batch
      squeeze + shape guard).
  (b) per-decode-token LOGIT drift on the first K decode tokens: max_abs, mean
      symmetric-ish KL(test||ref) over the captured rows, and the argmax-flip count.
      Extends the final-token argmax of scripts/fr13_apc_hit_first_divergence.py:410.

The MAGNITUDE settles the bug-vs-fp question:
  state/logit max_abs ~ 1 bf16 ULP .. ~0.0078  => fp reassociation noise (more
      cross-chunk folds at smaller block = slightly larger, but bounded).
  state/logit max_abs ~ tens, OR argmax flips that GROW as block shrinks => a real
      wrong-state bug (the restored GDN checkpoint is the wrong realization).

Capture layout (call-indexed, schema fr13.prefill_gdn_capture.v1 /
fr13.final_logit_capture.v1):
  <ref-gdn>.call*.pt          per-GDN-layer ref final_state (latest per layer)
  <ref-logit>.call*.pt        ref decode logits, one file per decode forward
  <rundir>/logs/on_b{B}_gdn.call*.pt    test final_state per layer for block B
  <rundir>/logs/on_b{B}_logit.call*.pt  test decode logits for block B
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


# ---- payload helpers (shared shape with fr13_apc_cacherow_diff.py /
#      fr13_apc_hit_first_divergence.py) ----

def _call_idx(p: str) -> int:
    m = re.search(r"\.call(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1


def _layer_idx(s: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", str(s))
    return int(m.group(1)) if m else None


def _payload_layer(pay: dict[str, Any]) -> int | None:
    for key in ("layer_name", "layer_prefix"):
        li = _layer_idx(pay.get(key, ""))
        if li is not None:
            return li
    return None


def _has_hit(pay: dict[str, Any]) -> bool:
    his = pay.get("has_initial_state")
    if his is None:
        return False
    try:
        return bool(his.any())
    except Exception:
        return False


def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Squeeze a leading batch dim if present, then require equal shape.

    Mirrors fr13_apc_cacherow_diff.py:_max_abs (final_state is [B, vh, hv, hk];
    a per-row state is [vh, hv, hk])."""
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        if a.ndim == b.ndim + 1 and a.shape[0] == 1:
            a = a[0]
        elif b.ndim == a.ndim + 1 and b.shape[0] == 1:
            b = b[0]
    if a.shape != b.shape:
        return None
    return a, b


def _state_drift(test: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if test is None or ref is None:
        return {"missing": True}
    al = _align(test, ref)
    if al is None:
        return {"shape_mismatch": [list(test.shape), list(ref.shape)]}
    d = (al[0] - al[1]).abs()
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "shape": list(al[0].shape),
    }


# ---- GDN state loaders ----

def _load_latest_per_layer(prefix: str, prefer_hit: bool) -> dict[int, dict[str, Any]]:
    """Latest call per GDN layer from a <prefix>.call*.pt set.

    prefer_hit=True (TEST/cache-ON): pick the latest call whose has_initial_state has
    a True row (= the actual seq49 cache-HIT prefill), reusing fr13_apc_hit_first_
    divergence.py:_load_on_hit. prefer_hit=False (REF/single-chunk no-cache): every
    call IS a fresh full prefill, so just take the latest (highest call index)."""
    out: dict[int, dict[str, Any]] = {}
    files = sorted(glob.glob(prefix + ".call*.pt"), key=_call_idx, reverse=True)
    for p in files:
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        li = _payload_layer(pay)
        if li is None or li in out:
            continue
        if prefer_hit and not _has_hit(pay):
            continue
        out[li] = pay
    return out


# ---- logit loaders + drift ----

def _load_logits_concat(prefix: str, k: int) -> torch.Tensor | None:
    """Concatenate the first k decode forwards' captured logit rows in call order
    into one [rows, vocab] tensor (schema fr13.final_logit_capture.v1, field
    'logits' = [n_rows, vocab])."""
    rows: list[torch.Tensor] = []
    files = sorted(glob.glob(prefix + ".call*.pt"), key=_call_idx)
    for p in files[:k] if k > 0 else files:
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        lg = pay.get("logits")
        if lg is None:
            continue
        lg = lg.float()
        if lg.ndim == 1:
            lg = lg.unsqueeze(0)
        rows.append(lg)
    if not rows:
        return None
    width = min(t.shape[-1] for t in rows)
    return torch.cat([t[:, :width] for t in rows], dim=0)


def _logit_drift(test: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if test is None or ref is None:
        return {"missing": True}
    n = min(test.shape[0], ref.shape[0])
    v = min(test.shape[1], ref.shape[1])
    if n == 0 or v == 0:
        return {"empty": True}
    t = test[:n, :v]
    r = ref[:n, :v]
    max_abs = float((t - r).abs().max().item())
    # KL(test||ref) per row over the softmax distributions, averaged over rows.
    lt = torch.log_softmax(t, dim=-1)
    lr = torch.log_softmax(r, dim=-1)
    pt = lt.exp()
    kl_rows = (pt * (lt - lr)).sum(dim=-1)  # KL(test||ref) per row, nats
    kl_mean = float(kl_rows.mean().item())
    kl_max = float(kl_rows.max().item())
    at = t.argmax(dim=-1)
    ar = r.argmax(dim=-1)
    flips = int((at != ar).sum().item())
    return {
        "rows": int(n),
        "vocab": int(v),
        "max_abs": max_abs,
        "kl_mean_nats": kl_mean,
        "kl_max_nats": kl_max,
        "argmax_flips": flips,
        "argmax_flip_frac": flips / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True, type=Path)
    ap.add_argument("--ref-gdn", required=True,
                    help="reference GDN capture prefix (without .call*.pt)")
    ap.add_argument("--ref-logit", required=True,
                    help="reference logit capture prefix (without .call*.pt)")
    ap.add_argument("--blocks", required=True,
                    help="space-separated mamba_block_size list, e.g. '1024 2048 4096 8192 16384'")
    ap.add_argument("--k-decode", type=int, default=8,
                    help="first K decode forwards' logits to compare")
    ap.add_argument("--state-thresh", type=float, default=1e-2,
                    help="per-layer state max_abs above which a layer is 'divergent'")
    ap.add_argument("--out", type=Path,
                    default=Path("output/fr13_apc_drift_curve.jsonl"))
    args = ap.parse_args()

    blocks = [int(b) for b in args.blocks.split()]

    ref_state = _load_latest_per_layer(args.ref_gdn, prefer_hit=False)
    ref_logit = _load_logits_concat(args.ref_logit, args.k_decode)
    if not ref_state:
        raise SystemExit(f"FATAL: no reference GDN captures at {args.ref_gdn}.call*.pt")

    logdir = args.rundir / "logs"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    curve: list[dict[str, Any]] = []
    print("=== FR13 APC drift curve (vs single-chunk no-cache reference) ===")
    print(f"reference: {len(ref_state)} GDN layers, "
          f"{0 if ref_logit is None else ref_logit.shape[0]} decode logit rows")
    print(f"{'block':>7} | {'state_max':>11} {'state_layer':>11} {'state_mean':>11} "
          f"| {'logit_max':>10} {'kl_mean':>10} {'flips':>6} {'rows':>5}")

    with args.out.open("w") as fh:
        # record the reference self-row (block=ref => zero drift by construction) for
        # an explicit x-axis origin / sanity anchor.
        for b in blocks:
            test_state = _load_latest_per_layer(str(logdir / f"on_b{b}_gdn"), prefer_hit=True)
            test_logit = _load_logits_concat(str(logdir / f"on_b{b}_logit"), args.k_decode)

            per_layer: list[dict[str, Any]] = []
            first_div_layer = None
            overall_max = 0.0
            overall_max_layer = None
            mean_of_max = 0.0
            ncmp = 0
            for li in sorted(set(ref_state) & set(test_state)):
                sd = _state_drift(test_state[li].get("final_state"),
                                  ref_state[li].get("final_state"))
                rec = {"layer": li, **sd}
                per_layer.append(rec)
                mx = sd.get("max_abs")
                if isinstance(mx, float):
                    ncmp += 1
                    mean_of_max += mx
                    if mx > overall_max:
                        overall_max = mx
                        overall_max_layer = li
                    if mx > args.state_thresh and first_div_layer is None:
                        first_div_layer = li
            mean_of_max = (mean_of_max / ncmp) if ncmp else 0.0

            ld = _logit_drift(test_logit, ref_logit)

            row = {
                "mamba_block_size": b,
                "max_num_batched_tokens": b,  # forced == block by the launcher
                "n_layers_compared": ncmp,
                "state": {
                    "overall_max_abs": overall_max,
                    "overall_max_layer": overall_max_layer,
                    "mean_of_per_layer_max_abs": mean_of_max,
                    "first_divergent_layer": first_div_layer,
                    "state_thresh": args.state_thresh,
                    "first_divergent_layer_max_abs": (
                        next((r["max_abs"] for r in per_layer
                              if r["layer"] == first_div_layer), None)
                        if first_div_layer is not None else None
                    ),
                },
                "logit": ld,
                "per_layer": per_layer,
            }
            curve.append(row)
            fh.write(json.dumps(row) + "\n")

            sm = row["state"]["overall_max_abs"]
            sl = row["state"]["overall_max_layer"]
            smean = row["state"]["mean_of_per_layer_max_abs"]
            lm = ld.get("max_abs")
            kl = ld.get("kl_mean_nats")
            fl = ld.get("argmax_flips")
            rw = ld.get("rows")
            print(f"{b:>7} | {sm:>11.6g} {str(sl):>11} {smean:>11.6g} "
                  f"| {('%.6g' % lm) if isinstance(lm, float) else str(lm):>10} "
                  f"{('%.4g' % kl) if isinstance(kl, float) else str(kl):>10} "
                  f"{str(fl):>6} {str(rw):>5}")

    # ---- verdict heuristic (bug vs fp) ----
    state_maxes = [c["state"]["overall_max_abs"] for c in curve]
    logit_flips = [c["logit"].get("argmax_flips") or 0 for c in curve]
    biggest = max(state_maxes) if state_maxes else 0.0
    FP_CEIL = 0.0078  # ~ the documented fp-reassociation ceiling for this kernel
    print()
    if biggest <= FP_CEIL * 4:
        print(f"  VERDICT (heuristic): peak state drift {biggest:.6g} <= ~{FP_CEIL*4:.4g} "
              f"-> consistent with fp REASSOCIATION noise (more cross-chunk folds at "
              f"smaller block). Inspect the curve monotonicity below.")
    else:
        print(f"  VERDICT (heuristic): peak state drift {biggest:.6g} >> fp ceiling "
              f"({FP_CEIL:.4g}) -> a WRONG-STATE bug is likely (restored checkpoint is "
              f"the wrong realization), not mere reassociation.")
    # monotonicity hint: does drift fall as block grows (fewer boundaries)?
    if len(curve) >= 2:
        trend = "DECREASES" if state_maxes[-1] < state_maxes[0] else "does NOT decrease"
        print(f"  state drift {trend} from block={blocks[0]} ({state_maxes[0]:.6g}) "
              f"to block={blocks[-1]} ({state_maxes[-1]:.6g}); argmax flips per block "
              f"= {dict(zip(blocks, logit_flips))}")
    print(f"\n  drift curve jsonl -> {args.out}")


if __name__ == "__main__":
    main()
