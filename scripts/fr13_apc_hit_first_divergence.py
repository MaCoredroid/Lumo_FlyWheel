#!/usr/bin/env python3
"""FR13 APC HIT-turn first-divergence localizer (offline, no GPU).

GOAL: on a single APC cache-HIT turn, localize the FIRST cache-ON vs
cache-OFF(no-spec oracle) divergence across the GDN-hybrid stack, in execution
order, over four candidate sites:

  (a) RESTORED conv-state seed   -- conv twin of initial_state, captured by
      FR13_APC_CONV_RESTORE_CAPTURE (payload field "conv_restore"). This is the
      K-1 conv prior-window read back by physical block_id BEFORE causal_conv1d
      overwrites it. FR13 memory pins the e2e-miss root at the conv prior-window,
      so this axis is the prime suspect.
  (b) RESTORED ssm-state seed    -- payload field "initial_state" (the recurrent
      state restored on the cache-hit prefill row). Prior confound work showed
      47/48 faithful, so this is expected mostly-clean.
  (c) per-GDN-layer OUTPUT       -- payload field "core_out" (the GDN core attn
      output for the hit-turn tokens). argmax/value divergence here means the
      seed difference (a/b) actually changed the layer's emitted activations.
  (d) final-token OUTPUT argmax  -- from FR13_FINAL_LOGIT_CAPTURE payloads
      (schema fr13.final_logit_capture.v1). The end-to-end question: did the
      hit-turn last-token argmax flip cache-ON vs oracle.

DECISION RULE ("first divergence"):
  We scan GDN layers in increasing index (= forward execution order). For each
  layer we test conv-seed then ssm-seed then layer-output, in that within-layer
  execution order (conv runs before the recurrent scan, whose output is
  core_out). A site is "divergent" when its confound-controlled ratio exceeds
  --ratio-thresh (default 3.0): ratio = restore_diff / natural_per_chunk_variation,
  where natural variation is the no-cache trajectory's own median per-chunk delta
  for that field+layer (so a restored state that merely sits at a different
  position on-trajectory is NOT flagged; only genuine off-trajectory corruption
  is). For core_out (an activation, not a recurrent state) we ALSO report a plain
  per-position argmax mismatch fraction, since core_out has no "trajectory".
  The FIRST (layer, site) tuple that crosses threshold, in (layer asc, site order
  conv<ssm<out) lexicographic order, is the carrier. If none cross AND the
  final-token argmax matches the oracle, the verdict is "no measurable carrier".

INPUTS (two capture dirs, each a dir of FR13_PREFILL_GDN_CAPTURE *.call*.pt):
  --on-dir   : cache-ON arm, the HIT turn. Use FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX
               large enough that the hit-turn prefill is captured; we pick, per
               layer, the LATEST capture whose has_initial_state has any True row
               (= an actual cache-hit prefill). If none of a layer's ON captures
               has a hit row, that layer is reported "no-hit (skipped)".
  --oracle-dir : cache-OFF / no-spec oracle arm = the no-cache ground-truth
               trajectory (every chunk's final_state + its own conv/core per call).
OPTIONAL (final-token argmax, axis d):
  --on-logits / --oracle-logits : a single FR13_FINAL_LOGIT_CAPTURE *.pt (or its
               .call0 variant) from each arm for the hit turn. If omitted, axis d
               is skipped and the verdict can only state per-layer findings.

Reuses the confound control of fr13_apc_restore_confound_diff.py and the layer
parsing of fr13_apc_cacherow_diff.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


# ---- payload helpers (shared shape with the existing diff scripts) ----

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


def _hit_rows(pay: dict[str, Any]) -> list[int]:
    his = pay.get("has_initial_state")
    if his is None:
        return []
    return [i for i in range(int(his.numel())) if bool(his[i])]


def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    a = a.float()
    b = b.float()
    while a.ndim > b.ndim and a.shape[0] == 1:
        a = a[0]
    while b.ndim > a.ndim and b.shape[0] == 1:
        b = b[0]
    if a.shape != b.shape:
        return None
    return a, b


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float | None:
    al = _align(a, b)
    if al is None:
        return None
    return float((al[0] - al[1]).abs().max())


# ---- load ON hit-turn captures (latest per layer that actually has a hit row) ----

def _load_on_hit(d: Path, max_call: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    seen_layers: set[int] = set()
    files = [p for p in glob.glob(str(d / "*.call*.pt")) if _call_idx(p) <= max_call]
    for p in sorted(files, key=_call_idx, reverse=True):
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        li = _payload_layer(pay)
        if li is None:
            continue
        seen_layers.add(li)
        if li in out:
            continue
        if _has_hit(pay):
            out[li] = pay  # latest hit-turn capture for this layer
    # record layers we saw but never with a hit row (for the no-hit report)
    out["__seen_layers__"] = seen_layers  # type: ignore[assignment]
    return out


# ---- load oracle trajectory: per layer, the list of every call's fields ----

def _load_oracle_traj(d: Path) -> dict[int, list[dict[str, Any]]]:
    traj: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(glob.glob(str(d / "*.call*.pt")), key=_call_idx):
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        li = _payload_layer(pay)
        if li is None:
            continue
        traj[li].append(pay)
    return traj


def _nat_per_chunk(traj: list[dict[str, Any]], field: str) -> float:
    """Median per-call delta of `field` across the no-cache trajectory."""
    vals = [pay.get(field) for pay in traj if pay.get(field) is not None]
    deltas = []
    for i in range(1, len(vals)):
        d = _max_abs(vals[i], vals[i - 1])
        if d is not None:
            deltas.append(d)
    deltas.sort()
    return deltas[len(deltas) // 2] if deltas else 0.0


def _min_dist_to_traj(on_t: torch.Tensor, traj: list[dict[str, Any]], field: str) -> float | None:
    """Closest the ON restored seed gets to ANY oracle trajectory checkpoint of
    `field` (controls the position-offset confound: a faithful restored seed
    coincides with SOME oracle chunk boundary)."""
    best = None
    for pay in traj:
        v = pay.get(field)
        if v is None:
            continue
        d = _max_abs(on_t, v)
        if d is None:
            continue
        if best is None or d < best:
            best = d
    return best


def _argmax_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    """Per-position argmax-over-last-dim mismatch fraction for an activation."""
    al = _align(a, b)
    if al is None:
        return None
    x, y = al
    if x.ndim < 1:
        return None
    ax = x.reshape(-1, x.shape[-1]).argmax(-1)
    ay = y.reshape(-1, y.shape[-1]).argmax(-1)
    n = int(ax.numel())
    mism = int((ax != ay).sum())
    return {"positions": n, "argmax_mismatch": mism, "frac": (mism / n) if n else 0.0,
            "max_abs": _max_abs(x, y)}


def _conv_window_match(win: torch.Tensor, pre_conv: torch.Tensor) -> float | None:
    """Best (min) max_abs of conv window `win` [C, state_len] against every
    length-state_len sliding window of an oracle pre_conv stream [T, C].
    Returns None if shapes are incompatible (different C)."""
    win = win.float()
    pc = pre_conv.float()
    # normalize win to [C, L]
    if win.ndim == 3 and win.shape[0] == 1:
        win = win[0]
    if win.ndim != 2:
        return None
    C, L = win.shape
    # pre_conv is [T, C] (channels last); transpose to [C, T] for windowing
    if pc.ndim != 2:
        return None
    if pc.shape[1] == C:
        stream = pc.transpose(0, 1)  # [C, T]
    elif pc.shape[0] == C:
        stream = pc  # already [C, T]
    else:
        return None
    T = stream.shape[1]
    if T < L:
        return None
    best = None
    # slide; T is at most a prefill length, fine for an offline localizer
    for p in range(L, T + 1):
        d = float((stream[:, p - L:p] - win).abs().max())
        if best is None or d < best:
            best = d
    return best


def _conv_min_dist(conv_on: torch.Tensor, hit_rows: list[int],
                   traj: list[dict[str, Any]]) -> float | None:
    best = None
    for r in hit_rows:
        if r >= conv_on.shape[0]:
            continue
        win = conv_on[r]
        for pay in traj:
            pc = pay.get("pre_conv")
            if pc is None:
                continue
            d = _conv_window_match(win, pc)
            if d is not None and (best is None or d < best):
                best = d
    return best


def _conv_nat(traj: list[dict[str, Any]]) -> float:
    """Median across calls of each oracle pre_conv's own internal window-to-window
    variation (max range of the trailing column minus its predecessor), giving a
    natural conv-window scale for the confound ratio."""
    scales = []
    for pay in traj:
        pc = pay.get("pre_conv")
        if pc is None:
            continue
        pcf = pc.float()
        if pcf.ndim != 2 or pcf.shape[0] < 2:
            continue
        # consecutive-row delta magnitude = how much the conv input moves per step
        d = (pcf[1:] - pcf[:-1]).abs().max()
        scales.append(float(d))
    scales.sort()
    return scales[len(scales) // 2] if scales else 0.0


# ---- final-token logit argmax (axis d) ----

def _load_logit(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    p = path
    if not p.exists():
        c0 = p.with_name(p.stem + ".call0" + p.suffix)
        if c0.exists():
            p = c0
        else:
            return None
    pay = torch.load(p, map_location="cpu", weights_only=False)
    lg = pay.get("logits")
    return lg.float() if lg is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-dir", required=True, type=Path,
                    help="cache-ON FR13_PREFILL_GDN_CAPTURE dir (hit turn)")
    ap.add_argument("--oracle-dir", required=True, type=Path,
                    help="cache-OFF/no-spec-oracle FR13_PREFILL_GDN_CAPTURE dir")
    ap.add_argument("--on-logits", type=Path, default=None,
                    help="ON FR13_FINAL_LOGIT_CAPTURE .pt (hit turn) -- axis d")
    ap.add_argument("--oracle-logits", type=Path, default=None,
                    help="oracle FR13_FINAL_LOGIT_CAPTURE .pt (hit turn) -- axis d")
    ap.add_argument("--max-call", type=int, default=10 ** 12,
                    help="ignore ON captures with call-index > this (depth-match)")
    ap.add_argument("--ratio-thresh", type=float, default=3.0,
                    help="confound ratio above which a seed site is OFF-TRAJECTORY")
    ap.add_argument("--argmax-frac-thresh", type=float, default=0.0,
                    help="core_out argmax mismatch fraction > this flags axis c "
                         "(default 0.0 = ANY flip)")
    ap.add_argument("--out", type=Path,
                    default=Path("output/fr13_apc_hit_first_divergence.jsonl"))
    args = ap.parse_args()

    on = _load_on_hit(args.on_dir, args.max_call)
    seen_layers: set[int] = on.pop("__seen_layers__", set())  # type: ignore[assignment]
    traj = _load_oracle_traj(args.oracle_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(set(int(k) for k in on) & set(traj))
    no_hit = sorted(seen_layers - set(int(k) for k in on))

    # site order within a layer = execution order: conv seed -> ssm seed -> output
    SITES = ["conv", "ssm", "out"]
    first: dict[str, tuple[int, dict[str, Any]] | None] = {s: None for s in SITES}
    records: list[dict[str, Any]] = []

    with args.out.open("w") as fh:
        for li in layers:
            pay = on[li]
            t = traj[li]
            rec: dict[str, Any] = {"layer": li, "hit_rows": _hit_rows(pay)}

            # tag: cached-full-block vs partial-tail. A row is a full cached block
            # iff its segment length (seg_starts diff) is a multiple of the mamba
            # block size (1024) AND has_initial_state True; we expose seg lengths
            # + block_ids so the operator can confirm. Here we just surface them.
            ss = pay.get("seg_starts")
            bids = pay.get("block_ids")
            if ss is not None:
                seg = ss.tolist()
                rec["seg_lens"] = [int(seg[i + 1] - seg[i]) for i in range(len(seg) - 1)]
            if bids is not None:
                rec["block_ids"] = [int(x) for x in bids.tolist()]

            # (a) conv seed.  conv_restore is [Nrows, C, state_len] (state_len =
            # kernel_width - 1): the LAST state_len conv INPUTS at the restored
            # block boundary.  Its faithful no-cache counterpart is the oracle's
            # own conv window at the SAME absolute boundary, reconstructed from the
            # oracle pre_conv stream (the conv INPUT, [T, C]): the trailing
            # state_len input columns ending at the boundary, transposed to
            # [C, state_len].  We min-dist the ON restored window against EVERY
            # length-state_len sliding window of every oracle call's pre_conv (so
            # we do not need to know the exact boundary index), which is the conv
            # analogue of the SSM min-dist-to-trajectory control.  Confound = the
            # oracle's own median per-window variation (nat).
            conv_on = pay.get("conv_restore")
            if conv_on is not None:
                hr = _hit_rows(pay)
                cmin = _conv_min_dist(conv_on, hr, t)
                cnat = _conv_nat(t)
                cratio = (cmin / cnat) if (cmin is not None and cnat > 1e-6) else None
                rec["conv"] = {"min_dist": cmin, "nat_per_window": cnat, "ratio": cratio,
                               "approx": "min-dist over oracle pre_conv sliding "
                                         "windows; for an EXACT per-boundary check "
                                         "use the paired-boundary capture (runbook)"}
                if cratio is not None and cratio > args.ratio_thresh and first["conv"] is None:
                    first["conv"] = (li, rec["conv"])
            else:
                rec["conv"] = {"missing": True,
                               "note": "set FR13_APC_CONV_RESTORE_CAPTURE=1 on the ON boot"}

            # (b) ssm seed
            ssm_on = pay.get("initial_state")
            if ssm_on is not None:
                smin = _min_dist_to_traj(ssm_on, t, "final_state")
                snat = _nat_per_chunk(t, "final_state")
                sratio = (smin / snat) if (smin is not None and snat > 1e-6) else None
                rec["ssm"] = {"min_dist": smin, "nat_per_chunk": snat, "ratio": sratio}
                if sratio is not None and sratio > args.ratio_thresh and first["ssm"] is None:
                    first["ssm"] = (li, rec["ssm"])
            else:
                rec["ssm"] = {"missing": True}

            # (c) per-layer output (core_out) -- argmax + max_abs vs the oracle
            # call whose core_out best matches (closest = same boundary). We pick
            # the oracle call with min max_abs of core_out and report argmax there.
            out_on = pay.get("core_out")
            best = None
            if out_on is not None:
                for opay in t:
                    ov = opay.get("core_out")
                    if ov is None:
                        continue
                    am = _argmax_mismatch(out_on, ov)
                    if am is None:
                        continue
                    if best is None or am["max_abs"] < best["max_abs"]:
                        best = am
            if best is not None:
                rec["out"] = best
                if best["frac"] > args.argmax_frac_thresh and first["out"] is None:
                    first["out"] = (li, best)
            else:
                rec["out"] = {"missing_or_shape_mismatch": True}

            fh.write(json.dumps(rec, default=str) + "\n")
            records.append(rec)

    # ---- axis (d): final-token argmax ----
    on_lg = _load_logit(args.on_logits)
    or_lg = _load_logit(args.oracle_logits)
    final_argmax: dict[str, Any] = {"available": on_lg is not None and or_lg is not None}
    if final_argmax["available"]:
        al = _align(on_lg, or_lg)
        if al is None:
            final_argmax["shape_mismatch"] = [list(on_lg.shape), list(or_lg.shape)]
        else:
            x, y = al
            ax = x.reshape(-1, x.shape[-1]).argmax(-1)
            ay = y.reshape(-1, y.shape[-1]).argmax(-1)
            n = int(ax.numel())
            flips = int((ax != ay).sum())
            final_argmax.update({
                "rows": n,
                "argmax_flips": flips,
                "last_row_on": int(ax[-1]),
                "last_row_oracle": int(ay[-1]),
                "last_row_flipped": bool(ax[-1] != ay[-1]),
            })

    # ---- verdict ----
    print("=== FR13 APC hit-turn first-divergence localization ===")
    print(f"layers compared: {len(layers)}  (on_hit_layers={len([k for k in on])} "
          f"oracle_layers={len(traj)})")
    if no_hit:
        print(f"  layers seen in ON but with NO cache-hit prefill row "
              f"(skipped): {no_hit[:12]}{'...' if len(no_hit) > 12 else ''}")
    for site in SITES:
        f = first[site]
        print(f"  first-divergent[{site:4s}] = "
              f"{f[0] if f else None}"
              + (f"  (ratio/frac payload: {f[1]})" if f else ""))

    # earliest (layer, site) in (layer asc, conv<ssm<out) order
    site_rank = {"conv": 0, "ssm": 1, "out": 2}
    candidates = [(f[0], site_rank[s], s, f[1]) for s, f in first.items() if f is not None]
    candidates.sort()
    if candidates:
        ly, _, site, payload = candidates[0]
        sitename = {"conv": "RESTORED conv-state seed",
                    "ssm": "RESTORED ssm-state seed",
                    "out": "per-GDN-layer output"}[site]
        verdict = (f"VERDICT: carrier first appears at <{site}> "
                   f"({sitename}) at GDN layer {ly}")
    else:
        # no seed/output divergence -> check final argmax
        if final_argmax.get("available"):
            if final_argmax.get("last_row_flipped") or final_argmax.get("argmax_flips", 0) > 0:
                verdict = ("VERDICT: carrier first appears at <logits> (final-token "
                           f"argmax flip; {final_argmax.get('argmax_flips')} row(s) "
                           "flipped) -- NO measurable per-layer seed/output carrier, "
                           "the divergence is in a non-captured downstream stage "
                           "(full-attn / MLP / lm-head accumulation).")
            else:
                verdict = ("VERDICT: no measurable carrier on cached blocks "
                           "(cache-ON argmax == cache-OFF on the hit turn): all seed "
                           "ratios within natural per-chunk variation, all per-layer "
                           "outputs argmax-identical, AND final-token argmax matches "
                           "the oracle.")
        else:
            verdict = ("VERDICT: no per-layer seed/output carrier above threshold. "
                       "Final-token argmax NOT available (pass --on-logits/--oracle-logits) "
                       "-> cannot confirm end-to-end losslessness; per-layer is clean.")

    print(f"\n  final-token argmax: {json.dumps(final_argmax, default=str)}")
    print(f"\n  {verdict}")
    print(f"  per-layer jsonl -> {args.out}")


if __name__ == "__main__":
    main()
