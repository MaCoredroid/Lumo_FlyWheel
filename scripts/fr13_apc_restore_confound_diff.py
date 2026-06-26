#!/usr/bin/env python3
# FR13 APC restore-frame confound-controlled per-GDN-layer diff.
#
# Measures whether the cache-RESTORED GDN recurrent state (initial_state on a cache-hit
# prefill) is genuinely off the no-cache ground-truth trajectory, CONTROLLED for the two
# confounds that make a raw max_abs meaningless: (1) the position offset (restored state is
# at the cache boundary, ground-truth final is the full-prefix end) and (2) the GDN state's
# large natural magnitude. Control = divergence vs EACH layer's OWN natural per-chunk
# variation (the no-cache final_state changes ~this much per 1024-token chunk; a restored
# state within that is on-trajectory, >>that is genuine corruption).
#
# Usage: --on-dir <dir> --on-prefix oncache_gdn --nc-dir <dir> --nc-prefix nocache_gdn
import argparse, glob, re, torch
from collections import defaultdict


def ci(p):
    m = re.search(r"\.call(\d+)\.pt$", p); return int(m.group(1)) if m else -1


def lyr(pay):
    m = re.search(r"layers\.(\d+)\.", str(pay.get("layer_name") or pay.get("layer_prefix") or ""))
    return int(m.group(1)) if m else None


def ma(a, b):
    a = a.float(); b = b.float()
    while a.ndim > b.ndim and a.shape[0] == 1: a = a[0]
    while b.ndim > a.ndim and b.shape[0] == 1: b = b[0]
    if a.shape != b.shape: return None
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-dir", required=True); ap.add_argument("--on-prefix", default="oncache_gdn")
    ap.add_argument("--nc-dir", required=True); ap.add_argument("--nc-prefix", default="nocache_gdn")
    ap.add_argument("--on-topk", type=int, default=200)
    # --on-field: which ON-arm tensor to diff. initial_state = the RESTORED seed
    # (cross-turn signal: did the fix make next-turn's restored seed faithful).
    # final_state = the post-recompute WRITE-BACK (disambiguator: did the recompute
    # itself produce a recurrent-exact boundary state). nc-field is the no-cache
    # trajectory field (final_state per chunk).
    ap.add_argument("--on-field", default="initial_state", choices=["initial_state", "final_state"])
    ap.add_argument("--nc-field", default="final_state", choices=["initial_state", "final_state"])
    # --max-call: ignore ON captures with call-index > this. The global call-index
    # is shared across runs for the first LIMIT_PER_PREFIX-per-layer (same forward
    # event order), so windowing an UNARMED run to the ARMED run's max call-index
    # gives a DEPTH-MATCHED control (same replay turns) with no new boot.
    ap.add_argument("--max-call", type=int, default=10**12)
    a = ap.parse_args()
    # latest-per-layer ON-arm state (restored seed or recompute write-back)
    on = {}
    _on_files = [p for p in glob.glob(f"{a.on_dir}/{a.on_prefix}.call*.pt") if ci(p) <= a.max_call]
    for p in sorted(_on_files, key=ci, reverse=True)[:a.on_topk]:
        try: pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception: continue
        li = lyr(pay)
        v = pay.get(a.on_field)
        if li is not None and li not in on and v is not None: on[li] = v.float()
    # full no-cache trajectory per layer (every chunk's final_state)
    traj = defaultdict(list)
    for p in sorted(glob.glob(f"{a.nc_dir}/{a.nc_prefix}.call*.pt"), key=ci):
        try: pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception: continue
        li = lyr(pay)
        v = pay.get(a.nc_field)
        if li is not None and v is not None: traj[li].append(v.float())
    print(f"# ON={a.on_prefix}.{a.on_field}  vs  NC={a.nc_prefix}.{a.nc_field}", flush=True)
    print(f"{'L':>3} {'mindist':>8} {'nat/chunk':>9} {'ratio':>6}  verdict", flush=True)
    real = []
    for li in sorted(set(on) & set(traj)):
        t = traj[li]
        mind = min((d for d in (ma(on[li], fs) for fs in t) if d is not None), default=None)
        nat = sorted(d for d in (ma(t[i], t[i - 1]) for i in range(1, len(t))) if d is not None)
        natmed = nat[len(nat) // 2] if nat else 0.0
        if mind is None: continue
        ratio = mind / natmed if natmed > 0.01 else 999
        v = "OFF-TRAJ(corruption)" if ratio > 3 else ("borderline" if ratio > 1.5 else "faithful")
        if ratio > 3: real.append((li, round(mind, 2), round(ratio, 1)))
        print(f"{li:>3} {round(mind,2):>8} {round(natmed,2):>9} {round(ratio,1):>6}  {v}", flush=True)
    print(f"\nOFF-TRAJECTORY layers (real corruption, not offset): {real}", flush=True)
    print(f"n off-trajectory = {len(real)} / {len(set(on)&set(traj))} GDN layers", flush=True)


if __name__ == "__main__":
    main()
