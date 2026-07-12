"""Cumulative-onset diff: GDN-layer-0 divergence at EARLY (pos150) vs GARBLE (pos192).

Captures: CLEAN_EARLY node0 (pos150) in call0-4, CLEAN_GARBLE node0 (pos191/192) in call5-9, LIVE node0 (all pos)
in call10+. For each target pos, find the CLEAN capture (call0-9, closest pos) and the LIVE capture (call10+,
closest pos), diff the GDN layers (0,1,2 -- pre-attention, no RoPE = clean). If layer-0 rel_l2 is HIGH at EARLY
too => CUMULATIVE; if LOW at EARLY and HIGH at GARBLE => STEP-SPECIFIC (the corrupting event is between).
"""
import glob, torch


def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def load(f):
    return torch.load(f, map_location="cpu", weights_only=False)


def main():
    calls = sorted(glob.glob("/logs/n0.call*.pt"), key=lambda p: int(p.split("call")[1].split(".")[0]))
    metas = []
    for f in calls:
        d = load(f)
        ci = int(f.split("call")[1].split(".")[0])
        metas.append((ci, d["positions"][0][0].item(), f, d))
    clean_pool = [m for m in metas if m[0] < 10]      # CLEAN_EARLY + CLEAN_GARBLE forwards
    live_pool = [m for m in metas if m[0] >= 10]       # LIVE forwards
    print(f"clean_pool={len(clean_pool)} live_pool={len(live_pool)}")

    def nearest(pool, pos):
        return min(pool, key=lambda m: abs(m[1] - pos)) if pool else None

    for label, pos in [("EARLY", 150), ("GARBLE", 191)]:
        c = nearest(clean_pool, pos)
        L = nearest(live_pool, pos)
        if not c or not L:
            print(f"{label}: missing capture"); continue
        C, G = c[3], L[3]
        print(f"\n=== {label} pos~{pos}: CLEAN call{c[0]}(pos{c[1]}) vs LIVE call{L[0]}(pos{L[1]}) ===")
        print(f"  input_hidden rel_l2={rel_l2(G['input_hidden'], C['input_hidden']):.4f}")
        for i in range(3):   # GDN layers 0,1,2 (pre-attention, no RoPE)
            lc, lg = C["layers"][i], G["layers"][i]
            print(f"  layer{i} ({lg.get('layer_type','?')[:14]}) rel_l2={rel_l2(lg['hidden'], lc['hidden']):.4f}")
    print("\n=> EARLY layer0 HIGH => cumulative; EARLY low + GARBLE high => step-specific corrupting event")
    return 0


if __name__ == "__main__":
    main()
