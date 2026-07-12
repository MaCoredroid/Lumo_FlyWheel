"""FR13 garble per-layer diff: CLEAN node0 (call0) vs garble node0 (LIVE '_rows' call).

Runs INSIDE the container. Finds the CLEAN call (argmax '_row') and the garble call (argmax '_rows') by
projecting final_norm_hidden through lm_head, then for each of the 64 layers computes the divergence
(relative L2 + cosine) between the two node0 per-layer hiddens. The FIRST layer where divergence jumps
localizes the corrupt op: a linear_attention (GDN/mamba) layer => mamba carried state; a full_attention layer
=> attention KV; a jump at the norm/mlp => elsewhere. Also reports input_hidden (embedding) divergence: if the
INPUTS already diverge, the corruption is upstream (wrong token/embedding), not a layer op.
"""
import sys, glob, torch
from safetensors import safe_open

MDIR = "/models/qwen3.6-27b-bf16diag"


def load_lm_head():
    for f in glob.glob(MDIR + "/*.safetensors"):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if k.endswith("lm_head.weight"):
                    return sf.get_tensor(k).float()
    raise RuntimeError("no lm_head")


def argmax_of(d, lm):
    h = d["final_norm_hidden"].float()
    return int((h @ lm.t()).argmax(-1)[0])


def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    lm = load_lm_head()
    ROWS = {1748, 10630}    # '_rows'
    ROW = 8268              # '_row'
    calls = sorted(glob.glob("/logs/n0.call*.pt"), key=lambda p: int(p.split("call")[1].split(".")[0]))
    clean_f = garble_f = None
    for f in calls:
        d = torch.load(f, map_location="cpu", weights_only=False)
        am = argmax_of(d, lm)
        ci = int(f.split("call")[1].split(".")[0])
        pos = d["positions"][0][0].item()
        if am == ROW and clean_f is None:
            clean_f = f; print(f"CLEAN = call{ci} pos={pos} argmax='_row'", flush=True)
        if am in ROWS and garble_f is None:
            garble_f = f; print(f"GARBLE = call{ci} pos={pos} argmax='_rows'", flush=True)
    if not clean_f or not garble_f:
        print("MISSING clean or garble call; calls found:", len(calls)); return 1
    C = torch.load(clean_f, map_location="cpu", weights_only=False)
    G = torch.load(garble_f, map_location="cpu", weights_only=False)
    print(f"\ninput_hidden (embedding) rel_l2={rel_l2(G['input_hidden'], C['input_hidden']):.4f} "
          f"cos={cos(G['input_hidden'], C['input_hidden']):.5f}  "
          f"(if ~0 divergence => inputs match, corruption is a LAYER op; if large => upstream)", flush=True)
    print("\nlayer | type | rel_l2 | cos", flush=True)
    prev = 0.0
    first_div = None
    for i, (lc, lg) in enumerate(zip(C["layers"], G["layers"])):
        r = rel_l2(lg["hidden"], lc["hidden"]); c = cos(lg["hidden"], lc["hidden"])
        lt = lg.get("layer_type", "?")
        jump = r - prev
        flag = ""
        if first_div is None and r > 0.05:
            first_div = (i, lt); flag = "  <<< FIRST DIVERGENCE"
        print(f"{i:3d} | {lt[:16]:16s} | {r:.4f} | {c:.5f}{flag}", flush=True)
        prev = r
    if first_div:
        print(f"\n=== FIRST DIVERGENT LAYER: {first_div[0]} type={first_div[1]} ===", flush=True)
        print("linear_attention => MAMBA carried-state (col-0) is the corruption;", flush=True)
        print("full_attention   => ATTENTION KV/pos is the corruption.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
