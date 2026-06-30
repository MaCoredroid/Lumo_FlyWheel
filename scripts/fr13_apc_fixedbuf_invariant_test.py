"""FR13_APC_FIXED_BUFFER lossless-invariant unit test (GPU-free).

Proves the buffer path (copy_ into pre-alloc [cap,...] + slice/clone drain + roll
remainder) produces BIT-IDENTICAL 64-chunks to the shipped list path (per-token
.clone() append + torch.stack drain + del remainder), across random accept
patterns. If the chunk INPUTS are bit-identical, the chunked kernel (same
_es_chunk call in both paths) yields identical published state -> lossless.

Mirrors the committer logic in fr10_phase4_patch_vllm_tree_gdn.py exactly:
  append: list -> _pend["k"].append(_k_tok[t].clone()) ; buf -> _fb["k"][n:n+t].copy_(_k_tok)
  drain : list -> torch.stack(_pend["k"][:64],0)        ; buf -> _fb["k"][:64].clone()
  drop  : list -> del _pend["k"][:64]                   ; buf -> roll _fb["k"][64:n]->[:rem]
The drain fires while (abs >= ck_pos+64) and (pending>=64); ck_pos += 64 each drain.
"""
import torch

_CHUNK = 64
_FB_CAP = 256
KH, DK, VH, DV = 4, 128, 8, 64  # representative GDN head shapes


def make_tok(t, dtype, g):
    return (
        torch.randn(t, KH, DK, generator=g, dtype=torch.float32).to(dtype),
        torch.randn(t, VH, DV, generator=g, dtype=torch.float32).to(dtype),
        torch.randn(t, VH, generator=g, dtype=torch.float32).to(dtype),
        torch.randn(t, VH, generator=g, dtype=torch.float32).to(dtype),
    )


def run_seq(seed, dtype, n_steps, n_pad):
    g = torch.Generator().manual_seed(seed)
    apat = torch.Generator().manual_seed(seed ^ 0x5151)
    # list-path state
    lp = {"k": [], "v": [], "a": [], "b": []}
    # buffer-path state
    bp = {"buf": None, "n": 0}
    abs_pos = 0   # ONE absolute-position counter (abs_map[_rid] in the real code)
    ckpos = 0     # both paths drain in lock-step -> one ck_pos suffices
    chunks_compared = 0
    for _ in range(n_steps):
        t = int(torch.randint(1, n_pad + 1, (1,), generator=apat).item())
        kt, vt, at, bt = make_tok(t, dtype, g)
        # ---- LIST append ----
        for i in range(t):
            lp["k"].append(kt[i].clone()); lp["v"].append(vt[i].clone())
            lp["a"].append(at[i].clone()); lp["b"].append(bt[i].clone())
        # ---- BUFFER append ----
        if bp["buf"] is None:
            bp["buf"] = {
                "k": torch.empty((_FB_CAP, KH, DK), dtype=dtype),
                "v": torch.empty((_FB_CAP, VH, DV), dtype=dtype),
                "a": torch.empty((_FB_CAP, VH), dtype=dtype),
                "b": torch.empty((_FB_CAP, VH), dtype=dtype),
            }
            bp["n"] = 0
        fb = bp["buf"]; n = bp["n"]
        assert n + t <= _FB_CAP, f"overflow n={n} t={t}"
        fb["k"][n:n + t].copy_(kt); fb["v"][n:n + t].copy_(vt)
        fb["a"][n:n + t].copy_(at); fb["b"][n:n + t].copy_(bt)
        bp["n"] = n + t
        abs_pos += t  # advance once, shared by both paths (as in the committer)
        # ---- DRAIN (both paths in lock-step; identical token counts) ----
        while abs_pos >= ckpos + _CHUNK and len(lp["k"]) >= _CHUNK:
            assert bp["n"] >= _CHUNK, "buffer pending desync vs list"
            # list drain
            lk = torch.stack(lp["k"][:_CHUNK], 0); lv = torch.stack(lp["v"][:_CHUNK], 0)
            la = torch.stack(lp["a"][:_CHUNK], 0).to(torch.float32)
            lb = torch.stack(lp["b"][:_CHUNK], 0).to(torch.float32)
            del lp["k"][:_CHUNK]; del lp["v"][:_CHUNK]
            del lp["a"][:_CHUNK]; del lp["b"][:_CHUNK]
            # buffer drain
            bk = fb["k"][:_CHUNK].clone(); bv = fb["v"][:_CHUNK].clone()
            ba = fb["a"][:_CHUNK].clone().to(torch.float32)
            bb = fb["b"][:_CHUNK].clone().to(torch.float32)
            nn = bp["n"]; rem = nn - _CHUNK
            if rem > 0:
                fb["k"][:rem].copy_(fb["k"][_CHUNK:nn].clone())
                fb["v"][:rem].copy_(fb["v"][_CHUNK:nn].clone())
                fb["a"][:rem].copy_(fb["a"][_CHUNK:nn].clone())
                fb["b"][:rem].copy_(fb["b"][_CHUNK:nn].clone())
            bp["n"] = max(0, rem)
            ckpos += _CHUNK
            # ---- BIT-EXACT compare (the lossless invariant) ----
            for nm, x, y in (("k", lk, bk), ("v", lv, bv), ("a", la, ba), ("b", lb, bb)):
                if not torch.equal(x, y):
                    md = (x.to(torch.float32) - y.to(torch.float32)).abs().max().item()
                    raise AssertionError(
                        f"CHUNK MISMATCH seed={seed} dtype={dtype} field={nm} "
                        f"maxdiff={md} at chunk#{chunks_compared}")
            chunks_compared += 1
    return chunks_compared


def main():
    total = 0
    cases = 0
    for dtype in (torch.bfloat16, torch.float32, torch.float16):
        for n_pad in (1, 4, 8, 16):
            for seed in range(40):
                c = run_seq(seed, dtype, n_steps=300, n_pad=n_pad)
                total += c; cases += 1
    print(f"OK: {cases} sequences, {total} 64-chunks compared BIT-EXACT "
          f"(list path == buffer path) across bf16/fp32/fp16 x n_pad{{1,4,8,16}}")
    print("LOSSLESS INVARIANT HOLDS: buffer drain == stack drain, "
          "remainder roll == del[:64].")


if __name__ == "__main__":
    main()
