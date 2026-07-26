#!/usr/bin/env python3
"""FR13 verifier attack V2 — attn-row geometry microbench.

Names the achievable win for the +14ms/event attn-rows tax before any kernel
surgery: benchmarks the FA2 fork's flash attention with varlen decode-style
calls at tree-verify geometry (M=22 rows/req) vs native-MTP geometry (M=6),
across KV lengths and batch sizes, plus a num_splits sweep.

Run INSIDE the serving container (fork bindings):
  docker exec <c> python3 /workspace/scripts/fr13_attn_mgeom_bench.py

Interpretation contract:
  - tax_ratio = t(M=22)/t(M=6) per (B, KV). If tax_ratio ~= 22/6 = 3.67 the
    kernel is row-linear (bandwidth-fair; win only from fewer rows = CLOSED).
  - If tax_ratio >> 3.67 the M=22 geometry wastes tiles/occupancy and a
    tile/num_splits fix has headroom worth (tax_ratio-3.67)*t6 per call.
  - If num_splits sweep moves t22 materially, the fix may be pure launch-param
    (no kernel edit) -> FR13_ATTN_NUM_SPLITS env, cheapest possible lever.
"""
import os, time, json, itertools
import torch


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.monotonic() - t0) / iters * 1e3  # ms


def main():
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # fork binding
    dev = "cuda"
    torch.manual_seed(0)
    NH_Q, NH_KV, HD = 32, 4, 128   # qwen3.6-27b GQA geometry
    PAGE = 16
    results = []
    for B, M, KV in itertools.product((1, 4), (6, 22), (2048, 8192, 32768)):
        q = torch.randn(B * M, NH_Q, HD, device=dev, dtype=torch.bfloat16)
        n_pages = (KV + PAGE - 1) // PAGE
        k = torch.randn(n_pages * B, PAGE, NH_KV, HD, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        block_table = (
            torch.arange(B * n_pages, device=dev, dtype=torch.int32)
            .reshape(B, n_pages)
        )
        cu_q = torch.arange(0, (B + 1) * M, M, device=dev, dtype=torch.int32)
        seqused_k = torch.full((B,), KV, device=dev, dtype=torch.int32)
        base_kwargs = dict(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, max_seqlen_q=M,
            seqused_k=seqused_k, max_seqlen_k=KV,
            causal=True, block_table=block_table,
        )
        row = {"B": B, "M": M, "KV": KV}
        try:
            row["t_default_ms"] = bench(lambda: flash_attn_varlen_func(**base_kwargs))
        except (TypeError, NotImplementedError) as e:
            # signature drift: record and bail loudly rather than fake numbers
            print("SIGNATURE_MISMATCH:", e)
            print("available:", flash_attn_varlen_func.__doc__)
            return
        results.append(row)
        print(json.dumps(row), flush=True)
        # num_splits sweep: FA2's varlen API raises NotImplementedError for
        # num_splits>1 (splitkv is auto-selected inside dispatch) — probe once,
        # record the outcome, never let it kill the geometry rows.
        if B == 1 and M == 6 and KV == 2048:
            try:
                bench(lambda: flash_attn_varlen_func(**base_kwargs, num_splits=2), iters=3, warmup=1)
                row["num_splits_supported"] = True
            except Exception as e:
                print(f"num_splits lever: DEAD ({type(e).__name__})", flush=True)
    # tax ratios
    print("\n=== tax ratios (t22/t6, same B,KV; row-linear fair = 3.67) ===")
    for B in (1, 4):
        for KV in (2048, 8192, 32768):
            t6 = next((r["t_default_ms"] for r in results if r["B"] == B and r["M"] == 6 and r["KV"] == KV), None)
            t22 = next((r["t_default_ms"] for r in results if r["B"] == B and r["M"] == 22 and r["KV"] == KV), None)
            if t6 and t22:
                print(f"B={B} KV={KV}: t6={t6:.3f} t22={t22:.3f} ratio={t22/t6:.2f}")
    out = os.environ.get("BENCH_OUT", "/logs/fr13_attn_mgeom_bench.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print("wrote", out)


if __name__ == "__main__":
    main()
