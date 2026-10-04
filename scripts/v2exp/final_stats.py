import json,glob,os,statistics as st,sys
os.chdir(sys.argv[1])
def summ(d):
    p=d+"/replay.jsonl"
    if not os.path.exists(p): return None
    recs=[json.loads(l) for l in open(p)]
    ok=[r for r in recs if not r.get("error") and r.get("completion_tokens") and r.get("t_decode_s")]
    if len(ok)<43: return None
    N=sum(r["completion_tokens"] for r in ok); D=sum(r["t_decode_s"] for r in ok); R=len(ok)
    nd=na=0
    for r in ok:
        for k,v in r["spec_delta"].items():
            if k.startswith("vllm:spec_decode_num_drafts"): nd+=v
            if k.startswith("vllm:spec_decode_num_accepted_tokens") and "per_pos" not in k: na+=v
    return round((N-R)/D,2), (round(1+na/nd,2) if nd else None)
G=lambda p: sorted(glob.glob(p))
arms={
 "lumotree":["tree-sampled-20261001T055908Z","tree-sampled-20261001T064332Z","tree-sampled-20261001T072623Z"],
 "lumotree_fused_select_off":G("tree-topk0-sampled-*"),"lumotree_stock_attention":G("tree-fa2stock-sampled-*"),
 "lumotree_host_prep_off":G("tree-h27nobake-sampled-*"),
 "sglang_eagle_s3_d4":G("sglang-s3k1d4-sampled-*"),"sglang_eagle_s5_d6":G("sglang-s5k1d6-sampled-*"),
 "sglang_eagle_s7_d8":G("sglang-s7k1d8-sampled-*"),"sglang_eagle_s9_d10":G("sglang-s9k1d10-sampled-*"),
 "vllm_mtp7":G("mtp7-sampled-*"),"vllm_mtp5":["mtp5-sampled-20261001T062146Z","mtp5-sampled-20261001T070447Z","mtp5-sampled-20261001T074648Z"],
 "vllm_mtp3":G("mtp3-sampled-*"),"vllm_mtp1":G("mtp1-sampled-*"),"vllm_ar":G("ar-sampled-*")}
out={}
for a,ds in arms.items():
    rs=[s for s in (summ(d) for d in ds) if s]
    rates=[r[0] for r in rs]; acc=[r[1] for r in rs]
    m=round(st.mean(rates),2); sd=round(st.stdev(rates),2) if len(rates)>1 else None
    out[a]={"runs":rates,"mean":m,"sd":sd,"accept_len":acc}
    print(f"{a:26s} n={len(rates)} mean={m:6.2f} sd={sd} runs={rates} acc={acc}")
json.dump(out,open("/tmp/v2exp_final.json","w"),indent=1)
