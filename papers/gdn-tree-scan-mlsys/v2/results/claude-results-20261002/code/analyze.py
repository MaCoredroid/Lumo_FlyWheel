import json,glob,os,sys,collections
root=sys.argv[1]
runs=sorted(glob.glob(os.path.join(root,'replay','*','replay.jsonl')))
rows={}
for f in runs:
    name=os.path.basename(os.path.dirname(f))
    recs=[json.loads(l) for l in open(f)]
    ok=[r for r in recs if not r.get('error') and r.get('completion_tokens') and r.get('t_decode_s')]
    if not ok: print(name,'no valid records',len(recs)); continue
    N=sum(r['completion_tokens'] for r in ok); R=len(ok)
    D=sum(r['t_decode_s'] for r in ok)
    pooled=(N-R)/D
    ttft=sorted(r['t_ttft_s'] for r in ok)
    sd=collections.Counter()
    for r in ok:
        for k,v in r['spec_delta'].items(): sd[k]+=v
    acc=None
    nd=sum(v for k,v in sd.items() if k.startswith('vllm:spec_decode_num_drafts'))
    na=sum(v for k,v in sd.items() if k.startswith('vllm:spec_decode_num_accepted_tokens') and 'per_pos' not in k)
    if nd: acc=1+na/nd
    capped=sum(1 for r in ok if r['completion_tokens']>=r['max_tokens'])
    print(f"{name:34s} req={R:2d}/{len(recs)} tokens={N:6d} pooled={pooled:6.2f} tok/s  mean_accept_len={acc and round(acc,2)}  median_ttft={ttft[len(ttft)//2]:.1f}s capped={capped}")
    rows[name]={r['request']:r for r in ok}
# greedy output match across arms
g=[k for k in rows if 'greedy' in k]
if len(g)>=2:
    a,b=g[0],g[1]
    same=diff=0; firstdiv=[]
    for req in rows[a]:
        if req in rows[b]:
            x=rows[a][req]; y=rows[b][req]
            sx=x['reasoning']+'\x00'+x['content']+'\x00'+x['tool_calls']; sy=y['reasoning']+'\x00'+y['content']+'\x00'+y['tool_calls']
            if sx==sy: same+=1
            else:
                diff+=1
                i=next((i for i,(p,q) in enumerate(zip(sx,sy)) if p!=q),min(len(sx),len(sy)))
                firstdiv.append(i)
    print(f"greedy exact output match {a} vs {b}: {same}/{same+diff}; first-divergence char offsets (sorted): {sorted(firstdiv)[:12]}")
