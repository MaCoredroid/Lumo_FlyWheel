"""Isolate: does state-neutral padding stay byte-identical as MAX_T grows (more pad tokens)?
Also test determinism + col0-collision sensitivity."""
import torch
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update as sg
dev="cuda"; B=4; NKH,NVH,DK,DV=16,32,128,128; ROWS=64; dt=torch.bfloat16
torch.manual_seed(0)
real=[3,6,2,5]
A_log=torch.randn(NVH,device=dev); dt_bias=torch.randn(NVH,device=dev)
bank0=torch.randn(ROWS,NVH,DK,DV,device=dev)
col0=torch.randint(0,ROWS,(B,),device=dev,dtype=torch.int32)
print("col0=",col0.tolist(),"collision" if len(set(col0.tolist()))<B else "distinct")
MAXBUF=20
kb=torch.randn(B,MAXBUF,NKH,DK,device=dev,dtype=dt); vb=torch.randn(B,MAXBUF,NVH,DV,device=dev,dtype=dt)
ab=torch.randn(B,MAXBUF,NVH,device=dev,dtype=dt); bb=torch.rand(B,MAXBUF,NVH,device=dev,dtype=dt)
def ssi(cols):
    s=torch.zeros(B,cols,device=dev,dtype=torch.int32)
    for b in range(B): s[b,:]=col0[b]
    return s
def varlen():
    bk=bank0.clone(); T=sum(real)
    k=torch.cat([kb[b,:real[b]] for b in range(B)],0).reshape(1,T,NKH,DK).contiguous()
    v=torch.cat([vb[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH,DV).contiguous()
    a=torch.cat([ab[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH).contiguous()
    bt=torch.cat([bb[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH).contiguous()
    cu=torch.tensor([0]+list(torch.tensor(real).cumsum(0).tolist()),device=dev,dtype=torch.int32)
    sg(A_log=A_log,a=a,b=bt,dt_bias=dt_bias,q=torch.zeros(1,T,NKH,DK,device=dev,dtype=dt),k=k,v=v,scale=1.0,
       initial_state=bk,inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=ssi(max(real)),use_qk_l2norm_in_kernel=True)
    return bk
def fixed(MT):
    bk=bank0.clone(); T=B*MT
    K=torch.zeros(B,MT,NKH,DK,device=dev,dtype=dt); V=torch.zeros(B,MT,NVH,DV,device=dev,dtype=dt)
    A=torch.full((B,MT,NVH),-1e4,device=dev,dtype=dt); Bt=torch.zeros(B,MT,NVH,device=dev,dtype=dt)
    for b in range(B):
        K[b,:real[b]]=kb[b,:real[b]]; V[b,:real[b]]=vb[b,:real[b]]; A[b,:real[b]]=ab[b,:real[b]]; Bt[b,:real[b]]=bb[b,:real[b]]
    cu=torch.tensor([i*MT for i in range(B+1)],device=dev,dtype=torch.int32)
    sg(A_log=A_log,a=A.reshape(1,T,NVH),b=Bt.reshape(1,T,NVH),dt_bias=dt_bias,q=torch.zeros(1,T,NKH,DK,device=dev,dtype=dt),
       k=K.reshape(1,T,NKH,DK),v=V.reshape(1,T,NVH,DV),scale=1.0,initial_state=bk,inplace_final_state=True,
       cu_seqlens=cu,ssm_state_indices=ssi(MT),use_qk_l2norm_in_kernel=True)
    return bk
rows=col0.tolist()
A=varlen(); A2=varlen()
print(f"determinism varlen: {(A-A2).abs().max().item():.3e}")
for MT in [8,12,16]:
    d=(A-fixed(MT)).abs()[rows].max().item()
    print(f"  MAX_T={MT:2d} -> varlen vs fixed max_diff={d:.3e}  {'IDENTICAL' if d==0 else ('~floor' if d<1e-3 else 'DIVERGE')}")
