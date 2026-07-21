"""Test WHICH fixed-shape padding scheme (if any) gives byte-identical committed state as the
variable-length committer -> which enables CUDA-graph capture."""
import torch
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update as sg
dev="cuda"; B=4; NKH,NVH,DK,DV=16,32,128,128; ROWS=64; dt=torch.bfloat16
torch.manual_seed(0)
real=[3,6,2,5]; MAX_T=8
A_log=torch.randn(NVH,device=dev); dt_bias=torch.randn(NVH,device=dev)
bank0=torch.randn(ROWS,NVH,DK,DV,device=dev); col0=torch.randint(0,ROWS,(B,),device=dev,dtype=torch.int32)
kb=torch.randn(B,MAX_T,NKH,DK,device=dev,dtype=dt); vb=torch.randn(B,MAX_T,NVH,DV,device=dev,dtype=dt)
ab=torch.randn(B,MAX_T,NVH,device=dev,dtype=dt); bb=torch.rand(B,MAX_T,NVH,device=dev,dtype=dt)
def ssi():
    s=torch.zeros(B,MAX_T,device=dev,dtype=torch.int32)
    for b in range(B): s[b,:]=col0[b]
    return s
def base():
    bk=bank0.clone(); T=sum(real)
    k=torch.cat([kb[b,:real[b]] for b in range(B)],0).reshape(1,T,NKH,DK).contiguous()
    v=torch.cat([vb[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH,DV).contiguous()
    a=torch.cat([ab[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH).contiguous()
    bt=torch.cat([bb[b,:real[b]] for b in range(B)],0).reshape(1,T,NVH).contiguous()
    cu=torch.tensor([0]+list(torch.tensor(real).cumsum(0).tolist()),device=dev,dtype=torch.int32)
    sg(A_log=A_log,a=a,b=bt,dt_bias=dt_bias,q=torch.zeros(1,T,NKH,DK,device=dev,dtype=dt),k=k,v=v,scale=1.0,
       initial_state=bk,inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=ssi(),use_qk_l2norm_in_kernel=True)
    return bk
def fixed(pad_kind, use_nacc):
    bk=bank0.clone(); T=B*MAX_T
    K=kb.clone(); V=vb.clone(); Ai=ab.clone(); Bi=bb.clone()
    for b in range(B):
        if pad_kind=="zero":
            K[b,real[b]:]=0; V[b,real[b]:]=0; Ai[b,real[b]:]=0; Bi[b,real[b]:]=0
        elif pad_kind=="neutral":
            K[b,real[b]:]=0; V[b,real[b]:]=0; Ai[b,real[b]:]=-1e4; Bi[b,real[b]:]=0
        elif pad_kind=="neutral2":
            K[b,real[b]:]=0; V[b,real[b]:]=0; Ai[b,real[b]:]=-1e4; Bi[b,real[b]:]=-1e4
    k=K.reshape(1,T,NKH,DK).contiguous(); v=V.reshape(1,T,NVH,DV).contiguous()
    a=Ai.reshape(1,T,NVH).contiguous(); bt=Bi.reshape(1,T,NVH).contiguous()
    cu=torch.tensor([i*MAX_T for i in range(B+1)],device=dev,dtype=torch.int32)
    kw={} if not use_nacc else {"num_accepted_tokens":torch.tensor(real,device=dev,dtype=torch.int32)}
    sg(A_log=A_log,a=a,b=bt,dt_bias=dt_bias,q=torch.zeros(1,T,NKH,DK,device=dev,dtype=dt),k=k,v=v,scale=1.0,
       initial_state=bk,inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=ssi(),use_qk_l2norm_in_kernel=True,**kw)
    return bk
A=base(); rows=col0.tolist()
for kind in ["zero","neutral","neutral2"]:
    for nacc in [False,True]:
        d=(A-fixed(kind,nacc)).abs()[rows].max().item()
        tag="IDENTICAL" if d==0 else ("~floor" if d<1e-3 else "DIVERGE")
        print(f"  pad={kind:8s} num_accepted={str(nacc):5s} -> max_diff={d:.3e}  {tag}")
