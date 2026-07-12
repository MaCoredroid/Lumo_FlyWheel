"""Empirically sweep the fused_sigmoid_gating multi-seq convention to find the bit-exact + OOB-safe one."""
import torch, itertools
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode as pd,
    fused_sigmoid_gating_delta_rule_update as fsg,
)
NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM = NUM_KH*DIM; CONV_DIM = KEY_DIM*2 + NUM_VH*DIM; SCALE = DIM**-0.5
dev="cuda"
def split4(m,D):
    return (m[:,0:KEY_DIM].reshape(1,D,NUM_KH,DIM).contiguous(),
            m[:,KEY_DIM:2*KEY_DIM].reshape(1,D,NUM_KH,DIM).contiguous(),
            m[:,2*KEY_DIM:].reshape(1,D,NUM_VH,DIM).contiguous())
def packed_ref(m,a,b,Al,dtb,D,h0):
    ssm=h0.clone().unsqueeze(0); idx=torch.zeros(1,device=dev,dtype=torch.int32); outs=[]
    for i in range(D):
        ob=torch.zeros(1,1,NUM_VH,DIM,device=dev,dtype=torch.bfloat16)
        pd(mixed_qkv=m[i:i+1].contiguous(),a=a[i:i+1].contiguous(),b=b[i:i+1].contiguous(),
           A_log=Al,dt_bias=dtb,scale=SCALE,initial_state=ssm,out=ob,ssm_state_indices=idx,use_qk_l2norm_in_kernel=True)
        outs.append(ob[0,0].clone())
    return torch.stack(outs,0)
def run(m,a,b,Al,dtb,seg,states,inplace,slot_start,ndim):
    D=int(sum(seg)); N=len(seg); q,k,v=split4(m,D)
    aa=a.reshape(1,D,NUM_VH).contiguous(); bb=b.reshape(1,D,NUM_VH).contiguous()
    if slot_start==0:
        bank=torch.stack(list(states),0).contiguous()          # slots 0..N-1
        base=0
    else:
        bank=torch.stack([torch.zeros_like(states[0])]+list(states),0).contiguous()  # slot0=null, 1..N
        base=1
    cu=torch.tensor([0]+list(torch.tensor(seg).cumsum(0).tolist()),device=dev,dtype=torch.int32)
    maxT=max(seg)
    if ndim==1:
        idx=torch.tensor([base+i for i in range(N)],device=dev,dtype=torch.int32)
    else:
        idx=torch.zeros(N,maxT,device=dev,dtype=torch.int32)
        for i in range(N): idx[i,:]=base+i
    out=fsg(A_log=Al,a=aa,b=bb,dt_bias=dtb,q=q,k=k,v=v,scale=SCALE,initial_state=bank,
            inplace_final_state=inplace,cu_seqlens=cu,ssm_state_indices=idx,use_qk_l2norm_in_kernel=True)
    core=out[0] if isinstance(out,(tuple,list)) else out
    return core.reshape(D,NUM_VH,DIM)
def main():
    g=torch.Generator(device=dev).manual_seed(1313)
    mk=lambda *s: torch.randn(*s,generator=g,device=dev,dtype=torch.bfloat16)
    Al=mk(NUM_VH); dtb=mk(NUM_VH)
    D=6; m=mk(D,CONV_DIM)*0.3; a=mk(D,NUM_VH)*0.5; b=mk(D,NUM_VH)*0.5; h0=mk(NUM_VH,DIM,DIM).float()*0.2
    ref=packed_ref(m,a,b,Al,dtb,D,h0).float()
    print("SINGLE-SEQ sweep (D=6, must be 0.0):",flush=True)
    for inplace,slot,ndim in itertools.product([True,False],[0,1],[1,2]):
        try:
            out=run(m,a,b,Al,dtb,[D],[h0],inplace,slot,ndim)
            mx=(out.float()-ref).abs().max().item()
            print(f"  inplace={inplace} slot_start={slot} ndim={ndim}: max|d|={mx:.3e} {'<-- BIT-EXACT' if mx==0.0 else ''}",flush=True)
        except Exception as e:
            print(f"  inplace={inplace} slot_start={slot} ndim={ndim}: EXC {type(e).__name__}: {str(e)[:50]}",flush=True)
main()

def multiseq_winner():
    g=torch.Generator(device=dev).manual_seed(1313)
    mk=lambda *s: torch.randn(*s,generator=g,device=dev,dtype=torch.bfloat16)
    Al=mk(NUM_VH); dtb=mk(NUM_VH)
    seg=[2,3,5]; Dt=sum(seg)
    m=mk(Dt,CONV_DIM)*0.3; a=mk(Dt,NUM_VH)*0.5; b=mk(Dt,NUM_VH)*0.5
    states=[mk(NUM_VH,DIM,DIM).float()*0.2 for _ in seg]
    # WINNER: inplace=True, distinct slot per path (0..N-1), ndim=2
    out=run(m,a,b,Al,dtb,seg,states,inplace=True,slot_start=0,ndim=2)
    cu=[0,2,5,10]
    print("\nMULTI-SEQ winner (inplace=True, distinct rows, ndim=2):",flush=True)
    allok=True
    for s in range(3):
        lo,hi=cu[s],cu[s+1]
        ref=packed_ref(m[lo:hi],a[lo:hi],b[lo:hi],Al,dtb,hi-lo,states[s]).float()
        mx=(out[lo:hi].float()-ref).abs().max().item()
        allok&=(mx==0.0)
        print(f"  path{s} (len {hi-lo}): max|d|={mx:.3e} {'OK' if mx==0.0 else 'FAIL'}",flush=True)
    # SHARED-col0 case: all paths start from the SAME state, copied to N distinct rows
    h0=mk(NUM_VH,DIM,DIM).float()*0.2
    shared=[h0.clone() for _ in seg]
    out2=run(m,a,b,Al,dtb,seg,shared,inplace=True,slot_start=0,ndim=2)
    print("SHARED-col0 (all paths same h0, N distinct rows):",flush=True)
    for s in range(3):
        lo,hi=cu[s],cu[s+1]
        ref=packed_ref(m[lo:hi],a[lo:hi],b[lo:hi],Al,dtb,hi-lo,h0).float()
        mx=(out2[lo:hi].float()-ref).abs().max().item()
        allok&=(mx==0.0)
        print(f"  path{s}: max|d|={mx:.3e} {'OK' if mx==0.0 else 'FAIL'}",flush=True)
    print(f"\n=> MULTI-PATH CONVENTION {'SOLVED (build the fix)' if allok else 'STILL BROKEN'}",flush=True)

multiseq_winner()
