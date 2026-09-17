"""Does main actually read the WRONG initial state for a zero-accept row?
Compare the resulting block-5 state under:
  Z  : num_accepted=0 (main: i_t=-1 -> reads planted block 7 before the tensor)
  Z1 : num_accepted=1 (correct resume: i_t=0 -> reads block 5)
If Z != Z1 on a variant, that variant resumed from the wrong state.
"""
import sys
sys.path.insert(0, "/home/mark/shared/tmp-scratch/B_probe")
import probe_zero_accept as P
import torch, importlib.util

def run(fn, nacc_list):
    dev, dtype = "cuda", torch.float32
    flat = torch.empty(1 + P.N * P.NCOL, dtype=torch.int32, device=dev)
    flat[0] = P.SENTINEL_BEFORE
    idx = flat[1:].view(P.N, P.NCOL)
    idx[0] = torch.tensor(P.ROW0, dtype=torch.int32, device=dev)
    idx[1] = torch.tensor(P.ROW1, dtype=torch.int32, device=dev)
    state = torch.zeros(P.NUM_BLOCKS, P.HV, P.V, P.K, device=dev, dtype=dtype)
    for bl in range(P.NUM_BLOCKS):
        state[bl] = float(bl) + 0.5
    ins = P.make_inputs(dev, dtype)
    fn(A_log=ins["A_log"], a=ins["a"], b=ins["b"], dt_bias=ins["dt_bias"],
       q=ins["q"], k=ins["k"], v=ins["v"], initial_state=state,
       inplace_final_state=True, cu_seqlens=ins["cu_seqlens"],
       ssm_state_indices=idx,
       num_accepted_tokens=torch.tensor(nacc_list, dtype=torch.int32, device=dev),
       use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    return state[5].clone()

print(f"{'variant':10s} {'block5 after Z(=0) vs Z1(=1)':34s} max|diff|")
for name in P.VARIANTS:
    fn = P.load(name)
    z  = run(fn, [0, 2])
    z1 = run(fn, [1, 2])
    same = torch.equal(z, z1)
    d = (z - z1).abs().max().item()
    verdict = "same (correct slot-0 resume)" if same else "DIFFERENT -> wrong initial state"
    print(f"{name:10s} {verdict:34s} {d:.6g}")
