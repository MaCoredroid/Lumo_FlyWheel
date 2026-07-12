"""Validate the PER-NODE NATIVE VERIFY mechanism OFFLINE before wiring into launch_tree_gdn_replay.

VERIFY analog of the committer (fr13_native_committer_validate.py). The committer validated the final STATE
deposit at 1.19e-7 with q=zeros. The verify path instead needs each tree node's GDN OUTPUT:

    out_i = state_i . q_i        (state_i = GDN state AFTER absorbing node i along root->i path; q_i = REAL query)

We compute node 4's OUTPUT by running its root->node ancestor path [0,1,4] (acc_len>1) through native
fused_sigmoid_gating_delta_rule_update as a single flattened seq from a NON-ZERO committed h0, and take the
core_out row at the LAST token of the path (= node 4's output). fused returns (core_out, final_state);
core_out has shape (1, T, HV, V) after the internal squeeze(0). We compare that last-token row to a
pytorch-fp32 manual GDN recurrence's output at the same node.

KERNEL OUTPUT CONVENTION (read from fused_sigmoid_gating_delta_rule_update_kernel):
  - per token: b_h is updated FIRST (decay, delta-rule rank-1), THEN b_o = sum(b_h * b_q, K).
    => output queries the POST-update state.
  - b_q is L2-normalized in-kernel (USE_QK_L2NORM_IN_KERNEL=True) THEN multiplied by `scale`.
  - GQA: value head i_hv reads k-head/q-head i_h = i_hv // (HV // H).
  - o layout (1, T, HV, V); o[0, t] = [HV, V] is per-value-head output for token t. V == DV == DIM.

RED-TEAM: the fp32 reference must itself be right. packed_decode with ssm_state_indices=0 hits the NULL_BLOCK
guard (state_idx<=0 => early return, no accumulation) -- do NOT use it. Use the pytorch manual recurrence
(l2norm div, softplus/exp gating, decay, rank-1), the SAME ground truth the committer validator used, extended
to also emit the per-token output.

Run INSIDE the vllm container (GPU).
"""
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update

NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM, VAL_DIM = NUM_KH * DIM, NUM_VH * DIM
SCALE = DIM ** -0.5
GQA = NUM_VH // NUM_KH   # 3


def native_verify_output(*, state_bank, h0_row, path_nodes, q_ring, k_ring, v_ring, a_ring, b_ring,
                         A_log, dt_bias, scale):
    """Run root->node path [0,1,4] through native fused single-seq from h0 (bank[h0_row]); return the
    core_out row at the LAST token = the terminal node's OUTPUT. Shape [NUM_VH, DIM]."""
    dev = state_bank.device
    nodes = torch.tensor(path_nodes, dtype=torch.long, device=dev)
    T = nodes.numel()
    q = q_ring[0, nodes].reshape(1, T, NUM_KH, DIM).contiguous()   # REAL query (not zeros)
    k = k_ring[0, nodes].reshape(1, T, NUM_KH, DIM).contiguous()
    v = v_ring[0, nodes].reshape(1, T, NUM_VH, DIM).contiguous()
    aa = a_ring[0, nodes].reshape(1, T, NUM_VH).contiguous()
    bb = b_ring[0, nodes].reshape(1, T, NUM_VH).contiguous()
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    # Non-spec continuous batching (num_accepted=None => IS_SPEC_DECODING False => init reads ssm col 0).
    # Point every ssm col at the h0 row so the initial-state read at col 0 = h0. (write-back is irrelevant
    # here -- we only consume core_out, not final_state.)
    max_T = T
    ssi = torch.full((1, max_T), h0_row, device=dev, dtype=torch.int32)
    core_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=aa, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=scale,
        initial_state=state_bank, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=ssi, use_qk_l2norm_in_kernel=True)
    # core_out: (1, T, HV, V) after internal squeeze(0). last token = terminal node output.
    co = core_out.reshape(T, NUM_VH, DIM)
    return co[-1], co   # [VH, DIM], and full per-token stack for diagnostics


def ref_recurrence_outputs(nodes, q_ring, k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, h0, scale):
    """pytorch-fp32 manual GDN recurrence over path `nodes` from h0 = the unambiguous GROUND TRUTH.
    Returns (per-token outputs [T,VH,DIM], final state). Output at each token queries the POST-update state
    with the L2-normed, scale-multiplied query -- matching the kernel exactly."""
    st = h0.clone()   # [VH, DV, DK]
    outs = []
    for n in nodes:
        n = int(n)
        q = q_ring[0, n].float()          # [KH, DK]
        k = k_ring[0, n].float()          # [KH, DK]
        v = v_ring[0, n].float()          # [VH, DV]
        a = a_ring[0, n].float(); bb = b_ring[0, n].float()   # [VH]
        gg = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())   # [VH]
        beta = torch.sigmoid(bb)          # [VH]
        # L2norm per head, then GQA-expand k and q to value heads
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * scale
        qge = q.repeat_interleave(GQA, dim=0)   # [VH, DK]
        kge = k.repeat_interleave(GQA, dim=0)   # [VH, DK]
        # decay + delta-rule rank-1 update (state absorbs node n)
        st = st * torch.exp(gg)[:, None, None]
        corr = (st * kge[:, None, :]).sum(-1)                 # [VH, DV]
        vv = (v - corr) * beta[:, None]
        st = st + vv[:, :, None] * kge[:, None, :]
        # output queries POST-update state
        out = (st * qge[:, None, :]).sum(-1)                  # [VH, DV]
        outs.append(out)
    return torch.stack(outs, 0), st   # [T,VH,DIM], [VH,DIM,DIM]


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1313)
    mk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    N_PAD = 16
    A_log = mk(NUM_VH); dt_bias = mk(NUM_VH)
    q_ring = mk(1, N_PAD, NUM_KH, DIM) * 0.3
    k_ring = mk(1, N_PAD, NUM_KH, DIM) * 0.3
    v_ring = mk(1, N_PAD, NUM_VH, DIM) * 0.3
    a_ring = mk(1, N_PAD, NUM_VH) * 0.5
    b_ring = mk(1, N_PAD, NUM_VH) * 0.5

    # bank: row 0 = NULL, row 5 = committed h0 (non-zero)
    H0_ROW = 5
    bank = torch.zeros(8, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    h0 = mk(NUM_VH, DIM, DIM).float() * 0.2
    bank[H0_ROW] = h0.clone()

    # tree path root->node 4 : [0, 1, 4]  (acc_len>1, terminal node = 4)
    path = [0, 1, 4]

    ref_outs, _ = ref_recurrence_outputs(path, q_ring, k_ring, v_ring, a_ring, b_ring,
                                         A_log, dt_bias, h0, SCALE)
    node4_ref = ref_outs[-1]   # [VH, DIM]

    node4_native, native_all = native_verify_output(
        state_bank=bank, h0_row=H0_ROW, path_nodes=path, q_ring=q_ring, k_ring=k_ring,
        v_ring=v_ring, a_ring=a_ring, b_ring=b_ring, A_log=A_log, dt_bias=dt_bias, scale=SCALE)

    e = (node4_native.float() - node4_ref)
    # The kernel STORES core_out in bf16 (o = q.new_empty(...) => p_o bf16), unlike the committer which
    # compared the fp32 state (1.19e-7). So the honest floor for an OUTPUT compare is bf16 quantization, not
    # 1e-4. Prove the residual IS that floor: round the fp32 truth to bf16 and re-diff.
    e_bf16 = (node4_native.float() - node4_ref.to(torch.bfloat16).float())
    bf16_ulp = torch.finfo(torch.bfloat16).eps * node4_ref.abs().max().item()  # ~1 ULP at the top of range
    print(f"=== native VERIFY output for node 4 (path [0,1,4]) vs pytorch-fp32 GDN recurrence ===", flush=True)
    print(f"  node4 OUTPUT  max|d|(vs fp32 truth)={e.abs().max().item():.3e}  mean|d|={e.abs().mean().item():.3e}", flush=True)
    print(f"  node4 OUTPUT  max|d|(vs bf16-rounded truth)={e_bf16.abs().max().item():.3e}  (1 bf16 ULP@peak={bf16_ulp:.3e})", flush=True)
    print(f"  (native out max|.|={node4_native.abs().max().item():.3e}  ref out max|.|={node4_ref.abs().max().item():.3e})", flush=True)

    # per-token diagnostic (each node's own output along the path) -- flat across tokens => bf16 store floor,
    # NOT accumulation drift.
    for t, n in enumerate(path):
        et = (native_all[t].float() - ref_outs[t]).abs().max().item()
        print(f"    node {n} (token {t}) output max|d| vs fp32 truth={et:.3e}", flush=True)

    # PASS criterion: native bf16 output matches the bf16-rounding of the fp32 truth to within 1 ULP.
    ok = e_bf16.abs().max().item() <= 1.001 * bf16_ulp
    print(f"\n=== VERDICT: {'PASS - per-node native verify OUTPUT mechanism correct (residual = bf16 core_out store floor); wire it in' if ok else 'FAIL - convention wrong, fix before wiring'} ===",
          flush=True)


if __name__ == "__main__":
    main()
