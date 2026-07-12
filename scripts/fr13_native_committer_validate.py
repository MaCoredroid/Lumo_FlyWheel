"""Validate the NATIVE committer logic OFFLINE before wiring into launch_tree_gdn_replay.

The committer rebuilds the committed-path GDN state and deposits it to col 0 (the running row) for the next
step. The garble is a GROSS wrong-state in this deposit at num_accepted>1 x branches. Fix: instead of the
custom _tree_gdn_replay_kernel, gather the committed path [node 0]+accepted_paths[:acc_len] from the rings
and run it through native fused_sigmoid_gating (single-seq per request, from col-0), depositing the final
state to col 0 via inplace_final_state.

This validates the GATHER + native-call + col-0-deposit convention against a packed_decode reference over the
SAME committed path. If col-0 == packed_decode final state (0.0), the native committer logic is correct.

Ring layout (from launch_tree_gdn_replay): k(B,N,KH,DK) v(B,N,VH,DV) a,b(B,N,VH); state_bank(rows,VH,DV,DK) fp32.
Run INSIDE the vllm container (GPU).
"""
import torch
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)

NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM, VAL_DIM = NUM_KH * DIM, NUM_VH * DIM
CONV_DIM = KEY_DIM * 2 + VAL_DIM
SCALE = DIM ** -0.5


def native_committer(*, state_bank, spec_state_indices, accepted_paths, accepted_lens,
                     k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, num_spec_decodes, output_scale):
    """Gather committed path [0]+accepted_paths[:acc_len] per request, run native single-seq from col-0,
    deposit final state to col-0 (spec_state_indices[:,0]) via inplace_final_state. Returns nothing (mutates
    state_bank in place, like the custom committer)."""
    dev = state_bank.device
    B = num_spec_decodes
    # committed path node ids per request: [0] ++ accepted_paths[b, :acc_len[b]]
    seg_k, seg_v, seg_a, seg_b, seg_len, col0_rows = [], [], [], [], [], []
    for b in range(B):
        L = int(accepted_lens[b].item())
        nodes = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                           accepted_paths[b, :L].to(torch.long)])   # [1+L]
        seg_k.append(k_ring[b, nodes])   # [1+L, KH, DK]
        seg_v.append(v_ring[b, nodes])   # [1+L, VH, DV]
        seg_a.append(a_ring[b, nodes])   # [1+L, VH]
        seg_b.append(b_ring[b, nodes])
        seg_len.append(nodes.numel())
        col0_rows.append(int(spec_state_indices[b, 0].item()))
    T = sum(seg_len)
    q = torch.zeros(1, T, NUM_KH, DIM, device=dev, dtype=torch.bfloat16)   # state-only: q doesn't affect state
    k = torch.cat(seg_k, 0).reshape(1, T, NUM_KH, DIM).contiguous()
    v = torch.cat(seg_v, 0).reshape(1, T, NUM_VH, DIM).contiguous()
    aa = torch.cat(seg_a, 0).reshape(1, T, NUM_VH).contiguous()
    bb = torch.cat(seg_b, 0).reshape(1, T, NUM_VH).contiguous()
    cu = torch.tensor([0] + list(torch.tensor(seg_len).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    # NON-spec continuous-batching (num_accepted=None => IS_SPEC_DECODING False => init read at col 0).
    # WRITE-BACK IS PER-TOKEN: token t writes state to ssm_state_indices[i_n, t]. So ssi must be [B, max_T]
    # with EVERY column = the col-0 running row: init reads col 0 = h0; the FINAL token (col T-1) writes the
    # final state to that same row = col-0 deposit; intermediate writes to the same row are overwritten
    # (b_h carries in registers, so this is harmless). => final committed-path state lands at col-0.
    max_T = max(seg_len)
    ssi = torch.zeros(B, max_T, device=dev, dtype=torch.int32)
    for b in range(B):
        ssi[b, :] = int(spec_state_indices[b, 0].item())
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=aa, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=output_scale,
        initial_state=state_bank, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=ssi, use_qk_l2norm_in_kernel=True)


def packed_ref(nodes, k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, h0):
    """pytorch-fp32 manual GDN recurrence over committed path nodes from h0 = the unambiguous GROUND TRUTH.
    (packed_decode with ssm_state_indices=0 hits the NULL_BLOCK guard and never accumulates — do NOT use it
    as the committer reference.) fused_sigmoid matches this at ~1e-7."""
    import torch.nn.functional as F
    st = h0.clone()  # [VH,DV,DK]
    for n in nodes:
        n = int(n)
        k = k_ring[0, n].float()          # [KH,DK]
        v = v_ring[0, n].float()          # [VH,DV]
        a = a_ring[0, n].float(); bb = b_ring[0, n].float()   # [VH]
        gg = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())   # [VH]
        beta = torch.sigmoid(bb)          # [VH]
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)           # l2norm per k-head
        kge = k.repeat_interleave(NUM_VH // NUM_KH, dim=0)                 # [VH,DK] GQA expand
        st = st * torch.exp(gg)[:, None, None]
        corr = (st * kge[:, None, :]).sum(-1)                             # [VH,DV]
        vv = (v - corr) * beta[:, None]
        st = st + vv[:, :, None] * kge[:, None, :]
    return st  # [VH,DIM,DIM]


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1313)
    mk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    N_PAD = 16
    A_log = mk(NUM_VH); dt_bias = mk(NUM_VH)
    k_ring = mk(1, N_PAD, NUM_KH, DIM) * 0.3
    v_ring = mk(1, N_PAD, NUM_VH, DIM) * 0.3
    a_ring = mk(1, N_PAD, NUM_VH) * 0.5
    b_ring = mk(1, N_PAD, NUM_VH) * 0.5
    # bank: row 0 = NULL, row 5 = col-0 running row (h0)
    bank = torch.zeros(8, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    h0 = mk(NUM_VH, DIM, DIM).float() * 0.2
    COL0_ROW = 5
    bank[COL0_ROW] = h0.clone()
    spec_state_indices = torch.zeros(1, 12, device=dev, dtype=torch.int32)
    spec_state_indices[0, 0] = COL0_ROW   # col 0 -> running row 5

    # committed path: a BRANCH winner [1, 4] (acc_len=2) => committed nodes [0,1,4] (num_accepted>1)
    accepted_paths = torch.zeros(1, 11, device=dev, dtype=torch.long)
    accepted_paths[0, 0] = 1; accepted_paths[0, 1] = 4
    accepted_lens = torch.tensor([2], device=dev, dtype=torch.int32)
    committed_nodes = [0, 1, 4]

    # DEBUG 1: single-node committed path [0] -- isolates gather/deposit from multi-token accumulation
    ap1 = torch.zeros(1, 11, device=dev, dtype=torch.long)
    al1 = torch.tensor([0], device=dev, dtype=torch.int32)   # acc_len=0 => committed nodes = [0] only
    bank1 = torch.zeros(8, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32); bank1[COL0_ROW] = h0.clone()
    native_committer(state_bank=bank1, spec_state_indices=spec_state_indices, accepted_paths=ap1,
                     accepted_lens=al1, k_ring=k_ring, v_ring=v_ring, a_ring=a_ring, b_ring=b_ring,
                     A_log=A_log, dt_bias=dt_bias, num_spec_decodes=1, output_scale=SCALE)
    ref1 = packed_ref([0], k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, h0)
    e1 = (bank1[COL0_ROW] - ref1)
    print(f"[DEBUG single-node [0]] col0-changed-from-h0={( bank1[COL0_ROW]-h0).abs().max().item():.3e}  "
          f"vs packed_ref max|d|={e1.abs().max().item():.3e}", flush=True)

    ref_final = packed_ref(committed_nodes, k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, h0)

    native_committer(state_bank=bank, spec_state_indices=spec_state_indices, accepted_paths=accepted_paths,
                     accepted_lens=accepted_lens, k_ring=k_ring, v_ring=v_ring, a_ring=a_ring, b_ring=b_ring,
                     A_log=A_log, dt_bias=dt_bias, num_spec_decodes=1, output_scale=SCALE)
    col0_after = bank[COL0_ROW]

    e = (col0_after - ref_final)
    print(f"=== native committer col-0 deposit vs packed_decode over committed path [0,1,4] (acc_len=2) ===", flush=True)
    print(f"  max|d|={e.abs().max().item():.3e}  mean|d|={e.abs().mean().item():.3e}", flush=True)
    ok = e.abs().max().item() < 1e-4
    print(f"\n=== VERDICT: {'PASS - native committer logic correct, wire it in' if ok else 'FAIL - convention wrong, fix before wiring'} ===",
          flush=True)


if __name__ == "__main__":
    main()
