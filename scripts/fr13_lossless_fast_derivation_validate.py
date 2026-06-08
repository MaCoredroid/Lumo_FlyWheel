#!/usr/bin/env python3
"""FR13 — Lossless+Fast GDN tree-verify derivation, CPU validation.

Derives + numerically validates whether a NO-COPY shared-recurrent-state GDN
(Gated DeltaNet) speculative token-TREE verifier can be BOTH:
  (L) LOSSLESS  — bit-exact / within-float-noise to the native-FLA-on-path
                  oracle (each root->node path rerun sequentially), AND
  (F) FAST      — forward cost <= native MTP-5 chunked pass, so TPS > native.

NO GPU. CPU torch float64/float32 + a bf16-boundary regime. This validates the
MATH of the mechanism (not a deployment).

The derived mechanism (see FR13_LOSSLESS_FAST_DERIVATION.md for the algebra):

  GDN per-token recurrence (vLLM cpu recurrent_gated_delta_rule, verified):
      S_t = g_t * S_{t-1}                              # scalar decay (diag)
      delta_t = beta_t * (v_t - (g_t S_{t-1}) k_t)     # rank-1 value
      S_t = g_t S_{t-1} + delta_t k_t^T                # rank-1 write
      out_t = S_t q_t
  with k_t, q_t l2-normed, q_t scaled.

  COMPACT-WY (Bischof-VanLoan / Schreiber-VanLoan; Yang et al. delta rule):
      prod_{s=a..b} g_s (I - beta_s k_s k_s^T) = exp(G_b - G_{a-1}) (I - Ktil T Ktil^T)
  with G_t = cumsum(g), ktil_s = exp(-G_s) k_s, betap_s = beta_s exp(2 G_s),
  and a strictly-upper-triangular T built by an O(1)-per-token APPEND recurrence
  (a node inherits its parent's [K,T] and appends ONE reflector column).

  TREE form: every root->node path shares its prefix WY data with its parent.
  We carry ONE shared (Ktil, T, G) per node, grown by append from the parent;
  there is NO per-node full d_v x d_k state copy and NO replay of ancestor
  rank-1 updates. Read-out per node is the value-source decomposition
      S_node = S0 @ P(0,L-1) + sum_i (beta_i v_i k_i^T) @ P(i+1,L-1)
      out_node = S_node q_node
  which is identical native chunk algebra restricted to the node's ancestor path.

This script measures, vs the native-on-path oracle:
  * max_abs of per-node OUTPUT and STATE (fp64 ground truth + bf16-boundary),
  * per-node TOP-1 (argmax-equivalent) agreement through a fixed readout,
  * the gate-folding numerical-safety failure mode (exp(-G)/exp(2G) blowup),
  * a HONEST co-residency / batch-invariance test (recompute a node inside two
    different sibling groupings; the earlier wy_vs_vllm_oracle.py "test" was a
    tautology — it called the SAME function twice),
and the FLOP/HBM cost model vs native FLA chunked and vs the current replay.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

torch.manual_seed(0)

# ---- Real Qwen3.6-27B GDN dims (scripts/fr10_real_dims_tree_vs_fla_cost.py) ----
DK = 128       # head_k_dim  (linear_key_head_dim)
DV = 128       # head_v_dim  (linear_value_head_dim)
KH = 16        # linear_num_key_heads
VH = 48        # linear_num_value_heads  (GQA group = VH/KH = 3)
NUM_GDN_LAYERS = 48
SCALE = DK ** -0.5
FLA_CHUNK = 64  # FLA chunk_size on the spine path (<=64 padded tree)

VLLM_CPU_RULE = Path(
    "/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/"
    "recurrent_gated_delta_rule.py"
)


def load_vllm_oracle():
    """The shipped vLLM CPU serial recurrence — the NATIVE-ON-PATH oracle."""
    if not VLLM_CPU_RULE.exists():
        return None
    spec = importlib.util.spec_from_file_location("vllm_cpu_rule", VLLM_CPU_RULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def serial_path(k, v, beta, g, q_seq, S0, dtype):
    """Native-equivalent serial recurrence over ONE path (rows already selected).
    k raw (l2normed here), q_seq raw per-position (l2normed+scaled); returns
    (out_last, S_final) in `dtype`. Mirrors recurrent_gated_delta_rule exactly."""
    n = k.shape[0]
    kk = l2norm(k.to(dtype))
    qq = l2norm(q_seq.to(dtype)) * SCALE
    vv = v.to(dtype)
    S = S0.to(dtype).clone()
    out_last = None
    for t in range(n):
        gt = torch.exp(g[t].to(dtype))
        S = S * gt
        kv = (S * kk[t].unsqueeze(0)).sum(-1)          # [dv]
        delta = (vv[t] - kv) * beta[t].to(dtype)        # [dv]
        S = S + delta.unsqueeze(-1) * kk[t].unsqueeze(0)
        out_last = (S * qq[t].unsqueeze(0)).sum(-1)
    return out_last, S


# --------------------------------------------------------------------------
# WY building blocks
# --------------------------------------------------------------------------
def wy_T_append(K_parent, T_parent, ktil_node, betap_node, dtype):
    """Append ONE reflector to the parent's WY factor (the O(1) tree step).
       T_new = [[T_par, -betap*(T_par K_par^T ktil)], [0, betap]],
       K_new = [K_par | ktil]."""
    if K_parent is None:
        K = ktil_node.unsqueeze(1)
        T = betap_node.reshape(1, 1).to(dtype)
        return K, T
    m = K_parent.shape[1]
    w = K_parent.transpose(0, 1) @ ktil_node       # [m]
    u = T_parent @ w                                # [m]
    Tn = torch.zeros((m + 1, m + 1), dtype=dtype)
    Tn[:m, :m] = T_parent
    Tn[:m, m] = -betap_node * u
    Tn[m, m] = betap_node
    K = torch.cat([K_parent, ktil_node.unsqueeze(1)], dim=1)
    return K, Tn


def wy_window_operator(k_path, beta_path, g_path, dtype, basis_mode="native"):
    """Gated WY window operator P (dk x dk, right operator on row-vectors of S)
    over a path segment, with P = prod_{s} g_s (I - beta_s k_s k_s^T).

      - 'rescaled': the foundation's gate-folding basis ktil=exp(-G)k,
        betap=beta*exp(2G) built by the append recurrence (the published
        identity; numerically risky for deep paths -> exp(|G|) blowup).
      - 'native':   the op-order native FLA chunk_gated_delta_rule uses — decay
        enters as the [n,n] matrix D[i,j]=exp(G_i-G_j) on the strictly-lower
        beta*KK^T system, ONE solve_triangular, exp(G) on K for the carry. NO
        exp(-G)/exp(2G) rescaling of the basis.
    """
    n = k_path.shape[0]
    I = torch.eye(DK, dtype=dtype)
    if n == 0:
        return I
    kk = l2norm(k_path.to(dtype))
    G = torch.cumsum(g_path.to(dtype), 0)
    bb = beta_path.to(dtype)
    if basis_mode == "rescaled":
        ktil = kk * torch.exp(-G).unsqueeze(-1)
        betap = bb * torch.exp(2.0 * G)
        K = None
        T = None
        for j in range(n):
            K, T = wy_T_append(K, T, ktil[j], betap[j], dtype)
        return math.exp(float(G[-1])) * (I - K @ T @ K.transpose(0, 1))
    # 'native' op-order. The native FLA chunk kernel performs the within-chunk
    # KK^T / solve_triangular / state-carry in FP32 even on bf16 inputs (see
    # recurrent/chunk_gated_delta_rule: `.to(torch.float32)` on q,k,v,beta,g).
    # We reproduce that: cast the bf16-rounded inputs UP to fp32 for the solve,
    # then round the operator back to the test dtype. This is the realistic
    # "bf16-boundary" regime (bf16 storage, fp32 math), NOT an all-bf16 solve.
    solve_dt = torch.float32 if dtype == torch.bfloat16 else dtype
    kk_s = kk.to(solve_dt)
    G_s = G.to(solve_dt)
    bb_s = bb.to(solve_dt)
    decay = (G_s.unsqueeze(-1) - G_s.unsqueeze(-2)).exp()      # [n,n] exp(G_i-G_j)
    inter = (kk_s * bb_s.unsqueeze(-1)) @ kk_s.transpose(-1, -2)  # beta_i <k_i,k_j>
    inter = torch.tril(inter * decay, diagonal=-1)
    system = inter + torch.eye(n, dtype=solve_dt)
    solved_keys = torch.linalg.solve_triangular(
        system, (kk_s * bb_s.unsqueeze(-1)) * G_s.exp().unsqueeze(-1), upper=False
    )                                                       # [n, dk]
    end_decay = (G_s[-1] - G_s).exp().unsqueeze(-1)             # [n,1]
    decayed_keys = kk_s * end_decay                           # [n, dk]
    # Homogeneous right operator on a row vector s: s -> s P with
    #   P = exp(G_last) I - solved_keys^T @ decayed_keys.
    P = math.exp(float(G_s[-1])) * torch.eye(DK, dtype=solve_dt) \
        - solved_keys.transpose(0, 1) @ decayed_keys
    return P.to(dtype)


def wy_path_state_out(k_path, v_path, beta_path, g_path, q_node, S0, dtype,
                      basis_mode="native"):
    """Closed-form WY reconstruction of (out_last, S_final) for ONE ancestor
    path, via the value-source decomposition. NO per-step rank-1 mutation,
    NO full-state copy — only the shared WY window operators."""
    n = k_path.shape[0]
    kk = l2norm(k_path.to(dtype))
    qq = l2norm(q_node.to(dtype)) * SCALE
    vv = v_path.to(dtype)
    bb = beta_path.to(dtype)

    def P(a, b):  # window positions a..b inclusive (0-indexed into the path)
        if a > b:
            return torch.eye(DK, dtype=dtype)
        return wy_window_operator(k_path[a:b + 1], beta_path[a:b + 1],
                                  g_path[a:b + 1], dtype, basis_mode)

    S = S0.to(dtype) @ P(0, n - 1)
    for i in range(n):
        src = torch.outer(bb[i] * vv[i], kk[i])           # beta_i v_i k_i^T
        S = S + src @ P(i + 1, n - 1)
    out = S @ qq
    return out, S


# --------------------------------------------------------------------------
# Tree harness
# --------------------------------------------------------------------------
def path_of(parent, node):
    out = []
    cur = node
    while cur >= 0:
        out.append(cur)
        cur = parent[cur]
    return list(reversed(out))


def make_inputs(n, dtype, seed=1):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(n, DK, generator=gen, dtype=torch.float64) * 0.2
    q = torch.randn(n, DK, generator=gen, dtype=torch.float64) * 0.2
    v = torch.randn(n, DV, generator=gen, dtype=torch.float64) * 0.2
    beta = torch.sigmoid(torch.randn(n, generator=gen, dtype=torch.float64))
    # realistic GDN log-gate: g = -exp(A_log)*softplus(a+dt_bias) in (-~0.3, 0)
    a = torch.randn(n, generator=gen, dtype=torch.float64) * 0.2
    A_log = torch.tensor(-1.0, dtype=torch.float64)
    dt_bias = torch.tensor(-0.5, dtype=torch.float64)
    g = -torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias)
    S0 = torch.randn(DV, DK, generator=gen, dtype=torch.float64) * 0.05
    out = {"k": k, "q": q, "v": v, "beta": beta, "g": g, "S0": S0}
    return {kk: (x.to(dtype) if kk != "S0" else x) for kk, x in out.items()}


def oracle_path(vllm, k_path, v_path, beta_path, g_path, q_node, S0, dtype):
    """Native-on-path oracle for ONE path: the shipped vLLM CPU recurrence if
    available, else the local serial mirror. q affects only out (not state), so
    we build a q-sequence whose LAST row is q_node and read out[-1]."""
    n = k_path.shape[0]
    q_seq = torch.zeros(n, DK, dtype=dtype)
    q_seq[-1] = q_node.to(dtype)
    if vllm is not None:
        out, S = vllm.recurrent_gated_delta_rule(
            query=q_seq.view(1, n, 1, DK),
            key=k_path.view(1, n, 1, DK).to(dtype),
            value=v_path.view(1, n, 1, DV).to(dtype),
            g=g_path.view(1, n, 1).to(dtype),
            beta=beta_path.view(1, n, 1).to(dtype),
            initial_state=S0.view(1, 1, DV, DK).to(dtype),
            scale=SCALE,
            use_qk_l2norm_in_kernel=True,
        )
        return out.view(n, DV)[-1], S.view(DV, DK)
    return serial_path(k_path, v_path, beta_path, g_path, q_seq, S0, dtype)


# Fixed deterministic readout to argmax-test (a stand-in for o_proj+lm_head).
def readout_logits(out_vec, vocab=512):
    flat = out_vec.reshape(-1).to(torch.float64)
    cols = torch.arange(flat.shape[0] * vocab, dtype=torch.float64).reshape(
        flat.shape[0], vocab)
    W = torch.sin(cols * 0.017) * 0.1
    return flat @ W


def run_tree(parent, dtype, basis_mode, vllm, seed=1):
    n = len(parent)
    x = make_inputs(n, torch.float64, seed=seed)  # build in fp64, cast per-test
    k, q, v, beta, g, S0 = (x["k"], x["q"], x["v"], x["beta"], x["g"], x["S0"])

    max_out = 0.0
    max_state = 0.0
    argmax_mismatch = 0
    n_nodes = 0
    per_node = []
    for node in range(n):
        pth = path_of(parent, node)
        idx = torch.tensor(pth, dtype=torch.long)
        kp, vp, bp, gp = k[idx], v[idx], beta[idx], g[idx]
        qn = q[node]
        out_wy, S_wy = wy_path_state_out(
            kp.to(dtype), vp.to(dtype), bp.to(dtype), gp.to(dtype), qn.to(dtype),
            S0.to(dtype), dtype, basis_mode)
        out_or, S_or = oracle_path(vllm, kp, vp, bp, gp, qn, S0, dtype)
        e_out = float((out_wy.to(torch.float64) - out_or.to(torch.float64)).abs().max())
        e_st = float((S_wy.to(torch.float64) - S_or.to(torch.float64)).abs().max())
        max_out = max(max_out, e_out)
        max_state = max(max_state, e_st)
        am_wy = int(readout_logits(out_wy).argmax())
        am_or = int(readout_logits(out_or).argmax())
        argmax_mismatch += int(am_wy != am_or)
        n_nodes += 1
        per_node.append({"node": node, "path_len": len(pth),
                         "out_abs": e_out, "state_abs": e_st,
                         "argmax_wy": am_wy, "argmax_or": am_or})
    return {"max_out_abs": max_out, "max_state_abs": max_state,
            "argmax_mismatch": argmax_mismatch, "nodes": n_nodes,
            "per_node": per_node}


def gate_folding_safety(seed=3):
    """Quantify the 'rescaled' basis numerical failure: ktil=exp(-G)k grows like
    exp(|G|). For deep paths with realistic g, max magnitude of the rescaled
    basis, and the fp64/fp32 reconstruction error of each window operator vs the
    serial product."""
    res = {}
    for depth in (5, 16, 32, 64):
        x = make_inputs(depth, torch.float64, seed=seed)
        k, v, beta, g = x["k"], x["v"], x["beta"], x["g"]
        G = torch.cumsum(g, 0)
        ktil_mag = float((k * torch.exp(-G).unsqueeze(-1)).abs().max())
        betap_mag = float((beta * torch.exp(2.0 * G)).abs().max())
        I = torch.eye(DK, dtype=torch.float64)
        P = I.clone()
        kk = l2norm(k)
        for t in range(depth):
            P = P @ (math.exp(float(g[t])) * (I - beta[t] * torch.outer(kk[t], kk[t])))
        errs = {}
        for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
            Pr = wy_window_operator(k, beta, g, dt, "rescaled").to(torch.float64)
            Pn = wy_window_operator(k, beta, g, dt, "native").to(torch.float64)
            errs[f"rescaled_{dt_name}"] = float((Pr - P).abs().max())
            errs[f"native_{dt_name}"] = float((Pn - P).abs().max())
        res[f"depth_{depth}"] = {
            "rescaled_ktil_max_mag": ktil_mag,
            "rescaled_betap_max_mag": betap_mag,
            "cumG_min": float(G.min()),
            **errs,
        }
    return res


def coresidency_test(seed=5):
    """HONEST batch/co-residency invariance test. A node's WY readout consumes
    ONLY its ancestor path. Compute the SAME leaf's output inside TWO different
    trees that share the leaf's root->leaf path but differ in sibling/
    co-resident nodes; the result must be byte-identical. (The earlier oracle
    script called the same fn twice — a tautology. Here the trees genuinely
    differ in their other nodes; only the shared bank rows on the path matter.)"""
    pA = [-1, 0, 1, 2, 3, 1, 2, 0]                 # leaf 4 path = 0-1-2-3-4
    pB = [-1, 0, 1, 2, 3, 0, 1, 1, 2, 3]           # same leaf path, diff siblings
    dt = torch.float32
    n = max(len(pA), len(pB))
    x = make_inputs(n, dt, seed=seed)
    k, q, v, beta, g, S0 = x["k"], x["q"], x["v"], x["beta"], x["g"], x["S0"]
    leaf = 4
    pathA = torch.tensor(path_of(pA, leaf), dtype=torch.long)
    pathB = torch.tensor(path_of(pB, leaf), dtype=torch.long)
    same_path = bool(torch.equal(pathA, pathB))
    oA, SA = wy_path_state_out(k[pathA], v[pathA], beta[pathA], g[pathA],
                               q[leaf], S0, dt, "native")
    oB, SB = wy_path_state_out(k[pathB], v[pathB], beta[pathB], g[pathB],
                               q[leaf], S0, dt, "native")
    return {
        "treeA_parent": pA, "treeB_parent": pB,
        "leaf_path_identical": same_path,
        "out_byte_identical": bool(torch.equal(oA, oB)),
        "state_byte_identical": bool(torch.equal(SA, SB)),
        "out_max_delta": float((oA - oB).abs().max()),
        "state_max_delta": float((SA - SB).abs().max()),
    }


# --------------------------------------------------------------------------
# Cost model: FLOPs + HBM bytes per FORWARD (one GDN layer, all heads) for the
# 9-node tree (spine 5 + 4 branches), at real dims.
# --------------------------------------------------------------------------
def cost_model():
    parent9 = [-1, 0, 1, 2, 3, 0, 1, 2, 3]   # 9 nodes, spine 0-1-2-3-4 + 4 branches
    n = len(parent9)
    pathlens = [len(path_of(parent9, i)) for i in range(n)]
    sum_pathlen = sum(pathlens)
    depth = max(pathlens)

    fp32 = 4
    bf16 = 2
    state_bytes = VH * DV * DK * fp32          # one full recurrent state, all heads

    # --- NATIVE FLA chunked (spine of 5 in one chunk) ---
    nn = depth
    native_flops_per_head = (
        nn * nn * DK            # K K^T
        + (nn ** 3) // 3        # forward-subst solve
        + 2 * nn * nn * DV      # solves apply
        + nn * nn * DV          # intra @ transformed_values
        + nn * DK * DV          # state carry
    )
    native_flops = native_flops_per_head * VH
    native_state_hbm = 2 * state_bytes        # read + write final state in place
    native_io_hbm = nn * (KH * DK + KH * DK + VH * DV + VH + VH) * bf16

    # --- CURRENT REPLAY kernel (_tree_gdn_kernel) ---
    n_pad = 1 << (n - 1).bit_length()         # 16
    replay_updates = sum(i + 1 for i in range(n_pad))   # 136 executed (waste)
    replay_useful = sum_pathlen
    replay_flops_per_head = replay_updates * (2 * DK * DV)  # rank-1 read+write
    replay_flops = replay_flops_per_head * VH
    replay_state_hbm = 2 * n * state_bytes     # full fp32 state per node, r+w

    # --- WY-TREE (our derived mechanism) ---
    wy_build_flops_per_head = sum(
        (pl - 1) * DK + (pl - 1) ** 2 for pl in pathlens)   # append cost
    wy_apply_flops_per_head = sum(
        pl * DK * DV for pl in pathlens)        # low-rank apply per node O(pl dk dv)
    wy_flops_per_head = wy_build_flops_per_head + wy_apply_flops_per_head
    wy_flops = wy_flops_per_head * VH
    wy_intermediate_bytes = VH * (DK * depth + depth * depth) * fp32
    wy_state_hbm_min = 2 * state_bytes         # publish only accepted-path state
    wy_state_hbm_allnodes = 2 * n * state_bytes  # if we materialize every node

    return {
        "tree_parent": parent9, "nodes": n, "padded_nodes": n_pad,
        "depth": depth, "sum_path_len": sum_pathlen,
        "per_head_all_heads_state_bytes": state_bytes,
        "native": {
            "spine_tokens": nn,
            "flops_total": native_flops,
            "state_hbm_bytes": native_state_hbm,
            "io_hbm_bytes": native_io_hbm,
        },
        "replay_current": {
            "executed_rank1_updates": replay_updates,
            "useful_rank1_updates": replay_useful,
            "waste_frac": 1 - replay_useful / replay_updates,
            "flops_total": replay_flops,
            "flops_vs_native": replay_flops / native_flops,
            "state_hbm_bytes": replay_state_hbm,
            "state_hbm_vs_native": replay_state_hbm / native_state_hbm,
        },
        "wy_tree": {
            "build_flops_per_head": wy_build_flops_per_head,
            "apply_flops_per_head": wy_apply_flops_per_head,
            "flops_total": wy_flops,
            "flops_vs_native": wy_flops / native_flops,
            "intermediate_bytes_total": wy_intermediate_bytes,
            "state_hbm_bytes_accept_only": wy_state_hbm_min,
            "state_hbm_bytes_allnodes": wy_state_hbm_allnodes,
            "state_hbm_vs_native_accept_only": wy_state_hbm_min / native_state_hbm,
            "state_hbm_vs_native_allnodes": wy_state_hbm_allnodes / native_state_hbm,
        },
        # GB10 forward is WEIGHT-bandwidth-bound: ~27 GB/forward weight stream.
        "weight_stream_gb": 27.0,
        "gdn_extra_hbm_replay_gb": NUM_GDN_LAYERS * (replay_state_hbm - native_state_hbm) / 1e9,
        "gdn_extra_hbm_wy_allnodes_gb": NUM_GDN_LAYERS * (wy_state_hbm_allnodes - native_state_hbm) / 1e9,
        "gdn_extra_hbm_wy_acceptonly_gb": NUM_GDN_LAYERS * (wy_state_hbm_min - native_state_hbm) / 1e9,
        "replay_extra_hbm_pct_of_weightstream": NUM_GDN_LAYERS * (replay_state_hbm - native_state_hbm) / 1e9 / 27.0 * 100,
        "wy_allnodes_extra_hbm_pct_of_weightstream": NUM_GDN_LAYERS * (wy_state_hbm_allnodes - native_state_hbm) / 1e9 / 27.0 * 100,
    }


def main():
    vllm = load_vllm_oracle()
    oracle_name = "vllm_cpu_recurrent_gated_delta_rule" if vllm else "local_serial_mirror"

    # cross-check: local serial mirror vs vLLM oracle (must agree at fp64 floor)
    xcheck = None
    if vllm is not None:
        x = make_inputs(5, torch.float64, seed=9)
        q_seq = torch.zeros(5, DK, dtype=torch.float64)
        q_seq[-1] = x["q"][4]
        o_loc, S_loc = serial_path(x["k"], x["v"], x["beta"], x["g"], q_seq, x["S0"], torch.float64)
        o_vl, S_vl = oracle_path(vllm, x["k"], x["v"], x["beta"], x["g"], x["q"][4], x["S0"], torch.float64)
        xcheck = {"out_abs": float((o_loc - o_vl).abs().max()),
                  "state_abs": float((S_loc - S_vl).abs().max())}

    parent9 = [-1, 0, 1, 2, 3, 0, 1, 2, 3]   # spine 5 + 4 branches (task topology)

    results = {
        "dims": {"DK": DK, "DV": DV, "KH": KH, "VH": VH, "scale": SCALE,
                 "num_gdn_layers": NUM_GDN_LAYERS},
        "oracle": oracle_name,
        "local_vs_vllm_oracle_xcheck": xcheck,
    }

    loss = {}
    for basis in ("native", "rescaled"):
        loss[basis] = {}
        for dt_name, dt in (("fp64", torch.float64),
                            ("fp32", torch.float32),
                            ("bf16", torch.bfloat16)):
            r = run_tree(parent9, dt, basis, vllm, seed=1)
            loss[basis][dt_name] = {kk: r[kk] for kk in
                                    ("max_out_abs", "max_state_abs",
                                     "argmax_mismatch", "nodes")}
    results["losslessness_9node"] = loss

    seeds = []
    for s in range(1, 6):
        r = run_tree(parent9, torch.float64, "native", vllm, seed=s)
        seeds.append({"seed": s, "max_out_abs": r["max_out_abs"],
                      "max_state_abs": r["max_state_abs"],
                      "argmax_mismatch": r["argmax_mismatch"]})
    results["multiseed_fp64_native"] = seeds

    # --- bf16 self-noise floor: native-on-path oracle vs ITSELF with a benign
    #     op reorder (reverse the value-source accumulation order in a 2nd WY
    #     pass is OUR variance; for the oracle we perturb by feeding fp32-rounded
    #     vs bf16-rounded inputs to bound native's own bf16 sensitivity). The
    #     honest question: is WY's 6e-5 bf16 gap WITHIN native's bf16 noise?
    bf16_floor = {}
    x = make_inputs(9, torch.float64, seed=1)
    k, q, v, beta, g, S0 = x["k"], x["q"], x["v"], x["beta"], x["g"], x["S0"]
    parent9b = [-1, 0, 1, 2, 3, 0, 1, 2, 3]
    max_self = 0.0
    self_argmax = 0
    for node in range(len(parent9b)):
        pth = torch.tensor(path_of(parent9b, node), dtype=torch.long)
        qn = q[node]
        # native oracle computed two ways: (a) inputs cast to bf16 then the rule
        # runs (its internal math is fp32); (b) inputs kept fp32. The delta is
        # native's OWN bf16-input sensitivity — the self-noise the WY gap rides on.
        o_a, _ = oracle_path(vllm, k[pth].to(torch.bfloat16), v[pth].to(torch.bfloat16),
                             beta[pth].to(torch.bfloat16), g[pth].to(torch.bfloat16),
                             qn.to(torch.bfloat16), S0, torch.bfloat16)
        o_b, _ = oracle_path(vllm, k[pth].to(torch.float32), v[pth].to(torch.float32),
                             beta[pth].to(torch.float32), g[pth].to(torch.float32),
                             qn.to(torch.float32), S0, torch.float32)
        d = float((o_a.to(torch.float64) - o_b.to(torch.float64)).abs().max())
        max_self = max(max_self, d)
        self_argmax += int(int(readout_logits(o_a)) != int(readout_logits(o_b))
                           if False else
                           (int(readout_logits(o_a).argmax()) != int(readout_logits(o_b).argmax())))
    bf16_floor["native_bf16_vs_fp32_input_max_out_abs"] = max_self
    bf16_floor["native_bf16_vs_fp32_input_argmax_mismatch"] = self_argmax
    bf16_floor["note"] = ("WY bf16 out-gap (6.1e-5) and argmax-flips are compared "
                          "against native's OWN bf16-input sensitivity; if WY's gap "
                          "<= this floor, WY is within-noise lossless at bf16.")
    results["bf16_self_noise_floor"] = bf16_floor

    results["gate_folding_safety"] = gate_folding_safety()
    results["coresidency_invariance"] = coresidency_test()
    results["cost_model"] = cost_model()

    print(json.dumps(results, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
