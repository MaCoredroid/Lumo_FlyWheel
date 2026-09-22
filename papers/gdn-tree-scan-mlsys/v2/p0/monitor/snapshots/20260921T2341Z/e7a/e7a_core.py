#!/usr/bin/env python3
"""E7a core algebra: three tree-GDN verification/commit mechanisms on identical operands.

Pure torch (CPU or GPU, no Triton). Everything here is an ALGEBRAIC realization used to
(1) establish the identities against a float64 serial oracle and (2) serve as the torch-level
mirror of each mechanism. Device (Triton) realizations live in e7a_device.py and are the only
things whose latency/memory may be reported as GPU results.

Mechanisms (all LOCAL reimplementations; B and C are our implementations of the published
mechanism families, not the authors' systems):
  A  sequential rank-1 scan over the tree (parent-checkpoint) + accepted-path REPLAY commit.
     Mirrors the op order of the production `_gdn_node_step` body and the native
     `fused_sigmoid_gating_delta_rule_update` recurrence.
  B  ancestor-masked triangular system solved by FORWARD SUBSTITUTION (WY / TreeWY family),
     then COMPACT reconstruction of the committed state from stored factors.
  C  the SAME ancestor-masked system solved by the FINITE NEUMANN series (Bole family),
     then the same compact reconstruction.

State orientation (matches native and our kernels): S in R^{DV x DK} per value head,
  S_t = exp(g_t) * S_{t-1};  u_t = beta_t * (v_t - S_t k_t);  S_t += u_t k_t^T;  o_t = S_t q_t.
Verified against Bole eq.(1)/(5)/(6)/(10) and TreeWY eq.(1)-(3) (primary arXiv HTML, 2026-09-21):
  (I + G) U = R,  G[i,j] = beta_i (P_i/P_j) <k_i,k_j> for j strict ancestor of i,
  R[i] = beta_i (v_i - P_i S0 k_i),  P_i = prod_{r on root..i} exp(g_r),
  S_a = P_a S0 + sum_{j <= a} (P_a/P_j) U_j k_j^T,   o_i = S_i q_i.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Iterable

import torch

SOFTPLUS_THRESHOLD = 20.0
L2NORM_EPS = 1e-6


# ----------------------------------------------------------------------------- tree utils
def ancestors(parents: list[int], i: int) -> list[int]:
    out = []
    cur = parents[i]
    while cur >= 0:
        out.append(cur)
        cur = parents[cur]
    out.reverse()
    return out


def path_to(parents: list[int], i: int) -> list[int]:
    return ancestors(parents, i) + [i]


def depth_of(parents: list[int], i: int) -> int:
    return len(ancestors(parents, i))


def max_strict_ancestors(parents: list[int]) -> int:
    return max(depth_of(parents, i) for i in range(len(parents)))


def strict_mask(parents: list[int], device=None) -> torch.Tensor:
    n = len(parents)
    m = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i in range(n):
        for j in ancestors(parents, i):
            m[i, j] = True
    return m


def visible_mask(parents: list[int], device=None) -> torch.Tensor:
    m = strict_mask(parents, device)
    return m | torch.eye(len(parents), dtype=torch.bool, device=device)


def children_of(parents: list[int]) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p >= 0:
            ch[p].append(i)
    return ch


def check_topological(parents: list[int]) -> None:
    if parents[0] != -1 or any(p < 0 for p in parents[1:]):
        raise ValueError("expected a single root at node 0")
    for i, p in enumerate(parents):
        if p >= i:
            raise ValueError(f"parent {p} must precede child {i}")


def chain_parents(depth: int) -> list[int]:
    """Root + `depth` draft nodes in a chain (depth = number of strict ancestors of the leaf)."""
    return [-1] + list(range(depth))


CATERPILLAR_10 = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]  # the served topology in every historical payload


def binary_parents(depth: int) -> list[int]:
    """Fixed branching control: full binary tree, root at 0, BFS order. depth=3 -> 15 nodes."""
    parents = [-1]
    frontier = [0]
    for _ in range(depth):
        nxt = []
        for p in frontier:
            for _c in range(2):
                parents.append(p)
                nxt.append(len(parents) - 1)
        frontier = nxt
    return parents


def sibling_reorder_permutation(parents: list[int], mode: str = "reverse") -> tuple[list[int], list[int]]:
    """Return (new_parents, perm) where perm[new_id] = old_id, produced by a DFS pre-order that visits
    children in a different order (reverse / rotate). Keeps parents[i] < i. Used as the
    sibling-reorder control: results must be invariant up to relabeling."""
    ch = children_of(parents)
    perm: list[int] = []
    new_parent_of_old: dict[int, int] = {}

    def visit(old: int, new_parent: int) -> None:
        new_id = len(perm)
        perm.append(old)
        new_parent_of_old[old] = new_parent
        kids = list(ch[old])
        if mode == "reverse":
            kids = kids[::-1]
        elif mode == "rotate" and len(kids) > 1:
            kids = kids[1:] + kids[:1]
        for c in kids:
            visit(c, new_id)

    visit(0, -1)
    new_parents = [new_parent_of_old[perm[i]] for i in range(len(perm))]
    return new_parents, perm


# ----------------------------------------------------------------------------- payloads
@dataclass
class Payload:
    q: torch.Tensor  # (N, KH, DK) bf16 or fp32
    k: torch.Tensor  # (N, KH, DK)
    v: torch.Tensor  # (N, VH, DV)
    a: torch.Tensor  # (N, VH)   raw gate a
    b: torch.Tensor  # (N, VH)   raw gate b
    A_log: torch.Tensor  # (VH,)
    dt_bias: torch.Tensor  # (VH,)
    h0: torch.Tensor  # (VH, DV, DK) fp32
    parents: list[int]
    output_scale: float
    label: str = ""
    provenance: dict = field(default_factory=dict)
    serving_out: torch.Tensor | None = None
    serving_state: torch.Tensor | None = None
    synthetic: bool = False

    @property
    def n(self) -> int:
        return len(self.parents)

    def to(self, device) -> "Payload":
        kw = {}
        for f in ("q", "k", "v", "a", "b", "A_log", "dt_bias", "h0", "serving_out", "serving_state"):
            t = getattr(self, f)
            kw[f] = t.to(device) if t is not None else None
        return Payload(parents=list(self.parents), output_scale=self.output_scale, label=self.label,
                       provenance=dict(self.provenance), synthetic=self.synthetic, **kw)

    def permuted(self, perm: list[int], new_parents: list[int]) -> "Payload":
        idx = torch.tensor(perm, dtype=torch.long, device=self.q.device)
        kw = {f: getattr(self, f).index_select(0, idx).contiguous() for f in ("q", "k", "v", "a", "b")}
        so = self.serving_out.index_select(0, idx) if self.serving_out is not None else None
        ss = self.serving_state.index_select(0, idx) if self.serving_state is not None else None
        return Payload(A_log=self.A_log, dt_bias=self.dt_bias, h0=self.h0, parents=list(new_parents),
                       output_scale=self.output_scale, label=self.label + f"+perm", provenance=dict(self.provenance),
                       serving_out=so, serving_state=ss, synthetic=self.synthetic, **kw)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_payload(path: str, *, with_hash: bool = True) -> Payload:
    p = torch.load(path, map_location="cpu", weights_only=True)
    schema = p.get("schema", "")
    parents = [int(x) for x in p["tree_parent"]]
    n = len(parents)
    if schema == "fr10.tree_gdn_scan_capture.v1":
        h0 = p["h0"]
        serving_out = p.get("serving_out")
        serving_state = p.get("serving_state")
    elif schema == "fr10.src_native_handoff_payload.v1":
        h0 = p["prev_h0"]
        serving_out = None
        serving_state = p.get("serving_tree_state")
    else:
        raise ValueError(f"unknown payload schema {schema!r} in {path}")

    def rows(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        return t[:n].contiguous()

    fresh = "/experiments/out-" in path and "e7a-step2" in path
    prov = {"path": path, "schema": schema, "layer_prefix": p.get("layer_prefix"), "batch_index": p.get("batch_index"),
            "status": "FRESH (captured in this worktree; see capture_request.json binding)" if fresh else "HISTORICAL (June 2026 old-repo capture)"}
    if with_hash:
        prov["sha256"] = sha256_file(path)
    # request binding written by serve_drivers.sh capture (sibling of the logs/ dir)
    binding = os.path.join(os.path.dirname(os.path.dirname(path)), "capture_request.json")
    if fresh and os.path.exists(binding):
        try:
            b = json.load(open(binding))
            prov["request_binding"] = {k: b.get(k) for k in ("prefix_id", "prefix_sha256", "prefix_tokens_pool", "source", "response_id", "usage")}
        except Exception as exc:  # binding is provenance, never silently dropped
            prov["request_binding_error"] = repr(exc)
    elif fresh:
        prov["request_binding"] = None
    for extra in ("accepted_len", "accepted_node_id", "accepted_node_path", "state_index", "n_pad"):
        if extra in p:
            val = p[extra]
            prov[extra] = [int(x) for x in val] if isinstance(val, (list, tuple)) else int(val)
    return Payload(q=rows(p["query_spec"]), k=rows(p["key_spec"]), v=rows(p["value_tree"]),
                   a=rows(p["a"]), b=rows(p["b"]), A_log=p["A_log"].contiguous(), dt_bias=p["dt_bias"].contiguous(),
                   h0=h0.contiguous(), parents=parents, output_scale=float(p["output_scale"]),
                   label=str(p.get("layer_prefix", "")).replace("language_model.model.", ""),
                   provenance=prov, serving_out=serving_out, serving_state=serving_state, synthetic=False)


def synth_payload(parents: list[int], seed: int, *, regime: str = "historical-like",
                  KH: int = 16, VH: int = 48, DK: int = 128, DV: int = 128,
                  h0_ref: torch.Tensor | None = None) -> Payload:
    """SYNTHETIC operands. Magnitudes follow the historical payload ranges (q/k/v bf16 ~N(0,1)-ish,
    a in [-8,11], b in [-5,7], A_log in [-3.5,-0.6], dt_bias in [-7,7], h0 in [-2.4,3.6]) unless a
    regime says otherwise. Labeled synthetic everywhere downstream."""
    check_topological(parents)
    n = len(parents)
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)
    q = (rn(n, KH, DK) * 1.0).to(torch.bfloat16)
    k = (rn(n, KH, DK) * 0.7).to(torch.bfloat16)
    v = (rn(n, VH, DV) * 0.6).to(torch.bfloat16)
    A_log = (torch.rand(VH, generator=gen) * (3.5 - 0.6) * -1.0 - 0.6).to(torch.float32)
    dt_bias = (rn(VH) * 3.0).to(torch.bfloat16)
    if regime == "historical-like":
        a = (rn(n, VH) * 3.0).to(torch.bfloat16)
        b = (rn(n, VH) * 2.0).to(torch.bfloat16)
    elif regime == "tiny-gates":
        # strong decay: a + dt_bias large positive -> softplus ~ x -> g ~ -exp(A_log)*x deeply negative
        a = (rn(n, VH).abs() * 4.0 + 12.0).to(torch.bfloat16)
        b = (rn(n, VH) * 2.0).to(torch.bfloat16)
        A_log = (torch.rand(VH, generator=gen) * 1.0 + 0.5).to(torch.float32)  # exp(A_log) in [1.6, 4.5]
    elif regime == "near-unit-gates":
        a = (-(rn(n, VH).abs() * 3.0) - 6.0).to(torch.bfloat16)  # softplus ~ 0 -> g ~ 0
        b = (rn(n, VH) * 2.0).to(torch.bfloat16)
    else:
        raise ValueError(regime)
    if h0_ref is not None:
        h0 = h0_ref.clone().float()
    else:
        h0 = (rn(VH, DV, DK) * 0.8).to(torch.float32)
    return Payload(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, h0=h0, parents=list(parents),
                   output_scale=DK ** -0.5, label=f"synthetic[{regime}] seed={seed} n={n}",
                   provenance={"status": "SYNTHETIC", "regime": regime, "seed": seed, "parents": list(parents)},
                   synthetic=True)


# ----------------------------------------------------------------------------- gates & norms
def gates_native(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, dtype: torch.dtype,
                 *, beta_bf16_roundtrip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """g = -exp(A_log) * softplus(a + dt_bias) with the native threshold; beta = sigmoid(b).
    All in `dtype` from the rounded inputs. `beta_bf16_roundtrip` mirrors the packed one-token decode
    kernel (sigmoid(b).to(bf16).to(fp32)); the spec-update kernel does NOT round-trip."""
    x = a.to(dtype) + dt_bias.to(dtype)[None, :]
    sp = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)) if dtype == torch.float64 else torch.log(1.0 + torch.exp(x)), x)
    g = -torch.exp(A_log.to(dtype))[None, :] * sp
    beta = torch.sigmoid(b.to(dtype))
    if beta_bf16_roundtrip:
        beta = beta.to(torch.bfloat16).to(dtype)
    return g, beta


def l2norm(x: torch.Tensor, mode: str = "rsqrt") -> torch.Tensor:
    ss = (x * x).sum(-1, keepdim=True) + L2NORM_EPS
    if mode == "rsqrt":
        return x * torch.rsqrt(ss)
    if mode == "divsqrt":
        return x / torch.sqrt(ss)
    raise ValueError(mode)


@dataclass
class Operands:
    """Rounded operands lifted to a working dtype. q is scaled by output_scale (as the kernels do)."""
    q: torch.Tensor  # (N, KH, DK) normalized*scale
    k: torch.Tensor  # (N, KH, DK) normalized
    v: torch.Tensor  # (N, VH, DV)
    g: torch.Tensor  # (N, VH)
    beta: torch.Tensor  # (N, VH)
    h0: torch.Tensor  # (VH, DV, DK)
    parents: list[int]
    dtype: torch.dtype

    @property
    def group(self) -> int:
        return self.v.shape[1] // self.q.shape[1]

    def kh_of(self, hv: int) -> int:
        return hv // self.group

    def expand_k(self) -> torch.Tensor:
        """k per VALUE head: (N, VH, DK)."""
        return self.k.repeat_interleave(self.group, dim=1)

    def expand_q(self) -> torch.Tensor:
        return self.q.repeat_interleave(self.group, dim=1)


def lift(pay: Payload, dtype: torch.dtype, *, l2norm_mode: str = "rsqrt", beta_bf16_roundtrip: bool = False) -> Operands:
    q = l2norm(pay.q.to(dtype), l2norm_mode) * pay.output_scale
    k = l2norm(pay.k.to(dtype), l2norm_mode)
    g, beta = gates_native(pay.a, pay.b, pay.A_log, pay.dt_bias, dtype, beta_bf16_roundtrip=beta_bf16_roundtrip)
    return Operands(q=q, k=k, v=pay.v.to(dtype), g=g, beta=beta, h0=pay.h0.to(dtype), parents=list(pay.parents), dtype=dtype)


def cum_gates(ops: Operands) -> torch.Tensor:
    """cum_g[i, vh] = sum of g over root..i inclusive (= log P_i)."""
    n = len(ops.parents)
    cum = torch.zeros_like(ops.g)
    for i in range(n):
        p = ops.parents[i]
        cum[i] = ops.g[i] + (cum[p] if p >= 0 else 0.0)
    return cum


# ----------------------------------------------------------------------------- A: sequential (oracle when fp64)
@dataclass
class SeqResult:
    out: torch.Tensor  # (N, VH, DV)
    state: torch.Tensor  # (N, VH, DV, DK)
    u: torch.Tensor  # (N, VH, DV) correction factor actually applied at node i
    cum_g: torch.Tensor  # (N, VH)


def sequential_tree(ops: Operands, *, checkpoint_parent: bool = True) -> SeqResult:
    """Mechanism A verifier. Op order mirrors `_gdn_node_step` / native fused_sigmoid_gating:
    state *= exp(g); v -= (state k); v *= beta; state += v (x) k; out = state q(scaled).
    checkpoint_parent=True starts node i from the PARENT's post-state (the scan's register h_cache);
    False replays the whole root path from h0 per node (identical algebra, different rounding path)."""
    n = len(ops.parents)
    VH, DV, DK = ops.h0.shape
    kx, qx = ops.expand_k(), ops.expand_q()
    state = torch.empty((n, VH, DV, DK), dtype=ops.dtype, device=ops.h0.device)
    out = torch.empty((n, VH, DV), dtype=ops.dtype, device=ops.h0.device)
    u = torch.empty((n, VH, DV), dtype=ops.dtype, device=ops.h0.device)

    def step(S, i):
        S = S * torch.exp(ops.g[i])[:, None, None]
        vv = ops.v[i] - torch.einsum("hvk,hk->hv", S, kx[i])
        vv = vv * ops.beta[i][:, None]
        S = S + vv[:, :, None] * kx[i][:, None, :]
        o = torch.einsum("hvk,hk->hv", S, qx[i])
        return S, vv, o

    for i in range(n):
        if checkpoint_parent:
            S = ops.h0 if ops.parents[i] < 0 else state[ops.parents[i]]
            S, vv, o = step(S, i)
        else:
            S = ops.h0
            for j in path_to(ops.parents, i):
                S, vv, o = step(S, j)
        state[i], u[i], out[i] = S, vv, o
    return SeqResult(out=out, state=state, u=u, cum_g=cum_gates(ops))


def replay_commit(ops: Operands, path: list[int]) -> torch.Tensor:
    """Mechanism A commit: re-execute h0 -> nodes in `path` (root first) sequentially; return final state."""
    kx = ops.expand_k()
    S = ops.h0
    for j in path:
        S = S * torch.exp(ops.g[j])[:, None, None]
        vv = ops.v[j] - torch.einsum("hvk,hk->hv", S, kx[j])
        vv = vv * ops.beta[j][:, None]
        S = S + vv[:, :, None] * kx[j][:, None, :]
    return S


# ----------------------------------------------------------------------------- B/C: masked system
@dataclass
class System:
    G: torch.Tensor  # (VH, N, N) strictly lower-triangular in topological order (masked)
    R: torch.Tensor  # (VH, N, DV)
    cum_g: torch.Tensor  # (N, VH)
    decay: torch.Tensor  # (VH, N, N) = P_i/P_j on visible pairs (0 elsewhere)
    strict: torch.Tensor  # (N, N) bool
    visible: torch.Tensor  # (N, N) bool
    ratio_mode: str


def decay_ratios(cum_g: torch.Tensor, g: torch.Tensor, parents: list[int], visible: torch.Tensor, mode: str) -> torch.Tensor:
    """decay[vh, i, j] = P_i / P_j for j on root..i (inclusive), else 0.
    mode 'expdiff': exp(cum_i - cum_j)   (what our WY probe and the tree kernels do)
    mode 'ratio'  : exp(cum_i) / exp(cum_j)   (literal P_i/P_j; underflow-prone for tiny gates)
    mode 'prodedge': product of exp(g_r) over the edges strictly after j up to i (recurrence-faithful)."""
    n = cum_g.shape[0]
    VH = cum_g.shape[1]
    cum = cum_g.t()  # (VH, N)
    visn = visible.to(torch.bool)[None]
    if mode == "expdiff":
        # SELECTION masking with the exponent input masked BEFORE exp: off-path pairs never see a
        # (possibly overflowing) exponent, so they are finite exact zeros (no inf*0 = NaN).
        diff = torch.where(visn, cum[:, :, None] - cum[:, None, :], torch.zeros((), dtype=cum.dtype, device=cum.device))
        d = torch.where(visn, torch.exp(diff), torch.zeros((), dtype=cum.dtype, device=cum.device))
    elif mode == "ratio":
        # DELIBERATE literal-ratio diagnostic (P_i / P_j): may underflow/overflow ON-PATH for tiny gates;
        # off-path pairs are selection-masked to exact zeros. Callers must check finiteness (labeled diagnostic).
        P = torch.exp(cum)
        d = torch.where(visn, P[:, :, None] / P[:, None, :], torch.zeros((), dtype=cum.dtype, device=cum.device))
    elif mode == "prodedge":
        d = torch.zeros((VH, n, n), dtype=cum_g.dtype, device=cum_g.device)
        eg = torch.exp(g.t())  # (VH, N)
        for i in range(n):
            path = path_to(parents, i)
            acc = torch.ones(VH, dtype=cum_g.dtype, device=cum_g.device)
            for j in reversed(path):
                d[:, i, j] = acc
                acc = acc * eg[:, j]
    else:
        raise ValueError(mode)
    return d


def build_system(ops: Operands, *, ratio_mode: str = "expdiff") -> System:
    n = len(ops.parents)
    strict = strict_mask(ops.parents, ops.h0.device)
    visible = visible_mask(ops.parents, ops.h0.device)
    cum_g = cum_gates(ops)
    decay = decay_ratios(cum_g, ops.g, ops.parents, visible, ratio_mode)
    kx = ops.expand_k()  # (N, VH, DK)
    kk = torch.einsum("ihk,jhk->hij", kx, kx)  # (VH, N, N)
    beta = ops.beta.t()  # (VH, N)
    G = beta[:, :, None] * decay * kk * strict.to(kk.dtype)[None]
    P = torch.exp(cum_g.t())  # (VH, N)
    s0k = torch.einsum("hvk,ihk->hiv", ops.h0, kx)  # (VH, N, DV): S0 k_i
    R = beta[:, :, None] * (ops.v.permute(1, 0, 2) - P[:, :, None] * s0k)
    return System(G=G, R=R, cum_g=cum_g, decay=decay, strict=strict, visible=visible, ratio_mode=ratio_mode)


def solve_forward_substitution(sys_: System, parents: list[int]) -> torch.Tensor:
    """Mechanism B solver: U_i = R_i - sum_{j in strict ancestors(i)} G[i,j] U_j, in topological order.
    Returns U (VH, N, DV)."""
    VH, n, DV = sys_.R.shape
    U = torch.empty_like(sys_.R)
    for i in range(n):
        acc = sys_.R[:, i, :].clone()
        for j in ancestors(parents, i):
            acc = acc - sys_.G[:, i, j][:, None] * U[:, j, :]
        U[:, i, :] = acc
    return U


def solve_neumann(sys_: System, parents: list[int], *, terms: int | None = None, matmul_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Mechanism C solver: U = sum_{m=0}^{d} (-G)^m R with Z <- -G Z (Bole eq. 6 / IV-B iteration).
    d = max strict-ancestor count (G^{d+1} = 0). `matmul_dtype` lets the product run at a lower
    precision (e.g. bfloat16) as a sensitivity variant; accumulation stays in the working dtype."""
    d = max_strict_ancestors(parents) if terms is None else terms
    G, R = sys_.G, sys_.R
    if matmul_dtype is not None and matmul_dtype != G.dtype:
        Gm = G.to(matmul_dtype)
        Z = R.clone()
        U = R.clone()
        for _ in range(d):
            Z = -(torch.bmm(Gm, Z.to(matmul_dtype))).to(R.dtype)
            U = U + Z
        return U
    Z = R.clone()
    U = R.clone()
    for _ in range(d):
        Z = -torch.bmm(G, Z)
        U = U + Z
    return U


def reconstruct_all_states(ops: Operands, sys_: System, U: torch.Tensor) -> torch.Tensor:
    """S_i = P_i S0 + sum_{j <= i} (P_i/P_j) U_j k_j^T for every node. (N, VH, DV, DK)."""
    n = len(ops.parents)
    P = torch.exp(sys_.cum_g.t())  # (VH, N)
    kx = ops.expand_k().permute(1, 0, 2)  # (VH, N, DK)
    W = sys_.decay[:, :, :, None] * U[:, None, :, :]  # (VH, i, j, DV) weights (P_i/P_j) U_j on visible
    corr = torch.einsum("hijv,hjk->ihvk", W, kx)  # (N, VH, DV, DK)
    base = P.t()[:, :, None, None] * ops.h0[None]  # (N, VH, DV, DK)
    return base + corr


def outputs_from_factors(ops: Operands, sys_: System, U: torch.Tensor) -> torch.Tensor:
    """o_i = P_i (S0 q_i) + sum_{j <= i} (P_i/P_j) U_j <k_j, q_i>  -- verifier output without forming S_i."""
    P = torch.exp(sys_.cum_g.t())  # (VH, N)
    kx = ops.expand_k()  # (N, VH, DK)
    qx = ops.expand_q()
    s0q = torch.einsum("hvk,ihk->hiv", ops.h0, qx)  # (VH, N, DV)
    qk = torch.einsum("ihk,jhk->hij", qx, kx)  # (VH, i, j) <q_i, k_j>
    C = sys_.decay * qk  # visible-masked by decay
    return (P[:, :, None] * s0q + torch.bmm(C, U)).permute(1, 0, 2)


def commit_compact(ops: Operands, sys_: System, U: torch.Tensor, path: list[int]) -> torch.Tensor:
    """Compact committed-state reconstruction for accepted path `path` (root first, last = a):
    S_a = P_a S0 + sum_{j in path} (P_a/P_j) U_j k_j^T.  (VH, DV, DK)"""
    a = path[-1]
    P_a = torch.exp(sys_.cum_g[a])  # (VH,)
    kx = ops.expand_k()
    S = P_a[:, None, None] * ops.h0
    for j in path:
        w = sys_.decay[:, a, j]  # (VH,)
        S = S + (w[:, None] * U[:, j, :])[:, :, None] * kx[j][:, None, :]
    return S


# ----------------------------------------------------------------------------- metrics
def _fp32_ordinal(x: torch.Tensor) -> torch.Tensor:
    """Monotone ordered-integer mapping of fp32: negative floats map to -(magnitude bits), so the
    ordinal is monotone in the real value and +0/-0 share ordinal 0 (signed-zero policy: equal).
    Non-finite values are NOT excluded here; compare() counts them separately and masks them out."""
    b = x.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(b >= 0, b, -(b + (1 << 31)))


def ulp_distance_fp32(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (_fp32_ordinal(x.float()) - _fp32_ordinal(ref.float())).abs()


def _bf16_ordinal(x: torch.Tensor) -> torch.Tensor:
    b = x.contiguous().view(torch.int16).to(torch.int64)
    return torch.where(b >= 0, b, -(b + (1 << 15)))


def ulp_distance_bf16(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (_bf16_ordinal(x.to(torch.bfloat16)) - _bf16_ordinal(ref.to(torch.bfloat16))).abs()


def compare(x: torch.Tensor, ref: torch.Tensor, *, ulp: str | None = None) -> dict:
    """Error of candidate x vs reference ref (ref usually fp64 oracle). Elementwise stats in fp64.
    ulp='fp32': x rounded to fp32 vs ref rounded to fp32; ulp='bf16' likewise at bf16."""
    if tuple(x.shape) != tuple(ref.shape):
        raise ValueError(f"compare(): shape mismatch candidate {tuple(x.shape)} vs reference {tuple(ref.shape)}")
    xd, rd = x.double(), ref.double()
    finite = torch.isfinite(xd) & torch.isfinite(rd)
    nonfinite_x = int((~torch.isfinite(xd)).sum().item())
    nonfinite_ref = int((~torch.isfinite(rd)).sum().item())
    n_total = int(xd.numel())
    if nonfinite_x or nonfinite_ref:
        # non-finite entries are reported, then EXCLUDED as PAIRS from every statistic below
        # (a masked pair is neither "exact" nor part of ref_max); nan if no finite pair remains
        xd, rd = xd[finite], rd[finite]
    n_finite = int(xd.numel())
    if n_finite == 0:
        nan = float("nan")
        res = {"nonfinite_candidate": nonfinite_x, "nonfinite_reference": nonfinite_ref, "n": n_total, "n_finite": 0,
               "max_abs": nan, "rms": nan, "ref_max_abs": nan, "max_rel_to_refmax": nan, "rms_rel": nan,
               "ulp_significant_frac": nan, "exact_frac_all": nan}
        if ulp == "fp32":
            res.update({"ulp32_max_sig": None, "ulp32_mean_sig": nan, "ulp32_frac_ge1_sig": nan, "ulp32_frac_ge4_sig": nan, "ulp32_frac_ge64_sig": nan})
        elif ulp == "bf16":
            res.update({"ulp16_max_sig": None, "ulp16_mean_sig": nan, "ulp16_frac_ge1_sig": nan, "ulp16_frac_ge2_sig": nan})
        return res
    diff = (xd - rd)
    refmax = rd.abs().max().item()
    res = {
        "nonfinite_candidate": nonfinite_x, "nonfinite_reference": nonfinite_ref, "n_finite": n_finite,
        "max_abs": diff.abs().max().item(),
        "rms": diff.pow(2).mean().sqrt().item(),
        "ref_max_abs": refmax,
        "max_rel_to_refmax": (diff.abs().max().item() / refmax) if refmax > 0 else float("nan"),
        "rms_rel": (diff.pow(2).mean().sqrt() / (rd.pow(2).mean().sqrt() + 1e-300)).item(),
        "n": n_total,
    }
    # ULP statistics are restricted to SIGNIFICANT elements (|ref| >= 2^-8 * max|ref|): ordinal distance
    # on near-zero/denormal entries is dominated by magnitude, not by the mechanism, and would swamp the max.
    sig = rd.abs() >= (refmax * 2.0 ** -8)
    res["ulp_significant_frac"] = sig.double().mean().item()
    # exact_frac_all = NUMERIC equality at the stated width (ordinal distance 0; +0 and -0 fold together).
    # bitwise_frac   = INTEGER-VIEW equality at the stated width (signed zeros differ). Both over finite pairs.
    if ulp == "fp32":
        u = ulp_distance_fp32(xd.float(), rd.float())
        us = u[sig] if sig.any() else u
        bw = (xd.float().contiguous().view(torch.int32) == rd.float().contiguous().view(torch.int32))
        res.update({"ulp32_max_sig": int(us.max().item()), "ulp32_mean_sig": us.double().mean().item(),
                    "ulp32_frac_ge1_sig": (us >= 1).double().mean().item(), "ulp32_frac_ge4_sig": (us >= 4).double().mean().item(),
                    "ulp32_frac_ge64_sig": (us >= 64).double().mean().item(),
                    "exact_frac_all": (u == 0).double().mean().item(), "bitwise_frac": bw.double().mean().item()})
    elif ulp == "bf16":
        u = ulp_distance_bf16(xd, rd)
        us = u[sig] if sig.any() else u
        bw = (xd.to(torch.bfloat16).contiguous().view(torch.int16) == rd.to(torch.bfloat16).contiguous().view(torch.int16))
        res.update({"ulp16_max_sig": int(us.max().item()), "ulp16_mean_sig": us.double().mean().item(),
                    "ulp16_frac_ge1_sig": (us >= 1).double().mean().item(), "ulp16_frac_ge2_sig": (us >= 2).double().mean().item(),
                    "exact_frac_all": (u == 0).double().mean().item(), "bitwise_frac": bw.double().mean().item()})
    return res


def gate_stats(ops: Operands, sys_: System) -> dict:
    cum = sys_.cum_g
    P = torch.exp(cum.double())
    vis = sys_.visible
    dec = sys_.decay.double()
    dvis = dec[:, vis]  # visible pairs only
    return {
        "g_min": ops.g.min().item(), "g_max": ops.g.max().item(),
        "beta_min": ops.beta.min().item(), "beta_max": ops.beta.max().item(),
        "cum_g_min": cum.min().item(), "cum_g_max": cum.max().item(),
        "P_min": P.min().item(), "P_max": P.max().item(),
        "decay_ratio_min_visible": dvis.min().item(), "decay_ratio_max_visible": dvis.max().item(),
        "fp32_underflow_risk": bool(P.min().item() < 1.2e-38),
        "max_depth": max_strict_ancestors(ops.parents),
        "ratio_mode": sys_.ratio_mode,
    }


def all_accept_paths(parents: list[int]) -> list[list[int]]:
    """Every valid committed prefix: root-only (zero drafts accepted) and root..node for each node."""
    return [path_to(parents, i) for i in range(len(parents))]


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)} {o.dtype}>"
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o
