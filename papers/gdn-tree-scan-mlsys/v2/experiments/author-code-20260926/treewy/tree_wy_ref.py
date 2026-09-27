# SPDX-License-Identifier: Apache-2.0
"""
Tree-WY reference for the gated delta rule (Gated DeltaNet) speculative-tree
verification.  PyTorch, dtype/layout-matched to vLLM's GDN forward
(``chunk_gated_delta_rule``) so it can serve as ground truth for the fused
tree-mask kernel and the reconstruct-on-commit rollback .

This is the torch port of the validated NumPy prototype, generalized to:
  * log-space gating ``g`` (alpha_t = exp(g_t)), as the FLA kernels use;
  * grouped heads (vectorized over H);
  * state layout ``[H, V, K]`` (value-dim x key-dim), matching ``initial_state``;
  * optional L2-norm of q,k and a query ``scale`` (the kernel's knobs).

A **chain** (parent = path) reduces to the ordinary gated delta rule, which is
what ``chunk_gated_delta_rule`` computes; a **tree** (``parent[t] < t``) swaps
the dense causal mask for the strict-ancestor mask.  See README.md /
the WY/UT transform of the gated delta rule for the math.

NOTE: reference clarity over speed.  O(N^2) dense masks; fine for spec trees
(N ~ 16-64) and for testing; the fused Triton kernel is the production path.
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_ancestor_masks(parent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict-ancestor and inclusive-ancestor masks from a parent array.

    parent: long tensor [N], parent[t] < t, root = -1.
    Returns (anc_strict, anc_incl), each [N, N] float, with [t, i] = 1 iff
    i is a (strict / inclusive) ancestor of t.  Strictly-lower-triangular by
    construction (parent[t] < t), so the WY system is unit-lower-triangular.
    """
    n = parent.shape[0]
    anc_incl = torch.zeros(n, n, dtype=torch.float32, device=parent.device)
    for t in range(n):
        j = t
        while j != -1:
            anc_incl[t, j] = 1.0
            j = int(parent[j].item())
    anc_strict = anc_incl.clone()
    anc_strict[range(n), range(n)] = 0.0
    return anc_strict, anc_incl


def path_cumulative_decay(g: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    """Inclusive path-cumulative log-decay G_cum[t] = sum_{i <= t} g_i along
    the root->t path.  g: [N, H] (log space).  Returns [N, H]."""
    n, h = g.shape
    gcum = torch.empty_like(g)
    for t in range(n):
        p = int(parent[t].item())
        gcum[t] = g[t] + (gcum[p] if p != -1 else 0.0)
    return gcum


class TreeWYResult:
    """Holds the single-solve outputs so the accepted state can be committed."""

    def __init__(self, output, vtilde, gcum, anc_incl, k, initial_state):
        self.output = output  # [N, H, V]  -- all node outputs
        self.vtilde = vtilde  # [H, N, V]  -- pseudo-values (the cache)
        self._gcum = gcum  # [N, H]
        self._anc_incl = anc_incl  # [N, N]
        self._k = k  # [N, H, K]  (post-l2norm)
        self._s0 = initial_state  # [H, V, K]

    def reconstruct_state(self, accepted_node: int) -> torch.Tensor:
        """Committed state S_a for the accepted node, rebuilt from vtilde:
            S_a = g_a S0 + sum_{i <= a} (g_a/g_i) vtilde_i k_i^T .
        Returns [H, V, K]."""
        a = accepted_node
        gcum, k = self._gcum, self._k
        # decay weights g_a / g_i along a's inclusive ancestor set, else 0
        w = self._anc_incl[a].unsqueeze(0) * torch.exp(
            (gcum[a].unsqueeze(1) - gcum.transpose(0, 1)).clamp(max=0.0)
        )  # [H, N]; clamp: masked (non-ancestor) entries can be >0 -> exp overflow
        kperm = k.permute(1, 0, 2)  # [H, N, K]
        delta = torch.einsum("hi,hiv,hik->hvk", w, self.vtilde, kperm)
        return torch.exp(gcum[a]).view(-1, 1, 1) * self._s0 + delta


def tree_wy_gated_delta_rule(
    q: torch.Tensor,  # [N, H, K]
    k: torch.Tensor,  # [N, H, K]
    v: torch.Tensor,  # [N, H, V]
    g: torch.Tensor,  # [N, H]  (log-space decay; alpha = exp(g))
    beta: torch.Tensor,  # [N, H]
    parent: torch.Tensor,  # [N]     long, parent[t] < t, root = -1
    initial_state: torch.Tensor,  # [H, V, K]
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> TreeWYResult:
    """Single-pass tree-WY verification of one draft tree (H heads).

    Returns a TreeWYResult with all node outputs and the pseudo-values; call
    .reconstruct_state(a) to commit the accepted node's recurrent state.
    """
    n, h, kdim = k.shape
    vdim = v.shape[-1]
    dev, dt = k.device, torch.float32
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    initial_state = initial_state.to(dt)
    if scale is None:
        scale = kdim**-0.5
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * scale

    anc_strict, anc_incl = build_ancestor_masks(parent)
    gcum = path_cumulative_decay(g, parent)  # [N, H]
    gcumT = gcum.transpose(0, 1)  # [H, N]

    # pairwise decay ratio g_t/g_i  ->  [H, N, N]
    dec = torch.exp((gcumT.unsqueeze(2) - gcumT.unsqueeze(1)).clamp(max=0.0))
    KK = torch.einsum("nhk,mhk->hnm", k, k)  # [H, N, N], KK[h,t,i]=k_t.k_i
    G = anc_strict.unsqueeze(0) * dec * KK  # [H, N, N]
    betaT = beta.transpose(0, 1)  # [H, N]
    M = torch.eye(n, device=dev, dtype=dt).unsqueeze(0) + betaT.unsqueeze(2) * G

    Sk = torch.einsum("hvk,nhk->hnv", initial_state, k)  # [H, N, V]  (S0 k_t)
    R = (
        betaT.unsqueeze(2) * v.permute(1, 0, 2)
        - (betaT * torch.exp(gcumT)).unsqueeze(2) * Sk
    )  # [H, N, V]
    vtilde = torch.linalg.solve(M, R)  # [H, N, V]

    base = torch.exp(gcumT).unsqueeze(2) * torch.einsum(
        "hvk,nhk->hnv", initial_state, q
    )  # [H,N,V]
    KQ = torch.einsum("nhk,mhk->hnm", k, q)  # [H, N, N], KQ[h,i,t]=k_i.q_t
    Co = anc_incl.unsqueeze(0) * dec * KQ.transpose(1, 2)  # [H,N,N] -> [h,t,i]
    out = (base + torch.einsum("hti,hiv->htv", Co, vtilde)).permute(1, 0, 2)  # [N,H,V]

    return TreeWYResult(out, vtilde, gcum, anc_incl, k, initial_state)


# ---------------------------------------------------------------------------
# Ground-truth: naive per-node recurrence over the tree (for tests).
# ---------------------------------------------------------------------------
def naive_tree_recurrence(
    q,
    k,
    v,
    g,
    beta,
    parent,
    initial_state,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
):
    """Per-node gated delta rule over the tree. Returns (outputs [N,H,V],
    states list of [H,V,K]) -- the ground truth the tree-WY form must match."""
    n, h, kdim = k.shape
    vdim = v.shape[-1]
    dt = torch.float32
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    s0 = initial_state.to(dt)
    if scale is None:
        scale = kdim**-0.5
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * scale
    alpha = torch.exp(g)  # [N, H]
    eye = torch.eye(kdim, device=k.device, dtype=dt)

    states = [None] * n
    out = torch.zeros(n, h, vdim, dtype=dt, device=k.device)
    for t in range(n):
        p = int(parent[t].item())
        spar = s0 if p == -1 else states[p]  # [H, V, K]
        kt = k[t]  # [H, K]
        T = eye.unsqueeze(0) - beta[t].view(h, 1, 1) * torch.einsum(
            "hk,hj->hkj", kt, kt
        )  # [H, K, K]
        vk = torch.einsum("hv,hk->hvk", v[t], kt)  # [H, V, K]
        st = (
            alpha[t].view(h, 1, 1) * torch.einsum("hvk,hkj->hvj", spar, T)
            + beta[t].view(h, 1, 1) * vk
        )
        states[t] = st
        out[t] = torch.einsum("hvk,hk->hv", st, q[t])
    return out, states


# ---------------------------------------------------------------------------
# BATCHED tree-WY over B sequences, each a variable-size tree (padded
# to a common N).  GPU-runnable; this is the call target for the rollback
# and the validation target for the fused Triton kernel .
# ---------------------------------------------------------------------------
def build_batched_masks(parent: torch.Tensor, valid: torch.Tensor):
    """parent [B, N] (parent[t] < t within each row, root/pad = -1),
    valid [B, N] bool.  Returns anc_strict, anc_incl [B, N, N], gcum-ready.
    Invalid (pad) nodes get all-zero mask rows/cols and never appear as
    ancestors, so they don't perturb the solve."""
    b, n = parent.shape
    anc_incl = torch.zeros(b, n, n, dtype=torch.float32, device=parent.device)
    for bi in range(b):
        for t in range(n):
            if not bool(valid[bi, t]):
                continue
            j = t
            while j != -1:
                anc_incl[bi, t, j] = 1.0
                j = int(parent[bi, j].item())
    anc_strict = anc_incl.clone()
    idx = torch.arange(n, device=parent.device)
    anc_strict[:, idx, idx] = 0.0
    return anc_strict, anc_incl


def path_cumulative_decay_batched(g, parent, valid):
    """g [B, N, H] log-space -> gcum [B, N, H] along each row's tree."""
    b, n, h = g.shape
    gcum = torch.zeros_like(g)
    for bi in range(b):
        for t in range(n):
            if not bool(valid[bi, t]):
                continue
            p = int(parent[bi, t].item())
            gcum[bi, t] = g[bi, t] + (gcum[bi, p] if p != -1 else 0.0)
    return gcum


def prepare_tree_wy(parent: torch.Tensor, valid: torch.Tensor, g: torch.Tensor):
    """Vectorized, sync-free construction of the topology-dependent quantities
    (done ONCE per draft, not per layer):

      anc_incl = (I - P)^-1  where P[b,t,parent[t]]=1  -> ancestor-or-self mask
      anc_strict = anc_incl - I
      gcum = anc_incl @ g    (path-cumulative log-decay)

    P is strictly lower-triangular (parent[t] < t) hence (I-P) is unit-lower-
    triangular and invertible by a triangular solve; the entries are 0/1 because
    each node has a unique path to the root.  No Python loops / .item() syncs.

    parent,valid: [B,N]; g: [B,N,H].  Returns anc_strict,anc_incl [B,N,N],
    gcumP [B,H,N].
    """
    b, n = parent.shape
    dev = parent.device
    P = torch.zeros(b, n, n, dtype=torch.float32, device=dev)
    m = valid & (parent >= 0)
    bi, ti = m.nonzero(as_tuple=True)
    P[bi, ti, parent[bi, ti]] = 1.0
    eye = torch.eye(n, dtype=torch.float32, device=dev).expand(b, n, n)
    anc_incl = torch.linalg.solve_triangular(
        eye - P, eye, upper=False, unitriangular=True
    )
    anc_incl = anc_incl * valid.to(anc_incl.dtype).unsqueeze(2)  # zero pad rows
    diag = torch.diagonal(anc_incl, dim1=1, dim2=2)
    anc_strict = anc_incl - torch.diag_embed(diag)
    gcum = torch.einsum("bti,bih->bth", anc_incl, g.float())  # [B,N,H]
    return anc_strict, anc_incl, gcum.permute(0, 2, 1)  # gcumP [B,H,N]


# --- Increment-1 topology cache --------------------------------------------
# The ancestor masks (anc_strict/anc_incl) depend ONLY on the (fixed) draft-tree
# parent, but prepare_tree_wy reran the solve_triangular+nonzero+index_put that
# builds them every GDN layer, every step (48x/step). cuSOLVER's triangular solve
# has high fixed launch overhead, so that redundancy is a real slice of the
# per-step CPU. Cache the single-topology [N,N] masks keyed by the IDENTITY of a
# stable parent tensor (the caller holds it across steps: chain parent cached by
# t below; tree parent cached on the layer). gcum stays per-step (depends on g).
_ANC_CACHE: dict = {}
_CHAIN_PARENT: dict = {}


def _chain_parent(t: int, device) -> torch.Tensor:
    """Stable [t] chain parent [-1,0,1,..,t-2], cached so its id() is invariant
    across steps (-> _anc_for cache hits)."""
    key = (t, str(device))
    p = _CHAIN_PARENT.get(key)
    if p is None:
        p = torch.tensor([-1] + list(range(t - 1)), device=device, dtype=torch.long)
        _CHAIN_PARENT[key] = p
    return p


def _anc_for(parent_base: torch.Tensor):
    """Cached strict/inclusive ancestor masks for ONE topology.
    parent_base: [N] long, parent[t] < t, root = -1 (valid = all nodes).
    Returns (anc_strict [N,N], anc_incl [N,N]). Identical construction to
    prepare_tree_wy's (validated), just computed once per topology."""
    # Key on the topology CONTENTS, not id(parent_base): a Python object id is
    # reused after the tensor is garbage-collected, so a new topology could land on
    # a freed address and silently receive another topology's ancestor masks. That
    # is order-dependent, which made it show up only when other tests had already
    # allocated and freed tensors.
    key = tuple(parent_base.tolist())
    c = _ANC_CACHE.get(key)
    if c is not None:
        return c
    n = parent_base.shape[0]
    dev = parent_base.device
    P = torch.zeros(n, n, dtype=torch.float32, device=dev)
    ti = (parent_base >= 0).nonzero(as_tuple=True)[0]
    P[ti, parent_base[ti]] = 1.0
    eye = torch.eye(n, dtype=torch.float32, device=dev)
    anc_incl = torch.linalg.solve_triangular(
        eye - P, eye, upper=False, unitriangular=True
    )  # [N,N]
    anc_strict = anc_incl - torch.diag_embed(torch.diagonal(anc_incl))
    _ANC_CACHE[key] = (anc_strict, anc_incl)
    return anc_strict, anc_incl


class BatchedTreeWYResult:
    def __init__(self, output, vtilde, gcumP, anc_incl, k, s0):
        self.output = output  # [B, N, H, V]
        self.vtilde = vtilde  # [B, H, N, V]
        self._gcumP = gcumP  # [B, H, N]
        self._anc_incl = anc_incl  # [B, N, N]
        self._k = k  # [B, N, H, K] (post-l2norm)
        self._s0 = s0  # [B, H, V, K]

    def reconstruct_states(self, accepted: torch.Tensor) -> torch.Tensor:
        """Committed state per sequence for its accepted node.
        accepted [B] long. Returns [B, H, V, K]."""
        b = accepted.shape[0]
        bi = torch.arange(b, device=accepted.device)
        incl_a = self._anc_incl[bi, accepted]  # [B, N]
        gcum_a = self._gcumP[bi, :, accepted]  # [B, H]
        w = incl_a.unsqueeze(1) * torch.exp(
            (gcum_a.unsqueeze(2) - self._gcumP).clamp(max=0.0)
        )  # [B, H, N]
        kperm = self._k.permute(0, 2, 1, 3)  # [B, H, N, K]
        delta = torch.einsum("bhi,bhiv,bhik->bhvk", w, self.vtilde, kperm)
        return torch.exp(gcum_a).unsqueeze(-1).unsqueeze(-1) * self._s0 + delta

    def reconstruct_all_states(self) -> torch.Tensor:
        """ALL nodes' committed states at once, [B, N, H, V, K]. Used when the
        next step may read any draft position (reconstruct rollback)."""
        gp = self._gcumP  # [B, H, N]
        # W[b,h,t,i] = anc_incl[t,i] * exp((gcum_t - gcum_i) clamped <= 0)
        dec = torch.exp((gp.unsqueeze(3) - gp.unsqueeze(2)).clamp(max=0.0))  # [B,H,N,N]
        w = self._anc_incl.unsqueeze(1) * dec  # [B, H, N(t), N(i)]
        kperm = self._k.permute(0, 2, 1, 3)  # [B, H, N, K]
        delta = torch.einsum("bhti,bhiv,bhik->bhtvk", w, self.vtilde, kperm)
        base = torch.exp(gp).unsqueeze(-1).unsqueeze(-1) * self._s0.unsqueeze(2)
        return (base + delta).permute(0, 2, 1, 3, 4)  # [B, N, H, V, K]


def _tree_wy_core(
    q, k, v, beta, anc_strict, anc_incl, gcumP, initial_state, scale, use_qk_l2norm
):
    """C2-lite compile boundary: the pure-tensor per-step math. Returns plain
    tensors (out, vtilde, k_normed, s0) so torch.compile traces the whole graph
    without breaking on a dataclass construction. Identical math to the eager
    body it replaced; `scale` is resolved by the caller (a float, not None)."""
    n = k.shape[1]
    dt = torch.float32
    q, k, v, beta = (t.to(dt) for t in (q, k, v, beta))
    s0 = initial_state.to(dt)
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * scale

    dec = torch.exp(
        (gcumP.unsqueeze(3) - gcumP.unsqueeze(2)).clamp(max=0.0)
    )  # [B,H,N,N]

    KK = torch.einsum("bnhk,bmhk->bhnm", k, k)  # [B,H,N,N]
    G = anc_strict.unsqueeze(1) * dec * KK
    betaP = beta.permute(0, 2, 1)  # [B,H,N]
    eye = torch.eye(n, device=k.device, dtype=dt).view(1, 1, n, n)
    M = eye + betaP.unsqueeze(3) * G  # [B,H,N,N]

    Sk = torch.einsum("bhvk,bnhk->bhnv", s0, k)  # [B,H,N,V]
    R = (
        betaP.unsqueeze(3) * v.permute(0, 2, 1, 3)
        - (betaP * torch.exp(gcumP)).unsqueeze(3) * Sk
    )  # [B,H,N,V]
    bsz, h = k.shape[0], k.shape[2]
    # M is UNIT lower-triangular (parent[t] < t) -> triangular solve, no pivoting.
    vt = torch.linalg.solve_triangular(
        M.reshape(bsz * h, n, n),
        R.reshape(bsz * h, n, -1),
        upper=False,
        unitriangular=True,
    ).reshape(bsz, h, n, -1)

    base = torch.exp(gcumP).unsqueeze(3) * torch.einsum("bhvk,bnhk->bhnv", s0, q)
    KQ = torch.einsum("bnhk,bmhk->bhnm", k, q)  # [b,h,i,t]
    Co = anc_incl.unsqueeze(1) * dec * KQ.transpose(2, 3)  # [b,h,t,i]
    out = (base + torch.einsum("bhti,bhiv->bhtv", Co, vt)).permute(0, 2, 1, 3)
    return out, vt, k, s0


def tree_wy_compute(
    q,
    k,
    v,
    beta,
    anc_strict,
    anc_incl,
    gcumP,
    initial_state,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> BatchedTreeWYResult:
    """The per-verify-step GPU work (the kernel-equivalent), given the
    precomputed topology (anc_strict, anc_incl, gcumP from prepare_tree_wy).
    q,k: [B,N,H,K]  v: [B,N,H,V]  beta: [B,N,H]  initial_state: [B,H,V,K]."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    out, vt, k_n, s0 = _tree_wy_core(
        q, k, v, beta, anc_strict, anc_incl, gcumP, initial_state, scale, use_qk_l2norm
    )
    return BatchedTreeWYResult(out, vt, gcumP, anc_incl, k_n, s0)


def tree_wy_batched(
    q,
    k,
    v,
    g,
    beta,
    parent_base,
    initial_state,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> BatchedTreeWYResult:
    """Convenience wrapper: CACHED topology (anc masks, once per topology) +
    per-step gcum + compute. parent_base is the stable 1-D topology ([N], the
    same object across steps so _anc_for hits). valid is implicitly all-ones
    (every spec node is real in the GDN path). The B rows of the old [B,N,N]
    masks were identical, so the cached [N,N] masks are broadcast (expand) to B."""
    B = q.shape[0]
    anc_s1, anc_i1 = _anc_for(parent_base)  # [N,N] cached
    anc_strict = anc_s1.unsqueeze(0).expand(B, -1, -1)  # view, B copies of one topo
    anc_incl = anc_i1.unsqueeze(0).expand(B, -1, -1)
    gcum = torch.einsum("ti,bih->bth", anc_i1, g.float())  # [B,N,H] (per-step)
    gcumP = gcum.permute(0, 2, 1)  # [B,H,N]
    return tree_wy_compute(
        q, k, v, beta, anc_strict, anc_incl, gcumP, initial_state, scale, use_qk_l2norm
    )


# ---------------------------------------------------------------------------
# reconstruct committed states for a batch of equal-length spec CHAINS
# packed as the GDN forward provides them.  This is the call target for the
# reconstruct rollback (the wiring captures s0 + slices these tensors, calls this,
# and writes/compares the result).  Conventions are the B200-confirmed ones:
# state [H,V,K] (no transpose), grouped heads via repeat_interleave (v-head h ->
# k-head h//2), g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b), l2norm q,k.
# ---------------------------------------------------------------------------
def reconstruct_chains_result(
    query_spec: torch.Tensor,  # [R*T, num_k_heads, head_k_dim]  (post-conv)
    key_spec: torch.Tensor,  # [R*T, num_k_heads, head_k_dim]
    value_spec: torch.Tensor,  # [R*T, num_v_heads, head_v_dim]
    a_spec: torch.Tensor,  # [R*T, num_v_heads]
    b_spec: torch.Tensor,  # [R*T, num_v_heads]
    A_log: torch.Tensor,  # [num_v_heads]
    dt_bias: torch.Tensor,  # [num_v_heads]
    s0: torch.Tensor,  # [R, num_v_heads, head_v_dim, head_k_dim]  (init per seq)
    spec_len: int,  # tokens per sequence T (uniform chains)
    scale: float | None = None,
    parent: torch.Tensor | None = None,  # [T] or [R,T] lifted tree parent; None=chain
) -> "BatchedTreeWYResult":
    """Build the tree-WY result for a batch of equal-length spec chains. Call
    .reconstruct_states(accepted-1) for the committed state, or
    .reconstruct_all_states() for every draft position (reconstruct rollback).

    ``parent`` is the per-position tree parent (DFS pre-order, parent[t] < t,
    root = -1). When None it defaults to a sequential chain [-1,0,1,..,T-2]
    (ordinary gated delta rule); for tree spec-decode pass the lifted tree
    parent so the strict-ancestor masks / path-cumulative decay match the
    actual draft tree instead of a chain."""
    r = s0.shape[0]
    t = spec_len
    nk, dk = query_spec.shape[1], query_spec.shape[2]
    nv, dv = value_spec.shape[1], value_spec.shape[2]
    rep = nv // nk
    g = -A_log.float().exp() * F.softplus(a_spec.float() + dt_bias.float())
    beta = b_spec.sigmoid()
    qb = query_spec.view(r, t, nk, dk).repeat_interleave(rep, dim=2)
    kb = key_spec.view(r, t, nk, dk).repeat_interleave(rep, dim=2)
    vb = value_spec.view(r, t, nv, dv)
    gb = g.view(r, t, nv)
    betab = beta.view(r, t, nv)
    # Stable 1-D topology for the anc cache (id-keyed). Chain: a t-cached tensor.
    # Tree: the caller passes a stable per-layer _parent_v (1-D); .to() is a no-op
    # (already long+device) so its id() persists across steps -> cache hits.
    if parent is None:
        parent_base = _chain_parent(t, s0.device)
    else:
        parent_base = parent.to(device=s0.device, dtype=torch.long)
        if parent_base.dim() == 2:
            parent_base = parent_base[0].contiguous()  # rows identical
    return tree_wy_batched(
        qb, kb, vb, gb, betab, parent_base, s0, scale=scale, use_qk_l2norm=True
    )


def tree_wy_reconstruct_chains(
    query_spec,
    key_spec,
    value_spec,
    a_spec,
    b_spec,
    A_log,
    dt_bias,
    s0,
    spec_len,
    num_accepted,
    scale: float | None = None,
) -> torch.Tensor:
    """Convenience: committed accepted state per seq, [R, nv, V, K]."""
    res = reconstruct_chains_result(
        query_spec,
        key_spec,
        value_spec,
        a_spec,
        b_spec,
        A_log,
        dt_bias,
        s0,
        spec_len,
        scale,
    )
    return res.reconstruct_states((num_accepted - 1).long())
