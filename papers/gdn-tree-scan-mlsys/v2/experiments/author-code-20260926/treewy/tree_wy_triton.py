# SPDX-License-Identifier: Apache-2.0
"""C2-full: hand-fused Triton kernel for the tree-WY per-verify-step forward.

WHY THIS EXISTS (the profiler verdict)
--------------------------------------
The GDN spec path is CPU-DISPATCH-bound, not GPU-compute-bound: on the B200 the
qwen_gdn_attention_core op burns ~11x more CPU than GPU and the GPU sits ~70%
idle, because the eager tree-WY forward issues ~15-20 tiny ops PER LAYER (mm/bmm/
mul/add/exp/solve_triangular/einsum...), each ~10-40us of Python+dispatch, while
the actual math is microseconds of GPU time. torch.compile cannot fix it here:
default mode only fuses launches (still dispatched from Python -> measured no-op),
and mode="reduce-overhead" crashes (cudagraph_trees cannot capture inside vLLM's
custom-op context). So the ONLY lever for dispatch reduction is collapsing the
whole per-step forward into ONE kernel launch. That is this file.

The kernel reproduces `_tree_wy_core` in tree_wy_ref.py EXACTLY (that torch fn is
the validated reference; tree_wy_ref's prototype passes at ~1e-7). One program per
(batch, value-head); N (=num spec nodes, lifted) is small (<=BN, padded to a power
of two >=16); K (key dim) and V (value dim) are the full head dims (128). The
triangular solve is done in-program by forward substitution (N unrolled steps),
which is why a single fused kernel is feasible despite the cross-row dependency.

Numerics: all matmuls use input_precision="ieee" (full fp32, NOT tf32) so the
kernel matches the torch reference to ~1e-6; validate with test_tree_wy_triton.py
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU dev box: kernel unavailable, host wrapper will raise.
    _HAS_TRITON = False


def _next_pow2_ge16(n: int) -> int:
    bn = 16
    while bn < n:
        bn *= 2
    return bn


if _HAS_TRITON:

    @triton.jit
    def _tree_wy_chain_commit_capture_vhoist_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        alog_ptr,
        dtb_ptr,
        ancs_ptr,
        anci_ptr,
        vt_ptr,
        kk_ptr,
        gc_ptr,
        kt_ptr,
        leaf_ptr,
        slots_ptr,
        ssm_ptr,
        out_ptr,
        R,
        NV,
        N,
        T,
        REP,
        scale,
        sqr,
        sqh,
        sqk,
        skr,
        skh,
        skk,
        svr,
        svh,
        svv,
        sar,
        sah,
        sbr,
        sbh,
        sasr,
        sasc,
        sair,
        saic,
        svtp,
        svth,
        svtn,
        svtv,
        skkp,
        skkn,
        skkh,
        skkk,
        sgcp,
        sgch,
        sgcn,
        sktr,
        sktc,
        ssmb,
        ssmh,
        ssmv,
        ssmk,
        sor,
        son,
        soh,
        sov,
        K: tl.constexpr,
        BV: tl.constexpr,
        NVT: tl.constexpr,
        BN: tl.constexpr,
        NUNR: tl.constexpr,
        DBF: tl.constexpr = False,
    ):
        """V-HOISTED commit+forward. Same math as
        _tree_wy_chain_commit_capture_kernel but ONE program per (seq b, value-head h)
        (grid r*nv, NOT r*nv*NVT). The V-INDEPENDENT work -- KK=k kᵀ, QKt=q kᵀ, the
        WY matrix M, Co, gc/dec, q/k L2-norm, the commit weights w and normalized
        prev-key kkp -- is computed ONCE, then an internal loop over the NVT V-tiles
        does only the V-DEPENDENT work (s0 commit, Sk, R, WY-solve vt, base, out).
        The per-program register footprint stays bounded because only ONE V-tile's
        s0[BV,K]/vt[BN,BV] is live at a time (the tiling that keeps this under the
        spill ceiling is preserved); what's removed is recomputing the V-independent
        matmuls/norms NVT times (the r*nv*NVT-grid kernel's redundancy). The
        V-independent kk/gc stash is still deferred to _tree_wy_stash_kgc_kernel
        (2nd launch). Grid (r*nv,)."""
        pid = tl.program_id(0)
        h = pid % NV
        b = pid // NV
        kg = h // REP
        base_row = b * T
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, K)
        nmask = offs_n < N

        dst = tl.load(slots_ptr + b).to(tl.int64)
        do_write = dst != 0

        # ================= V-INDEPENDENT (computed ONCE) =================
        # ---- commit weights (depend on gc/keep only) ----
        leaf = tl.load(leaf_ptr + b).to(tl.int32)
        gcp = tl.load(
            gc_ptr + dst * sgcp + h * sgch + offs_n * sgcn, mask=nmask, other=0.0
        ).to(tl.float32)
        # A block freshly (re)initialized by a prefill carries a POSITIVE sentinel in
        # gcum, which a real cumulative log-decay can never be (it sums
        # g = -exp(A_log)*softplus(...) <= 0). It means "the stash in this block
        # belongs to a previously-finished sequence" -> skip this one commit, which is
        # exactly what committing from an all-zero stash would do (exp(0)*s0 + 0 ==
        # s0). Lets the prefill path mark the tiny gcum stash instead of zeroing the
        # two large ones. Uniform per program, so this is a scalar branch.
        do_commit = tl.load(gc_ptr + dst * sgcp + h * sgch) <= 0.0
        keep = tl.load(kt_ptr + leaf * sktr + offs_n * sktc, mask=nmask, other=0.0).to(
            tl.float32
        )
        kkp = tl.load(
            kk_ptr
            + dst * skkp
            + offs_n[:, None] * skkn
            + h * skkh
            + offs_k[None, :] * skkk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        gc_leaf = tl.sum(tl.where(offs_n == leaf, gcp, 0.0))
        w = keep * tl.exp(tl.minimum(gc_leaf - gcp, 0.0))  # [BN]
        exp_gc_leaf = tl.exp(gc_leaf)

        # ---- forward setup (gates, q/k norm, masks, M, Co) ----
        a_n = tl.load(
            a_ptr + (base_row + offs_n) * sar + h * sah, mask=nmask, other=0.0
        ).to(tl.float32)
        b_n = tl.load(
            b_ptr + (base_row + offs_n) * sbr + h * sbh, mask=nmask, other=0.0
        ).to(tl.float32)
        A = tl.load(alog_ptr + h).to(tl.float32)
        dtb = tl.load(dtb_ptr + h).to(tl.float32)
        x = a_n + dtb
        softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = -tl.exp(A) * softplus
        beta = tl.sigmoid(b_n)
        q = tl.load(
            q_ptr
            + (base_row + offs_n)[:, None] * sqr
            + kg * sqh
            + offs_k[None, :] * sqk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + (base_row + offs_n)[:, None] * skr
            + kg * skh
            + offs_k[None, :] * skk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        nn = nmask[:, None] & nmask[None, :]
        ancs = tl.load(
            ancs_ptr + offs_n[:, None] * sasr + offs_n[None, :] * sasc,
            mask=nn,
            other=0.0,
        ).to(tl.float32)
        anci = tl.load(
            anci_ptr + offs_n[:, None] * sair + offs_n[None, :] * saic,
            mask=nn,
            other=0.0,
        ).to(tl.float32)
        gc = tl.sum(anci * g[None, :], axis=1)
        kinv = 1.0 / tl.maximum(tl.sqrt(tl.sum(k * k, axis=1)), 1e-12)
        k = k * kinv[:, None]
        # ---- kk/gc stash for NEXT step's commit, FOLDED IN (no 2nd launch) ----
        # Written HERE (not at kernel end) so the fp32 `k` need not stay live across
        # the V-loop (in the bf16 path it's dead after the kb cast below) -> no extra
        # register pressure on this register-bound kernel. Safe: vhoist is ONE program
        # per (seq b, head h), the SOLE writer of its (dst,h) stash slot, and the
        # commit at the top already consumed the PREV step's kk/gc into registers, so
        # overwriting with THIS step's values is a read-before-write in program order.
        # k is the L2-normalized key, gc the path-cumulative gate — exactly what the
        # separate _tree_wy_stash_kgc launch re-loaded/re-derived (folding drops that
        # launch + its redundant recompute). Bit-identical to the 2-launch write.
        if do_write:
            tl.store(
                kk_ptr
                + dst * skkp
                + offs_n[:, None] * skkn
                + h * skkh
                + offs_k[None, :] * skkk,
                k,
                mask=nmask[:, None],
            )
            tl.store(gc_ptr + dst * sgcp + h * sgch + offs_n * sgcn, gc, mask=nmask)
        qinv = 1.0 / tl.maximum(tl.sqrt(tl.sum(q * q, axis=1)), 1e-12)
        q = (q * qinv[:, None]) * scale
        dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
        eye = (offs_n[:, None] == offs_n[None, :]).to(tl.float32)
        exp_gc = tl.exp(gc)
        beta_exp_gc = beta * exp_gc
        if DBF:
            kb = k.to(tl.bfloat16)
            qb = q.to(tl.bfloat16)
            kkpb = kkp.to(tl.bfloat16)
            KK = tl.dot(kb, tl.trans(kb))
            M = eye + beta[:, None] * (ancs * dec * KK)
            Co = anci * dec * tl.dot(qb, tl.trans(kb))
        else:
            KK = tl.dot(k, tl.trans(k), input_precision="ieee")
            M = eye + beta[:, None] * (ancs * dec * KK)
            QKt = tl.dot(q, tl.trans(k), input_precision="ieee")
            Co = anci * dec * QKt

        # ================= V-DEPENDENT (loop over V-tiles) =================
        for jv in range(NVT):
            offs_v = jv * BV + tl.arange(0, BV)
            s0 = tl.load(
                ssm_ptr
                + dst * ssmb
                + h * ssmh
                + offs_v[:, None] * ssmv
                + offs_k[None, :] * ssmk
            ).to(tl.float32)  # [BV,K]
            # ---- COMMIT (skipped for a sentinel-marked, i.e. stale, stash) ----
            if do_commit:
                vtp = tl.load(
                    vt_ptr
                    + dst * svtp
                    + h * svth
                    + offs_n[:, None] * svtn
                    + offs_v[None, :] * svtv,
                    mask=nmask[:, None],
                    other=0.0,
                ).to(tl.float32)  # [BN,BV]
                Aw = w[:, None] * vtp
                if DBF:
                    delta = tl.dot(tl.trans(Aw).to(tl.bfloat16), kkpb)
                else:
                    delta = tl.dot(tl.trans(Aw), kkp, input_precision="ieee")
                s0 = exp_gc_leaf * s0 + delta
            if do_write:
                tl.store(
                    ssm_ptr
                    + dst * ssmb
                    + h * ssmh
                    + offs_v[:, None] * ssmv
                    + offs_k[None, :] * ssmk,
                    s0,
                )
            # ---- FORWARD from s0 in registers ----
            v = tl.load(
                v_ptr
                + (base_row + offs_n)[:, None] * svr
                + h * svh
                + offs_v[None, :] * svv,
                mask=nmask[:, None],
                other=0.0,
            ).to(tl.float32)  # [BN,BV]
            if DBF:
                s0b = s0.to(tl.bfloat16)
                Sk = tl.dot(kb, tl.trans(s0b))  # [BN,BV]
            else:
                Sk = tl.dot(k, tl.trans(s0), input_precision="ieee")
            R_ = beta[:, None] * v - beta_exp_gc[:, None] * Sk
            vt = tl.zeros([BN, BV], dtype=tl.float32)
            for tt in range(NUNR):
                rowm = offs_n == tt
                Mrow = tl.sum(tl.where(rowm[:, None], M, 0.0), axis=0)
                Rrow = tl.sum(tl.where(rowm[:, None], R_, 0.0), axis=0)
                acc = tl.sum(Mrow[:, None] * vt, axis=0)
                vt = tl.where(rowm[:, None], (Rrow - acc)[None, :], vt)
            if DBF:
                base = exp_gc[:, None] * tl.dot(qb, tl.trans(s0b))
                out = base + tl.dot(Co.to(tl.bfloat16), vt.to(tl.bfloat16))
            else:
                base = exp_gc[:, None] * tl.dot(q, tl.trans(s0), input_precision="ieee")
                out = base + tl.dot(Co, vt, input_precision="ieee")
            tl.store(
                out_ptr
                + b * sor
                + offs_n[:, None] * son
                + h * soh
                + offs_v[None, :] * sov,
                out,
                mask=nmask[:, None],
            )
            if do_write:
                tl.store(
                    vt_ptr
                    + dst * svtp
                    + h * svth
                    + offs_n[:, None] * svtn
                    + offs_v[None, :] * svtv,
                    vt,
                    mask=nmask[:, None],
                )
        # kk/gc stash already written pre-loop (folded; see the hoisted section) so
        # fp32 k stays dead across the V-loop -> no 2nd launch, no extra registers.

    @triton.jit(do_not_specialize=["R", "N", "T"])
    def _tree_wy_chain_commit_capture_vhoist_maskfree_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        alog_ptr,
        dtb_ptr,
        vt_ptr,
        kk_ptr,
        gc_ptr,
        leaf_ptr,
        slots_ptr,
        ssm_ptr,
        out_ptr,
        R,
        NV,
        N,
        T,
        REP,
        scale,
        sqr,
        sqh,
        sqk,
        skr,
        skh,
        skk,
        svr,
        svh,
        svv,
        sar,
        sah,
        sbr,
        sbh,
        svtp,
        svth,
        svtn,
        svtv,
        skkp,
        skkn,
        skkh,
        skkk,
        sgcp,
        sgch,
        sgcn,
        ssmb,
        ssmh,
        ssmv,
        ssmk,
        sor,
        son,
        soh,
        sov,
        K: tl.constexpr,
        BV: tl.constexpr,
        NVT: tl.constexpr,
        BN: tl.constexpr,
        NUNR: tl.constexpr,
        DBF: tl.constexpr = False,
    ):
        """MASK-FREE variant of _tree_wy_chain_commit_capture_vhoist_kernel.

        Bit-identical math, but the CHAIN ancestor masks are computed IN-KERNEL from
        index comparisons instead of being loaded from ancs_ptr/anci_ptr/kt_ptr:
            anci[i,j] = (j <= i)   inclusive lower-triangular  (== the loaded anc_i)
            ancs[i,j] = (j <  i)   strict   lower-triangular  (== the loaded anc_s)
            keep[j]   = (j <= leaf)                            (== row `leaf` of kt)
        This drops 3 pointer + 6 stride args off the launch (less Triton per-arg
        dispatch on the host-bound eager/PIECEWISE path) and the host mask build.
        It applies to a chain AND to a width-1 draft tree, whose lifted parent is
        [-1,0,1,..] -- i.e. exactly a chain, so the same triangles are correct. A
        width>1 tree needs real masks and must use the mask-taking kernel above.

        Two structural notes:
        * cudagraph-padded rows carry slot == NULL_BLOCK_ID (0) and write nothing, so
          their whole commit+forward is dead work; they emit zeros and return early.
          No real request can hold block 0 (it is popped as the reserved null block),
          so this never skips live work.
        * a sentinel in gcum marks a block whose stash belongs to a previously
          finished sequence, and skips that one commit -- see `do_commit` below.

        Dots accumulate via `tl.dot(x, y, acc)` rather than materializing the product
        and adding. Same math and (measured) the same register count -- Triton
        already fused it -- but it keeps the accumulation explicit."""
        pid = tl.program_id(0)
        h = pid % NV
        b = pid // NV
        kg = h // REP
        base_row = b * T
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, K)
        nmask = offs_n < N

        dst = tl.load(slots_ptr + b).to(tl.int64)
        do_write = dst != 0
        if not do_write:
            # padded row: emit zeros (its token slots are discarded downstream) and
            # skip the entire commit+forward.
            for jv in range(NVT):
                offs_vz = jv * BV + tl.arange(0, BV)
                tl.store(
                    out_ptr
                    + b * sor
                    + offs_n[:, None] * son
                    + h * soh
                    + offs_vz[None, :] * sov,
                    tl.zeros([BN, BV], dtype=out_ptr.dtype.element_ty),
                    mask=nmask[:, None],
                )
            return

        # ================= V-INDEPENDENT (computed ONCE) =================
        # ---- commit weights (depend on gc/keep only) ----
        leaf = tl.load(leaf_ptr + b).to(tl.int32)
        gcp = tl.load(
            gc_ptr + dst * sgcp + h * sgch + offs_n * sgcn, mask=nmask, other=0.0
        ).to(tl.float32)
        # A block freshly (re)initialized by a prefill carries a POSITIVE sentinel
        # in gcum, which a real cumulative log-decay can never be (it sums
        # g = -exp(A_log)*softplus(...) <= 0). It means "the stash in this block
        # belongs to a previously-finished sequence" -> skip this one commit, which
        # is exactly what committing from an all-zero stash would do
        # (exp(0)*s0 + 0 == s0). Uniform per program, so this is a scalar branch.
        do_commit = tl.load(gc_ptr + dst * sgcp + h * sgch) <= 0.0
        # keep == row `leaf` of the inclusive chain keep-table == (j <= leaf).
        keep = tl.where(nmask & (offs_n <= leaf), 1.0, 0.0)
        kkp = tl.load(
            kk_ptr
            + dst * skkp
            + offs_n[:, None] * skkn
            + h * skkh
            + offs_k[None, :] * skkk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        gc_leaf = tl.sum(tl.where(offs_n == leaf, gcp, 0.0))
        w = keep * tl.exp(tl.minimum(gc_leaf - gcp, 0.0))  # [BN]
        exp_gc_leaf = tl.exp(gc_leaf)

        # ---- forward setup (gates, q/k norm, masks, M, Co) ----
        a_n = tl.load(
            a_ptr + (base_row + offs_n) * sar + h * sah, mask=nmask, other=0.0
        ).to(tl.float32)
        b_n = tl.load(
            b_ptr + (base_row + offs_n) * sbr + h * sbh, mask=nmask, other=0.0
        ).to(tl.float32)
        A = tl.load(alog_ptr + h).to(tl.float32)
        dtb = tl.load(dtb_ptr + h).to(tl.float32)
        x = a_n + dtb
        softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = -tl.exp(A) * softplus
        beta = tl.sigmoid(b_n)
        q = tl.load(
            q_ptr
            + (base_row + offs_n)[:, None] * sqr
            + kg * sqh
            + offs_k[None, :] * sqk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + (base_row + offs_n)[:, None] * skr
            + kg * skh
            + offs_k[None, :] * skk,
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        nn = nmask[:, None] & nmask[None, :]
        # chain ancestor masks IN-KERNEL (row i, col j over offs_n)
        anci = tl.where(nn & (offs_n[None, :] <= offs_n[:, None]), 1.0, 0.0)
        ancs = tl.where(nn & (offs_n[None, :] < offs_n[:, None]), 1.0, 0.0)
        gc = tl.sum(anci * g[None, :], axis=1)
        kinv = 1.0 / tl.maximum(tl.sqrt(tl.sum(k * k, axis=1)), 1e-12)
        k = k * kinv[:, None]
        # ---- kk/gc stash for NEXT step's commit, FOLDED IN (no 2nd launch) ----
        if do_write:
            tl.store(
                kk_ptr
                + dst * skkp
                + offs_n[:, None] * skkn
                + h * skkh
                + offs_k[None, :] * skkk,
                k,
                mask=nmask[:, None],
            )
            tl.store(gc_ptr + dst * sgcp + h * sgch + offs_n * sgcn, gc, mask=nmask)
        qinv = 1.0 / tl.maximum(tl.sqrt(tl.sum(q * q, axis=1)), 1e-12)
        q = (q * qinv[:, None]) * scale
        dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
        eye = (offs_n[:, None] == offs_n[None, :]).to(tl.float32)
        exp_gc = tl.exp(gc)
        beta_exp_gc = beta * exp_gc
        if DBF:
            kb = k.to(tl.bfloat16)
            qb = q.to(tl.bfloat16)
            kkpb = kkp.to(tl.bfloat16)
            KK = tl.dot(kb, tl.trans(kb))
            M = eye + beta[:, None] * (ancs * dec * KK)
            Co = anci * dec * tl.dot(qb, tl.trans(kb))
        else:
            KK = tl.dot(k, tl.trans(k), input_precision="ieee")
            M = eye + beta[:, None] * (ancs * dec * KK)
            QKt = tl.dot(q, tl.trans(k), input_precision="ieee")
            Co = anci * dec * QKt

        # ================= V-DEPENDENT (loop over V-tiles) =================
        for jv in range(NVT):
            offs_v = jv * BV + tl.arange(0, BV)
            s0 = tl.load(
                ssm_ptr
                + dst * ssmb
                + h * ssmh
                + offs_v[:, None] * ssmv
                + offs_k[None, :] * ssmk
            ).to(tl.float32)  # [BV,K]
            # ---- COMMIT (skipped for a sentinel-marked, i.e. stale, stash) ----
            if do_commit:
                vtp = tl.load(
                    vt_ptr
                    + dst * svtp
                    + h * svth
                    + offs_n[:, None] * svtn
                    + offs_v[None, :] * svtv,
                    mask=nmask[:, None],
                    other=0.0,
                ).to(tl.float32)  # [BN,BV]
                Aw = w[:, None] * vtp
                if DBF:
                    s0 = tl.dot(tl.trans(Aw).to(tl.bfloat16), kkpb, exp_gc_leaf * s0)
                else:
                    s0 = tl.dot(
                        tl.trans(Aw), kkp, exp_gc_leaf * s0, input_precision="ieee"
                    )
            if do_write:
                tl.store(
                    ssm_ptr
                    + dst * ssmb
                    + h * ssmh
                    + offs_v[:, None] * ssmv
                    + offs_k[None, :] * ssmk,
                    s0,
                )
            # ---- FORWARD from s0 in registers ----
            v = tl.load(
                v_ptr
                + (base_row + offs_n)[:, None] * svr
                + h * svh
                + offs_v[None, :] * svv,
                mask=nmask[:, None],
                other=0.0,
            ).to(tl.float32)  # [BN,BV]
            if DBF:
                s0b = s0.to(tl.bfloat16)
                Sk = tl.dot(kb, tl.trans(s0b))  # [BN,BV]
            else:
                Sk = tl.dot(k, tl.trans(s0), input_precision="ieee")
            R_ = beta[:, None] * v - beta_exp_gc[:, None] * Sk
            vt = tl.zeros([BN, BV], dtype=tl.float32)
            for tt in range(NUNR):
                rowm = offs_n == tt
                Mrow = tl.sum(tl.where(rowm[:, None], M, 0.0), axis=0)
                Rrow = tl.sum(tl.where(rowm[:, None], R_, 0.0), axis=0)
                acc = tl.sum(Mrow[:, None] * vt, axis=0)
                vt = tl.where(rowm[:, None], (Rrow - acc)[None, :], vt)
            if DBF:
                out = tl.dot(
                    Co.to(tl.bfloat16),
                    vt.to(tl.bfloat16),
                    exp_gc[:, None] * tl.dot(qb, tl.trans(s0b)),
                )
            else:
                out = tl.dot(
                    Co,
                    vt,
                    exp_gc[:, None] * tl.dot(q, tl.trans(s0), input_precision="ieee"),
                    input_precision="ieee",
                )
            tl.store(
                out_ptr
                + b * sor
                + offs_n[:, None] * son
                + h * soh
                + offs_v[None, :] * sov,
                out,
                mask=nmask[:, None],
            )
            if do_write:
                tl.store(
                    vt_ptr
                    + dst * svtp
                    + h * svth
                    + offs_n[:, None] * svtn
                    + offs_v[None, :] * svtv,
                    vt,
                    mask=nmask[:, None],
                )


def tree_wy_chain_commit_capture_triton(
    query_spec,
    key_spec,
    value_spec,
    a_spec,
    b_spec,
    A_log,
    dt_bias,
    vt_stash,
    kk_stash,
    gc_stash,
    leaf_full,
    slots,
    ssm_state,
    spec_len,
    scale,
    BV=32,
    num_warps=2,
    num_stages=3,
    dot_bf16=True,
):
    """V-hoisted, mask-free FUSED commit+forward in ONE launch: one program per
    (seq, value-head) loops the V-tiles internally and folds the kk/gc stash write
    into the same launch (no ssm HBM round-trip). The chain ancestor masks are
    computed in-kernel from index comparisons (no anc_s/anc_i/kt args). bf16 dot
    operands (DBF) + fp32 accum keep it under the register-spill ceiling.

    This is the CHAIN (and width-1 tree) mixer. width>1 draft trees need real
    ancestor masks -> tree_wy_tree_commit_capture_triton. Validate against the
    torch reference via tests/kernels/mamba/test_tree_wy_chain_capture.py."""
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton unavailable; cannot run tree_wy_chain_commit_capture."
        )
    rt, nk, dk = query_spec.shape
    nv, dv = value_spec.shape[1], value_spec.shape[2]
    r = slots.shape[0]
    N = spec_len
    rep = nv // nk
    if scale is None:
        scale = dk**-0.5
    assert dv % BV == 0, f"dv={dv} not divisible by BV={BV}"
    nvt = dv // BV
    dev = query_spec.device
    out = torch.empty(r, N, nv, dv, device=dev, dtype=query_spec.dtype)
    BN = _next_pow2_ge16(N)
    _tree_wy_chain_commit_capture_vhoist_maskfree_kernel[(r * nv,)](
        query_spec,
        key_spec,
        value_spec,
        a_spec,
        b_spec,
        A_log,
        dt_bias,
        vt_stash,
        kk_stash,
        gc_stash,
        leaf_full,
        slots,
        ssm_state,
        out,
        r,
        nv,
        N,
        N,
        rep,
        float(scale),
        query_spec.stride(0),
        query_spec.stride(1),
        query_spec.stride(2),
        key_spec.stride(0),
        key_spec.stride(1),
        key_spec.stride(2),
        value_spec.stride(0),
        value_spec.stride(1),
        value_spec.stride(2),
        a_spec.stride(0),
        a_spec.stride(1),
        b_spec.stride(0),
        b_spec.stride(1),
        vt_stash.stride(0),
        vt_stash.stride(1),
        vt_stash.stride(2),
        vt_stash.stride(3),
        kk_stash.stride(0),
        kk_stash.stride(1),
        kk_stash.stride(2),
        kk_stash.stride(3),
        gc_stash.stride(0),
        gc_stash.stride(1),
        gc_stash.stride(2),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        K=dk,
        BV=BV,
        NVT=nvt,
        BN=BN,
        NUNR=N,
        DBF=dot_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def tree_wy_tree_commit_capture_triton(
    query_spec,
    key_spec,
    value_spec,
    a_spec,
    b_spec,
    A_log,
    dt_bias,
    anc_s,
    anc_i,
    vt_stash,
    kk_stash,
    gc_stash,
    kt,
    leaf_full,
    slots,
    ssm_state,
    spec_len,
    scale,
    BV=32,
    num_warps=2,
    num_stages=3,
    dot_bf16=True,
):
    """MASK-TAKING FUSED commit+forward, for a width>1 draft TREE.

    Identical math to tree_wy_chain_commit_capture_triton but the ancestor structure
    is supplied as tensors (anc_s strict / anc_i inclusive / kt keep-table) instead of
    being derived from index comparisons, so it expresses an arbitrary DFS-pre-order
    tree. A chain (or width-1 tree) should use the mask-free launcher: same result,
    fewer launch args, no host mask build.

    V-HOISTED and fused into ONE launch: one program per (seq, value-head) loops the
    V-tiles internally, computing the V-INDEPENDENT work (KK, QKt, the WY matrix M,
    Co, gc/dec, q/k L2-norm, commit weights) once instead of V/BV times, and folds
    the (kk, gcum) stash write for the next step's commit into the same launch -- it
    owns each (dst, head) slot alone, so there is no cross-V-tile race. bf16 dot
    operands with fp32 accumulation keep it under the register-spill ceiling (114
    regs / 0 spills, B200).
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton unavailable; cannot run tree_wy_tree_commit_capture."
        )
    rt, nk, dk = query_spec.shape
    nv, dv = value_spec.shape[1], value_spec.shape[2]
    r = slots.shape[0]
    N = spec_len
    rep = nv // nk
    if scale is None:
        scale = dk**-0.5
    assert dv % BV == 0, f"dv={dv} not divisible by BV={BV}"
    nvt = dv // BV
    dev = query_spec.device
    out = torch.empty(r, N, nv, dv, device=dev, dtype=query_spec.dtype)
    BN = _next_pow2_ge16(N)
    # --- launch 1/2: FUSED commit+forward (s0 in registers) -> ssm/out/vt ---
    # grid r*nv when vhoist (one program loops all V-tiles), else r*nv*nvt.
    _tree_wy_chain_commit_capture_vhoist_kernel[(r * nv,)](
        query_spec,
        key_spec,
        value_spec,
        a_spec,
        b_spec,
        A_log,
        dt_bias,
        anc_s,
        anc_i,
        vt_stash,
        kk_stash,
        gc_stash,
        kt,
        leaf_full,
        slots,
        ssm_state,
        out,
        r,
        nv,
        N,
        N,
        rep,
        float(scale),
        query_spec.stride(0),
        query_spec.stride(1),
        query_spec.stride(2),
        key_spec.stride(0),
        key_spec.stride(1),
        key_spec.stride(2),
        value_spec.stride(0),
        value_spec.stride(1),
        value_spec.stride(2),
        a_spec.stride(0),
        a_spec.stride(1),
        b_spec.stride(0),
        b_spec.stride(1),
        anc_s.stride(0),
        anc_s.stride(1),
        anc_i.stride(0),
        anc_i.stride(1),
        vt_stash.stride(0),
        vt_stash.stride(1),
        vt_stash.stride(2),
        vt_stash.stride(3),
        kk_stash.stride(0),
        kk_stash.stride(1),
        kk_stash.stride(2),
        kk_stash.stride(3),
        gc_stash.stride(0),
        gc_stash.stride(1),
        gc_stash.stride(2),
        kt.stride(0),
        kt.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        K=dk,
        BV=BV,
        NVT=nvt,
        BN=BN,
        NUNR=N,
        DBF=dot_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
