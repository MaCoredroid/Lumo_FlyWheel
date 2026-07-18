#!/usr/bin/env python3
"""FR13_PIGGYBACK V0(d) -- the decisive TWO-STEP RING INDUCTION gate.

Run INSIDE the vllm container (GPU):

    docker exec <vllm-container> python3 /workspace/scripts/fr13_piggyback_v0d_ring_induction.py

WHY THIS GATE EXISTS (FR13_PIGGYBACK_PHASE3_APPLY_REPORT.md V0(d); S1 spec):
V0(c) proved the SINGLE-step law: export@stream-7 == replay deposit when the
chain rows carry the right bytes. It CANNOT distinguish row-8-rooted from
row-0-rooted cross-step gathers (both degenerate to the same fixture in one
step). The live system composes steps: step N+1's chain rows are fed from step
N's RINGS via the B5 row-8-rooted gather (root = prev ring row 8 = the LIVE-8
bonus twin; drafts = prev ring rows of the accepted subtree streams). This
gate proves that COMPOSITION byte-exactly:

  (A) TWO-STEP INDUCTION, L1 in {0..5}: step-1 = fresh request (all-identity
      chain) on the 18-stream extended tree; its 18 per-stream rows ARE the
      staged ring (live staging = ring row i <- stream i). Accept a spine path
      of length L1 in step-1's subtree. Step-2's chain streams 1..1+L1 carry
      [step-1 ring row 8] + [step-1 ring rows of the accepted path streams]
      (the B5 gather), rest identity. ASSERT step-2's export@7 ==
      launch_tree_gdn_replay(root_node=8, step-1 ring, path) from the same
      pre-step state, BYTE-EXACT. L1=0 is the all-zero-accept case (chain =
      [row 8] only) -- S1 new-risk 1's fixture.
  (B) ROOT_NODE THREADING IDENTITY: replay(root_node=8, ring, path=streams)
      == replay(root_node=0 DEFAULT, ring re-indexed so row 0 = old row 8,
      rows 1..L1 = the path rows). Validates the S1-4(a-f) kernel/launcher
      threading against the stock gather it replaced, and doubles as the
      V0(b) codegen check for the ROOT_NODE=0 fold (the default-arg launch is
      exercised and must byte-match the pre-thread convention).
  (C) THREE-STEP COMPOUNDING (L1=2 then L2=3): a third step chained the same
      way must land on the composed replay reference -- catches any cross-step
      state pollution a single induction hop can mask.

SCOPE NOTE (deliberate): piggyback_catchup_replay's PLUMBING (module/layer
staging, dead-col-1 doctoring, rid partitioning) is exercised by the live
V2.5 interleave-stress gate with its fail-loud counters -- it requires the
full gdn-module state and is NOT mocked here. The STATE MATH it relies on
(root_node=8 replay over staged rings) is exactly checks (A)/(B).

Conventions inherited from fr13_piggyback_v0c_validate (imported): int-view
byte compares (never atol), no-exact-zero h0 (-0.0 caveat N3), scrubbed
env/sidecars, BV=8 geometry override, deployed custom replay kernel (refuses
under FR13_COMMITTER_NATIVE, N4). Ring here is 32 rows (stream ids up to 17
are ring rows; launcher requires power-of-two <= 32).
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fr13_piggyback_v0c_validate as V0C  # noqa: E402  (module-level topology asserts run)

NUM_KH, NUM_VH, DIM = V0C.NUM_KH, V0C.NUM_VH, V0C.DIM
SCALE = V0C.SCALE
N_STREAMS, N_PAD = V0C.N_STREAMS, V0C.N_PAD
CHAIN_END, SUBTREE_ROOT = V0C.CHAIN_END, V0C.SUBTREE_ROOT
IDENT = V0C.IDENT
SPEC_COLS, BANK_ROWS, COL0_ROW = V0C.SPEC_COLS, V0C.BANK_ROWS, V0C.COL0_ROW
N_RING = 32  # stream ids ARE ring rows here (max 17) -> 32 (pow2 <= 32)

# subtree spine stream ids: choice (0,)^(8+j) -> stream = choices-rank + 1
_SPINE_STREAMS = tuple(
    V0C.EXT_CHOICES.index((0,) * (8 + j)) + 1 for j in range(1, 6)
)
assert _SPINE_STREAMS[0] == 9 or V0C.STREAM_PARENTS[_SPINE_STREAMS[0]] == SUBTREE_ROOT
for _j in range(1, 5):  # each spine stream's parent is the previous spine stream
    assert V0C.STREAM_PARENTS[_SPINE_STREAMS[_j]] == _SPINE_STREAMS[_j - 1]


def _mk_step_rows(g, dev):
    """One step's per-stream tree rows (k/v/a/b for all 18 streams) + q pad."""
    def mk(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=torch.bfloat16)

    return dict(
        k=(mk(N_STREAMS, NUM_KH, DIM) * 0.3).contiguous(),
        v=(mk(N_STREAMS, NUM_VH, DIM) * 0.3).contiguous(),
        a=(mk(N_STREAMS, NUM_VH) * 0.5).contiguous(),
        b=(mk(N_STREAMS, NUM_VH) * 0.5).contiguous(),
    )


def _rows_to_ring(rows, dev):
    """Live staging convention: ring row i == tree stream i (rows 18..31 junk)."""
    g = torch.Generator(device=dev)
    g.manual_seed(97)
    ring = dict(
        k=torch.randn(1, N_RING, NUM_KH, DIM, generator=g, device=dev,
                      dtype=torch.bfloat16).contiguous() * 0.3,
        v=torch.randn(1, N_RING, NUM_VH, DIM, generator=g, device=dev,
                      dtype=torch.bfloat16).contiguous() * 0.3,
        a=torch.randn(1, N_RING, NUM_VH, generator=g, device=dev,
                      dtype=torch.bfloat16).contiguous() * 0.5,
        b=torch.randn(1, N_RING, NUM_VH, generator=g, device=dev,
                      dtype=torch.bfloat16).contiguous() * 0.5,
    )
    for f in ("k", "v", "a", "b"):
        ring[f][0, :N_STREAMS] = rows[f]
    return ring


def _scan_buffers(g, dev, rows, chain_srcs):
    """Assemble the N_PAD scan buffers for one step.

    chain_srcs: list of (stream_rows_dict, src_stream) feeding chain streams
    1..len(chain_srcs) (the B5 gather: [prev row 8] + prev accepted rows).
    Streams beyond the chain up to 7, and stream 0, are E9 identity-masked.
    Streams 8..17 take THIS step's rows (bonus twin + subtree).
    """
    def mk(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=torch.bfloat16)

    q = (mk(N_PAD, NUM_KH, DIM) * 0.3).contiguous()
    k = (mk(N_PAD, NUM_KH, DIM) * 0.3).contiguous()
    v = (mk(N_PAD, NUM_VH, DIM) * 0.3).contiguous()
    ra = (mk(N_PAD, NUM_VH) * 0.5).contiguous()
    rb = (mk(N_PAD, NUM_VH) * 0.5).contiguous()
    for i, (src_rows, src_stream) in enumerate(chain_srcs):
        s = 1 + i
        k[s] = src_rows["k"][src_stream]
        v[s] = src_rows["v"][src_stream]
        ra[s] = src_rows["a"][src_stream]
        rb[s] = src_rows["b"][src_stream]
    for s in range(SUBTREE_ROOT, N_STREAMS):
        k[s] = rows["k"][s]
        v[s] = rows["v"][s]
        ra[s] = rows["a"][s]
        rb[s] = rows["b"][s]
    for s in [0] + list(range(1 + len(chain_srcs), CHAIN_END + 1)):
        ra[s] = IDENT
        rb[s] = IDENT
    return dict(q=q, k=k, v=v, ra=ra, rb=rb)


def _run_scan(KM, buf, bank, spec, aux):
    KM.launch_tree_gdn_prepared(
        q=buf["q"], k=buf["k"], v=buf["v"],
        g=buf["ra"], beta=buf["rb"],
        h0=bank,
        n_actual=N_STREAMS, n_pad=N_PAD,
        strict_mask=aux["strict"], visible_mask=aux["visible"],
        output_scale=SCALE, use_qk_l2norm_in_kernel=True,
        h0_indices=spec.reshape(-1), h0_is_bank=True,
        h0_index_row=0, h0_batch_index=0, h0_use_accepted_column=False,
        raw_a=buf["ra"], raw_b=buf["rb"],
        A_log=aux["A_log"], dt_bias=aux["dt_bias"],
        piggyback_export=True, chain_end_idx=CHAIN_END,
    )
    torch.cuda.synchronize()
    return bank[COL0_ROW].clone()


def _run_replay(KM, ring, bank, spec, aux, path_streams, root_node, dev):
    paths = torch.zeros(1, SPEC_COLS - 1, device=dev, dtype=torch.int32)
    for i, s in enumerate(path_streams):
        paths[0, i] = s
    KM.launch_tree_gdn_replay(
        state_bank=bank, spec_state_indices=spec,
        prev_lens=torch.zeros(1, device=dev, dtype=torch.int32),
        accepted_paths=paths,
        accepted_lens=torch.tensor([len(path_streams)], device=dev, dtype=torch.int32),
        k_ring=ring["k"], v_ring=ring["v"], a_ring=ring["a"], b_ring=ring["b"],
        A_log=aux["A_log"], dt_bias=aux["dt_bias"],
        num_spec_decodes=1, output_scale=SCALE, use_qk_l2norm_in_kernel=True,
        runrow_commit=True, runrow_init=True, burn_node_bank=False,
        root_node=root_node,
    )
    torch.cuda.synchronize()
    return bank[COL0_ROW].clone()


def _fresh_bank(h0, dev):
    bank = torch.zeros(BANK_ROWS, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    bank[COL0_ROW] = h0
    return bank


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA unavailable -- run inside the vllm container.",
              file=sys.stderr, flush=True)
        return 2
    V0C._scrub_env_and_sidecars()
    dev = "cuda"
    print(f"[env] device={torch.cuda.get_device_name(0)} torch={torch.__version__}",
          flush=True)

    import lumo_flywheel_serving.fr10_gdn_tree_kernel as KM

    aux_g = torch.Generator(device=dev)
    aux_g.manual_seed(20260718)
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import Tree
    strict, visible = Tree(V0C.STREAM_PARENTS).masks(torch.device(dev), N_PAD)
    aux = dict(
        strict=strict, visible=visible,
        A_log=torch.randn(NUM_VH, generator=aux_g, device=dev, dtype=torch.bfloat16),
        dt_bias=torch.randn(NUM_VH, generator=aux_g, device=dev, dtype=torch.bfloat16),
    )
    spec = (COL0_ROW + torch.arange(SPEC_COLS, device=dev, dtype=torch.int32)
            ).reshape(1, SPEC_COLS)

    rows_out: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        rows_out.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""), flush=True)

    def h0_of(seed):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        for _try in range(16):  # resample on exact zeros (N3) instead of dying
            h0 = torch.randn(NUM_VH, DIM, DIM, generator=g, device=dev,
                             dtype=torch.float32) * 0.2
            if not bool((h0 == 0).any()):
                return h0
        raise RuntimeError("h0 fixture: could not draw a zero-free tensor in 16 tries")

    # ---- (A) two-step induction + (B) threading identity, per L1 -------------
    print(f"[check A/B] two-step ring induction + root_node threading, L1 in {V0C.LS}",
          flush=True)
    try:
        for L1 in V0C.LS:
            g = torch.Generator(device=dev)
            g.manual_seed(31337 + L1)
            h0 = h0_of(1000 + L1)
            step1 = _mk_step_rows(g, dev)
            ring1 = _rows_to_ring(step1, dev)
            path1 = list(_SPINE_STREAMS[:L1])

            # step 1: fresh request -> all-identity chain; sanity export@7 == h0
            buf1 = _scan_buffers(g, dev, step1, chain_srcs=[])
            bank = _fresh_bank(h0, dev)
            col0_1 = _run_scan(KM, buf1, bank, spec, aux)
            add(f"(A) L1={L1} step-1 identity sanity (export@7 == h0)",
                V0C._bytes_eq(col0_1, h0), V0C._diff_stats(col0_1, h0))

            # step 2: chain = B5 gather [step-1 row 8] + step-1 accepted rows
            step2 = _mk_step_rows(g, dev)
            chain_srcs = [(step1, SUBTREE_ROOT)] + [(step1, s) for s in path1]
            buf2 = _scan_buffers(g, dev, step2, chain_srcs=chain_srcs)
            col0_2 = _run_scan(KM, buf2, bank, spec, aux)  # bank still holds col0_1

            # reference: dropped-replay deposit, root_node=8 over step-1's ring
            ref = _run_replay(KM, ring1, _fresh_bank(h0, dev), spec, aux,
                              path1, root_node=SUBTREE_ROOT, dev=dev)
            add(f"(A) L1={L1} induction: step-2 export@7 == replay(root=8) (BYTE)",
                V0C._bytes_eq(col0_2, ref), V0C._diff_stats(col0_2, ref))

            # (B) threading identity vs stock root-0 replay on a re-indexed ring
            ring0 = {f: ring1[f].clone() for f in ("k", "v", "a", "b")}
            for f in ("k", "v", "a", "b"):
                ring0[f][0, 0] = ring1[f][0, SUBTREE_ROOT]
                for i, s in enumerate(path1):
                    ring0[f][0, 1 + i] = ring1[f][0, s]
            ref0 = _run_replay(KM, ring0, _fresh_bank(h0, dev), spec, aux,
                               list(range(1, L1 + 1)), root_node=0, dev=dev)
            add(f"(B) L1={L1} replay(root=8) == stock replay(root=0, re-indexed) (BYTE)",
                V0C._bytes_eq(ref, ref0), V0C._diff_stats(ref, ref0))
    except Exception as e:  # noqa: BLE001
        add("(A/B) induction group crashed", False, f"{type(e).__name__}: {e}")

    # ---- (C) three-step compounding (L1=2 then L2=3) -------------------------
    print("[check C] three-step compounding (L1=2, L2=3)", flush=True)
    try:
        g = torch.Generator(device=dev)
        g.manual_seed(777001)
        h0 = h0_of(9001)
        step1 = _mk_step_rows(g, dev)
        ring1 = _rows_to_ring(step1, dev)
        path1 = list(_SPINE_STREAMS[:2])
        bank = _fresh_bank(h0, dev)
        _run_scan(KM, _scan_buffers(g, dev, step1, []), bank, spec, aux)

        step2 = _mk_step_rows(g, dev)
        ring2 = _rows_to_ring(step2, dev)
        path2 = list(_SPINE_STREAMS[:3])
        chain2 = [(step1, SUBTREE_ROOT)] + [(step1, s) for s in path1]
        _run_scan(KM, _scan_buffers(g, dev, step2, chain2), bank, spec, aux)

        step3 = _mk_step_rows(g, dev)
        chain3 = [(step2, SUBTREE_ROOT)] + [(step2, s) for s in path2]
        col0_3 = _run_scan(KM, _scan_buffers(g, dev, step3, chain3), bank, spec, aux)

        # composed reference: replay commit-1 then commit-2 on a fresh bank
        rbank = _fresh_bank(h0, dev)
        _run_replay(KM, ring1, rbank, spec, aux, path1, SUBTREE_ROOT, dev)
        ref = _run_replay(KM, ring2, rbank, spec, aux, path2, SUBTREE_ROOT, dev)
        add("(C) three-step: step-3 export@7 == composed replay refs (BYTE)",
            V0C._bytes_eq(col0_3, ref), V0C._diff_stats(col0_3, ref))
    except Exception as e:  # noqa: BLE001
        add("(C) compounding group crashed", False, f"{type(e).__name__}: {e}")

    # ---- table ---------------------------------------------------------------
    print("=" * 92)
    print(f"{'FR13_PIGGYBACK V0(d) RING-INDUCTION CHECK':74s} RESULT")
    print("-" * 92)
    for name, ok, detail in rows_out:
        print(f"{name:74s} {'PASS' if ok else 'FAIL'}")
        if detail and not ok:
            print(f"    -> {detail}")
    n_fail = sum(1 for _, ok, _ in rows_out if not ok)
    print("-" * 92)
    if n_fail == 0:
        print("=== VERDICT: PASS -- the cross-step B5 row-8-rooted carry composes "
              "byte-exactly; root_node threading == stock; safe to proceed to V1 ===")
    else:
        print(f"=== VERDICT: FAIL -- {n_fail}/{len(rows_out)} failed; do NOT arm "
              "FR13_PIGGYBACK (ship rule: V0(d)+V1 must be green first) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
