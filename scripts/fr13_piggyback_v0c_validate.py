#!/usr/bin/env python3
"""FR13_PIGGYBACK V0(c) offline validator -- Scheme A export / identity / committer gates.

Run INSIDE the vllm container (GPU):

    docker exec <vllm-container> python3 /workspace/scripts/fr13_piggyback_v0c_validate.py

(the launchers mount the repo at /workspace; PYTHONPATH is not needed -- the script
inserts <repo>/src and <repo>/scripts into sys.path itself). Deterministic seeds;
single-request fixtures (bank = 12 rows, ~38 MB) so everything after the two Triton
compiles runs in seconds. Exit 0 iff EVERY check passes.

WHAT THIS SETTLES (FR13_PIGGYBACK_BUNDLE_SPEC.md 0 + 4/V0(c); unresolved Risk 2 =
the 7-vs-8 export-index dispute), for prev-accept-len L in {0,1,2,3,4,5}:

  (i)  EXPORT SCHEME (the 7-vs-8 decider). Build the 17-choice extended cat9_pb
       tree (chain (0,)^1..(0,)^8 ++ re-rooted locked cat9 => 18 scan streams,
       n_pad=32, BV=8 override = the E23 serve geometry). Pack per Scheme A:
       stream 0 (vLLM's pos-0 prev-bonus copy) identity-masked; streams 1..L+1 =
       [prev pos-0 root token] + THIS commit's accepted drafts (real rows);
       streams L+2..7 identity-masked (raw_a=raw_b=-1e9, the E9 mask); stream 8 =
       (0,)^8 = the CURRENT bonus (the ONE active application, bonus-once-at-
       subtree-root); streams 9..17 = the re-rooted cat9 subtree (real rows).
       Run launch_tree_gdn_prepared(piggyback_export=True, chain_end_idx=7) from
       h0 = the pre-step state, and separately launch_tree_gdn_replay on the SAME
       [prev root]+drafts committed path from the same pre-step state
       (runrow_init=True, runrow_commit=True = the deployed stateless col-0
       lifecycle). ASSERT export deposit(col-0) == replay deposit(col-0)
       BYTE-EXACT (torch int-view equality -- NEVER atol). Then run
       chain_end_idx=8 and ASSERT it DIFFERS from the replay deposit AND EQUALS
       the replay of [prev root]+drafts+BONUS: 8 exports the POST-bonus state,
       which the next step's chain would re-apply => the double-applied-bonus
       garble class. That is WHY chain_end_idx=7 is correct under Scheme A.

  (ii) IDENTITY (the first-step L=0 edge: by_req absent => chain lens 0 => ALL
       chain rows 0..7 identity-masked). ASSERT the state after the chain
       segment (= export@7) == h0 byte-exact: raw_a=raw_b=-1e9 rows are true
       no-ops (softplus(-1e9 + dt_bias) = 0 => exp(g) = 1; sigmoid(-1e9) = 0 =>
       beta = 0 => zero rank-1 update).

  (iii) COMMITTER SHIFT. fr13_device_multidraft_commit on the extended 17-node
       choices-rank fixture with the pb walk root armed (env FR13_PIGGYBACK=1 =>
       _fr13_pb_walk_root() = cap()-1 = 7, CHOICES-rank space) == the base-cat9
       commit on the stripped 9-node subtree, modulo the +8 node-index shift on
       accepted_node_paths / accepted_row (accepted_row stays 0 on zero-accept
       in BOTH -- TOPOLOGY.edits[8]'s max(0,walk_root) variant was NOT applied,
       per bundle Group 5). out_rows / accepted_lens / accepted_token_rows are
       asserted EXACTLY equal (same host-generator seed => same derived device
       seed => same draws: the walks are isomorphic). Gated on BOTH the legacy
       per-node walk and the FR13_DM_DEPTHSYNC twin, plus a negative test: the
       armed walk on a BASE (unextended) tree must fail loud
       (_fr13_pb_validate_extended_tree).

NOTES / CAVEATS (all deliberate):
  N1 SPACES: committer walk root 7 = CHOICES-rank (choice i = scan stream i+1);
     kernel chain_end_idx 7 = scan-STREAM (0,)^7. Equal numbers, DIFFERENT
     spaces (bundle 2.8) -- do not "unify" them.
  N2 GEOMETRY: the scan compiles at BV=8 (FR13_TREE_GDN_GEOM_OVERRIDE=BV=8,
     required by the n_pad=32 h_cache register wall; E23 sets the same), while
     _tree_gdn_replay_kernel is fixed BV=16. Byte-equality across the two
     compilations is NOT spec-guaranteed (shared _gdn_node_step, same fp32
     sequence, but codegen identity is empirical) -- settling it empirically IS
     this gate, same class as the FR13_REPLAY_ROUTE byte A/B obligation.
  N3 -0.0: identity rows multiply state by exp(-0.0)=1.0 and add +-0.0; a -0.0
     state lane could flip sign-bit to +0.0 (the scan's one-hot parent handoff
     also flips -0.0, which the replay reproduces via `state + 0.0`). The
     fixture asserts h0 contains no exact zeros so the compare stays byte-exact.
  N4 The replay entry reroutes to the NATIVE fused_sigmoid committer when
     FR13_COMMITTER_NATIVE (env or /logs|/tmp sidecar) is armed with
     runrow_init. This gate targets the DEPLOYED custom replay kernel
     (fr13_launch_locked.sh does NOT set the flag) and REFUSES to run armed.
  N5 q is state-inert (native-validator note: q never touches state), so random
     q rows are fine; the scan `out` product is not compared.
  N6 Stale /logs|/tmp/fr13_piggyback.arm sidecars are cleared in setup (bundle
     Risk 9: a stale arm would flip _fr13_pb_walk_root/_cap mid-gate). NEVER run
     this against a container that is concurrently SERVING piggyback.

SIGNATURE AUDIT (2026-07-18, against the working tree -- bundle rule: on any
mismatch do NOT improvise, record it here and fail the affected check):
  * launch_tree_gdn_prepared(q,k,v,g,beta,h0,*,...,piggyback_export=False,
    chain_end_idx=0)  -- src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:1982
    (params :2009-2010, threaded to constexprs :2218-2219).       MATCHES SPEC.
  * PIGGYBACK_EXPORT/CHAIN_END_IDX constexprs :739-740; export block :909-920
    deposits h_cache[CHAIN_END_IDX] -> h0_indices[H0_INDEX_ROW+0] (col-0).
                                                                  MATCHES SPEC.
  * launch_tree_gdn_replay(...):1216 with runrow_commit/runrow_init/
    burn_node_bank kwargs; native reroute guard :1325 (see N4).   MATCHES SPEC.
  * _fr13_piggyback_on :22 / _fr13_piggyback_cap(default=8) :34.  MATCHES SPEC.
  * fr13_device_multidraft_commit(num_draft_tokens, draft_token_ids,
    tree_parent_indices, target_logits, tree_self_logits, draft_probs,
    bonus_token_ids, max_spec_len, *, generators, all_greedy) --
    scripts/fr13_device_multidraft_kernel.py:454; _fr13_pb_walk_root :97;
    _fr13_pb_validate_extended_tree :130; legacy walk arming :544-580;
    depthsync twin arming :738-750.                               MATCHES SPEC.
  NO DISCREPANCIES FOUND -- every check below asserts live (none is structured
  as an expected-fail).
"""
from __future__ import annotations

import os
import sys

# Pure-CPU imports only at module level: py_compile / host-venv import must not
# touch GPU deps (triton / lumo kernels / the MD module are imported in main()).
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- constants (native-validator conventions: NUM_KH/NUM_VH/DIM) -------------
NUM_KH, NUM_VH, DIM = 16, 48, 128
SCALE = DIM ** -0.5

N_STREAMS = 18        # root + 17 choices (served tree_n=18, bundle E8)
N_PAD = 32            # padded_nodes(18); needs BV<=8 (register wall)
CHAIN_END = 7         # scan-STREAM space: (0,)^7 -- Scheme A PRE-bonus export
SUBTREE_ROOT = 8      # scan-STREAM space: (0,)^8 -- bonus applied ONCE here
WALK_ROOT = 7         # CHOICES-rank space: cap()-1 (N1: != CHAIN_END's space)
SHIFT = 8             # base-cat9 choice c <-> extended choice c+8
N_RING = 16           # replay ring rows (power of two <= 32)
SPEC_COLS = 8
BANK_ROWS = 12
COL0_ROW = 2          # bank row backing spec col 0 (the running row)
IDENT = -1e9          # E9 identity-mask raw gating value
VOCAB = 64
MAX_SPEC_LEN = 9      # identical for base+ext committer runs (loop-count parity)
LS = (0, 1, 2, 3, 4, 5)

# --- extended cat9_pb topology (module-level pure python; fail-loud asserts) --
_BASE_CAT9_RAW = (
    (0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0),
    (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1),
)  # the LOCKED cat9 (fr13_launch_locked.sh TREE)
# LIVE sort key: tree_attn sorts choices by (len, choice) -- length-first,
# NOT plain lexicographic (they DIFFER here: lex puts (0,)^11 before
# (0*9,1); length-first reverses that, and flips the base subtree order
# too, e.g. (0,0,0,1) before (0,0,0,0,1)). Fixture stream numbering MUST
# match the live layout (caught by fr13_piggyback_v0d_bias_conv check b).
_LIVE_KEY = lambda _p: (len(_p), _p)  # noqa: E731 -- mirrors tree_attn/A1
BASE_CHOICES = tuple(sorted(_BASE_CAT9_RAW, key=_LIVE_KEY))
CHAIN_CHOICES = tuple((0,) * i for i in range(1, 9))  # (0,)^1 .. (0,)^8
EXT_CHOICES = tuple(sorted(CHAIN_CHOICES + tuple((0,) * 8 + c for c in BASE_CHOICES),
                           key=_LIVE_KEY))


def _choices_to_parents(choices):
    """CHOICES-rank parents: parent = rank of the prefix tuple; root children = -1."""
    idx = {c: i for i, c in enumerate(choices)}
    return [(-1 if len(c) == 1 else idx[c[:-1]]) for c in choices]


BASE_PARENTS = _choices_to_parents(BASE_CHOICES)
EXT_PARENTS = _choices_to_parents(EXT_CHOICES)
# stream space: stream 0 = root, choice i = stream i+1 (bundle 0 indexing facts)
STREAM_PARENTS = (-1,) + tuple(p + 1 for p in EXT_PARENTS)

assert len(EXT_CHOICES) == 17 and len(STREAM_PARENTS) == N_STREAMS
assert EXT_CHOICES[:8] == CHAIN_CHOICES, "chain must occupy choice ranks 0..7"
assert EXT_CHOICES[8:] == tuple((0,) * 8 + c for c in BASE_CHOICES), (
    "re-rooted base choices must occupy ranks 8..16 in base-sorted order")
assert EXT_PARENTS[:8] == [-1, 0, 1, 2, 3, 4, 5, 6], "chain parents must be j-1"
for _c in range(9):  # the +8 shift structure the committer check relies on
    _bp = BASE_PARENTS[_c]
    assert EXT_PARENTS[SHIFT + _c] == (WALK_ROOT if _bp == -1 else SHIFT + _bp), (
        f"extended choice {SHIFT + _c} breaks the +{SHIFT} re-root shift")
assert STREAM_PARENTS[SUBTREE_ROOT] == CHAIN_END and STREAM_PARENTS[9] == SUBTREE_ROOT


# --- helpers -----------------------------------------------------------------
def _bytes_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    """BYTE compare: int-view equality (never atol -- feedback_fr13_lossless_compare_target)."""
    return bool(torch.equal(a.contiguous().view(torch.int32),
                            b.contiguous().view(torch.int32)))


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> str:
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    bi = b.contiguous().view(torch.int32).to(torch.int64)
    n = int((ai != bi).sum().item())
    if n == 0:
        return "byte-identical"
    d = (a.float() - b.float()).abs()
    return (f"{n}/{ai.numel()} lanes differ, max|d|={d.max().item():.3e}, "
            f"max int32-delta={int((ai - bi).abs().max().item())}")


def _scrub_env_and_sidecars():
    """Pin the LOCKED scan config + disarm every flag that could reroute a path.

    Bundle Risk 9: stale piggyback arm sidecars on the host/container would arm
    the committer walk (and _cap) underneath this gate -- clear them, fail loud
    if uncleanable. FR13_COMMITTER_NATIVE would reroute launch_tree_gdn_replay
    away from the deployed custom replay kernel (N4) -- refuse to guess."""
    for k in (
        "FR13_SCAN_ALIGN", "FR13_NPAD_INVARIANT", "FR13_PARENT_GATHER",
        "FR13_PARENT_GATHER_SELFCHECK", "FR13_REPLAY_GPU_TIMER",
        "FR13_MULTIDRAFT_GPU_TIMER", "FR13_DM_DEPTHSYNC",
        "LUMO_TREE_SAMPLER_DEBUG_LOG", "FR13_PIGGYBACK",
        "FR13_PIGGYBACK_PREFIX_CAP", "FR13_COMMITTER_NATIVE",
    ):
        os.environ.pop(k, None)
    # E23 geometry: n_pad=32 launches only with effective BV<=8 (register wall).
    os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = "BV=8"
    for p in ("/logs/fr13_piggyback.arm", "/tmp/fr13_piggyback.arm"):
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[setup] cleared stale piggyback arm sidecar: {p}", flush=True)
            except OSError as e:
                raise RuntimeError(
                    f"FR13_PIGGYBACK V0(c): stale arm sidecar {p} exists and "
                    f"cannot be removed ({e}); it would arm _fr13_pb_walk_root/"
                    "_cap underneath this gate. Clear it and rerun.")
    for p in ("/logs/fr13_committer_native.flag", "/tmp/fr13_committer_native.flag"):
        if os.path.exists(p):
            raise RuntimeError(
                f"FR13_PIGGYBACK V0(c): {p} exists -- launch_tree_gdn_replay "
                "would reroute to the NATIVE fused_sigmoid committer instead of "
                "the deployed custom replay kernel this gate compares against "
                "(N4). Remove the flag (it is not part of the locked pipeline) "
                "or run in a clean container.")


# --- check (i)/(ii) fixtures -------------------------------------------------
def _build_case(L: int, seed: int, dev: str, all_identity: bool = False) -> dict:
    """One single-request fixture: scan buffers + replay ring sharing BYTES.

    Chain payload tokens: row 0 = prev pos-0 root token, rows 1..L = accepted
    drafts, row L+1 = the CURRENT bonus. Ring rows 0..L+1 carry the SAME bytes
    (replay path node 0 = prev root; ring rows 1..L = drafts; L+1 = bonus).
    """
    assert 0 <= L <= 5, "chain must fit: 1+L <= 7 (E5 raise)"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    def mk(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, device=dev, dtype=dtype)

    A_log = mk(NUM_VH)
    dt_bias = mk(NUM_VH)
    n_tok = L + 2
    tok_k = mk(n_tok, NUM_KH, DIM) * 0.3
    tok_v = mk(n_tok, NUM_VH, DIM) * 0.3
    tok_a = mk(n_tok, NUM_VH) * 0.5
    tok_b = mk(n_tok, NUM_VH) * 0.5

    # scan buffers (random fill; identity rows' k/v/q values are inert -- the
    # E5 packer repeat-pads token VALUES for the same reason)
    q = mk(N_PAD, NUM_KH, DIM) * 0.3
    k = mk(N_PAD, NUM_KH, DIM) * 0.3
    v = mk(N_PAD, NUM_VH, DIM) * 0.3
    ra = mk(N_PAD, NUM_VH) * 0.5
    rb = mk(N_PAD, NUM_VH) * 0.5
    if not all_identity:
        k[1:L + 2] = tok_k[:L + 1]
        v[1:L + 2] = tok_v[:L + 1]
        ra[1:L + 2] = tok_a[:L + 1]
        rb[1:L + 2] = tok_b[:L + 1]
    k[SUBTREE_ROOT] = tok_k[L + 1]
    v[SUBTREE_ROOT] = tok_v[L + 1]
    ra[SUBTREE_ROOT] = tok_a[L + 1]
    rb[SUBTREE_ROOT] = tok_b[L + 1]
    # E9 identity mask: stream 0 ALWAYS; unused chain slots L+2..7 (all 1..7 on
    # the all-identity first-step edge). Streams 9..17 stay real (subtree).
    first_masked = 1 if all_identity else L + 2
    for s in [0] + list(range(first_masked, CHAIN_END + 1)):
        ra[s] = IDENT
        rb[s] = IDENT

    from lumo_flywheel_serving.fr10_gdn_tree_kernel import Tree
    strict, visible = Tree(STREAM_PARENTS).masks(torch.device(dev), N_PAD)

    # replay ring: same bytes at rows 0..L+1
    ring_k = (mk(1, N_RING, NUM_KH, DIM) * 0.3).contiguous()
    ring_v = (mk(1, N_RING, NUM_VH, DIM) * 0.3).contiguous()
    ring_a = (mk(1, N_RING, NUM_VH) * 0.5).contiguous()
    ring_b = (mk(1, N_RING, NUM_VH) * 0.5).contiguous()
    ring_k[0, :n_tok] = tok_k
    ring_v[0, :n_tok] = tok_v
    ring_a[0, :n_tok] = tok_a
    ring_b[0, :n_tok] = tok_b

    h0 = mk(NUM_VH, DIM, DIM, dtype=torch.float32) * 0.2
    assert not bool((h0 == 0).any()), "fixture h0 must contain no exact zeros (N3)"
    bank = torch.zeros(BANK_ROWS, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    bank[COL0_ROW] = h0
    spec = (COL0_ROW + torch.arange(SPEC_COLS, device=dev, dtype=torch.int32)).reshape(1, SPEC_COLS)

    paths = torch.zeros(1, SPEC_COLS - 1, device=dev, dtype=torch.int32)
    paths[0, :L] = torch.arange(1, L + 1, device=dev, dtype=torch.int32)
    paths_bonus = paths.clone()
    paths_bonus[0, L] = L + 1  # ring row L+1 = the bonus token
    return dict(
        dev=dev, L=L, q=q, k=k, v=v, ra=ra, rb=rb, A_log=A_log, dt_bias=dt_bias,
        strict=strict, visible=visible, ring_k=ring_k, ring_v=ring_v,
        ring_a=ring_a, ring_b=ring_b, h0=h0, bank=bank, spec=spec,
        paths=paths, paths_bonus=paths_bonus,
        prev_lens=torch.zeros(1, device=dev, dtype=torch.int32),
    )


def _scan_deposit(KM, case: dict, chain_end_idx: int) -> torch.Tensor:
    """Extended-tree scan with piggyback export; returns the col-0 deposit."""
    bank = case["bank"].clone()
    KM.launch_tree_gdn_prepared(
        q=case["q"], k=case["k"], v=case["v"],
        g=case["ra"], beta=case["rb"],  # inert under RAW_GATING (shape-only)
        h0=bank,
        n_actual=N_STREAMS, n_pad=N_PAD,
        strict_mask=case["strict"], visible_mask=case["visible"],
        output_scale=SCALE, use_qk_l2norm_in_kernel=True,
        h0_indices=case["spec"].reshape(-1), h0_is_bank=True,
        h0_index_row=0, h0_batch_index=0, h0_use_accepted_column=False,
        raw_a=case["ra"], raw_b=case["rb"],
        A_log=case["A_log"], dt_bias=case["dt_bias"],
        piggyback_export=True, chain_end_idx=chain_end_idx,
    )
    torch.cuda.synchronize()
    return bank[COL0_ROW].clone()


def _replay_deposit(KM, case: dict, paths: torch.Tensor, acc_len: int) -> torch.Tensor:
    """Deployed committer replay ([node 0] ++ path) from col-0; returns col-0."""
    bank = case["bank"].clone()
    KM.launch_tree_gdn_replay(
        state_bank=bank, spec_state_indices=case["spec"],
        prev_lens=case["prev_lens"], accepted_paths=paths,
        accepted_lens=torch.tensor([acc_len], device=case["dev"], dtype=torch.int32),
        k_ring=case["ring_k"], v_ring=case["ring_v"],
        a_ring=case["ring_a"], b_ring=case["ring_b"],
        A_log=case["A_log"], dt_bias=case["dt_bias"],
        num_spec_decodes=1, output_scale=SCALE, use_qk_l2norm_in_kernel=True,
        runrow_commit=True, runrow_init=True, burn_node_bank=False,
    )
    torch.cuda.synchronize()
    return bank[COL0_ROW].clone()


# --- check (iii) fixtures ----------------------------------------------------
def _committer_fixture(seed: int) -> dict:
    """Base 9-node cat9 fixture + its 17-node extension (shared subtree BYTES)."""
    rng = np.random.default_rng(seed)
    # scale-8 logits => near-point-mass softmax => the drafted spine is accepted
    # with high probability, giving len>=2 coverage (scale-3 + p=0.7 measured
    # hist {0:25,1:7} => coverage check failed; equality held on all trials)
    tl_b = (rng.normal(size=(9, VOCAB)) * 8.0).astype(np.float32)
    sl_b = (rng.normal(size=(9, VOCAB)) * 8.0).astype(np.float32)
    drafts_b = [0] * 9
    by_parent: dict = {}
    for node, p in enumerate(BASE_PARENTS):
        by_parent.setdefault(p, []).append(node)
    for _par, kids in by_parent.items():
        used: set = set()
        greedy_tok = int(np.argmax(tl_b[kids[0]]))
        for i, node in enumerate(kids):
            if i == 0 and rng.random() < 0.95:
                tok = greedy_tok
            else:
                tok = int(rng.integers(0, VOCAB))
                while tok in used or (i != 0 and tok == greedy_tok):
                    tok = int(rng.integers(0, VOCAB))
            used.add(tok)
            drafts_b[node] = tok
    bonus = int(rng.integers(0, VOCAB))
    # extension: chain choices 0..7 get junk rows/drafts (never walked/committed)
    chain_drafts = [int(x) for x in rng.integers(0, VOCAB, size=8)]
    chain_tl = (rng.normal(size=(8, VOCAB)) * 3.0).astype(np.float32)
    chain_sl = (rng.normal(size=(8, VOCAB)) * 3.0).astype(np.float32)
    return dict(
        base=dict(parents=list(BASE_PARENTS), drafts=drafts_b,
                  target=tl_b, self=sl_b, bonus=bonus),
        ext=dict(parents=list(EXT_PARENTS), drafts=chain_drafts + drafts_b,
                 target=np.concatenate([chain_tl, tl_b], axis=0),
                 self=np.concatenate([chain_sl, sl_b], axis=0), bonus=bonus),
    )


def _run_commit(MD, fx: dict, gen_seed: int, dev: str):
    gens = {0: torch.Generator(device="cpu").manual_seed(gen_seed)}
    out = MD.fr13_device_multidraft_commit(
        [len(fx["parents"])],
        torch.tensor(fx["drafts"], dtype=torch.long, device=dev),
        torch.tensor(fx["parents"], dtype=torch.long, device=dev),
        torch.tensor(fx["target"], dtype=torch.float32, device=dev),
        torch.tensor(fx["self"], dtype=torch.float32, device=dev),
        None,
        torch.tensor([fx["bonus"]], dtype=torch.long, device=dev),
        MAX_SPEC_LEN,
        generators=gens,
        all_greedy=False,
    )
    rows, arows, alens, paths, toks = out
    return rows[0], arows[0], alens[0], list(paths[0]), list(toks[0])


def _compare_shift(base, ext):
    """ext(armed) == base modulo the +8 shift on node ids (path + accepted_row)."""
    b_row, b_arow, b_alen, b_path, b_tok = base
    e_row, e_arow, e_alen, e_path, e_tok = ext
    if e_row != b_row:
        return False, f"out_rows differ: base={b_row} ext={e_row}"
    if e_alen != b_alen:
        return False, f"accepted_lens differ: base={b_alen} ext={e_alen}"
    if e_path != [x + SHIFT for x in b_path]:
        return False, f"paths not +{SHIFT}-shifted: base={b_path} ext={e_path}"
    if e_tok != b_tok:
        return False, f"accepted_token_rows differ: base={b_tok} ext={e_tok}"
    if b_alen > 0:
        if e_arow != b_arow + SHIFT:
            return False, f"accepted_row not +{SHIFT}: base={b_arow} ext={e_arow}"
    elif not (b_arow == 0 and e_arow == 0):
        return False, f"zero-accept accepted_row: base={b_arow} ext={e_arow} (both must be 0)"
    return True, ""


# --- main --------------------------------------------------------------------
def main() -> int:
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is unavailable. This validator launches the FR13 tree-GDN\n"
            "Triton kernels and MUST run INSIDE the vllm container (GPU), e.g.:\n"
            "  docker exec <vllm-container> python3 /workspace/scripts/fr13_piggyback_v0c_validate.py",
            file=sys.stderr, flush=True)
        return 2
    _scrub_env_and_sidecars()
    dev = "cuda"
    print(f"[env] device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"geom_override={os.environ['FR13_TREE_GDN_GEOM_OVERRIDE']}", flush=True)

    import lumo_flywheel_serving.fr10_gdn_tree_kernel as KM
    import fr13_device_multidraft_kernel as MD

    rows: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = ""):
        rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""),
              flush=True)

    # ---- (0) sanity: cap/arm defaults this gate (and E8) rely on
    add("(0) _fr13_piggyback_cap()==8 (default; chain_end=cap-1=7)",
        int(KM._fr13_piggyback_cap()) == 8, f"cap={KM._fr13_piggyback_cap()}")
    add("(0) piggyback disarmed after scrub (_fr13_piggyback_on()==False)",
        not KM._fr13_piggyback_on())
    add("(0) committer-native disarmed (deployed custom replay kernel path)",
        not KM._fr13_committer_native_on())

    # ---- (i) export scheme, per L
    print("[check i] export scheme (7-vs-8 decider), L in " + str(LS), flush=True)
    try:
        for L in LS:
            case = _build_case(L, seed=1313 + L, dev=dev)
            dep7 = _scan_deposit(KM, case, CHAIN_END)
            dep8 = _scan_deposit(KM, case, SUBTREE_ROOT)
            rep = _replay_deposit(KM, case, case["paths"], L)
            rep_bonus = _replay_deposit(KM, case, case["paths_bonus"], L + 1)
            add(f"(i) L={L} export@7 == replay deposit (BYTE, int-view)",
                _bytes_eq(dep7, rep), _diff_stats(dep7, rep))
            add(f"(i) L={L} export@8 != replay deposit (7-vs-8 decider)",
                not _bytes_eq(dep8, rep),
                "export@8 EQUALS the pre-bonus replay deposit -- decider vacuous")
            add(f"(i) L={L} export@8 == replay+bonus (8 = double-applied bonus)",
                _bytes_eq(dep8, rep_bonus), _diff_stats(dep8, rep_bonus))
    except Exception as e:  # noqa: BLE001 -- keep the table; fail the group loud
        add("(i) export-scheme group crashed", False, f"{type(e).__name__}: {e}")

    # ---- (ii) identity (first-step edge: ALL chain rows masked)
    print("[check ii] identity mask no-op (all-identity chain, L=0 edge)", flush=True)
    try:
        case_id = _build_case(0, seed=4242, dev=dev, all_identity=True)
        dep7 = _scan_deposit(KM, case_id, CHAIN_END)
        add("(ii) all-identity chain: export@7 == h0 (BYTE, int-view)",
            _bytes_eq(dep7, case_id["h0"]), _diff_stats(dep7, case_id["h0"]))
    except Exception as e:  # noqa: BLE001
        add("(ii) identity group crashed", False, f"{type(e).__name__}: {e}")

    # ---- (iii) committer shift (legacy + depthsync twin + negative)
    print("[check iii] committer walk shift (FR13_PIGGYBACK=1 vs base)", flush=True)
    try:
        def _trials(n_trials: int, seed0: int, gseed0: int):
            mism, lens_seen = [], []
            for t in range(n_trials):
                fx = _committer_fixture(seed0 + t)
                os.environ.pop("FR13_PIGGYBACK", None)
                base = _run_commit(MD, fx["base"], gseed0 + t, dev)
                os.environ["FR13_PIGGYBACK"] = "1"
                try:
                    if MD._fr13_pb_walk_root() != WALK_ROOT:
                        raise RuntimeError(
                            f"armed _fr13_pb_walk_root()={MD._fr13_pb_walk_root()} != {WALK_ROOT}")
                    ext = _run_commit(MD, fx["ext"], gseed0 + t, dev)
                finally:
                    os.environ.pop("FR13_PIGGYBACK", None)
                ok, why = _compare_shift(base, ext)
                lens_seen.append(base[2])
                if not ok:
                    mism.append(f"trial {t}: {why}")
            return mism, lens_seen

        mism, lens_seen = _trials(32, 20260718, 5000)
        hist = {v: lens_seen.count(v) for v in sorted(set(lens_seen))}
        add(f"(iii) legacy walk: ext(armed) == base +{SHIFT}-shift [32 trials]",
            not mism, "; ".join(mism[:3]))
        add(f"(iii) accept-len coverage len==0 AND len>=2 seen (hist={hist})",
            (0 in hist) and any(v >= 2 for v in hist), "widen trials/logit scale")

        os.environ["FR13_DM_DEPTHSYNC"] = "1"
        try:
            mism_ds, _ = _trials(16, 40000, 6000)
        finally:
            os.environ.pop("FR13_DM_DEPTHSYNC", None)
        add(f"(iii) depthsync twin: ext(armed) == base +{SHIFT}-shift [16 trials]",
            not mism_ds, "; ".join(mism_ds[:3]))

        # negative: armed walk on a BASE tree must fail loud (bundle E18 guard)
        fx = _committer_fixture(777)
        os.environ["FR13_PIGGYBACK"] = "1"
        try:
            _run_commit(MD, fx["base"], 999, dev)
            neg_ok, neg_msg = False, "no raise -- base tree walked silently under the armed root"
        except RuntimeError as e:
            neg_ok, neg_msg = ("FR13_PIGGYBACK" in str(e)), str(e)[:120]
        finally:
            os.environ.pop("FR13_PIGGYBACK", None)
        add("(iii) negative: armed walk on BASE tree raises fail-loud",
            neg_ok, neg_msg)
    except Exception as e:  # noqa: BLE001
        os.environ.pop("FR13_PIGGYBACK", None)
        os.environ.pop("FR13_DM_DEPTHSYNC", None)
        add("(iii) committer group crashed", False, f"{type(e).__name__}: {e}")

    # ---- table + verdict
    print("=" * 92)
    print(f"{'FR13_PIGGYBACK V0(c) CHECK':74s} RESULT")
    print("-" * 92)
    for name, ok, detail in rows:
        print(f"{name:74s} {'PASS' if ok else 'FAIL'}")
        if detail and not ok:
            print(f"    -> {detail}")
    n_fail = sum(1 for _, ok, _ in rows if not ok)
    print("-" * 92)
    if n_fail == 0:
        print("=== VERDICT: PASS -- Scheme A holds: chain_end_idx=7 (pre-bonus) is the "
              "correct export; identity mask is a true no-op; committer +8 shift exact ===")
    else:
        print(f"=== VERDICT: FAIL -- {n_fail}/{len(rows)} checks failed; do NOT land the "
              "bundle's 7-vs-8 binding until this gate passes (bundle Risk 2) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
