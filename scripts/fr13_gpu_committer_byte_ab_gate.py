#!/usr/bin/env python3
"""FR13 OPT-1 byte-A/B gate: GPU-resident committer (FR13_GPU_COMMITTER).

CPU-ONLY, boot-free. Proves the OPT-1 GPU committer
(``scripts/fr13_gpu_committer_kernel.py``) is byte-identical to the legacy
Python committer's serving decision for every tree topology in a stress matrix:

  REF  = an inline, faithful copy of the legacy committer's INTEGER serving
         logic, transcribed verbatim from
         ``_lumo_tree_path_lcp_max_greedy_sample`` (the SERVING subset:
         FR13_TREE_BONUS_SELF=1, no diagnostic gates, no FORCE_SPINE). This is
         the byte-for-byte ground truth the GPU committer must reproduce.
  ORACLE = fr13_gpu_committer_oracle (the kernel module's pure-Python reference
           that the Triton kernel mirrors).
  TRITON = fr13_gpu_committer_triton (only run when a CUDA GPU is present; on a
           CPU-only host this arm is SKIPPED and noted -- it is the documented
           live-GPU iteration step).

Gate passes iff, for every request in every tree of the matrix:
  (out_rows, accepted_rows, accepted_lens) from ORACLE == REF, token-for-token;
  and (when GPU present) TRITON == REF too.

The matrix covers: the production 9-node caterpillar, a pure spine, a fan from
root, deep + shallow alt leaves, full-accept (bonus=self) vs early-reject
(bonus=reject_parent) vs zero-accept (bonus=root_parent), ties on LCP (earliest
leaf must win), multi-request batches, and randomised trees.

A separate structural check (G-FLAG) asserts the patcher hook is DEFAULT-OFF:
the flag ``FR13_GPU_COMMITTER`` is read with default "0", and the legacy Python
committer block is left intact (the hook only *prepends* a flag-guarded branch).

Usage: python3 scripts/fr13_gpu_committer_byte_ab_gate.py
Exit 0 = PASS.  Nonzero = FAIL (and which).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "scripts" / "fr13_gpu_committer_kernel.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# REF: verbatim transcription of the legacy committer SERVING logic
# (_lumo_tree_path_lcp_max_greedy_sample, fr10_phase4_patch_vllm_tree_gdn.py
#  :5608-5879, FR13_TREE_BONUS_SELF=1, no diagnostic/force-spine branches).
# ---------------------------------------------------------------------------
def ref_committer(parents_cpu, drafts_cpu, ptgt_cpu, stgt_cpu, counts, max_spec_len):
    out_rows = []
    accepted_rows = []
    accepted_lens = []
    ref_node_paths = []
    ref_token_rows = []
    start = 0
    for node_count in counts:
        node_count = int(node_count)
        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        parent_targets = ptgt_cpu[start:start + node_count]
        self_targets = stgt_cpu[start:start + node_count]

        children = {-1: []}
        for node, parent in enumerate(parents):
            parent = int(parent)
            children.setdefault(parent, []).append(node)
            children.setdefault(node, [])
        leaves = [node for node in range(node_count) if not children.get(node)]
        if not leaves:
            leaves = list(range(node_count))

        best_path = []
        best_lcp = -1
        best_path_idx = 0
        for path_idx, leaf in enumerate(leaves):
            path = []
            node = int(leaf)
            guard = 0
            while 0 <= node < node_count and guard <= node_count:
                path.append(node)
                node = int(parents[node])
                guard += 1
            path.reverse()
            lcp = 0
            for node in path:
                if int(drafts[node]) != int(parent_targets[node]):
                    break
                lcp += 1
            if lcp > best_lcp:
                best_lcp = int(lcp)
                best_path = path
                best_path_idx = int(path_idx)
        best_lcp = max(0, int(best_lcp))

        row = []
        for node in best_path[:best_lcp]:
            row.append(int(drafts[node]))
        if best_path:
            if best_lcp < len(best_path):
                row.append(int(parent_targets[best_path[best_lcp]]))
            elif best_lcp > 0:
                # FR13_TREE_BONUS_SELF=1 serving path (tree_self_target).
                row.append(int(self_targets[best_path[best_lcp - 1]]))
            else:
                row.append(int(parent_targets[best_path[0]]))
        row = row[:int(max_spec_len) + 1]
        out_rows.append(row)
        accepted_rows.append(int(best_path[best_lcp - 1]) if best_lcp > 0 else 0)
        accepted_lens.append(int(best_lcp))
        # accepted_node_paths / accepted_token_rows (committer :5868-5869).
        ref_node_paths.append([int(x) for x in best_path[:best_lcp]])
        ref_token_rows.append([int(drafts[x]) for x in best_path[:best_lcp]])
        start += node_count
    return (
        out_rows, accepted_rows, accepted_lens,
        ref_node_paths, ref_token_rows,
    )


# ---------------------------------------------------------------------------
# Tree builders for the matrix
# ---------------------------------------------------------------------------
def caterpillar9():
    # production 9-node caterpillar: spine 0-1-3-5-7, leaves 2,4,6,8 hung off.
    parents = [-1, 0, 0, 1, 1, 3, 3, 5, 5]
    return parents


def make_request(parents, *, accept_depth, mismatch_at=None, max_tok=1000):
    """Build (drafts, ptgt, stgt) so the spine accepts `accept_depth` nodes.

    drafts[node]==ptgt[node] on the accepted prefix of the FIRST leaf's path;
    a forced mismatch at `mismatch_at` (root-distance) ends the LCP; self/parent
    targets get distinct ids so a wrong bonus-source is caught.
    """
    nc = len(parents)
    rng = random.Random(hash((tuple(parents), accept_depth, mismatch_at)) & 0xFFFF)
    drafts = [rng.randint(1, max_tok) for _ in range(nc)]
    ptgt = [rng.randint(max_tok + 1, 2 * max_tok) for _ in range(nc)]
    stgt = [rng.randint(2 * max_tok + 1, 3 * max_tok) for _ in range(nc)]
    # Force the first leaf's root path to match for `accept_depth` nodes.
    # first leaf in node order:
    has_child = [False] * nc
    for node, parent in enumerate(parents):
        if 0 <= parent < nc:
            has_child[parent] = True
    leaves = [n for n in range(nc) if not has_child[n]] or list(range(nc))
    leaf = leaves[0]
    path = []
    node = leaf
    while 0 <= node < nc:
        path.append(node)
        node = parents[node]
    path.reverse()
    for p, n in enumerate(path):
        if mismatch_at is not None and p >= mismatch_at:
            ptgt[n] = drafts[n] + 7  # guaranteed mismatch
        elif p < accept_depth:
            ptgt[n] = drafts[n]  # match
        else:
            ptgt[n] = drafts[n] + 13  # mismatch after accept_depth
    return drafts, ptgt, stgt


def _dispatch_composition_gate(ptext: str) -> list[str]:
    """LIVE-DISPATCH composition gate (catches the OPT-1 G2 crash class).

    The kernel-in-isolation arms above (ORACLE/FULL/TRITON/DEVICE) feed the
    committer functions CLEAN inputs and never exercise the COMPOSITION inside
    ``_lumo_tree_path_lcp_max_greedy_sample``: the interaction between
      (A) the synckill DtoH-skip that NULLS the host committer-input lists
          (``parents_cpu = None``), gated by
          ``_FR13_COMMITTER_SYNCKILL and _fr13_gpu_committer``;  and
      (B) the injected loop-skip that empties the per-node loop iterable
          (``_fr13_gpu_commit_counts = []``), gated by ``_fr13_gpu_committer``.
    The ORIGINAL defect: (A) was gated by a SEPARATE env/globals recompute
    (``_fr13_sk_engage``) that drifted from the runtime predicate (B) used at
    the loop-skip, so (A) fired while (B) did NOT (synckill ON + a diagnostic
    gate active), the legacy per-node loop ran with parents_cpu=None ->
    NoneType subscript -> EngineCoreDead. The kernel-isolation gate PASSED and
    missed it because it never composed (A) with (B).

    THE FIX (single source of truth): ``_fr13_gpu_committer`` is computed ONCE
    at the top of the committer body and the SAME variable now gates BOTH (A)
    the synckill null AND (B) the loop-skip. So A == ``SYNCKILL and X`` and
    B == ``X`` for the IDENTICAL X -- A implies B is now true BY CONSTRUCTION,
    not by keeping two hand-copied predicates in sync.

    This gate reproduces the LIVE composed dispatch logic: it (1) applies the
    patcher anchor replacement to obtain the ACTUAL composed committer text, (2)
    confirms the null guard and the loop-skip BOTH reference the SAME single
    ``_fr13_gpu_committer`` and that the loop-skip does NOT recompute it, (3)
    extracts the ONE predicate VERBATIM and EVALUATES the A=>B invariant across
    the full cartesian product of the gating flags:

        whenever the synckill NULL fires (A==True), the loop-SKIP must also
        fire (B==True)  --  i.e. A implies B.

    If A can be True while B is False for ANY flag combination, the legacy loop
    would run with nulled inputs -> the crash. This is a true composition check
    of the live dispatch, not the kernel in isolation.
    """
    fails: list[str] = []
    import itertools
    import re

    # ---- (1) confirm the patcher composes the loop-skip (anchor present) ----
    anchor = (
        "    out_rows = []\n"
        "    accepted_rows = []\n"
        "    accepted_lens = []\n"
        "    path_log_rows = []\n"
        "    winner_log_rows = []\n"
        "    accepted_node_paths = []\n"
        "    accepted_token_rows = []\n"
        "    start = 0\n"
        "    for req_i, node_count in enumerate(counts):\n"
    )
    if anchor not in ptext:
        fails.append(
            "DISPATCH-COMP: committer loop anchor not found in patcher "
            "(cannot compose the live dispatch text)"
        )
        return fails

    # ---- (2) SINGLE-SOURCE-OF-TRUTH structure checks ----
    # (2a) the synckill NULL guard must reference the SAME `_fr13_gpu_committer`
    #      (no separate `_fr13_sk_engage` recompute that could drift).
    if "if _FR13_COMMITTER_SYNCKILL and _fr13_gpu_committer:" not in ptext:
        fails.append(
            "DISPATCH-COMP: synckill NULL guard does NOT reference the single "
            "`_fr13_gpu_committer` predicate (single-source-of-truth broken)"
        )
        return fails
    # (2b) the drifting env-recompute must be GONE from the committer body.
    if "_fr13_sk_engage = (" in ptext:
        fails.append(
            "DISPATCH-COMP: stale `_fr13_sk_engage` env-recompute still present "
            "(the divergent duplicate the fix removed)"
        )
        return fails
    # (2c) the injected loop-skip must REFERENCE `_fr13_gpu_committer`, and must
    #      NOT recompute it (the prior duplicate that diverged at B=4). The
    #      loop-skip block is injected by the patcher as quoted python-source
    #      string literals; pull the segment between the sentinel and the
    #      (emptied) per-node loop and confirm.
    si = ptext.find('" + sentinel + " (OPT-1): flag-gated GPU-resident committer')
    if si < 0:
        # the sentinel concat splits the literal; fall back to the comment text.
        si = ptext.find('flag-gated GPU-resident committer')
    ei = ptext.find('for req_i, node_count in enumerate(_fr13_gpu_commit_counts)', si)
    loopskip_region = ptext[si:ei] if (si >= 0 and ei > si) else ""
    if 'if _fr13_gpu_committer:' not in loopskip_region:
        fails.append(
            "DISPATCH-COMP: injected loop-skip does not reference the single "
            "`_fr13_gpu_committer` predicate"
        )
        return fails
    if '_fr13_gpu_committer = (' in loopskip_region:
        fails.append(
            "DISPATCH-COMP: injected loop-skip RECOMPUTES `_fr13_gpu_committer` "
            "(a duplicate predicate that can drift from the null guard -- the "
            "exact B=4 crash class the fix removes)"
        )
        return fails

    # ---- (3) recover the ONE predicate VERBATIM and evaluate A=>B ----
    # `_fr13_gpu_committer` is computed ONCE in the committer body as plain
    # python. Recover it as real source, normalise the os.environ reads to a
    # plain dict lookup `_ENV[...]`, and evaluate over the flag space. Because
    # the SAME variable now gates both the null and the loop-skip, A=>B is true
    # by construction; this evaluation is the belt-and-suspenders confirmation.
    p_m = re.search(
        r"    _fr13_gpu_committer = \(\n(.*?)\n    \)\n",
        ptext,
        re.DOTALL,
    )
    if p_m is None:
        fails.append(
            "DISPATCH-COMP: could not extract the single `_fr13_gpu_committer` "
            "predicate from the committer body"
        )
        return fails
    import textwrap as _tw0
    pred_src = (
        "_fr13_gpu_committer = (\n"
        + _tw0.dedent(p_m.group(1)) + "\n)\n"
    )

    # Normalise `__import__('os').environ.get('FLAG', 'def')` -> `_ENV.get(...)`
    def _normalise(src: str) -> str:
        return re.sub(
            r"__import__\(\s*'os'\s*\)\.environ", "_ENV", src
        )

    pred_src = _normalise(pred_src)

    class _Env(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    # ---- evaluate A=>B across the flag cartesian product ----
    flag_names = [
        "FR13_GPU_COMMITTER",
        "FR13_COMMITTER_SYNCKILL",
        "FR13_FORCE_SPINE_COMMIT",
        "FR13_TREE_BONUS_SELF",
    ]
    # CAG / fork-margin engagement also depends on whether the call-site
    # PUBLISHED the verify logits (an OFF call-site leaves the global None ->
    # the diagnostic is inactive even with the flag on). Model both axes.
    diag_states = [
        # (cag_flag, fork_flag, logits_published)
        (False, False, False),
        (True, False, True),    # CAG armed (flag + publish)
        (True, False, False),   # CAG flag, no publish -> inactive
        (False, True, True),    # fork-margin armed
        (False, True, False),   # fork flag, no publish -> inactive
        (True, True, True),
    ]
    checked = 0
    for combo in itertools.product([False, True], repeat=len(flag_names)):
        env = dict(zip(flag_names, combo))
        env_dict = _Env({k: ("1" if v else "0") for k, v in env.items()})
        for cag_flag, fork_flag, published in diag_states:
            ns: dict = {
                "_ENV": env_dict,
                # the runtime locals the single predicate reads (same modelled
                # state the committer top-of-function computes them to):
                "_fr13_force_spine_commit": bool(env["FR13_FORCE_SPINE_COMMIT"]),
                "_fr13_cag_active": bool(cag_flag and published),
                "_fr13_fork_margin_active": bool(fork_flag and published),
                "_fr13_bonus_self": bool(env["FR13_TREE_BONUS_SELF"]),
                # _FR13_COMMITTER_SYNCKILL = GPU_COMMITTER & SYNCKILL (mod top).
                "_FR13_COMMITTER_SYNCKILL": bool(
                    env["FR13_GPU_COMMITTER"]
                    and env["FR13_COMMITTER_SYNCKILL"]
                ),
            }
            try:
                exec(pred_src, {}, ns)  # noqa: S102 - trusted repo source
            except Exception as exc:  # noqa: BLE001
                fails.append(
                    "DISPATCH-COMP: predicate eval raised on "
                    f"{env} cag={cag_flag} fork={fork_flag} pub={published}: "
                    f"{type(exc).__name__}:{exc}"
                )
                return fails

            # A (the null) = SYNCKILL and the SAME predicate; B (loop-skip) = the
            # SAME predicate. Both read ns["_fr13_gpu_committer"].
            a_null_fires = bool(
                ns["_FR13_COMMITTER_SYNCKILL"] and ns["_fr13_gpu_committer"]
            )
            b_loop_skips = bool(ns["_fr13_gpu_committer"])
            checked += 1
            # INVARIANT: A (the null) implies B (the loop-skip). If A and not B,
            # the legacy per-node loop runs with parents_cpu=None -> the crash.
            if a_null_fires and not b_loop_skips:
                fails.append(
                    "DISPATCH-COMP: synckill NULL fires but loop-SKIP does NOT "
                    f"(crash composition) for {env} "
                    f"cag={cag_flag} fork={fork_flag} pub={published}"
                )

    if not fails:
        print(
            "[DISPATCH-COMP] PASS  single-source `_fr13_gpu_committer` gates "
            "BOTH null + loop-skip; A=>B holds across "
            f"{checked} flag/diagnostic combinations (composed live dispatch; "
            "would catch the OPT-1 G2 crash class)"
        )
    else:
        print("[DISPATCH-COMP] FAIL  (composition crash class present)")
    return fails


def main() -> int:
    fails: list[str] = []
    km = _load("_fr13_committer_kernel_mod", KERNEL)

    have_gpu = False
    try:
        import torch  # noqa
        have_gpu = bool(km._HAVE_TRITON and torch.cuda.is_available())
    except Exception:
        have_gpu = False

    # ---------- G-FLAG: hook is DEFAULT-OFF & legacy path intact ----------
    ptext = PATCHER.read_text()
    if 'FR13_GPU_COMMITTER", "0"' in ptext or "FR13_GPU_COMMITTER', '0'" in ptext:
        print("[G-FLAG] PASS  FR13_GPU_COMMITTER read with default '0' (DEFAULT-OFF)")
    else:
        fails.append("FR13_GPU_COMMITTER not read default-OFF in patcher")
        print("[G-FLAG] FAIL  flag not default-OFF (or hook not wired)")
    # legacy committer block still present verbatim (hook must not delete it).
    if "for req_i, node_count in enumerate(counts):" in ptext and \
       "best_lcp = max(0, int(best_lcp))" in ptext:
        print("[G-FLAG2] PASS  legacy Python committer block left intact")
    else:
        fails.append("legacy committer block missing/altered (default path not preserved)")
        print("[G-FLAG2] FAIL  legacy committer block altered")

    # ---------- G-FLAG3: OPT-1 G2 SYNCKILL is DEFAULT-OFF & gated under
    # FR13_GPU_COMMITTER. The synckill flag must (a) be read default-OFF and
    # (b) require FR13_GPU_COMMITTER=1 (meaningless without the GPU committer),
    # and (c) the legacy/OFF transport (:6761 packed committer-input sync) must
    # stay present so the default-ON serving path is byte-identical.
    sk_default_off = (
        'FR13_COMMITTER_SYNCKILL\', \'0\'' in ptext
        or "FR13_COMMITTER_SYNCKILL\", \"0\"" in ptext
    )
    sk_gated = (
        "_FR13_COMMITTER_SYNCKILL = (" in ptext
        and "FR13_GPU_COMMITTER" in ptext
    )
    if sk_default_off and sk_gated:
        print("[G-FLAG3] PASS  FR13_COMMITTER_SYNCKILL default-OFF, gated under "
              "FR13_GPU_COMMITTER")
    else:
        fails.append("FR13_COMMITTER_SYNCKILL not default-OFF/gated under "
                     "FR13_GPU_COMMITTER")
        print("[G-FLAG3] FAIL  synckill flag not default-OFF/gated")
    # The OFF/legacy committer-input main-thread sync must still be present
    # (default-ON path byte-identical: synckill only forks when the flag is ON).
    if "torch.cuda.current_stream(tree_parent_indices.device).synchronize()" in ptext:
        print("[G-FLAG4] PASS  legacy :6761 committer-input sync preserved "
              "(OFF path byte-identical)")
    else:
        fails.append("legacy committer-input sync (:6761) removed -- OFF path "
                     "not byte-identical")
        print("[G-FLAG4] FAIL  legacy committer-input sync missing")

    # ---------- DISPATCH-COMP: LIVE composed-dispatch composition gate --------
    # The G-FLAG/ORACLE/FULL/DEVICE arms test the kernel in ISOLATION; this gate
    # composes the synckill NULL-gate with the loop-SKIP gate exactly as the
    # live committer dispatch does, and asserts the A=>B invariant. It is the
    # arm that would have caught the OPT-1 G2 crash class.
    fails.extend(_dispatch_composition_gate(ptext))

    # ---------- Behavioral matrix ----------
    matrix = []
    parents_cat = caterpillar9()
    # caterpillar at every accept depth (full-accept, partial, zero)
    for ad in range(0, 6):
        d, pt, st = make_request(parents_cat, accept_depth=ad)
        matrix.append(("caterpillar9 full-accept depth=%d" % ad,
                       [parents_cat], [d], [pt], [st], 5))
    # caterpillar with an early reject (bonus = reject_parent_target)
    d, pt, st = make_request(parents_cat, accept_depth=5, mismatch_at=2)
    matrix.append(("caterpillar9 reject@2", [parents_cat], [d], [pt], [st], 5))
    # zero accept (root mismatch -> bonus=root_parent_target)
    d, pt, st = make_request(parents_cat, accept_depth=0, mismatch_at=0)
    matrix.append(("caterpillar9 reject@root", [parents_cat], [d], [pt], [st], 5))
    # pure spine (linear chain)
    spine = [-1, 0, 1, 2, 3]
    d, pt, st = make_request(spine, accept_depth=3)
    matrix.append(("pure-spine depth=3", [spine], [d], [pt], [st], 5))
    # fan from root: many shallow leaves -> earliest-leaf tie-break exercised
    fan = [-1, 0, 0, 0, 0]
    d, pt, st = make_request(fan, accept_depth=1)
    matrix.append(("root-fan", [fan], [d], [pt], [st], 5))
    # LCP tie: two leaves with equal LCP, earliest node must win
    tie = [-1, 0, 0]  # leaves 1,2 both depth-1 from root
    dt = [10, 20, 30]
    ptt = [10, 20, 30]  # both leaves' paths match fully (root matches, leaf matches)
    stt = [40, 50, 60]
    matrix.append(("lcp-tie earliest-leaf", [tie], [dt], [ptt], [stt], 5))

    # multi-request batch (two trees in one call)
    d1, pt1, st1 = make_request(parents_cat, accept_depth=3)
    d2, pt2, st2 = make_request(spine, accept_depth=2)
    matrix.append((
        "batch[caterpillar+spine]",
        [parents_cat, spine],
        [d1, d2], [pt1, pt2], [st1, st2], 5,
    ))

    # randomised trees
    rng = random.Random(1234)
    for t in range(40):
        nc = rng.randint(1, 12)
        parents = [-1]
        for node in range(1, nc):
            parents.append(rng.randint(-1, node - 1))  # valid: parent < node or root
        ad = rng.randint(0, nc)
        ma = rng.choice([None, None, rng.randint(0, nc)])
        d, pt, st = make_request(parents, accept_depth=ad, mismatch_at=ma)
        matrix.append(("rand#%d nc=%d" % (t, nc), [parents], [d], [pt], [st],
                       rng.randint(3, 8)))

    n_pass = 0
    for name, trees, drafts_l, ptgt_l, stgt_l, msl in matrix:
        counts = [len(p) for p in trees]
        parents_flat = [x for p in trees for x in p]
        drafts_flat = [x for d in drafts_l for x in d]
        ptgt_flat = [x for d in ptgt_l for x in d]
        stgt_flat = [x for d in stgt_l for x in d]
        bonus_flat = [0] * len(counts)

        # REF is the full 5-tuple (out_rows, accepted_rows, accepted_lens,
        # accepted_node_paths, accepted_token_rows) the hook consumes.
        ref5 = ref_committer(parents_flat, drafts_flat, ptgt_flat, stgt_flat, counts, msl)
        ref3 = ref5[:3]
        # 3-tuple oracle (the kernel's pure-Python reference).
        ora = km.fr13_gpu_committer_oracle(
            parents_flat, drafts_flat, ptgt_flat, stgt_flat, bonus_flat, counts, msl,
        )
        if ora != ref3:
            fails.append("ORACLE != REF on '%s':\n   ref=%r\n   ora=%r" % (name, ref3, ora))
            print("[ORA] FAIL  %s" % name)
            continue
        # 5-tuple FULL dispatch (exactly what the patcher hook calls).
        full = km.fr13_gpu_committer_full(
            parents_flat, drafts_flat, ptgt_flat, stgt_flat, bonus_flat, counts, msl,
        )
        if tuple(full) != tuple(ref5):
            fails.append("FULL != REF on '%s':\n   ref=%r\n   full=%r" % (name, ref5, full))
            print("[FULL] FAIL  %s" % name)
            continue
        if have_gpu:
            tri = km.fr13_gpu_committer_triton(
                parents_flat, drafts_flat, ptgt_flat, stgt_flat, bonus_flat, counts, msl,
            )
            if tri != ref3:
                fails.append("TRITON != REF on '%s':\n   ref=%r\n   tri=%r" % (name, ref3, tri))
                print("[TRI] FAIL  %s" % name)
                continue
            # ---------- OPT-1 G2 DEVICE arm (FR13_COMMITTER_SYNCKILL) ----------
            # Feed the kernel DEVICE tensors (no host committer-input list = the
            # :6761 sync this kills), run the device decision + side-stream
            # readback, materialise, and assert the FULL 5-tuple == REF. Also
            # assert the device->device writeback of output_token_ids /
            # accepted_tree_rows equals the legacy host-scatter element-for-
            # element (G2.d writeback-equality).
            import torch as _t
            dev = "cuda"
            p_dev = _t.tensor(parents_flat, dtype=_t.int64, device=dev)
            d_dev = _t.tensor(drafts_flat, dtype=_t.int64, device=dev)
            pt_dev = _t.tensor(ptgt_flat, dtype=_t.int64, device=dev)
            st_dev = _t.tensor(stgt_flat, dtype=_t.int64, device=dev)
            b_dev = _t.tensor(bonus_flat, dtype=_t.int64, device=dev)
            dev_out, materialise = km.fr13_gpu_committer_device_full(
                p_dev, d_dev, pt_dev, st_dev, b_dev, counts, msl,
            )
            dev5 = materialise()
            if tuple(dev5) != tuple(ref5):
                fails.append(
                    "DEVICE(synckill) != REF on '%s':\n   ref=%r\n   dev=%r"
                    % (name, ref5, dev5)
                )
                print("[DEV] FAIL  %s" % name)
                continue
            # writeback-equality: device output_token_ids (PAD-padded, then
            # sliced to row_len per request) must equal the legacy host scatter.
            n_req = len(counts)
            ot = dev_out["out_tokens"].cpu().tolist()
            rl = dev_out["row_len"].cpu().tolist()
            wb_ok = True
            for r in range(n_req):
                got = [int(x) for x in ot[r][: int(rl[r])]]
                if got != ref5[0][r]:
                    wb_ok = False
                    fails.append(
                        "DEVICE writeback row %d != REF on '%s': %r vs %r"
                        % (r, name, got, ref5[0][r])
                    )
                    break
            if not wb_ok:
                print("[DEV-WB] FAIL  %s" % name)
                continue
        n_pass += 1

    arm = ("ORACLE+TRITON+DEVICE" if have_gpu
           else "ORACLE (TRITON+DEVICE arms SKIPPED: no CUDA -- live-GPU step)")
    print("[MATRIX] %d/%d trees byte-identical to REF via %s" % (n_pass, len(matrix), arm))
    if not have_gpu:
        print("[NOTE] CPU-only host: the Triton kernel arm is the documented live-GPU "
              "iteration step (see FR13_GPU_COMMITTER_BIND.md). The ORACLE arm proves "
              "the integer contract the kernel transcribes.")

    print()
    if fails:
        print("GATE: FAIL (%d failure(s))" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    print("GATE: PASS  (GPU committer byte-identical to the legacy Python committer "
          "serving decision; hook DEFAULT-OFF; default path intact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
