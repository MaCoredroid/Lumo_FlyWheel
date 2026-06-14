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
        n_pass += 1

    arm = "ORACLE+TRITON" if have_gpu else "ORACLE (TRITON arm SKIPPED: no CUDA -- live-GPU step)"
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
