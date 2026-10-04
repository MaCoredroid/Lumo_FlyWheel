#!/usr/bin/env python3
"""CPU pre-flight for FR13_TREE_PENALTY_HISTORY.
1. Exec the helper source embedded in the patcher; build per-row histories for the hydra27 tree from a fake
   tree_parent_indices (physical rows, root first) and check every target/self row against the per-path sets
   analyze_pen.py uses (the reference the 2026-10-02 audit was scored with).
2. Replay the 2026-10-02 pendiag capture: the chain prediction must reproduce the captured defect (all rows
   flattened) and the fixed combiner must equal the correct per-path set on every captured row.
3. Apply the two rejection-sampler patch functions to a copy of the stock vLLM file and byte-compile it."""
import importlib.util, json, os, py_compile, re, sys, tempfile
sys.path.insert(0, "/home/mark/shared/lumotree-v2exp-20260930/scripts")
sys.path.insert(0, "/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/pendiag")
import fr13_fixed32_topology as T
PATCHER = "/home/mark/shared/lumotree-v2exp-20260930/scripts/fr10_phase4_patch_vllm_tree_gdn.py"
STOCK = sys.argv[1]            # stock rejection_sampler.py extracted from the image
TRACE = sys.argv[2]            # pen_trace.jsonl from the 2026-10-02 capture

spec = importlib.util.spec_from_file_location("fr10p", PATCHER)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ns = {}; exec(mod._FR13_TREE_PENALTY_HISTORY_HELPERS, ns)
combine, parents_of = ns["_fr13_tree_combine_outputs"], ns["_fr13_tree_draft_parents"]
DP = [int(p) for p in T.DRAFT_PARENT]; N = len(DP); assert N == 31
PHYS = [-1] + [p + 1 for p in DP]  # physical-row view (unused by the runner; kept for reference)

class FakeT:
    def __init__(self, v): self.v = v
    def data_ptr(self): return id(self.v)
    def numel(self): return len(self.v)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.v)
class Meta: pass

def path_tokens(u, spec):
    out = []
    while u >= 0:
        out.append(spec[u]); u = DP[u]
    return set(out)

# --- 1. synthetic: the runner's layout (draft-space parents per tree request, request-local, root -1;
#        requests without drafts omitted from the tensor), B=1 and B=3 with an empty-draft request in the middle
for specs in ([list(range(N))], [[10 + j for j in range(N)], [], [50 + j for j in range(N)]]):
    tree_reqs = [i for i, s in enumerate(specs) if s]
    m = Meta(); m.tree_parent_indices = FakeT(DP * len(tree_reqs))
    tabs = parents_of(m, specs)
    assert len(tabs) == len(specs) and all((tabs[i] is None) == (not specs[i]) for i in range(len(specs)))
    assert all(list(tabs[i]) == DP for i in tree_reqs), tabs
    outs = [[1000 + r, 2000 + r] for r in range(len(specs))]
    for kind in ("target", "self"):
        rows = combine(outs, specs, tabs, kind)
        assert len(rows) == N * len(tree_reqs)
        k = 0
        for r in tree_reqs:
            for j in range(N):
                row = rows[k]; k += 1
                assert row[:2] == outs[r], row
                extra = set(row[2:])
                node = DP[j] if kind == "target" else j
                exp = path_tokens(node, specs[r]) if node >= 0 else set()
                assert extra == exp, (kind, r, j, extra, exp)
                assert len(row[2:]) == len(exp), "path tokens must appear once each (frequency penalty)"
print("synthetic: target/self histories match per-path sets for B=1 and B=3 with an empty-draft request")
# cache hit path
m = Meta(); m.tree_parent_indices = FakeT(list(DP)); one = [list(range(N))]
assert parents_of(m, one) is parents_of(m, one)
# malformed tables must raise, never silently fall back
for bad in ([0] + DP[1:], [-1, -1, 5] + DP[3:], DP[:-1], DP + [0]):
    m = Meta(); m.tree_parent_indices = FakeT(bad)
    try:
        parents_of(m, one); raise SystemExit(f"malformed table accepted: {bad[:6]} len={len(bad)}")
    except RuntimeError:
        pass
print("malformed or mis-sized parent tables raise")
# no tree => None (chain path)
assert parents_of(Meta(), one) is None

# --- 2. captured rows
calls = [json.loads(l) for l in open(TRACE)]
calls = [c for c in calls if "rows" in c and c["nrows"] == 31]
steps = [(calls[i], calls[i + 1]) for i in range(0, len(calls) - 1, 2)]
m = Meta(); m.tree_parent_indices = FakeT(list(DP)); tabs = parents_of(m, [list(range(N))])
n_rows = flat_ok = fixed_ok = 0
for tgt, slf in steps:
    spec_ = tgt["spec"][0]; assert len(spec_) == 31
    base = set(tgt["rows"][0]["pen_ids"])
    for kind, call in (("target", tgt), ("self", slf)):
        rows = combine([[]], [spec_], tabs, kind)
        for j, r in enumerate(call["rows"]):
            pen = set(r["pen_ids"]); n_rows += 1
            flat_ok += (pen == base | set(spec_[:j]))                      # what the deployed run did
            node = DP[j] if kind == "target" else j
            correct = base | (path_tokens(node, spec_) if node >= 0 else set())
            fixed_ok += (base | set(rows[j]) == correct)                    # what the fix produces
print(f"capture: rows={n_rows} captured==flattened-chain {flat_ok}/{n_rows}; fixed==per-path {fixed_ok}/{n_rows}")
assert n_rows == 2480 and flat_ok == n_rows and fixed_ok == n_rows

# --- 3. apply the sampler patches to the stock file and compile
tmp = tempfile.mkdtemp(); dst = os.path.join(tmp, "rejection_sampler.py")
open(dst, "w").write(open(STOCK).read())
from pathlib import Path
mod.REJECTION_SAMPLER_PATH = Path(dst)
assert mod._patch_rejection_sampler_target_logits_handoff() is True
assert mod._patch_rejection_sampler_tree_penalty_history() is True
assert mod._patch_rejection_sampler_tree_penalty_history() is False   # idempotent
assert mod._patch_rejection_sampler_bonus_handoff() is True
py_compile.compile(dst, doraise=True)
txt = open(dst).read()
for s in ("def apply_logits_processors_tree_self", "_fr13_tree_combine_outputs(", "_FR13_SG_TL_QUEUE", "_fr13_tree_draft_parents(metadata, sampling_metadata.spec_token_ids)"):
    assert s in txt, s
print("patched stock rejection_sampler.py compiles; handoff + tree-history + bonus patches coexist")
print("ALL PRE-FLIGHT CHECKS PASSED")
