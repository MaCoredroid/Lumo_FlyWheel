#!/usr/bin/env python3
"""One-shot edit of scripts/fr10_phase4_patch_vllm_tree_gdn.py: add the FR13_TREE_PENALTY_HISTORY patch
(per-path penalty histories for tree verification rows) and route the tree self-row processor calls through
RejectionSampler.apply_logits_processors_tree_self. Idempotent: refuses if already applied. Every anchor must
match exactly the expected number of times."""
import sys, re
P = sys.argv[1]
t = open(P).read()
if "_patch_rejection_sampler_tree_penalty_history" in t:
    print("already applied"); sys.exit(0)

HELPERS = r'''# ---- FR13_TREE_PENALTY_HISTORY --------------------------------------------------------------------
# vLLM's _combine_outputs_with_spec_tokens gives verification row j the history output + spec[:j], which
# is the ancestry of a CHAIN. TreeHost verifies a TREE: target row j is the distribution at parent(j),
# so its penalty history is output + the tokens on the root->parent(j) path; the tree "self" row j is the
# distribution at node j and additionally contains node j's own token. Penalties (presence / frequency /
# repetition) are therefore built per path from metadata.tree_parent_indices. The kind of the current call
# ("target" by default, "self" inside apply_logits_processors_tree_self) selects which history a row gets.
_FR13_TREE_HIST_KIND = "target"
_FR13_TREE_DP_CACHE: dict = {}
_FR13_TREE_HIST_STATS = {"calls": 0, "tree_calls": 0, "self_calls": 0, "rows": 0}


def _fr13_tree_draft_parents(metadata, spec_token_ids):
    """Per-request draft-space parent tables (root = -1) from metadata.tree_parent_indices. The runner
    builds that tensor as the concatenation, over the requests that carry drafts, of the tree's parent
    template in draft order (request-local indices, no batch offset); requests with no drafts contribute
    nothing. Returns a list aligned with spec_token_ids (None for a request without drafts), or None
    when the batch carries no tree."""
    tpi = getattr(metadata, "tree_parent_indices", None)
    if tpi is None or spec_token_ids is None:
        return None
    lens = tuple(len(s) for s in spec_token_ids)
    key = (int(tpi.data_ptr()), int(tpi.numel()), lens)
    hit = _FR13_TREE_DP_CACHE.get(key)
    if hit is not None:
        return hit
    phys = [int(x) for x in tpi.detach().cpu().tolist()]
    if sum(lens) != len(phys):
        raise RuntimeError(
            f"FR13_TREE_PENALTY_HISTORY: tree_parent_indices numel={len(phys)} but the batch carries "
            f"{sum(lens)} draft tokens over {len(lens)} requests")
    tables = []
    pos = 0
    for r, n in enumerate(lens):
        if n == 0:
            tables.append(None)
            continue
        seg = phys[pos:pos + n]
        pos += n
        for j, p in enumerate(seg):
            if not (-1 <= p < j):
                raise RuntimeError(
                    f"FR13_TREE_PENALTY_HISTORY: request {r} draft {j} has parent {p}; "
                    "parents must be -1 (root) or an earlier draft")
        tables.append(tuple(seg))
    if len(_FR13_TREE_DP_CACHE) >= 64:
        _FR13_TREE_DP_CACHE.clear()
    _FR13_TREE_DP_CACHE[key] = tables
    return tables


def _fr13_tree_combine_outputs(output_token_ids, spec_token_ids, draft_parents, kind):
    """Row histories for tree verification rows (len(spec) rows per request, like the chain combiner).
    'target': row j gets output + path(parent(j)); 'self': row j gets output + path(j), path inclusive."""
    if spec_token_ids is None:
        return output_token_ids
    if kind not in ("target", "self"):
        raise RuntimeError(f"FR13_TREE_PENALTY_HISTORY: unknown history kind {kind!r}")
    result = []
    for ri, (out, spec) in enumerate(zip(output_token_ids, spec_token_ids)):
        if len(spec) == 0:
            continue
        dp = draft_parents[ri]
        if dp is None or len(dp) != len(spec):
            raise RuntimeError(
                f"FR13_TREE_PENALTY_HISTORY: request {ri} has {len(spec)} draft tokens but a "
                f"{0 if dp is None else len(dp)}-node tree parent table")
        paths = []
        for j in range(len(spec)):
            p = dp[j]
            paths.append((paths[p] if p >= 0 else []) + [spec[j]])
        for j in range(len(spec)):
            if kind == "self":
                hist = paths[j]
            else:
                p = dp[j]
                hist = paths[p] if p >= 0 else []
            result.append([*out, *hist])
        _FR13_TREE_HIST_STATS["rows"] += len(spec)
    return result
# ---- end FR13_TREE_PENALTY_HISTORY ----------------------------------------------------------------


'''

NEWFUNC = r'''

def _patch_rejection_sampler_tree_penalty_history() -> bool:
    """FR13_TREE_PENALTY_HISTORY: per-path penalty histories for tree verification rows (defect found by
    the 2026-10-02 pendiag capture: all 2,480 captured tree rows carried vLLM's flattened-chain history,
    output + spec[:j]; only the first token of each step was penalised correctly). Adds module helpers,
    a tree-aware branch in apply_logits_processors (chain path untouched when the batch has no tree), and
    RejectionSampler.apply_logits_processors_tree_self for the tree self rows (row j = distribution AT
    node j, history includes node j's token). The forward's tree_self call and the STEP_GRAPH=2 wrapper's
    self calls are routed through the new method by the runner-side patches."""
    text = REJECTION_SAMPLER_PATH.read_text()
    if "_FR13_TREE_HIST_KIND" in text:
        return False
    cls_anchor = "class RejectionSampler(nn.Module):\n"
    if text.count(cls_anchor) != 1:
        raise RuntimeError(f"FR13_TREE_PENALTY_HISTORY class anchor count={text.count(cls_anchor)}")
    text = text.replace(cls_anchor, _FR13_TREE_PENALTY_HISTORY_HELPERS + cls_anchor, 1)
    call_anchor = (
        "        output_token_ids = sampling_metadata.output_token_ids\n"
        "        if any_penalties_or_bad_words:\n"
        "            output_token_ids = self._combine_outputs_with_spec_tokens(\n"
        "                output_token_ids,\n"
        "                sampling_metadata.spec_token_ids,\n"
        "            )\n"
    )
    if text.count(call_anchor) != 1:
        raise RuntimeError(f"FR13_TREE_PENALTY_HISTORY combiner anchor count={text.count(call_anchor)}")
    call_new = (
        "        output_token_ids = sampling_metadata.output_token_ids\n"
        "        if any_penalties_or_bad_words:\n"
        "            # FR13_TREE_PENALTY_HISTORY: tree rows take per-path histories; chain rows keep\n"
        "            # the stock combiner (output + spec[:j]).\n"
        "            _fr13_tdp = _fr13_tree_draft_parents(metadata, sampling_metadata.spec_token_ids)\n"
        "            _FR13_TREE_HIST_STATS[\"calls\"] += 1\n"
        "            if _fr13_tdp is not None:\n"
        "                _FR13_TREE_HIST_STATS[\"tree_calls\"] += 1\n"
        "                output_token_ids = _fr13_tree_combine_outputs(\n"
        "                    output_token_ids,\n"
        "                    sampling_metadata.spec_token_ids,\n"
        "                    _fr13_tdp,\n"
        "                    _FR13_TREE_HIST_KIND,\n"
        "                )\n"
        "            else:\n"
        "                output_token_ids = self._combine_outputs_with_spec_tokens(\n"
        "                    output_token_ids,\n"
        "                    sampling_metadata.spec_token_ids,\n"
        "                )\n"
    )
    text = text.replace(call_anchor, call_new, 1)
    meth_anchor = "    @staticmethod\n    def apply_penalties(\n"
    if text.count(meth_anchor) != 1:
        raise RuntimeError(f"FR13_TREE_PENALTY_HISTORY method anchor count={text.count(meth_anchor)}")
    meth_new = (
        "    def apply_logits_processors_tree_self(\n"
        "        self,\n"
        "        logits: torch.Tensor,\n"
        "        sampling_metadata: SamplingMetadata,\n"
        "        metadata: SpecDecodeMetadata,\n"
        "    ) -> torch.Tensor:\n"
        "        \"\"\"FR13_TREE_PENALTY_HISTORY: processors for the tree SELF rows (row j is the\n"
        "        distribution at node j, so its penalty history includes node j's own token).\n"
        "        Delegates to apply_logits_processors with the history kind set to 'self'.\"\"\"\n"
        "        global _FR13_TREE_HIST_KIND\n"
        "        prev = _FR13_TREE_HIST_KIND\n"
        "        _FR13_TREE_HIST_KIND = \"self\"\n"
        "        _FR13_TREE_HIST_STATS[\"self_calls\"] += 1\n"
        "        try:\n"
        "            return self.apply_logits_processors(logits, sampling_metadata, metadata)\n"
        "        finally:\n"
        "            _FR13_TREE_HIST_KIND = prev\n"
        "\n"
    )
    text = text.replace(meth_anchor, meth_new + meth_anchor, 1)
    REJECTION_SAMPLER_PATH.write_text(text)
    return True
'''

def must(count, what, expect):
    if count != expect:
        raise SystemExit(f"anchor {what}: found {count}, expected {expect}")

# 1. helpers constant + new patch function, inserted before the handoff patch function
a = "\n\ndef _patch_rejection_sampler_target_logits_handoff() -> bool:\n"
must(t.count(a), "handoff def", 1)
t = t.replace(a, "\n\n_FR13_TREE_PENALTY_HISTORY_HELPERS = " + repr(HELPERS) + "\n" + NEWFUNC + a, 1)

# 2. register in main() right after the handoff entry
a = "        (REJECTION_SAMPLER_PATH, _patch_rejection_sampler_target_logits_handoff()),\n"
must(t.count(a), "main handoff entry", 1)
t = t.replace(a, a + "        # FR13_TREE_PENALTY_HISTORY: per-path penalty histories for tree rows (self rows\n"
                   "        # routed through apply_logits_processors_tree_self by the runner patches below).\n"
                   "        (REJECTION_SAMPLER_PATH, _patch_rejection_sampler_tree_penalty_history()),\n", 1)

# 3. forward's tree_self call (eager route, the deployed FR13_STEP_GRAPH=0 path)
a = ("            tree_self_logits = self.apply_logits_processors(\n"
     "                tree_self_logits, sampling_metadata, metadata\n"
     "            )\n")
must(t.count(a), "eager tree_self call", 1)
t = t.replace(a, a.replace("self.apply_logits_processors(", "self.apply_logits_processors_tree_self("), 1)

# 4. STEP_GRAPH=2 wrapper self calls (three sites; inert on the deployed =0 route)
a = '        "                    _fr13_sg_stls = self.rejection_sampler.apply_logits_processors(\\n"\n'
must(t.count(a), "sg stls assign", 1)
t = t.replace(a, a.replace("apply_logits_processors(", "apply_logits_processors_tree_self("), 1)
a = '        "                    _fr13_sg_stls.copy_(self.rejection_sampler.apply_logits_processors(\\n"\n'
must(t.count(a), "sg stls copy_", 1)
t = t.replace(a, a.replace("apply_logits_processors(", "apply_logits_processors_tree_self("), 1)
a = ('        "                    _fr13_sg_ent[\\"stls\\"].copy_(\\n"\n'
     '        "                        self.rejection_sampler.apply_logits_processors(\\n"\n')
must(t.count(a), "sg ent stls copy_", 1)
t = t.replace(a, a.replace("apply_logits_processors(", "apply_logits_processors_tree_self("), 1)

# the remaining apply_logits_processors( occurrences must all be target-row calls
rest = [m.start() for m in re.finditer(r"apply_logits_processors\(", t)]
print("remaining apply_logits_processors( occurrences:", len(rest))
open(P, "w").write(t)
print("applied")
