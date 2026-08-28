# Comment for RFC #54080 (TreeWY) — v8 (owners-audit edit pass applied). Mark posts.

> DECOUPLED: posts FIRST, no RFC number ("I'll link it here"). One item needs
> Mark's confirmation before posting: the capture-set qualifier "decode-only
> ragged batches" — confirm that matches what our fixed32 serve actually
> captures (mixed prefill/decode was NOT in our capture set, correct?).
> All owner-facing claims below were verified against their primary sources.

---

Congrats on the paper and the branch — the `N+1`-snapshot framing and the
memory table at 397B are the part I hadn't seen quantified anywhere.

One data point on the capture blocker, and I read the branch first: your GDN
commit is *written* to capture — static slot-indexed stash, masks built at
`__init__`, no host sync — and what takes it away at width>1 is the global
`cudagraph_mode` PIECEWISE downgrade, which by your §4 evicts the GDN mixer
from graphs too. `config/vllm.py` names the constraint exactly: the width>1
mask "routes them to FlashInfer's prefill wrapper. vLLM builds that wrapper
without cudagraph buffers, so the mask cannot be replayed." That's a
property of how vLLM constructs that wrapper, not of branching verify. In
our out-of-tree fork (27B Qwen hybrid, FA2 varlen) the ancestor mask never
goes through a host-side `plan()`: it is a persistent device buffer added to
the score tile, so the graph replays it like any other static-address input.
Full capture held at width > 1 on decode-only ragged batches — no
`cudagraph_mode` PIECEWISE fallback.

Our sign matches yours: in our matched B=4 control the tree accepted 4.286
vs native MTP-5's 3.422 yet ran 32.85 vs 42.74 tok/s (tied 10/16 on tasks
resolved); a later lever stack brought the best arm to ~parity with our
native fit. Your docs put the chain at 11,523 tok/s and `[3,3,3]` at 1,614 —
not comparable point-for-point to ours (different model, GPU, drafter, and
21 nodes against your 39), and I'm not apportioning your gap between the
PIECEWISE downgrade and the extra verify tokens; your README names both
causes, and one B=4 point on other hardware can't separate them. The only
claim is that a captured branching verify exists. (Details:
https://macoredroid.github.io/Lumo_FlyWheel/.)

On correctness: your all-ones-width control is byte-identical to the chain
within `reconstruct`; ours is a different anchor — greedy output byte-exact
against a *no-speculation* reference at fan-out 1, enforced as a boot-time
and per-generation contract. That anchor is what made this visible:
byte-exact state does not imply byte-exact output — a reduction-order change
alone cost us 0.087 tok/event of acceptance at greedy. You already compare
acceptance length end-to-end (§5's 175 matched points, mean |Δ| 0.039), as
does the ReplaySSM series — the gap is resolution, not method: at
max |Δ| 0.33 that comparison can't resolve an 0.087-class shift, and your
Appendix B Table 4 has treewy below storeall in all three shapes
(mean −0.085), captioned as sampling noise — the same sign three times is
worth a deterministic check. Your closed form is pinned tight (1e-5 against
the naive recurrence, ~1e-15 in fp64); where drift can live is the bf16
kernel layer (2e-2 vs `chunk_gated_delta_rule`, 1.95e-3 for the production
V-hoisted mixer) — and F21HGG's report today on #49887 shows the same class
live upstream ("first diverged … at output token 8 … the chunked replay
transform changes arithmetic order"). We have deterministic CPU fixtures for
exactly this — reduction-order reassociation, and greedy tie-break
determinism (`logits.argmax` versus the chain Triton kernel's greedy path on
vocab ties — sibling ties you already break deterministically). They'd apply
to #49847/#49887 and your branch alike; I'd like to contribute them either
way.

Last thing, read from the branch. Of the three substrate pieces we have
ready — per-node parent indexing, a declared carry budget, a
replay-on-commit hook — your branch carries working equivalents:
`SpecDecodeMetadata.draft_parents`/`draft_depths`; the
`num_speculative_blocks = 0` stash shape (which `main` now also carries from
#51855 under a different predicate, and which ReplaySSM wrote first in
July); and the lazy commit keyed on `accepted_leaf_ids`. #49847/#49887 and
#54103 are each growing their own. Your framing names the design question: a
commit that is "an ancestor-masked reduction rather than a cursor move," and
ReplaySSM's ring — whose circular indexing was chosen "for tree-based
speculative decoding, where accepted tokens are no longer contiguous in the
buffer" — are two answers to one question. Neither a scalar cursor nor a
mask alone covers both; an accepted-node index list derives both. We're
preparing an interface RFC proposing that shared surface, with a draft PR —
I'll link it here. One question that is yours to answer: do those shapes fit
TreeWY's commit path as-is, or does the ancestor-masked reduction need
something a shared hook doesn't carry yet?

_Disclosure: AI-assisted analysis; I ran the benchmarks and reviewed the
traces myself._
