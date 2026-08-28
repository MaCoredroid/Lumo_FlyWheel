# Comment for RFC #54080 (TreeWY) — v6 (branch-read relevance rewrite). Mark posts.

> Every paragraph now hooks a verified line of HER code/docs (quotes checked
> at source: config/vllm.py:1450-1458 mask-channel constraint;
> rejection_sampler.py:828-832 unverified-p limitation). Cut: fixed-shape
> lead (she has it), 28.8/GB10 line (marketing to her thread), the hedge
> "if your branch already covers it" (it does — we say what we found).

---

Congrats on the paper and the branch — the `N+1`-snapshot framing and the
memory table at 397B are the part I hadn't seen quantified anywhere.

One data point on the capture blocker, and I read the branch first: your GDN
commit already captures, and `config/vllm.py` names the real constraint —
the width>1 mask "routes them to FlashInfer's prefill wrapper. vLLM builds
that wrapper without cudagraph buffers, so the mask cannot be replayed."
That's a property of the mask *channel*, not of branching verify. We run a
tree verifier for GDN hybrids out-of-tree (27B Qwen hybrid, our own vLLM
fork) where the ancestor mask never goes through a host-side `plan()`: it is
a persistent device buffer added to the score tile inside an FA2 varlen
fork, so the graph replays it like any other static-address input. Full
capture held at width > 1, no PIECEWISE. Fixed node count and padded rows
were necessary but not sufficient — you already have both. A second data
point, not a rebuttal of your diagnosis.

What it's worth, roughly: your docs put `[3,3,3]` at 11,523 → 1,614 tok/s.
Our captured tree ran 32.85 vs 42.74 tok/s against native MTP-5 at B=4
while accepting 4.286 vs 3.422 — same sign as your sweep, ~23% rather than
~7×. So capture looks like most of that gap, and verify width is the rest.
(Details: https://macoredroid.github.io/Lumo_FlyWheel/.)

On correctness, one thing that bears on the paper's "identical acceptance
length." We hold the same bar you state for all-ones widths — greedy output
byte-exact against native decode at fan-out 1 — enforced as a boot-time and
per-generation contract. What bit us is that byte-exact *state* does not
imply byte-exact output: a reduction-order change alone, state still
matching to tolerance, cost us 0.087 tok/event of acceptance at greedy.
Your WY reconstruct is checked against `chunk_gated_delta_rule` at
`atol=2e-2`, and your rejection sampler already flags the same shape of
risk — the reconstructed `p` "has NOT been verified to agree bit-for-bit …
a disagreement would bias acceptance rather than corrupt state." That's
exactly the failure an acceptance-length parity harness catches and a
numerical-closeness test does not. We have CPU fixtures for it —
reduction-order reassociation, and greedy tie-break determinism aimed at
`logits.argmax` versus the chain sampler's greedy path, which your width-1
byte-identical claim rests on. Implementation-agnostic; they'd cover #47572
equally and I'd like to contribute them either way.

One question, and I read the branch before asking. Of the three substrate
pieces I have ready — per-node parent indexing, a declared carry budget, a
replay-on-commit hook — your branch already has working equivalents of all
three: `SpecDecodeMetadata.draft_parents`/`draft_depths`, the stash shape
with `num_speculative_blocks = 0`, and the lazy commit keyed on
`accepted_leaf_ids`. They're private to TreeWY, and #47576 and #54103 are
each growing their own. Your own framing is the argument for sharing them:
if a commit is "an ancestor-masked reduction rather than a cursor move,"
then a shared hook has to carry a mask, not a cursor — which is the one
thing none of the three in-flight commit paths agree on. So the question
isn't whether you need these; it's whether yours become the shared ones.
Under this RFC as its substrate piece, or a separate interface RFC yours
depends on? You mention wanting to collaborate with the ReplaySSM team —
this is the smallest surface that makes that mechanical.

_Disclosure: AI-assisted analysis; I ran the benchmarks and reviewed the
traces myself._
