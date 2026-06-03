# FR10: What "lossless" means for MTP + token-tree verification

Author: Claude (online researcher + red-team). Date: 2026-06-03.
Purpose: pin the precise definition(s) of "lossless" for our setting, where the
candidate **tree is drafted by the Qwen3.6 MTP head** and verified by the base
model. This grounds Gates B/C/D and the tree rejection sampler we still owe.

## The reference: lossless *with respect to what?*

"Lossless" is always relative to a target distribution. Ours is the **base
Qwen3.6-27B LM-head next-token distribution at the serving temperature
(temp=0.6)**, conditioned on the true committed prefix. Greedy (temp→0) special
case = the base model's argmax stream. Lossless ⇔ the served output is
distributed *identically* to running the base model autoregressively with **no**
speculation. It is **NOT** the MTP head's distribution.

## MTP's role: it is the drafter q, never the reference

Qwen3.6 has `mtp_num_hidden_layers=1` — an MTP-1 module. At inference it is a
self-speculative **drafter**: it proposes candidate tokens (draft distribution
`q = M_s`). A *tree* is built from MTP by taking **top-k candidates as branch
width** at a position and/or **chaining the MTP module / suffix decoding for
depth** (spec §6/§8). Accepting MTP tokens *directly* (no verification) would
serve the MTP head's distribution — that is **lossy**. Lossless MTP =
MTP drafts → base model verifies → output == base model.

Key theorem (Traversal Verification, arXiv:2505.12398, Thm 3.3; same property in
SpecInfer arXiv:2305.09781 and Sequoia arXiv:2402.12374): **output distribution
is preserved for ANY draft `q`.** Draft/tree quality changes only the *acceptance
rate* (speed), never correctness. ⇒ We may pick any MTP tree shape for speed and
stay lossless, *provided the three requirements below hold.*

## "Lossless" decomposes into THREE independent requirements

### L1 — Verifier-forward parity  (spec Gate D ; = FR10 P1)
For every tree node, the verifier must produce **exactly** the target conditional
`p = M_b(· | root-path-prefix(node))`. On a hybrid GDN model this requires
per-node recurrent-**state** parity: `packed_tree_state[node] ==
serial_per_path_state[node]` (proven bit-exact in P1, max delta ~3e-8 fp32). If a
sibling contaminates a node's GDN state (the FR9 spine-2 failure), `p` is wrong,
so the acceptance rule below operates on the wrong target ⇒ **lossy**. L1 is a
*precondition*: "the tree came from the MTP head" does NOT rescue a contaminated
verifier state.

### L2 — Acceptance-rule correctness  (spec Gate C)
Given correct per-node `p` (L1) and the MTP draft `q`, accept the tree via a
*valid* tree speculative-sampling rule (SpecInfer multi-step / Sequoia /
Traversal Verification). It must be **sequence-level**, not a token-greedy
shortcut. Required properties:
- Chain acceptance uses the rescaled ratio `min(p/q, 1)` composed along the path;
  on rejection, resample from the renormalized residual `M_b' = norm([p − q]_+)`.
- **If a node is rejected, ALL its descendants are rejected.** Children were
  sampled conditional on the parent; keeping them samples a sequence the target
  never draws ⇒ bias. (This is the sampler-side analogue of L1's trunk integrity.)
- **No "longest-accepted hidden winner" / max-over-branches order-statistic
  selector** — it is a biased estimator ⇒ lossy. This is exactly the FR9 banned
  mode and the Gate C negative control (`test_lossless_selector_gate_c_stub_design.py`).
- Output distribution == `M_b` for any `q`.

### L3 — Greedy special case  (spec Gate B)
At temp→0 the rule collapses to: accept the drafted token iff it equals
`argmax(p)` at that node, else stop and emit `argmax(p)`. Output == base model
greedy decode, **byte-exact**. Enforce by running the public path0 token through
the **identical** verifier kernel (identity-by-construction), not a tolerance
bound. (This is why the P1 greedy gate asserts argmax/token identity, not a
logit-margin threshold.)

## One-line summary

`lossless = L1 (GDN tree-state parity, done) × L2 (valid tree rejection sampler
over the MTP tree — sequence-level, descendant-rejection, no max-selector) ×
L3 (greedy byte-exactness via identical kernel)`. MTP only sets the tree's `q`;
it moves acceptance rate, never the served distribution. So speed (tree shape /
MTP depth / suffix) and losslessness are **separable** — which is exactly why we
can chase both at once on the new kernel.

## Sources
- Leviathan et al. 2023 (arXiv:2211.17192) & Chen et al. 2023 (arXiv:2302.01318):
  speculative sampling preserves the target distribution exactly.
- SpecInfer (arXiv:2305.09781): tree-based multi-step speculative sampling, lossless.
- Sequoia (arXiv:2402.12374): lossless tree verification; sampling-without-replacement.
- Traversal Verification (arXiv:2505.12398): sequence-level rule; losslessness
  independent of draft quality; descendant-rejection requirement.
- DeepSeek-V3 / Qwen3-Next MTP: MTP-1 self-speculative drafter, ~80%+ accept ⇒ speed only.
