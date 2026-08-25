# NVIDIA contact — draft for Mark to send BEFORE filing the RFC

> **Where:** a comment on PR #51674 (the fused post-conv MTP decode for
> Qwen3-GDN) is the natural venue — public, low-pressure, on-topic. Slack DM
> only if a reply invites it.
> **Why first:** they are actively shipping into the exact surface our phase 0
> touches (#51674, #52539, #51855). Contacting before filing converts a
> collision risk into co-review, and the RFC can then CC them as people we're
> already talking to rather than strangers.

---

Hi — I've been following this work closely (this PR, the head-ratio
parameterization in #52539, and the RecoverSSM slot-demand hook in #51855);
we're building in the same area and I'd rather coordinate than collide.

We run Qwen-GDN hybrids in low-batch agentic serving and have an out-of-tree
implementation of **token-tree** speculative decoding for hybrid models —
tree proposals over an MTP head + suffix cache, verified losslessly, with the
recurrent-state handling that trees force: branch-local scans with per-node
parent-state selection, and re-linearization of the accepted path on commit
(the failure mode when you skip it is nasty — sibling-state near-neighbor
corruption that passes numerical gates; happy to share the war stories).
Measured on a 27B GDN hybrid: 5.66 committed tokens/step vs 3.36 for a chain
EAGLE config on the same silicon.

We're about to file an RFC proposing this in phases, starting with
tree-capable interfaces on `MambaSpecDecodeGPUContext` (substrate-shared,
no behavior change for chains — your MTP work would be unaffected and, we
think, eventually faster with tree proposals on top). Before filing I wanted
to (a) make sure we're not about to duplicate anything on your roadmap, and
(b) ask if you'd be open to being CC'd as reviewers — you own the kernels
this eventually feeds.

Also, small thing found while reading `fused_recurrent.py` closely:
`stride_indices_tok` is computed and passed but never read in the kernel body
(the spec path indexes bare `+ i_t`). Looks like a dead parameter — or a
latent bug if a non-contiguous layout is ever passed. Happy to send the
one-line fix with a test if useful.
