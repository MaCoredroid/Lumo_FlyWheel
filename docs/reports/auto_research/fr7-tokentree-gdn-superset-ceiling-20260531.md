# FR7 Token-Tree GDN Superset Ceiling Close-Out

Date: 2026-05-31

FR7 proved the token-tree verifier itself was internally correct, but it did not
prove the production target: winner acceptance greater than or equal to E5 on
every event.

## Result

The per-path LCP-max verifier reported `superset_violations=0` against the
tree's own path0 chain. That is only an internal tree property. The same run
showed path0 was not the E5 baseline:

| Metric | E5 control | FR7 token-tree path0 |
| --- | ---: | ---: |
| Accepted tokens / event | 3.150 | 1.66 |
| Full accept rate | 43.6% | 1.7% |
| acc=0 rate | 13.3% | 13.6% |

The position-1 behavior was close to E5, but the chain collapsed at depth.
Therefore the experiment was strictly below E5 in roughly 42% of events, and
the real superset claim, `winner >= E5`, was false even though the in-tree
`winner >= path0` claim was true.

## Root Cause

Qwen3.6 is a GDN/SSM hybrid. A single packed token tree shares one recurrent
Markov state across all spines. Sibling spine state contaminates the deeper
state of the top-1 chain, so path0 is no longer byte-identical to native E5.

This matches the STree finding from arXiv 2505.14969, which requires a
tree-aware recurrent scan such as a custom `A_tree` / `TreeScan` CUDA kernel to
make packed tree decoding lossless for hybrid state-space models.

## Decision

Do not port the STree-style kernel in this round. vLLM does not provide native
tree speculative decoding for hybrids on this stack, and implementing a
tree-aware GDN recurrent kernel would be a research-grade CUDA project rather
than a cheap speed path.

The next route is multi-spine independent recurrent state: run each spine as a
separate co-resident persistent sequence, copy the fixed-size post-prefix
recurrent state into each spine's own slot, verify ordinary uniform-batch MTP
chains, and select the longest accepted spine at the accept stage. Spine A must
remain the native MTP top-1 depth-5 chain so `path0 == E5` by construction.
