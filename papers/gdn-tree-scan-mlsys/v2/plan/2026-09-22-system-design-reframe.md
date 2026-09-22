# Approved system-design reframing

User approved the proposed shift from an audit-led paper to the mechanism, implementation and optimization of the tree verifier. Approval: "go ahead with those" on 2026-09-22. Existing source revisions and evidence remain preserved.

## Writing contract

Contributions: integrated tree verification; GPU scan/replay execution and state lifetime; optimization mechanisms under backend, numerical and lifecycle constraints. Explain parent state, descriptor maps, drafter/acceptance interface, accepted-state publication, register/cache tradeoffs, and implemented optimization choices. Contract and failures motivate implementation choices. Local compact-solver comparisons evaluate the design tradeoff. The accounting audit belongs in methods/appendix, not the contribution list.

Separate the historical forked-FA2 design from the fresh stock-TREE_ATTN flat-map route, and separate source existence from enabled/executed/individually measured components. No historical numerical results return. Preserve all fresh adverse findings, frozen-rule failures, seed deviation and continuation divergence.

## Outline

1. Introduction: problem, system idea, three implementation contributions, bounded evaluation.
2. Background and direct related work.
3. Tree-verifier architecture: descriptor, ancestry and selected continuation.
4. GPU scan/replay: state policy, shared update, memory/compute tradeoff, algorithm.
5. Optimization mechanisms: draft-logit reuse, device acceptance, backend layout and grouping, launch/batch candidates; explicit route/evidence map.
6. Evaluation: fresh numerical study, continuation checks, whole-configuration timing.
7. Discussion: what measured design establishes and what optimization ablations remain.
8. Conclusion. Appendix: measurement/provenance details and deferred validation.

## Execution contract

Audit exact current source and E1 snapshots before selecting any new experiment. Prefer existing evidence and tests. A new ablation needs a compatible active route, fixed inputs/outputs/reference and source identities, correctness checks, matched intervals and predetermined stopping rules. Do not conflate topology/backend/model changes with a one-factor toggle. Existing original frozen campaigns remain immutable. Unsupported candidates remain explicitly unmeasured rather than being presented as wins. No public submission.

## QA

Independent source/route audit and manuscript red team; unchanged fresh-results checks; citation lint; LaTeX compile; render and inspect all revised pages; preserve prior receipt and export a new private review package when settled.
