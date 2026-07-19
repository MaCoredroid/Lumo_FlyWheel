# FR13 R3 — verify tree-tax trim (design, 2026-07-19)

Sequencing-locked: R1 (rg matrix) holds the GPU; this design lands flag-gated impl ready for the
first free window. Owner: Claude (no-delegate rule). Target: verify 103ms (cat9pb pbm1 measured
s_per_fwd_gpu) -> ~70ms; ladder projection ~34 TPS with R1+R2.

## The gap to explain
native MTP5 verify 58.1ms vs cat9pb 103.3ms = +45.2ms for 18 packed cols vs 6. Decode is
weight-read-bound (~98.6ms floor for the FULL forward at B=1 was the old figure; at eff-conc ~2
the per-request attribution halves) — the DENSE per-token compute scales weakly with cols, so the
+45ms must live in the TREE-SPECIFIC kernels/host seams, NOT the FFN GEMMs:

| candidate slice | mechanism | est | trim idea |
|---|---|---|---|
| GDN tree-scan (48 layers) | 18 streams/req in the BV=8 register-tiled kernel vs native's rank-1 fused path | ? (profile) | chain streams NEED full scan (state advance = their purpose) — no trim there; subtree streams already minimal. Kernel-level: BV=8 halves value-tile parallelism -> occupancy loss vs BV=16@9-node. Candidate: per-shape BV (9-node subtree fits BV=16; chain-8 separate pass?) |
| FA2 tree-attention fork | tree mask + 18 Q rows vs 6; ghost rows (chain 0..6 + row 0) are -inf masked but STILL COMPUTED | ? (profile) | skip ghost Q rows in the kernel grid (they produce discarded outputs); exact-mask semantics unchanged => byte-identical for live rows |
| logits GEMM row-trim | chain cols 0..6 logits never read (walk starts at chain end; committer reads subtree + root twin only) | ~1-2ms only (LM-head is weight-read-bound; row count barely matters) | LOW priority |
| tree bias/remap/host prep | build-once bias claimed; attn-KV remap per step; conv src indices | ? (profile) | verify build-once actually engaged under pb; fold remap into graph capture (R4 synergy) |

## Step 1 (first GPU window, ~20 min): kernel-level attribution
torch.profiler one-shot on a standalone pb server, single request, 30 decode steps, export
chrome trace + top-20 kernels by GPU time, tree kernels grouped (gdn tree scan / FA2 fork /
logits / elementwise glue). NO trim work before this lands — the table above is hypothesis
ranking, and this project's history (drafter "GPU-bound" refuted, FR-Spec refuted, deep-tail
"cold" refuted) says profile first.

## Constraints
- Flag-gated FR13_VERIFY_TRIM=1; OFF = byte-identical (bug-class 9).
- Chain streams' GDN scan is sacred (pb state advance) — any trim must leave scan bytes identical.
- Ghost-row attention skip must preserve the exact softmax of LIVE rows (ghosts only ever
  contribute -inf columns to others; their own row outputs are discarded — verify BOTH directions
  in the mask audit before skipping).
- A/B: same-session vs pb reference arm, wall-free, measured-wall basis + resolve band per the
  standing bake rule.
