# B1 CFWD/U8 composed real-task gate

Status: **FAIL** for U8 promotion; the served reference path remained lossless.

- Source commit: `ca55b0236a5f2e8b701ee86d902016b8efdef339`
- Workload: SWE-Verified `astropy__astropy-12907`, batch 1, Hydra27 physical32, K64/root1
- Task outcome: resolved, 1/1
- Fixed32 work census: 1,434 complete decode events, forward steps 0 through 1,433
- Candidate coverage: 7,170 full-logit comparisons, 469,893,120 BF16 elements
- Candidate result: 75,916 raw BF16 mismatches (0.0161560%)
- Per-depth comparisons: 1,434 each for root and MTP depths 1 through 4
- Per-depth mismatches: root 15,065; depth 1 15,310; depth 2 14,914; depth 3 15,266; depth 4 15,361
- Serving behavior: incumbent BF16 logits were always returned; candidate output was shadow-only
- Terminal result: the fixed32 final flush correctly failed closed with `error:RuntimeError`

The prior CFWD capture-scope fix passed its runtime lifecycle: the FULL target graph captured, the external committer route ran for every event, and no binding drift or fallback occurred. This run does not issue a U8 production credential and is not timing eligible.

Evidence files are copied byte-for-byte from:

`output/fr13_b1_cfwd_u8_composed_ca55b0236_20260805T175629Z/hydra27_fixed32_k64_root_cfwd_u8_ca55b0236_20260805T175629Z`
