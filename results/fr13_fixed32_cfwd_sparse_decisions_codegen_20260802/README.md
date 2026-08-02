# Fixed32 CFWD sparse all-parent decisions

Status: **CPU/oracle and offline SM121a codegen passed; default off; real-task
byte and timing qualification pending**.

## Scope

This candidate targets the float decision storm in
`_fr13_fixed32_taw_all_parent_decisions`. It is separate from the mature
key-group CFWD recurrent-state candidate, which updates 48-layer GDN state.

The candidate replaces dense per-parent `q_mix`, residual, and full-row CDF
tensor operations with four fixed Triton launches:

1. Reduce the 13 self rows and 17 target rows into 256-token block masses.
2. Fuse child lookup, overlap mass, source selection, duplicate-aware q_mix at
   the selected token, and the accept test for all 17 target parents.
3. Reduce sparse-corrected target residuals into blocks without materializing
   a dense q_mix or residual tensor.
4. Select a block and perform one in-block inverse-CDF scan for all 13 self and
   17 rejection rows.

The existing one-program integer walk remains the fifth and final commit
launch. Workspaces are preseeded before graph capture only when
`FR13_FIXED32_CFWD_FUSED_DECISIONS=1`; the selector defaults to `0`.

## Duplicate rule

Every child occurrence contributes to q_mix. For target token `t`, child-token
multiplicity `m_t`, and overlap mass `S`, q_mix is `m_t * p[t] / S`. The
residual at a drafted token is therefore `max(p[t] * (1 - m_t / S), 0)`.
The implementation does not use the distinct-token shortcut when duplicates
are present. See `math_contract.json` and the randomized property tests.

## Offline codegen

Two fresh CUDA-hidden compiles produced byte-identical `codegen_summary.json`
files. The source-bound SM121a resource results are:

| Kernel | B1 / B4 CTAs | Warps | Registers | Launch / reported shared | Spills | Stack/local/calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Probability block sums | 30,720 / 122,880 | 4 | 16 | 16 / 1,024 B | 0 / 0 | 0 / 0 / 0 |
| Parent setup | 17 / 68 | 8 | 34 | 32 / 1,024 B | 0 / 0 | 0 / 0 / 0 |
| Residual block sums | 17,408 / 69,632 | 4 | 29 | 16 / 1,024 B | 0 / 0 | 0 / 0 / 0 |
| Block plus in-block CDF | 30 / 120 | 8 | 42 | 128 / 1,024 B | 0 / 0 | 0 / 0 / 0 |

The logical preseeded workspace is 193,721 bytes at B1 and 774,884 bytes at
B4, versus full-vocabulary dense q_mix/residual rows.

## Verification boundary

- Candidate-specific CPU/oracle/property suite: 24 passed.
- Combined focused source/runner/census suite: 64 passed.
- Python byte compilation, shell syntax, Ruff, and `git diff --check`: passed.
- Exact source-contract schema/hash: `fr13-fixed32-taw-all-parent-v8` /
  `0f856a9e3d43f7c218d1b1f05d724fd82b5f0d3ce501c42bc742cc8cfc9acd0d`.
- No GPU query, CUDA launch, Docker action, real task, synthetic/probe timing,
  or throughput measurement was performed.

## Limitations

This candidate still consumes the existing 13-row and 17-row probability
caches, so the two full-row softmax calls remain. The block reductions also
use many CTAs. Their wall-time value is unmeasured.

Partitioned reductions can round differently from PyTorch's dense reductions
at measure-zero CDF/accept boundaries. Distributional rejection sampling is
preserved, but raw-byte equality is not claimed until the source-bound real
SWE-Verified B1 diagnostic and exact4 B4 gate pass. Production remains off,
and this artifact makes no hardware-floor or acceptance claim.

The legacy script-level `--self-test` still fails its duplicate acceptance
assertion on the unchanged base commit as well as this branch. Focused pytest
coverage and the independent duplicate oracle pass; the pre-existing failure
is not represented as a candidate pass.
