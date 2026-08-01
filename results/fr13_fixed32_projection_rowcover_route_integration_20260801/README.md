# FR13 fixed32 projection row-cover route integration

Status: **static/install/runtime route pass; not byte-qualified and not timing
eligible**.

This reduced artifact records the production binding for the combined B1-M32
and B4-M128 projection candidate. It contains no prompts, responses, private
environment data, process/container identifiers, raw logs, or performance
samples.

## Bound candidate

- Source commit: `d51e2b4dcf2c76d1cf8e957ddd06f61ba399c042`
- Branch: `agent/fixed32-projection-route-integration-20260801`
- Candidate family: `projection_rowcover_pair`
- Binary SHA256:
  `af48592c748ba80b1c614dc7a96c8250ae3bcca4c185c92939b4d308f8ef31f6`
- Binary bytes: `113078080`
- Patch source SHA256:
  `32ee5747eeff597f7eacec530f86658ba26b6fe8560591c21c305e594953935a`
- Patched dispatch SHA256:
  `ba18f08dcbd17a52c1b7293be0cc6eb4ee57176388d4e2ccba9bfb62c9b31c45`

The B1 production selector is `static_persistent_stocktile`, with
`static_persistent_stocktile_byte_ab` reserved for its byte comparison gate.
It dispatches the candidate only at exact physical `M=32`, root depth 1, and
`FR13_DRAFT_VOCAB_K=65536`. Its diagnostic credential is restricted to the
allowed one-task real SWE-Verified task and never authorizes timing.

The B4 production selector is `persistent_b4_m128`, with
`persistent_b4_m128_byte_ab` reserved for its exact4 byte comparison gate. It
dispatches the candidate only at exact `M=128` under the K64 profile. All
other rows, profiles, and shapes fall back to stock.

The verifier remains full-vocabulary. The fixed K64 draft block-map identity
is `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`.

## Qualification

The deterministic static qualifier passed against the exact combined binary:

- four candidate resource records matched exactly;
- all four candidate kernels have zero stack and zero local memory;
- six of six stock kernel resource records matched;
- 873 strong dynamic exports matched;
- both B1 selectors and both B4 selectors resolve to the same pinned binary;
- installer, launcher, runtime patcher, and reducer contracts are covered by
  the focused test suite.

The focused suite passed `164` tests. Python byte compilation, shell syntax,
and `git diff --check` also passed. Ruff was unavailable in this worktree.

## Live blockers

Fresh real SWE-Verified byte gates are still required for both Tail23 and
Hydra27 before timing:

1. B1: the allowed one-task K64/root1 diagnostic at exact physical `M=32`.
2. B4: the canonical exact4 K64 diagnostic at exact `M=128`.

The earlier B1 full-vocabulary/root0 evidence cannot qualify this route. The
runtime manifest was not regenerated in this isolated worktree because its
private and untracked campaign inputs were intentionally not copied. No GPU,
Docker, byte gate, or timing campaign ran while producing this artifact.

This artifact makes no throughput, acceptance, hardware-floor, or speedup
claim.
