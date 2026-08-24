# How our vLLM upstreaming works (mechanics)

## The flow, one time through

1. **Fork** `vllm-project/vllm` to your GitHub account (one-time; `gh repo fork`).
2. **Branch** off `main` in the fork — one branch per PR, small scope.
3. **Commit with DCO sign-off** (`git commit -s`) — every commit needs the
   `Signed-off-by:` trailer or the DCO bot blocks the PR. AI assistance must be
   disclosed in the PR body; you (the human) review every changed line before it
   goes anywhere public.
4. **Internal review first (our step):** push the branch to OUR fork and open a
   PR *within the fork* (base = fork's `main`). Mark reviews there. Nothing is
   visible to upstream reviewers at this stage.
5. **Upstream:** once approved internally, open the PR from the fork branch
   against `vllm-project/vllm:main` with the proper title prefix (`[Bugfix]`,
   `[Kernel]`, `[Core]`…). Reviewer updates come every 2–3 days; ping after 7.
6. **RFCs are issues, not PRs.** The tree RFC gets filed via the RFC issue
   template on `vllm-project/vllm`, then posted to `#contributors` on vLLM
   Slack. Code follows the RFC conversation, never precedes it.

## What we never do

- **PR the whole thing.** ~19k LOC would be auto-tagged `rfc-required`, is
  humanly unreviewable, and violates the no-pure-agent-PR policy. The unit of
  contribution is: one small PR, one idea, tests included.
- Ship the patch-at-boot architecture. Upstream gets real diffs.
- Let a PR carry campaign machinery (fixed32 audits, credentials, sidecars).

## Current queue (post-HEAD-recheck, 2026-08-24)

G1 (uniform-dispatch V1 bugfix) → C-residual + head_dtype interaction bug →
F-PR1 (plural proposal_methods config) → E-a (mask_mod tree mask) →
RFC (parallel) → D-argmax → gated: G2, G4 → phases of the tree RFC.
