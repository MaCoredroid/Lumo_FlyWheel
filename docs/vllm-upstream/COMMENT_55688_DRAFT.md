# Review comment for PR #55688 (FI ReplaySSM lifecycle unification) — v2 (codex round-1 applied)
> **POSTED 2026-09-10 22:54 UTC (Mark greenlight, Codex final GO):** https://github.com/vllm-project/vllm/pull/55688#issuecomment-5626523659

> v2 (2026-09-10): codex red-team round 1 returned NO-GO on v1 with one
> blocker (v1 quoted `plan_flush_count = accept_token_bias + 1 +
> old_committed`; the real kernel is `... + tl.where(checkpointed, 0,
> old_committed)` — verified at head e4be47b, ssu_dispatch.py:228–230).
> Applied: dropped the equation and the ring-capacity formula; dropped the
> "peer question, not a proposal" self-description; stated the current
> materialize contract instead of asking about it (both runners call
> `materialize()` right after `postprocess()` in the same post-sampling
> path); qualified "need not be contiguous"; removed the "small follow-up"
> promise. Body below is codex's replacement with voice tightened.
> Placement: one general PR comment on #55688 (not on a CodeRabbit thread,
> not a "request changes" review, not on #52928). Not a merge condition.
> Public body: 92 words by whitespace count; paste only below the separator.
> **Codex round 2 (2026-09-10 02:46): GO, no edits. Claude: GO.** Awaiting
> Mark's review; Mark posts. Before pasting, refresh the PR once (head was
> e4be47b, needs-rebase) — symbol names only, so a rebase should not break it.

---

We run tree speculative decoding out-of-tree on GDN hybrids; one follow-up
question, related to #54080.

`postprocess()` takes a per-request accepted count and updates contiguous
replay history; when a prefix snapshot is needed, `materialize()` publishes
it in the same post-sampling path.

For a tree whose verification records stay in node order, the accepted path
need not be contiguous. Would you keep the node-to-ring mapping outside this
lifecycle and preserve its contiguous-history contract, or take
accepted-node metadata here in a follow-up? Either way needs matching kernel
and storage work.

Happy to bring a concrete follow-up once this lands.
