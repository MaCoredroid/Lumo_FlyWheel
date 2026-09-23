# #pr-reviews follow-up for #53651 — v2 (Codex replacement verbatim — dropped "small" and the reviewer-nonresponse sentence; GO for a threaded reply; AWAITING MARK GO; post NOT before 2026-09-24T00:09Z = 7 days after the first #pr-reviews post ts 1789603743.165569)
> This is the week's one unsolicited ask. Form: a reply IN THE THREAD of our 2026-09-17 #pr-reviews
> post (channel C07QT0LUF4K, thread_ts 1789603743.165569), not a new channel message, unless Codex
> prefers a targeted GitHub mention. Facts: PR open since Aug 24; reviewer 22quinn requested, no
> response since our Sep 10 ping; DCO passes; pre-commit/format gated (pre-run-check fails without the
> `ready` label); no needs-rebase label; CodeRabbit green.

---

Following up on https://github.com/vllm-project/vllm/pull/53651: could someone review the remaining excluded-head weight re-tying fix and the materialized-weight guard for head_dtype? The description includes regression coverage and recorded local validation. DCO passes; upstream pre-commit remains gated. A review or a pointer to the appropriate quantization/model-loader reviewer would be appreciated.
