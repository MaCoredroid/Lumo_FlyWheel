# #55122 scope note after jschmied's 2026-09-21 reframing — v2 (Codex replacement verbatim: NO-GO on v1 — dropped "re-run stands for the current head" and "No objection to the reframing"; GO on this text; AWAITING MARK GO)
> Trigger: jschmied retitled #55122 ("persistent_topk: deterministic select, faster than the
> exact-topk workaround"), reframed it as a performance PR with a correctness side-benefit, and
> @-mentioned MaCoredroid: "your measurements and review are what the remaining case rests on;
> the change is to the framing, not to the kernel." New head b2312b2de79 = merge of main; both
> kernel sources byte-identical to 19588c898e5 (git diff --quiet; sha256 below).
> Purpose of the reply: scope our evidence so it is not read as supporting the performance claim.
> One PR comment, ≤110 words, no position on merge, no re-run (nothing changed to re-run).

---

To clarify the scope of our contribution: our [GB10 fallback results](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5648007112) and [rerun at `85f61e24b`](https://github.com/vllm-project/vllm/pull/55122#discussion_r4043382139) checked exact-reference agreement and repeatability on the reported float32 grid. We measured no performance, so our results do not validate the 21–28% speedup. The tested persistent path remains unchanged in source, but we have not executed the current head. Operator integration and CUDA graphs were outside our harness coverage.

AI assistance was used.
