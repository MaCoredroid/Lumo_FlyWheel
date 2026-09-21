# #55122 scope note after jschmied's 2026-09-21 reframing — v1 (my draft; Codex check pending; AWAITING MARK GO)
> Trigger: jschmied retitled #55122 ("persistent_topk: deterministic select, faster than the
> exact-topk workaround"), reframed it as a performance PR with a correctness side-benefit, and
> @-mentioned MaCoredroid: "your measurements and review are what the remaining case rests on;
> the change is to the framing, not to the kernel." New head b2312b2de79 = merge of main; both
> kernel sources byte-identical to 19588c898e5 (git diff --quiet; sha256 below).
> Purpose of the reply: scope our evidence so it is not read as supporting the performance claim.
> One PR comment, ≤110 words, no position on merge, no re-run (nothing changed to re-run).

---

Scope note on the measurements you flagged, since part of the remaining case rests on them. Ours are correctness measurements only: on GB10 the low-shared-memory single-CTA fallback returns the exact reference and repeats bit-for-bit across the float32 grid ([09-12](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5648007112)), re-run against `85f61e24b` ([09-18](https://github.com/vllm-project/vllm/pull/55122#discussion_r4043382139)). They say nothing about performance; the 21–28 % figure is @k3dani's, not ours. `persistent_topk.cuh` and `topk.cu` at `b2312b2d` are byte-identical to `19588c89`, so that re-run stands for the current head. No objection to the reframing. AI assistance was used.
