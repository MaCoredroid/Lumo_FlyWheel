# vLLM contribution scan — 2026-09-17
Lane: GDN / Mamba hybrid / spec-decode determinism / GB10. vllm-project/vllm, updated since 2026-09-03. Read-only; verified 2026-09-17. Excluded (our footprint): #53651, #53798, #55688, #54080, #54928, #55122.

## 1. PR #51508 — "GDN/KDA: silent recurrent-state corruption (and CUDA crash) for stale zero-accept spec rows"
maxpla3 · OPEN · `bug, speculative-decoding, needs-rebase, nvidia, kimi, k3` · +421/-11, 12 files · last activity 2026-09-16 (mergify conflict).
**Missing:** zero human reviews (only the `claude` bot). Three open PRs fix the same `num_accepted_tokens == 0 → index -1` defect, unadjudicated: #48475 (08-17), #50021 (09-02), #51508. Author pinged maintainers 2026-08-19: *"@tdoublep @ZJY0516 … Three PRs are now circling this (#48475, #50021, this one) and the other two are clamp-only. They stop the fault but still let the kernel advance the recurrent state of a discarded step."* ZJY0516's only review on clamp-only #48475, in full: *"I'm hesitant, as this is only a workaround. I'd prefer to identify the root cause"* — #51508 **is** the root-cause variant and nobody has told him.
**Our evidence:** PR says *"Whether it crashes depends on allocation layout, so in the wild this bug may present as bad output rather than a clean crash."* GB10's 117 GiB unified pool is a different layout from every box in the thread.
**Cost:** zero-code (+ optional short GPU run). **Asset:** stock-v0.28.0 GB10 rig, first-divergence attribution.
**Dup check:** no human reviewing any of the three; #50021 has only bot + self-comments.
**First action:** comparative review on #51508 naming it the only non-clamp fix, answering ZJY0516's #48475 objection by number, and asking for a rebase.

## 2. Align-mode Mamba state-seed cluster — issues #53142 / #55600; PRs #55507, #55601 (+ our #53798)
#53142 OPEN (09-14); #55600 OPEN (09-06) duplicates it. PRs: #55507 (Karl0007, `bug, mrv2`, 09-14), #55601 (pondzikk, `bug, mrv2`, 09-06, zero human comments), #54076 (wickist, scheduler-side sibling, 09-16). The first two patch `mamba_hybrid.py` with different divisors for the same one-line seed bug; none is human-reviewed.
**Missing:** nobody has said these are one bug. Our #53798 body still claims *"No open PR touches align-mode seeding"* — now false. On #53142, Davan-Etelamaki (09-05): *"There is still no PR linked here. **I can open one** if no one else is on it."* — a fifth duplicate imminent.
**Cost:** zero-code. **Asset:** model-free align-mode restore-fidelity test (worker layer) — runs without weights, discriminates the divisors directly.
**Dup check:** only bots and the authors; #55601 has no human engagement at all.
**First action:** comment on #53142 linking #53798/#55507/#55601/#55600 as one cluster, with the model-free test result showing which divisor is correct.

## 3. PR #54146 — "GDN readout: honor fp32 SSM state precision to avoid fp16 overflow → NaN"
JackDanger · OPEN · `bug` · +64/-1, 2 files · 09-08. Closes #54308. Review-requested ZJY0516, mgoin, yewentao256; no human review.
**Missing:** validated **only on AMD MI50 (gfx906)**. PR asserts *"The fp16 path is byte-for-byte identical, so tensor-core / NVIDIA behavior is unaffected by construction"* — untested on any NVIDIA GPU. #54308 carries a ~20-line model-free, device-agnostic kernel repro. Unanswered: JartX (09-08) *"this PR has been helpful, but its incomplete for me, can you check my commit and take my second patch on your pr please?"*
**Cost:** GPU-time, minutes (one kernel call, no model). **Asset:** standalone kernel harness; GB10.
**Dup check:** nobody has run it on NVIDIA.
**First action:** run #54308's repro on GB10 (sm_121); post before/after non-finite counts and which GDN backend sm_121 picks.

## 4. PR #45819 — "Add batch invariance support to GDN_ATTN backend"
yuvalluria · OPEN · `v1` · +299/-99 · 09-16. Review-requested ZJY0516, LucasWilkinson, MatthewBonanni, tdoublep; yewentao256 actively reviewing (13 review events).
**Missing:** two yewentao256 asks open — *"Could you also run e2e accuracy using `lm_eval ..` with / without batch invariance to make sure we don't hurt acc[uracy]"* and *"Could you also shrink the diff? Idealy < 200 LOC"*. Every posted run is H100 NVL / SM90; **zero sm_12x data**. The blocker in linked issue #48613 is item 1 of yuvalluria's own list: *"Cross-chunk conv/recurrent state carry is not bit-exact when chunks don't align to 64"* — exactly what our 64-chunk-alignment work measures.
**Cost:** GPU-time (Qwen3.5-0.8B, ~13 min/suite). **Asset:** public evidence on bit-exact GDN prefix caching (64-chunk alignment).
**Dup check:** cm2435 offered H100 validation in July, silent since; tolleybot posted H100/TP4 only.
**First action:** run `tests/v1/determinism/test_batch_invariance.py -k GDN_ATTN` on GB10; post the sm_121 result with the 64-alignment finding.

## 5. Issue #55291 — "Qwen3.6-27B-FP8 eventually collapses into repeated ! tokens"
dikongfeixing8 · OPEN · `bug` · last activity 2026-09-04; ZJY0516 commented (*"vLLM 0.21.0 is too old"*).
**Missing:** unanswered ask from sizzlecar (09-04): *"The useful next step is reproducing on 0.28.0 or current main; if it disappears there, debugging the old cache path is not actionable."* We run this exact model on a pinned stock-v0.28.0 rig. The `!!!!` signature with `mamba_ssm_cache_dtype=float32` matches #54146/#54308's mechanism, so one run serves both.
**Cost:** GPU-time; trigger is non-deterministic ("after running for some time") — timebox it.
**Dup check:** dormant since 09-04; nobody has run 0.28.0.
**First action:** run the reporter's config on stock v0.28.0 on GB10; report whether it reproduces, plus the GDN backend / SSM dtype selected.

## Reject list
- **#54521, #51782, #55872** — saturated; jschmied/jahnclawdmonet/hclsys already posting GB10 sm_121 data; #55872 `needs-rebase`, stale since 09-09.
- **#57105 / #56500 / #56457** — jahnclawdmonet posted the GB10 measurement on #57105 (09-16).
- **#49760** — hebo1221 reviewed twice with GB10 arithmetic; author quiet since 09-02, three unresolved conflicts.
- **#56824 / #56830 / #55569 / #50011** — other Spark owners already supplying datapoints; #56830 row 3 covered by #55828.
- **#53912** — resolved in-thread: #50729 ships in v0.28.1; reporter confirmed.
- **#52244** — contested; ZJY0516 prefers #50897, and a GB10 measurement is posted.
- **#52817, #55876, #54993/#55627/#55895/#55905/#56244** — new subsystem / kernel-heavy; over budget.
- **#55533** — three contributors on it; the outstanding ask targets the reporter.
- **#57125** — root-caused by jahnclawdmonet within a day of filing.
- **#56307, #50264, #53623** — ROCm/RDNA; no hardware.
