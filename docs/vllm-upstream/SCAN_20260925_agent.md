# vLLM upstream scan — agent, 2026-09-25 (window 2026-09-17 → 2026-09-25)

Read-only; nothing posted. REST search worked (no 403). 37 queries → 1200 unique threads, 970
lane-relevant, 378 created in window. Evidence in `tmp-scratch/agent_scan_sep25_evidence/`.

## Top 5 (EV × engagement ÷ cost)

**1. PR #58718 — sm121 batch-invariant matmul table tuned on GB10 (hclsys, opened today).**
Missing: a second GB10's numbers. The PR flags that from M=512 up the table is 7–18% *slower* than
default and argues this is intrinsic (12/15 shapes pick BLOCK_K=128 vs the default 64, so they cannot
fall back without changing K-reduction order). Nobody has tested whether that penalty and the
0.53–0.81× decode win reproduce on other sm_121 silicon, or re-run the bitwise M-invariance gate off
his box. Ownership: hclsys; LioEinaudi (sm_120 table author, #57456 merged 09-19,
#58495 open `ready`) re-ran only the cc 12.0 side and LGTM'd. No competing PR. Cost: GPU ≤ half day
(Qwen3-1.7B/4B/8B shapes). Engagement: highest in the scan — both replied within 30 min today. **First action:** at the PR head run
`tests/v1/determinism/test_matmul_batch_invariant.py`, re-time the 15 (N,K) keys at
M ∈ {1,32,256,512,1024,2048} tuned-vs-default (CUDA-event p50), and sha256 the M=1 rows against
M ∈ {8,32,2048}.

**2. PR #57908 — preserve aligned prefix-cache snapshots with CoW (cbertucci33).**
Our P8/P9 lane exactly: full block-aligned hits stay shared while the resumed request writes the
terminal state in place, so the hash describes a superseded state. Four days old, no human comment. Missing: (a) do its two new `test_single_type_kv_cache_manager.py` regressions *discriminate* (fail
at merge-base, pass at head) — the #55506 check; (b) is the aliasing reachable on the served path or
manager-only (#54076 method); (c) does it collide with the align-seed cluster or merged #58434. Ownership: nobody; no competing CoW PR. `kv-cache-manager` maintainers active this week: njhill,
LucasWilkinson, netanel-haber (#58368 merged). Cost: **CPU only.** Engagement: author unproven, but
our fixture has adoption here. **First action:** run both tests at merge-base and head; post the
discrimination matrix.

**3. Issue #58624 / PR #58681 — SM12x DeepGEMM contiguous alignment 128 vs 64.** lucamotz measured
−15% MoE decode on a GB10 (sm_121, Qwen3.6-35B-A3B-FP8 + DFlash). Missing: an independent sm_121
measurement *at the new pin* — hclsys cannot produce the 128 (his tree vendors DeepGEMM 2.5.0) and
corroborated only the baseline; bluemelov1's is 8×sm_120, so the regression is still single-source on
our hardware class. Ownership: dajiaohuang's #58681 (+ DeepGEMM#18); hclsys did the boundary/guard
review, so only the measurement is open. Cost: GPU ≤ half day. Engagement: 4 active participants.
**Caveat:** unverified whether our pinned Qwen3.6-27B-FP8 hits the grouped-MoE path. **First action:** at the #56876 pin assert the alignment for our decode shapes, then paired
decode-step timing ±#58681 at c1/c16 with greedy byte-identity.

**4. PR #57605 — honor scheduler lookahead in mamba align-mode allocation (jsolman).** Two mechanisms (unmaterialised next page at an exact page boundary; null padding displacing the
running-state column → permanent zero acceptance), diagnosed only on 4-node Thor SM110 / GLM-5.3-Flash. Missing: a model-free `MambaManager` test discriminating both (our p9 playbook) plus confirmation it
composes with #57050's `physical_block_cap` clamp. Ownership: jsolman, responsive (rebased same day);
no maintainer, no competing PR. Cost: CPU only.
**First action:** write the two-case manager test; confirm fail at merge-base / pass at his head.

**5. Issue #57266 + PR #57278 — `conv_ssm_forward` empty strided slice kills EngineCore.** #57266 (Yunzez):
8 days, **zero comments**, reproduced on two vendors' hybrids (granite-4.0-h-tiny, Falcon-H1-90M),
sm_89 and sm_90, with `--mamba-block-size 32 --enable-prefix-caching`. #57278 (twu3202) claims the
fix but frames its invariant around `mamba_cache_mode="all"`, which that repro never sets. Missing:
an adjudication that the guard fires on #57266's flags, plus a model-free replay of the index
arithmetic as a failing regression. Ownership: twu3202, unreviewed; no competing fix. Cost: CPU replay + GPU ≤ half day (90M model).
Engagement: low — neither author nor maintainer has spoken.

## Reject / defer
#57267 BPbruce claimed it; #58653 willweimike working it; #58726 hclsys already posted the
second-GB10 repro; #58422 needs an NVFP4 artifact we lack, blocked on #46329; #58638 RFC and #58207/#58380
maintainer-owned (lucamotz supplying B200 data); #57539 part of the #55627/#55524/#55905 series,
broad; #58644 bot-filed CI flake, no owner; #58303 needs an interleaved workload we'd invent;
#57613/#57610 sliding-window, author supplies both halves; #52244/#57261/#57807 needs-rebase;
#57318/#57352 kernel perf, no cheap evidence; #58485/#58080/#57941 outside hybrid geometry.

## Plan changes
**(a) Our P8 fixture is now third-party adjudication infrastructure.** Karl0007 published
`Karl0007/vllm:oracle/53142-adapted-arms`, *based on our `p8-restore-fidelity` branch*, as a 3-arm
oracle: A (#53798) 10 passed, B (#55507 as written) 2 failed, C (#55507 + `adc7d30`) passes. Our
fixture found a real hole in #55507 — its unbound fallback used `cache_config.block_size`, the very
divisor the PR removes — and he fixed it. The align-seed cluster (#53798/#55507/#55601/#55688) now
shares an oracle that is ours and **still has no maintainer verdict**. **#55507 is not on our
used-threads list.**
**(b) njhill now owns the padded-tail region our p9 test guards.** #58434 (padded prompt tails as
spec-decode rows for hybrid models) merged 09-25; #58462 (dummy `idx_mapping` dtype) merged 09-23;
#58400 folds benchislett's "classify from request state earlier" into the same code — where our
#55506 finding lives. The p9 test's value rises, but its natural home is now main, not #55506.
**(c) LucasWilkinson RFC #58638** (+#58207, #58380) sets hybrid KV-cache-grouping direction for spec
drafters, naming Qwen3.6 + DFlash; lucamotz feeds it B200 data and drafts #58762/#58763. Independent
block-size/geometry PRs filed now risk being superseded.
**(d) #56792** changed state 09-20: Leslie360 added a cross-repo update (SGLang #40144, same root
cause, fix in flight); still no vLLM maintainer.
