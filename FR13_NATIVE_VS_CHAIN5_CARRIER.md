# FR13 — Native vs Chain5 Carrier Synthesis

Read-only synthesis of two divergent astropy SWE tasks (native RESOLVED, chain5 FAILED).
Goal: (1) a discriminating GATE, (2) a CARRIER verdict with honest confidence.

**Constraint honored:** the chain5 arm is live on the single GPU. This document is READ-ONLY.
No server was booted, no inference/docker/GPU touched. Only files read + grep.

---

## 1. Per-task diff table

| Axis | astropy-13453 (HTML `formats=`) | astropy-14508 (`_format_float` FITS) |
|---|---|---|
| Native outcome | RESOLVED (tests_passed) | RESOLVED (tests_passed) |
| Chain5 outcome | FAILED (tests_failed) | FAILED (tests_failed) |
| File/method chosen | SAME — `io/ascii/html.py::HTML.write()` | SAME — `io/fits/card.py::_format_float` |
| Root cause verbalized | SAME — "write() calls `_set_fill_values()` but NOT `_set_col_formats()`" | SAME — `f"{value:.16G}"` is the culprit; use `str(value)` |
| Insertion point | SAME — right after `self.data._set_fill_values(cols)` | SAME — top of `_format_float` |
| **Delta (the miss)** | Native prepends load-bearing `self.data.cols = cols`; **chain5 omits that one line** → `_set_col_formats()` iterates an empty `self.cols`, no-op | Native rewrites body + adds `e`→`E` FITS exponent normalization; **chain5 bolts on an early `return str(value)`** → lowercase-`e`/unpadded sci-notation bypasses the `elif "E"` FITS branch, regresses existing tests |
| Nature of miss | Incomplete patch — one missing dependency-setup line | Incomplete patch — short-circuit skips one normalization branch |
| Chain5 got corrective feedback? | NO — self-verify script died on broken-install ImportError ("could not determine astropy package version") → no signal the fix was a no-op | NO — first attempt aborted (apply_patch unsupported + JSON EOF at col 1350); empty_patch_retry produced the 857B patch; no functional verification |
| Shared infra artifacts | char-8-style silence; broken-install verify | **Both arms** hit identical `Unterminated string … (char 8)` BadRequestError at turn end; chain5 also hit apply_patch-unsupported + JSON EOF |
| Kernel/seed status | CONFOUNDED (native=stock qwen3_5_mtp/FLASH_ATTN; chain5=forked-fa2 TREE_ATTN; temp 0.6, N=1) | CONFOUNDED (same, N=1, temp 0.6) |
| Per-task lean | leans-seed | leans-seed |

---

## 2. Is chain5's failure SYSTEMATIC or HETEROGENEOUS?

**Verdict: HETEROGENEOUS / one-off ⇒ leans SEED.** Stated pattern:

- **The only shared attribute is a low-resolution one:** "landed on the right file/method/root cause,
  then produced an INCOMPLETE patch that missed one load-bearing element." That is a *completeness*
  descriptor, not a decode signature. The two misses are mechanically unrelated:
  - 13453: missing a **prerequisite assignment** (`self.data.cols = cols`) before a correct call.
  - 14508: adding an **early-return short-circuit** that skips a downstream normalization branch.
  These are different failure *classes* (omitted-setup vs premature-return), not one recurring
  wrong-pattern. A kernel carrier would show a *repeatable, characterizable* decode oddity
  (a recurring token-level derail, a systematic drop of a construct, a degeneration signature) —
  we see coherent, correctly-reasoned traces in both arms reaching the same diagnosis.
- **The errors that ARE shared across tasks are infra/serialization, not correctness:** the
  `char-8` `Unterminated string` BadRequestError and JSON-EOF/apply_patch-unsupported glitches.
  Critically these DO NOT cause the wrong patch — on 14508 **native hit the identical char-8 error
  and still landed correct**. So the shared signature is orthogonal to the correctness delta.
- **What decided each task was the absence of corrective feedback**, not a decode defect: chain5's
  self-verification crashed on a broken-install ImportError in BOTH tasks, so an incomplete patch
  went uncaught. Native's advantage was *iteration that converged* (13453: 3 retries → included the
  line), not a different token distribution. That is a run-to-run/seed-level property.

**Conclusion:** N=2 divergences, both task-specific incomplete-fix slips, both with kernel AND seed
confounded, both leaning seed. No kernel signature. This is a heterogeneous seed lean.

---

## 3. Reconciliation with known facts

- The **char-8 degeneration gate** (`scripts/fr13_char8_degeneration_gate.py`) already REJECTED
  char-8 as carrier: chain5 has 0 char-8 while native had char-8 yet resolved 5/5. This synthesis is
  consistent — char-8 is present in native's 14508 trace too and did not stop it resolving. char-8 is
  an infra/tool-call-serialization artifact, not the correctness carrier.
- Chain5's losses here are **`tests_failed` with real, well-targeted patches** — not empty patches,
  not degenerate output, not truncation. The model reasons correctly and edits the right lines; it
  just stops one element short. That is the profile of a *seed/iteration-luck* gap, not a systematic
  kernel corruption.
- **Resolve-rate is too noisy to adjudicate:** native 5/5 vs chain5 ~1–2/5 at **N=1, temp 0.6**.
  A one-line-short miss caused by an uncaught broken-install verification is exactly the
  high-variance, environment-coupled outcome that N=1 cannot separate from kernel. The rate delta is
  real but **not attributable** without de-confounding.

---

## 4. Proposed discriminating GATE (reusable, beyond resolve-rate)

### 4A. PRIMARY — Teacher-forced per-token logit-argmax gate (DE-CONFOUNDS kernel vs seed)

**Tool:** `scripts/fr13_apc_teacher_forced_logit_gate.py` (571 L; live-server required, boots nothing).

**Why it de-confounds:** resolve-rate mixes kernel + seed + environment. This gate removes seed and
temperature: it TEACHER-FORCES both kernels through ONE fixed token sequence (the cache-OFF / native
continuation) and compares the per-position **next-token argmax + top1–top2 logit margin**. There is
no free-running feedback, so temperature is irrelevant — at fixed inputs the two arms either produce
the same argmax or they do not. This is precisely the tool for kernel-vs-seed here: adapt the
ON/OFF pairing to **native-kernel vs chain5-kernel (forked-fa2 TREE_ATTN)**, both on this boot.

**Inputs:**
- A normalized `/v1/responses` turn from a *divergent* transcript (the 13453 or 14508 hit turn),
  normalized via `inference_proxy.normalize_responses_request_payload` (`--request-json`), OR a raw
  `--prompt-ids-json` list to skip render.
- Reference continuation = the native (cache-OFF) generated continuation, `--max-cont-tokens 192`.
- `--margin-floor 2.3026` (nats; the established `fr13_gold_margin_probe` threshold).
- Both kernels reachable on the same live boot.

**Outputs (`verdict.json`):** per-position `{argmax_diff, margin_off, margin_on, within_boot_det,
class}` plus `first_argmax_diff`, `confident_flips`, `near_tie_flips`, `noise_nondet_positions`.

**Per-position classification (temperature-independent, logit-space):**
- `CONFIDENT flip` — argmax differs AND both arms within-boot deterministic AND both margins >
  margin_floor ⇒ a REAL kernel-induced decision change.
- `NEAR-TIE flip` — max(margin) ≤ floor ⇒ noise-level, not a defect.
- `NOISE-NONDET` — an arm's two reps disagree ⇒ autotune (BATCH_INVARIANT=0), not adjudicable.

**Pass/fail:**
- **LOSSLESS** (exit 0): 0 CONFIDENT flips → forked-fa2 TREE_ATTN does NOT alter the decode
  distribution beyond noise ⇒ **kernel EXONERATED**, the ~1–2/5 vs 5/5 gap is SEED.
- **NOT-LOSSLESS** (exit 1): ≥1 CONFIDENT flip → the kernel confidently changes a decision the model
  held ⇒ **kernel is a plausible carrier**; report first flip position + count and inspect whether it
  sits at the patch-content divergence (the `self.data.cols = cols` line / the early-return).

### 4B. First-divergence localizer (which internal site moves first, if 4A says NOT-LOSSLESS)

**Tool:** `scripts/fr13_apc_hit_first_divergence.py` (offline, no GPU). Given captured per-layer
activations + final-token logits for the hit turn, reports the earliest divergent site across seed /
per-layer `core_out` argmax-mismatch / final-token argmax flip. Verdict names where a carrier first
appears, or "no measurable carrier" if all argmax-identical. Use only to attribute a 4A NOT-LOSSLESS.

### 4C. Behavioral marker the trace diff revealed (cheap N-replica screen, complements 4A)

The trace diff exposed a concrete, gradeable behavioral gap independent of resolve-rate:
**"did the arm's self-verification actually execute and return a functional signal?"** In BOTH chain5
losses the verify step died (broken-install ImportError / JSON-EOF) and the incomplete patch went
uncaught. Proposed marker `verify_signal_present ∈ {ran_ok, crashed_import, aborted_parse, absent}`
parsed from `codex_trace*.jsonl` / `codex_stdout`. Pass = arm's final patch was preceded by a verify
that ran and returned a result. This isolates the *feedback-loop* failure from any kernel claim and is
runnable over replicas without the GPU-heavy teacher-force. It is a SCREEN, not a carrier verdict —
4A remains the adjudicator.

---

## 5. CARRIER verdict (honest confidence)

**LEAN: SEED. Confidence: LOW-to-MODERATE. NOT a kernel verdict.**

- Both divergences are heterogeneous, task-specific incomplete-fix slips; no systematic decode
  signature. The shared errors (char-8, JSON-EOF) are infra artifacts already rejected as carrier and
  are orthogonal to correctness (native survived char-8 and still resolved).
- **A single-run trace diff CANNOT de-confound kernel vs seed.** Native and chain5 differ in BOTH
  kernel (stock qwen3_5_mtp/FLASH_ATTN vs forked-fa2 TREE_ATTN) AND seed (temp 0.6, N=1). The
  observed 5/5 vs ~1–2/5 gap is real but unattributed. The verdict is a **LEAN pending the
  teacher-forced gate (§4A) or many replicas** — not a conclusion.
- **cache:** not implicated — chain5 losses are `tests_failed` real patches, not cache-corruption/
  degeneration; no cache signature in the diff. Cache stays de-prioritized unless 4A flags it.
- **If §4A returns NOT-LOSSLESS**, the plausible mechanism is: forked-fa2 TREE_ATTN subtly shifts the
  decode distribution at patch-construction time, nudging the model onto a *nearby-but-incomplete*
  solution path (drop the `self.data.cols = cols` setup line / take the early-return short-circuit)
  rather than the complete edit. That mechanism is only credible if a CONFIDENT flip lands at or
  before the patch-content divergence; otherwise the gap remains SEED.

---

## 6. De-confound plan (ordered, cheapest first)

1. **Run §4A teacher-forced gate** on the 13453 and 14508 divergent turns, native-kernel vs
   chain5-kernel, both on one boot, force native's continuation. Read `VERDICT`:
   LOSSLESS ⇒ kernel exonerated, gap = SEED (stop). NOT-LOSSLESS ⇒ go to 2.
2. **Localize** any CONFIDENT flip with §4B; check whether it coincides with the patch-content
   divergence (the missing line / the early-return).
3. **Replica de-confound (if no live paired boot):** run chain5-kernel at N≥5, temp 0.6, on the same
   two tasks. Heterogeneous misses across replicas ⇒ SEED; the SAME missing line/short-circuit
   recurring ⇒ kernel-shaped, escalate.
4. **Environment control:** fix the broken-astropy-install so chain5's self-verification returns a
   real signal, then re-run. If chain5 self-corrects (converges like native's 3 retries on 13453),
   the gap was feedback-loop/env, NOT kernel — the strongest confound to remove before any kernel
   claim.
5. Apply §4C behavioral marker as a fast pre-screen across all replicas to quantify how often the
   verify step failed to fire.

**Do not** upgrade the lean to a kernel verdict on N=1 trace evidence. The teacher-forced gate (§4A)
is the single adjudicating instrument here.
