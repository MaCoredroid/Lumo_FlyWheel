# FR13 — DIFFUSE-vs-FIXABLE BIG-DENOMINATOR TEST PLAN (ready-to-run GPU spec)

Date 2026-06-15. CPU-only DESIGN, READ-ONLY (the K1 same-boot mechanism proof
`fr13_k1_mechanism_proof_workflow.js` holds the single GPU; this doc edits no code, boots
nothing). vLLM source ground = `scripts/vllm_src.sh` (pinned image sha `3dbe092e` =
0.19.2rc1.dev134). int-view NEVER atol.

## 0. THE PROBLEM (why this plan exists)

The banked flip numbers are SMALL-SAMPLE. Denominator = 4 `prompts_swe4` × up to 128 tokens,
truncated by EOS:
- native-E5: **3 clear-margin flips / 512 tokens = 0.59%**
- cat9 OFF: **23 / 435 = 5.29%**
- cat9 + K1: **20 / 466 = 4.29%** (re-confirmed: `output/fr13_recurrent_oracle/rescore_cat9.json`
  = 20 clear / 462 positions, per-prompt `[4,7,4,5]`, RECURRENT_PATH_ENGAGED, 43968 recurrent calls)

Cross-boot autotune noise is **±a few flips** (`feedback_no_cross_boot_byte_gate`,
`464013ce`: same-boot flag-OFF 26 vs banked 22 ⇒ ~±4 autotune variance), which is LARGE
relative to ~3–23 counts over ~450 tokens. ⇒ loose error bars, wobble (21/22/23), and
K1 18→12 cannot be certified above noise. The user wants a **BIG DENOMINATOR** via a
SWE-Verified ~30-min task to get a tight RATE + a real diffuse-vs-fixable verdict.

**Playbook rows this plan lives or dies by (quote in the run prompt):**
- **#9 (silent fallback / vacuous instrument):** a run "passes" while measuring nothing.
  Engagement asserts BEFORE any number — recurrent-decode path counter > 0, spec counters
  advance on the served arm + DO NOT advance during oracle rescore, draft-shape == 9.0/5.0,
  flags live in the worker `/proc/<pid>/environ`. The oracle already FAILS LOUD on zero
  `_forward_core_decode_non_spec` calls (`fr13_recurrent_decode_oracle.py:308/463`).
- **#12 (measurement traps / length / denominator):** raw counters only; capture-once pinned
  prompts; **per-turn token re-derivation must be validated** (re-tokenize round-trip ==
  served text byte-for-byte) or the denominator is a fiction; label every estimate; the
  binomial CI is on the REAL position count, not a hand-rolled rate.
- **#8 (offline ≠ live):** the rescore is in-process recurrent decode; it is NOT the served
  stream — the served stream is the cat9/native serving path. We score the SERVED tokens
  against the recurrent oracle, never substitute one for the other.

The diffuse hypothesis to **TEST (not assume):** the cat9 residual is the multi-layer
GDN/full-attn realization gap (`FR13_DIFFUSION_DEEP_DIVE`: geometric ~1.05–1.3x/layer,
LARGEST jumps at deep full-attn L35/47/51/59/62-63, **no single dominant alignable layer**)
+ the LCP-committer trajectory-fork superset (`FR13_APPLE_TO_APPLE_FORK`: of 23 forks,
R=8 genuine confident verify-vs-decode, C=8 co-residency near-ties, W=7 instrument artifacts),
NOT a single per-forward seam (scan-recompute refuted e2e `318f8b9f`; K1 partial ~1/3;
N_PAD null `c007d32e`; conv/FA2 amplifiers not originators). FIXABLE = a concentration that
re-opens a lever (one layer, one token-class, one fork-kind dominating).

---

## 1. THE SWE-VERIFIED BIG RUN

### 1.1 The harness that exists (RUNNABLE-NOW)
`scripts/run_swe_bench_q36_a.py` is the per-instance orchestrator: hydrate the SWE-Bench base
commit, drop AGENTS.md with the problem_statement, launch `codex-runner:v1` against the proxy
at `:8022` (which fronts vLLM at `:9950`), diff workspace→`patch.diff`, run
`codex-bench-eval-swe` (the patch+tests gate), write per-task artifacts under
`output/swe_bench_q36_a_temp06/<dataset>/per_task/<instance_id>/`. Wall budget default = 25 min
agent + 5 min eval (`DEFAULT_AGENT_WALL_S=1500`). Subsets are pre-registered by
`scripts/build_swe_bench_subset.py` (tier 0 = 20 instances, deterministic seeded). The gold
gate (`FR13_B1_SWE_GOLD_BIND.md`) already proved this runs clean at B=1: 6/6 rc=0, full wall,
draft-shape 9.0 (tree) / 5.0 (native), tasks astropy-12907 (resolved both arms) + astropy-13033
(failed both — a hard instance, symmetric).

### 1.2 Task selection (the denominator)
A SWE patch turn is **hundreds-to-few-thousand served tokens across the agent's turns**, NOT
30k. To get a big-but-bounded denominator inside ~30 min wall:

- **PRIMARY: `astropy__astropy-12907`** — the gold-gate task that RESOLVED on both arms at
  B=1 (cat9 patch passed tests, native patch passed tests). Re-using it means the SWE-quality
  gate (§4) has a known-good reference: we already know both arms can SOLVE it. Wall ~1565 s
  per arm (under the 30-min agent budget). Completion-token denominator: the gold-gate
  task1 ran 41 engine iterations on the first turn alone (proxy capture sample =
  `completion_tokens` 146 on one turn); a full agent rollout = tens of turns × tens-to-hundreds
  of completion tokens ⇒ **~1.5k–4k served tokens per arm** (≈4–9× the current 450). That alone
  takes the binomial CI half-width from ~±2% to ~±0.7% at 5% (see §2.4).
- **SECONDARY (only if PRIMARY's denominator < ~1500 tokens): add `django__django-11099`** or
  another tier-0 instance with a longer expected rollout. Keep total ≤ 2 tasks/arm so the wall
  stays inside the 30-min frame and the GPU is freed for the K1 proof promptly.
- **DO NOT** pick a task both arms fail with a 0-byte patch (no served stream of substance);
  astropy-13033 failed symmetrically — it is the SWE-quality control (both fail ⇒ not a tree
  regression) but its served stream is fine to ALSO score (forks are forks regardless of pass).

### 1.3 Capture every served token (the big-denominator vehicle)
The proxy already has the instrument, **default OFF**: `inference_proxy.py:33-69`
`_pair_dump_upstream`, gated on env `LUMO_PROXY_PAIR_DUMP_DIR`. On every UPSTREAM `/v1/responses`
call (initial + each auto-continue retry — `:2066` / `:2148`) it writes one JSON pairing the
exact upstream **request payload** with the parsed upstream **response** (output items =
reasoning/message text + function_call args, + `usage` token counts). Diagnostic-only; served
traffic unchanged.

**Run config:**
- BOTH arms launched per the locked launchers (§1.4), env `LUMO_PROXY_PAIR_DUMP_DIR=/logs/pair_dump_<arm>/`
  set on the PROXY process (not the vLLM container).
- Run `run_swe_bench_q36_a.py` once per arm on the chosen task(s), agent wall = 1500 s, eval = 1800 s.
- Each turn ⇒ one `pair_<ts>_<seq>_<kind>.json`. The per-turn **request** is the chat input
  (the oracle "prompt"); the per-turn **response** output text is the served stream for that turn.

**Token-ID recovery (the #12 hazard — handle explicitly).** The pair-dump captures TEXT, not
token IDs (the `/v1/responses` SSE path does not emit token ids). The recurrent oracle needs
`served_token_ids`. Two routes, in preference order:
  (A) **Add `return_token_ids` to the served arm IF the proxy can pass it through** — read
      `inference_proxy.py` normalize path; `/v1/responses` upstream is vLLM's Responses API,
      which does NOT reliably carry per-token ids back through the codex `responses` wire_api.
      Treat (A) as best-effort; if it does not round-trip, use (B).
  (B) **Re-tokenize per turn with a VALIDATED round-trip (the runnable default).** For each
      pair-dump turn, concatenate that turn's served output items into the exact served text,
      tokenize with the model tokenizer, and **assert detokenize(ids) == served_text byte-for-byte**
      (class #12 guard — a turn that fails the round-trip is DROPPED from the denominator and
      counted, never silently mis-scored). The oracle's forced-decode is internally consistent
      under (B): it forces `served_ids[i]` and reads the clean recurrent argmax BEFORE forcing,
      so as long as `served_ids` is a faithful tokenization of the served text the per-position
      flip is well-defined. This is the `fr13_recurrent_decode_oracle.py rescore` `--src` schema:
      `{prompts:[str...], records:[{served_token_ids:[int...]}...]}` — one (prompt, served_ids)
      pair per turn.

**This (B) re-tokenization + round-trip validation is the ONE new piece of harness** (a ~80-line
reducer `scripts/fr13_swe_stream_to_oracle_src.py`, NOT yet written): glob the pair-dump dir,
emit the oracle `--src` JSON per arm, with a `dropped_turns` count + `roundtrip_ok` per turn in
the artifact. Everything else is runnable now.

### 1.4 Flags (the DEPLOYED pipeline, both arms)
- **cat9 (deployed tree):** `scripts/fr13_launch_locked.sh` (== main HEAD b7887c89 default-ON,
  the gold-gate serving path). num_spec=9, TREE_ATTN, FIX-1/2/3/A ON, REPLAY_ROUTE=1,
  FA2_TREE_BIAS=1, CONV_COMMITTED_PATH=1, in_proj_ba pad `LUMO_FB_PROJ_PAD_ROWS=16`,
  BATCH_INVARIANT=0. **+ K1 if the concurrent K1 proof passes and the user bakes it**
  (`FR13_SCAN_ALIGN=1 MODE=body`) — otherwise cat9 OFF (K1 unset). Run BOTH cat9 variants only
  if the user wants the K1 18→12 re-test at scale; default this plan = cat9-as-currently-locked.
- **native-E5:** `scripts/fr10_launch_speed_server.sh` run as `naive_mtp` / FLASH_ATTN /
  num_spec=5 (the E5 baseline `output/fr10_native_mtp5_same8_20260604T210257Z`). This is the
  ~3-flip BAR / within-floor reference.
- Model: `/models/qwen3.6-27b-fp8`, served name `qwen3.6-27b`. **Pin E5's launch config** so the
  oracle scores the SAME model build both arms (`FR13_FLAGS.md` §"ACTION (validity)").

### 1.5 Engagement asserts at serve time (class #9, BEFORE trusting the denominator)
- Snapshot `/metrics` spec counters before/after each arm; the served arm's
  `spec_decode_num_drafts` MUST advance and `draft_tokens/drafts` == 9.0 (cat9) / 5.0 (native)
  (the gold-gate "draft-shape ✓" check; `run_swe_bench_q36_a.py` already snapshots /metrics).
- Assert the launcher flags are live in the vLLM worker `/proc/<pid>/environ` (class #9, the
  `3babafbe` ray-env-allowlist trap: env not reaching EngineCore = vacuous).
- Assert each arm produced ≥ N turns of pair-dump with non-empty served text.

---

## 2. FLIP SCORING AT SCALE (the recurrent oracle over a few-thousand-token denominator)

### 2.1 The scoring engine (RUNNABLE-NOW, deployment-correct)
`scripts/fr13_recurrent_decode_oracle.py rescore` — loads the model ONCE in-process (offline
`vllm.LLM`, NO speculation, FLASH_ATTN, eager, `max_num_seqs=1`), and for each (prompt,
served_ids) teacher-forces the served stream ONE single-token decode per position. Each oracle
distribution is produced by the **RECURRENT** `_forward_core_decode_non_spec` path
(causal_conv1d_update + fused_recurrent_gated_delta_rule_packed_decode), carrying conv/ssm
state forward in the KV cache — **the deployment no-spec decode path, NOT chunked prefill, NOT
streamed logprobs** (the HTTP teacher-force `fr13_oracle_stream_teacher_force.py` re-prefills
`prompt+served[:i]` per position ⇒ `num_prefills>0` ⇒ chunked WY ⇒ WRONG frame; the script
header documents this at length, L1-72). This is the correct compare target per
`feedback_fr13_lossless_compare_target`: no-spec recurrent decode = ground truth.

### 2.2 Per-token clear-margin flip (the metric, identical both arms)
At each served position i the LP records, BEFORE forcing:
- `oracle_argmax_id` = recurrent argmax; `flip := served_id != oracle_argmax_id`
- `deviation_nat := oracle_argmax_logprob − served_logprob_in_oracle`
- `clear_margin := flip AND (served_id outside oracle top-k OR deviation_nat > 1.0)`
The binding count is **clear-margin flips** (deviation_nat > 1.0 nat gold-margin), NOT raw flips
(near-ties are float noise / genuine ties = lossless). top_k = 20. THRESHOLD = 1.0 nat (same as
all banked rescores; matches the gold-margin probe). Within-process rep1==rep2 per-position
determinism is recorded (class #8 same-boot gate; the oracle runs each prompt twice).

### 2.3 Bounding the bottleneck (HONEST GPU cost — the rescore IS the cost)
The rescore is `sum(served_len) × 2 reps` single-token recurrent decode forwards, in-process,
B=1, eager. At GB10 B=1 decode ~17 TPS (~59 ms/token forward; eager + LP overhead pushes this
to ~80–120 ms/position):
- Current 4-prompt 462-position rescore: ~462×2 ≈ 924 forwards ≈ **1–2 min** of decode + model
  load (~2–4 min). Confirmed feasible (it is banked).
- Big run, ~3k served tokens/arm × 2 reps = ~6k forwards/arm ≈ **8–12 min/arm**; 2 arms ≈
  **16–24 min** of rescore + 2× model load (~5–8 min). **Fits one ~30-min GPU window per arm-pair**
  but is the dominant GPU cost — the SWE generation itself (§1) is the OTHER ~50 min (1500 s ×
  2 arms, serialized). So the full campaign is **~2 GPU windows**: window-1 = serve+capture both
  arms (~50–60 min), window-2 = rescore both arms (~25 min). Drop rep2 to halve rescore if the
  CI is already tight (the within-boot determinism is already banked clean; rep2 is a re-confirm,
  not load-bearing at scale — make it a `--reps 1` flag).
- **Cost-reduction levers (cheap, no correctness loss):** (a) rescore is incremental-friendly —
  cap each turn at the served_len (no re-prefill across turns is needed because each turn's prompt
  is self-contained chat input); (b) `--reps 1` after the determinism re-confirm; (c) score only
  cat9 + native (skip cat9+K1 unless the K1-at-scale question is live).

### 2.4 The tight rate + binomial CI (class #12: CI on the REAL denominator)
Per arm: `rate = total_clear_margin_flips / total_positions`. Report the **Wilson 95% CI**
(better than normal at small p). Half-widths at p≈5%:
- n=462 (current): ±~2.0% ⇒ cat9 [3.3%, 7.5%], native [0.2%, 1.9%] — already separated, but
  loose, and within the ±4-flip autotune wobble.
- n≈3000 (big run): ±~0.8% ⇒ cat9 ≈ [4.2%, 6.4%], native ≈ [0.3%, 1.2%] — **non-overlapping
  with margin**, and the half-width (~24 flips of slack at 3k) now exceeds the ±4 autotune
  wobble ⇒ the rate is certified above noise.
- The CI is computed on the position count AFTER dropping round-trip-fail turns (§1.3); the
  artifact records `n_positions_scored`, `n_turns_dropped`, `n_positions_dropped`.

---

## 3. DIFFUSE-vs-FIXABLE DISCRIMINATOR (the actual test)

Four measurements over the big sample; a DECISION RULE at the end. (a)+(b)+(c) are
RUNNABLE-NOW reducers over the rescore artifact + the served stream; (d) is a targeted GPU
sub-capture.

### (a) Rate stable + significantly above native (CI non-overlapping) — RUNNABLE
The Wilson CIs from §2.4. PASS-for-diffuse condition: cat9 CI lies entirely above native CI AND
both arms' rep1==rep2 determinism holds. If cat9 collapses to native at scale (CIs overlap), the
banked 5% was an autotune-wobble artifact ⇒ there is nothing to fix and nothing diffuse — relax.

### (b) Structural-boundary clustering (token-class distribution) — RUNNABLE
For every clear-margin flip, classify the served token + its left/right context into a
token-class bucket using the served TEXT (already in the pair-dump): {code-fence/backtick,
JSON/tool-call arg (inside function_call arguments), prose, identifier/code, whitespace/newline,
punctuation/format-fixed}. Compute the flip rate PER class and the share of flips in each class.
- **DIFFUSE signature:** flips concentrate at small-clean-margin **structural boundaries**
  (code-fence / prose↔code / tool-call JSON boundaries) — the `FR13_DIFFUSION_DEEP_DIVE` +
  gold-gate finding (the p3 ` ```` `-vs-`Let` 168× flip clustered at a code fence; the gold-gate
  forks at structural/template boundaries). This is realization-gap: the argmax is decided by a
  small-margin race at a boundary, lost to deep-layer accumulation.
- **FIXABLE signature:** flips concentrate in ONE format-deterministic class (e.g. all flips are
  JSON-key tokens inside tool-call args, or all are a single repeated token) ⇒ a wiring/commit
  bug in that path, not diffuse.

### (c) Leaf-fork vs spine-realization split at scale — RUNNABLE (reuse the FIXED reducer)
This is the cat9-specific superset axis. Reuse the apple-to-apple classifier with the
**CORRECTED reducer** (`FR13_APPLE_TO_APPLE_FORK` §2: `fr13_fork_margin_classify.py::_deciding_margin`
must join the verify margin AT the oracle FLIP position's own committed row, and read the
`self_logits` row for `tree_self_target` bonus flips — the dump must additionally emit it via
`FR13_FORK_MARGIN_DUMP`). To get the dump at scale, RE-RUN the SWE serve arm for cat9 with
`--arm FR13_FORK_MARGIN_DUMP` armed (read-only, default-OFF, byte-identical serving path per the
locked launcher), then classify each clear-margin flip as:
- **W** (instrument row/position misalignment) — must be ~0 with the corrected reducer; any
  residual W = denominator hygiene, dropped.
- **C** (co-residency-perturbed near-tie, verify margin <1 nat at the right row) — the no-copy /
  verify-row-isolation lever.
- **R** (genuine verify-vs-decode realization gap, verify ≥1 nat at the right row prefers a token
  decode rejects) — the diffuse GDN/full-attn front.
At small sample R=8, C=8 (of 23). At scale the **R:C ratio + their absolute rates** are the
signal: stable R:C ≈ 1:1 spread across positions = diffuse; a spike in C (co-residency) at one
tree depth or one leaf slot = a fixable co-residency seam; a spike in R at one layer = (d).

### (d) Per-layer first-nonzero attribution on a sample of flips — TARGETED GPU SUB-CAPTURE
Only on a random sample of ~8–12 clear-margin R flips (NOT the whole stream — too expensive).
For each sampled flip, run the same-boot input-aligned per-layer ladder
(`FR10_LAYER_HIDDEN_CAPTURE` + the node5/node7 ladder reducer, `output/fr13_node5_ladder/`
pattern): live tree-verify deep row vs clean teacher-forced single-forward of the accepted
prefix, both entering L0 byte-exact (input_maxabs=0.0), residual-L2 per layer.
- **DIFFUSE signature (the banked one):** no single dominant alignable layer — monotone smooth
  growth ~1.05–1.3x/layer, largest jumps at deep full-attn (L35/47/51/59/62-63), residual born
  in L0's GDN compute. The per-flip first-nonzero is L0-GDN for ALL sampled flips, and no single
  later layer carries a disproportionate share.
- **FIXABLE signature:** a clean→broken SPIKE at ONE layer reproducible across the sampled flips
  (a layer where residual jumps >>2x and is small everywhere else) ⇒ that layer/kernel is the
  seam, re-open it.

### DECISION RULE (the explicit verdict)
Let cat9 big-denominator clear-margin rate = `p̂` with Wilson 95% CI `[lo, hi]`, native rate
CI `[nlo, nhi]`.

**DIFFUSE (no fixable seam — RELAX, the residual is the realization gap, ship as
lossless-enough IF §4 passes):**
ALL of:
1. `lo > nhi` (cat9 significantly above native) AND `hi − lo < 0.015` (tight, ~3× below the
   banked spread) AND rep1==rep2 determinism holds both arms; AND
2. (b) flips are structural-boundary-clustered — NO single non-structural token-class holds
   > 50% of flips, AND the top structural classes (code-fence + prose↔code + tool-call boundary)
   together hold the majority; AND
3. (c) corrected W ≈ 0, and R+C spread across positions/depths/leaf-slots with **no single tree
   depth or leaf slot holding > ~40% of forks**, R:C within ~1:2..2:1; AND
4. (d) on the sampled flips, first-nonzero is L0-GDN and NO single later layer carries a
   reproducible >2× spike (the monotone-diffuse curve).

**FIXABLE (a concentration that re-opens a lever — do NOT relax, attack it):**
ANY of:
1. one token-class (non-structural, format-deterministic) holds > 50% of flips ⇒ commit/wiring
   bug in that path; OR
2. one tree depth or leaf slot holds > ~40% of forks, or C spikes at a specific depth ⇒
   co-residency seam (verify-row isolation / no-copy lever); OR
3. (d) a reproducible single-layer clean→broken spike ⇒ that kernel is the seam; OR
4. cat9 CI collapses onto native (CIs overlap) ⇒ the 5% was autotune wobble, no real gap
   (also a "relax" but for the opposite reason — there was never a defect).

**Non-vacuity for the verdict itself (class #9/#12):** the denominator is the validated
round-trip token count (not text length); the oracle counter > 0 and the spec counters did NOT
advance during rescore; flags live in the worker; W (instrument) is driven to ~0 before R/C are
read. A "diffuse" verdict reached with W>0 or a fictional denominator is INVALID.

---

## 4. SWE-QUALITY GATE (the deployable answer)

The decisive deployable question: **does cat9 SOLVE the SWE-Verified task (patch applies +
tests pass) vs native-E5, DESPITE the ~5% clear-margin flips?** If YES, the flips are
quality-irrelevant structural-boundary token choices = the deployable lossless-enough answer
(the flip is a small-margin boundary token the agent recovers from; the gold gate already showed
codex SOLVING task1 on the tree despite a ~6% non-argmax commit rate — `FR13_B1_SWE_GOLD_BIND`).

**Capture (RUNNABLE-NOW, it is what `run_swe_bench_q36_a.py` already does):**
- Per task, per arm: `codex-bench-eval-swe` runs the generated `patch.diff` against the
  instance's test suite ⇒ `resolved` / `failed` in the per-task `report.json` / campaign summary.
- **PASS/FAIL capture:**
  - PASS (deployable-lossless-enough): cat9 `resolved` on the task(s) native-E5 also `resolves`
    (matched outcome on the solvable instance) — flips did not change the task outcome.
  - FAIL (quality-relevant): cat9 `failed` where native-E5 `resolved` on the SAME instance ⇒
    the flips changed the trajectory enough to break the solve ⇒ NOT lossless-enough, the diffuse
    relaxation is unsafe, re-open.
  - CONTROL: a task BOTH fail (astropy-13033) is a draw, not a tree regression — record but it
    does not gate.
- Run the solvable PRIMARY task with **B=N≥2 same-instance repeats per arm** (the gold gate ran
  tree_a/tree_b, native_a/native_b) so "resolved" is not a single coin-flip; the gate is "cat9
  resolves at the same rate as native on the solvable instance."
- This is the **binding deployable verdict**: §3 says WHAT the residual is (diffuse vs fixable);
  §4 says whether it MATTERS. Diffuse + SWE-quality PASS = ship cat9 as lossless-enough. Fixable
  OR SWE-quality FAIL = do not relax.

---

## 5. RUNNABLE-NOW vs NEW-HARNESS

| piece | status |
|---|---|
| SWE serve + patch/test gate (`run_swe_bench_q36_a.py`) | RUNNABLE-NOW (gold-gate-proven) |
| Served-stream capture (`LUMO_PROXY_PAIR_DUMP_DIR` pair-dump) | RUNNABLE-NOW (in `inference_proxy.py`, default-OFF) |
| Recurrent decode oracle rescore (`fr13_recurrent_decode_oracle.py rescore`) | RUNNABLE-NOW (deployment-correct, engaged) |
| cat9 + native-E5 launchers (locked + speed) | RUNNABLE-NOW |
| Fork-margin dump + CORRECTED classify (`FR13_FORK_MARGIN_DUMP` + `fr13_fork_margin_classify.py`) | dump RUNNABLE; **classify reducer needs the §2-of-APPLE_TO_APPLE fix** (flip-position-row join + self_logits row) before scale use |
| Per-layer ladder on sampled flips (`FR10_LAYER_HIDDEN_CAPTURE` + node5 reducer) | RUNNABLE-NOW (targeted, ~8–12 captures) |
| **pair-dump → oracle `--src` re-tokenizer + round-trip validator** (`scripts/fr13_swe_stream_to_oracle_src.py`) | **NEW HARNESS** (~80 lines; the ONE load-bearing new piece; class #12 round-trip assert is its core) |
| Token-class clustering reducer (b) over the rescore + served text | NEW HARNESS (~60 lines, pure CPU reduce) |
| Wilson CI helper | trivial (inline) |

The campaign is feasible with one ~80-line new reducer + a known fix to the classify reducer;
everything GPU-side is existing scripts.

---

## 6. HONEST GPU COST + the 2-window schedule

- **Window 1 (serve + capture, ~50–60 min):** native-E5 SWE run (1 task ×2 repeats, ~1565 s) +
  cat9 SWE run (same), serialized; pair-dump on both. Optionally a 2nd cat9 boot with
  `--arm FR13_FORK_MARGIN_DUMP` for (c) if the dump must be at full scale (or reuse the small-sample
  dump for the R/C ratio and only confirm the RATE at scale — cheaper).
- **Window 2 (rescore, ~25 min):** load model once, rescore native served stream (~8–12 min) +
  cat9 served stream (~8–12 min), `--reps 1` after a determinism re-confirm. The rescore is the
  bottleneck; the SWE generation is the bigger wall but is the existing campaign cost.
- Both windows are GPU-SERIALIZED behind the K1 mechanism proof (currently holding the GPU). Run
  ONLY when free: pre-boot `recover_host_memory()`, MemAvailable ≥ 100 GiB, `docker ps` empty
  (the standard FR13 hygiene). Teardown + recover between arms (memory collection bind).
- Skipping cat9+K1 at scale (default) keeps it to 2 served arms + 2 rescore arms. Add the K1
  arm ONLY if the user wants the 18→12 question re-asked at scale (then 3 served + 3 rescore).

Links: `FR13_BUG_CLASS_PLAYBOOK` (#9/#12/#8), `FR13_DIFFUSION_DEEP_DIVE` (diffuse per-layer
account), `FR13_APPLE_TO_APPLE_FORK` (R/C/W, the corrected reducer), `FR13_B1_SWE_GOLD_BIND`
(the gold gate this scales), `feedback_fr13_lossless_compare_target` (US vs native-E5 vs
no-spec recurrent oracle), `feedback_no_cross_boot_byte_gate` (the ±-flip wobble this beats with
n), `reference_scalar_metric_per_token_blindspot` (per-token clear-margin probe is binding).
