# FR13 — Canonical SPEED / lossless / temp-0.6-drift MEASUREMENT infra

CPU build (no GPU boot here). The GPU validation (Phase 2 of the workflow) boots native E5 + cat9
and runs this infra, reconciling against the banked numbers. All vLLM citations read fresh from the
pinned image (`scripts/vllm_src.sh`, sha `3dbe092e…`, never `/tmp`). int-view never atol.

Code delivered:
- `scripts/fr13_measure.py` — the ONE canonical entry point (subcommands `speed`, `capture-q`,
  `temp06-drift`, `bag-tv`, `diag-residue`, `reconcile`).
- `scripts/fr13_measure_orchestrate.sh` — serialized GPU boot orchestration reusing the validated
  launchers + the prelaunch host-memory protocol.

---

## 1. THE REGIME BUG — DEFINITIVE diagnosis (code + artifact verified)

The workflow's proximate hypothesis was "fr10_quick sends a RAW STRING → 1.70; gold-gate sends
TOKENIZED ids → 3.161". **That framing is WRONG in the mechanism but right in the symptom.** Verified:

**Both probes send a RAW STRING to `/v1/completions`.** `fr10_quick_decode_tps_probe.py:166-179`
sends `"prompt": prompt` (raw string); `fr13_gold_margin_probe.py:88` (the `capture` subcommand that
produced the banked stream) ALSO sends `"prompt": prompt` (raw string). Only the `teacher_force`
subcommand (`:411`, a max_tokens=1 re-prefill, NOT the accept-producing run) sends tokenized ids.

**The server tokenizes the identical 681-token context in both runs (offline-confirmed).** For
prompt[0], `tok(prompt[0], add_special_tokens=True) == tok(..., add_special_tokens=False)` (681 ids,
Qwen has no BOS so the flag is a no-op) `== phase0 record's prompt_token_ids` byte-for-byte, decoding
to `## Codex CLI invocation prompt\n\n"`. `/v1/completions` `add_special_tokens` defaults `True`
(`completion/protocol.py:88`) and there is no chat-template path in the completion serving code. So
**the prefill bytes are identical** — raw-vs-tokenized is not the discriminator.

**The served greedy streams FORK across the two runs.** Comparing `native_e5_tps.json` (phase0, accept
1.70) vs `gold_margin/native_capture.json` (accept ~3.16) for prompt[0]: both share the first 6 served
ids `[271, 248068, 271, 248069, 271, 40]` (= `</think>\n\nI`) then DIVERGE at served index 6:
- phase0 → `1144, 310, 3418, 279 …` = "I need to understand the problem and applied the fix … None" —
  a **degenerate `<think></think>…None` loop** that accepts at **1.70**.
- gold → `3172, 1151, 539, 23218 …` = "I'll start by exploring the workspace …" — a **coherent**
  stream that accepts at **3.16**.

Each stream is **within-boot deterministic** (all 4 phase0 samples identical; gold `within_boot_det =
[T,T,T,T]`). So this is the **same-prefill cross-run greedy trajectory fork** — the GB10 autotune /
realization floor (`feedback_no_cross_boot_byte_gate`: "fresh B=1 forks at tokens 11-71 = autotune
floor, NOT a behavior change"), here forking as early as token 6 and amplified by the differing
harness regime.

**The amplifier — what differs between the 1.70 and 3.16 harness regimes:**
1. `fr13_speed_phase0.sh:60-63` fires a `/v1/chat/completions` warmup (`messages`, chat-templated,
   `<think>` tags) on the SAME booted server right before the probe load. The gold-gate does no chat
   warmup (the gold probe's own warmup is the raw load itself).
2. `samples_per_prompt=4` (phase0) vs `=1` (gold-gate) — 4 identical samples per prompt at
   `batch_size=1` (still sequential, co-residency ruled out) but a different decode schedule / KV
   layout that perturbs the per-step fp realization.
3. `MAX_NUM_SEQS=4` (phase0 launcher default) vs `=1` (the gold/current-gate bind).

Any of these perturbs the per-step bf16/fp8 realization enough to push the early-token greedy argmax
to the other side of a near-tie, forking the whole downstream trajectory.

**The binding lesson (bug-class #12, "non-like-for-like trajectories"):** `accept/event` is
**trajectory-dependent** and is NOT apple-to-apple across two free-running boots. A native run that
lands on a degenerate trajectory scores 1.70; the same native engine on a coherent trajectory scores
3.16. The fix is not "send tokenized ids" — it is **pin the served stream** (capture-once) and bind
every `accept/event` to its served-stream fingerprint, never comparing accept across two boots that
forked.

---

## 2. THE CANONICAL REGIME (baked in `fr13_measure.py`, no agent re-rolls it)

| knob | value | where |
|---|---|---|
| prompts | `output/fr13_acceptance_ladder/prompts_swe4.json` (4 SWE prompts) | `CANONICAL_PROMPTS` |
| seed | 1313 | `CANONICAL_SEED` |
| max_tokens | 128 | `CANONICAL_MAX_TOKENS` |
| request | `/v1/completions`, RAW prompt, `return_token_ids=True` | `_one_request` |
| greedy | temp 0 / top_p 1.0 | `speed` default |
| sampling | temp 0.6 / top_p 0.95 | `capture-q` default |
| flags | `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0` | launcher pins; orchestrator passes |
| decode mode | `vllm_xargs.fr10_decode_mode` = `naive_mtp` (native) / `tree_mtp` (tree) | `parse_arm` |
| warmup | **ONE raw self-warm** (NOT chat template) + `reset_prefix_cache` | `_self_warm` |
| B=1 | SEQUENTIAL (one prompt, one sample at a time) | `cmd_speed` |
| B=4 | client batch of 4 identical samples per prompt | `cmd_speed` |

The chat-template warmup is structurally absent (the diagnosed confound). The same module measures any
arm: `native_eN` (N=3..8) or a TREE caterpillar (`--tree '[(0,), (0,0), …]'`); a shape the booted
server did not build FAILS LOUD at the engagement assert (`tok/draft != len(TREE)`).

---

## 3. TRUTHFUL SPEED ACCOUNTING (the only allowed basis)

```
s/fwd        = d(vllm:request_decode_time_seconds_sum) / d(vllm:spec_decode_num_drafts_total)
accept/event = d(vllm:spec_decode_num_accepted_tokens_total) / d(spec_drafts)      [B-DEPENDENT]
committed/ev = accept/event + 1                                                     [the bonus token]
TPS          = committed/event / s/fwd                                              [DERIVED, not measured]
```

- `s/fwd` is decode-only, per spec-event, from RAW `/metrics` counter deltas — `~B-invariant` for the
  bandwidth-bound GB10 B=1 decode.
- `accept/event` is **B-DEPENDENT** (co-residency degrades it): B=1 and B=4 are DIFFERENT NUMBERS,
  each labelled with `batch_size` + a `served_stream_fingerprint` (trajectory-bound).
- **BANNED as the basis** (blocked + asserted in `assert_speed_basis`, raises): TPS, accept,
  wall-clock, per-request HTTP `req_elapsed`. `derived_tps` is emitted with `"DERIVED = … ; NOT
  measured."` and TPS can never be the `s/fwd` source.

**INSTRUMENT ON/OFF separation** (the instrumentation affects speed → SPEED and LOSSLESS are separate
boots/numbers):
- `mode=OFF` (CLEAN deployment): `FR10_METRICS=0`, NO logprobs/q-capture, NO flip/oracle, NO FR12/13
  diagnostics = the bytes the user ships. `speed` runs OFF and stamps `"instrument": "OFF"`. **SPEED
  is read ONLY from OFF.**
- `mode=ON` (lossless/drift instrumentation): the `capture-q` top-K logprobs + flip/temp-0.6 reduce.
  Adds real decode tax (extra top-K logprob + DtoH). **LOSSLESS/drift is read ONLY from ON.** An ON
  boot's `s/fwd` is used ONLY to quantify the tax.
- `assert_no_mode_mix()` raises if a `speed` number cites `instrument=ON` or a `lossless`/`drift`
  number cites `instrument=OFF`. Every emitted record carries `{"kind","instrument"}`.
- `diag-residue` computes `(on_s_fwd − off_s_fwd)/off_s_fwd` per arm (expect ≤2.5% per 46e89f22,
  MEASURED not assumed; `tax_within_expectation`).
- `OFF == deployment`: the OFF path uses the locked launcher defaults (no diagnostic flags armed), so
  the OFF speed run is byte-identical to the committed serving path.

---

## 4. temp-0.6 DISTRIBUTIONAL CAPTURE (first-class, the NEW q-capture)

The deployment-binding lossless gate is NOT the temp-0 argmax flip (necessary-not-sufficient,
class-#9). It is per-position `TV(softmax(q/0.6), softmax(p/0.6))`.

- **`capture-q` (NEW — nothing recorded this before).** Per served position over the WHOLE stream:
  the top-K (`--top-k 20`) `log_softmax` = the spec verify-forward dist `q` (extends the gold_margin
  capture from 4 fork positions to the full stream), PLUS `per_position_tail_mass = 1 − Σ
  exp(topk_logprobs)` as the truncation error bar. `q` is recorded at T=1; the reduce applies `/0.6`.
  Runs rep1/rep2 for within-boot determinism. ON mode.
- **`p` = the no-spec RECURRENT decode oracle** (`fr13_recurrent_decode_oracle.py rescore`,
  `FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN, single-token roll, NOT chunked) on the SAME served
  stream; asserts `RECURRENT_PATH_ENGAGED` (class-#9 fail-loud), already records per-position top-K.
- **`temp06-drift` (CPU reduce).** Per position: `TV(softmax(q/0.6), softmax(p/0.6))` + `KL(p‖q)` +
  the per-position over-floor vector (paired with the scalar, `reference_scalar_metric_per_token_blindspot`).
  **Temp recovery is valid** (unit-tested): `softmax(logprob/0.6)` over the captured support == 
  `softmax(logit/0.6)` because the per-position additive constant `C/0.6` cancels in the re-softmax.
  Each arm vs its OWN oracle (apples-to-apples).
  - *Known limitation, flagged in-band:* the gold-style `q` is keyed by token STRING; the oracle `p`
    by token ID. The reduce records `align_status` per position and the served-token probability gap
    as a lower bound when keys are not alignable; the GPU `capture-q` pass should emit `q` by token-id
    (the artifact already records `served_token_ids`) so the full cross-key TV is exact. This is a
    one-line extension at GPU-capture time, not a redesign — and it FAILS LOUD (records
    `align_status`) rather than silently producing a vacuous 0.
- **`bag-tv` (deployment cross-check).** N=6-8 native temp-0.6/top_p0.95 seeds → floor = p95 of
  C(N,2) native-vs-native bag-TV draws (upgrades the single-draw 0.1133, class-#12); cat9 same seeds →
  cat9-vs-native bag-TV PASS iff ≤ floor_p95. B=4 deployed shape for the verdict tier.

---

## 5. ENGAGEMENT + DETERMINISM asserts (BEFORE any number)

- **class-9 (silent fallback / vacuous):** `assert_engaged` — `tok/draft` over the window MUST ==
  `N` (native) / `len(TREE)` (tree) from raw `/metrics` (`num_draft_tokens/num_drafts`); else raise,
  record NOTHING. The oracle additionally asserts `RECURRENT_PATH_ENGAGED`. (`has_tree_parent_indices`
  / `tree_sample_accept` are asserted by the existing tree-engagement reducer in
  `fr10_quick_decode_tps_probe._assert_tree_engagement`, which the orchestrator can chain when the
  sampler-debug logs are armed.)
- **class-8 (determinism):** `capture-q` records rep1==rep2 within-boot.
- **Cross-boot byte gate is BANNED** (`feedback_no_cross_boot_byte_gate`) — never used; the trajectory
  fork above is exactly why.

---

## 6. HOW TO INVOKE (any arm, one regime)

```bash
# native MTP-5: OFF speed B=1+B=4 + ON capture-q + diag-residue (serialized boot)
scripts/fr13_measure_orchestrate.sh native e5

# cat9 (locked launcher): same battery
scripts/fr13_measure_orchestrate.sh tree cat9 \
  "[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,1), (0,0,1), (0,0,0,1), (0,0,0,0,1)]"

# any other caterpillar (forked launcher derives num_spec + tree from len(TREE)):
scripts/fr13_measure_orchestrate.sh tree cat10 "<10-node TREE>"

# reconcile all OFF speed records vs the banked historic numbers:
scripts/fr13_measure_orchestrate.sh reconcile

# temp-0.6 drift reduce (CPU, after q + recurrent-oracle p captured):
python3 scripts/fr13_measure.py temp06-drift \
  --q output/fr13_measure/native_e5_q_temp06_on.json \
  --p output/fr13_recurrent_oracle/native_e5_rescore.json \
  --out output/fr13_measure/native_e5_temp06_drift.json
```

Single-arm primitives (`fr13_measure.py speed|capture-q|temp06-drift|bag-tv|diag-residue|reconcile`)
all default to the canonical regime; only `--arm` (+ `--tree` / `--batch-size` / `--temperature`)
varies.

---

## 7. HISTORIC NUMBERS to reconcile (Phase-2 GPU)

| arm | banked s/fwd | banked accept/event | src |
|---|---:|---:|---|
| native E5 (MTP-5, FLASH) | 0.218160 | 3.161290 | FR13_B1_CURRENT_GATE_BIND |
| cat9 (tree_mtp, locked) | 0.2248 | 3.18 | FR13_B1_FIX3_GATE_BIND |

The Phase-0 `s/fwd` already MATCHED (native 0.2159, cat9 0.2241 vs banked 0.2182/0.2248 — within the
trajectory-length noise) **even on the forked trajectory** — confirming `s/fwd` is trajectory-robust
(it is per-event decode time, ~length-invariant). Only `accept/event` forked (native 1.70 vs 3.16),
exactly because it is trajectory-bound. The canonical regime + fingerprint binding makes the GPU
run reproduce 3.161/3.18 when it lands on the gold trajectory, and SURFACE (via the fingerprint +
the `reproduces_banked_accept` flag) when a boot forks — instead of silently reporting 1.70 as
"native accept".

---

## MEASURED vs INFERRED ledger
- **MEASURED (this build, banked artifacts + offline tokenizer):** both probes send raw strings
  (code-read); phase0 `prompt_token_ids == tok(prompt[0])` byte-for-byte (offline); served streams
  share 6 tokens then fork at index 6 (artifact diff); within-boot determinism on both
  (`[T,T,T,T]`); s/fwd matched but accept forked (1.70 vs 3.16). Temp-recovery softmax invariant
  (unit test). reconcile/bag-tv/temp06-drift reduce paths exercised on banked/synthetic data.
- **INFERRED / awaits GPU (Phase 2):** the live reproduction of 3.161/3.18 on this canonical regime;
  the B=4 accept degradation; the temp-0.6 TV(q,p) excess vs the native floor; the N=6-8 p95 bag-TV
  floor; the diag-residue tax magnitude (expected ≤2.5%).
