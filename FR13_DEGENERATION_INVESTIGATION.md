# FR13 — 13033 cache-ON garbled-degeneration: lossiness or temp-0.6 flake?

**Date:** 2026-07-01
**Scope:** synthesis of CHAR / HISTORY / CODE / RESEARCH sub-agents + independent verification.
**Constraint honored:** READ-ONLY. No server boot, no inference, no container ops, no GPU touch.

---

## VERDICT

**INCONCLUSIVE, leaning LOSSINESS (cache-ON-specific degeneration), NOT a temp-0.6 sampling flake.**
Confidence: **medium (~60% lossiness / ~25% flake / ~15% the newly-shipped .cpu()-drop or FIX-A LRU cap
introduced a fresh regression).**

The single garbled trace by itself does **not** meet the project's *byte-garble = lossy* criterion, but the
banked history's ON-vs-OFF asymmetry (0/80 OFF vs 4/59 ON) and the literature's symptom-match to
recurrent-state / spec-decode correctness bugs both point away from sampling. This is not yet a proof; the
**LIVE run's ON-vs-OFF re-test of 13033 is the decisive experiment** (see below).

---

## 1. Byte-garble vs coherent-degenerate (the project's own lossy criterion)

**VERIFIED.** The seq-34 initial served stream
(`output/fr13_tree_cache_matrix/run_20260701T063919Z/m_e5_ON/proxy_pair_dumps/pair_01782889020934756976_000034_initial.json`):

- reasoning: `Thestandling thetask\n`
- message: `\n\nI need tounderderstandtheprourposeofthisprompt.Itseemstobeaskingall-to-actionforanAGENTS.mdfilethatthe/workspacedirectory.Letmecheckwhat'sinthe/workspaceandfindoutmoreaboutthetask.`

Independent byte-check: message = 182 chars / **182 bytes, 100% ASCII, zero non-UTF-8, zero broken tokens.**
The pathology is **intra-word corruption**: dropped spaces (spaceless run) + **doubled subwords**
(`understand`→`underderstand`, `purpose`→`prourpose`, `call-to-action`→`all-to-action`).

**By FR13_PAPER_RESULTS_AND_VERDICTS.md:144-145** ("fully coherent UTF-8 (0 garbled bytes) … garbled-tokens =
lossy; coherent-but-off-task = not"), **this single trace does NOT satisfy the byte-garble=lossy criterion.**
CHAR is correct on the letter of the criterion.

**BUT — the criterion is under-specified for this symptom.** Spaceless + doubled-subword is *token-stream
integrity* corruption, not "coherent-but-off-task." It is qualitatively different from the char-8 flake
(which is fully-coherent English + a malformed tool-call JSON). Treating "all bytes are valid ASCII" as
"not lossy" would wrongly clear a genuine decode-path corruption whose bytes happen to be ASCII. So the
byte-criterion is **necessary but not sufficient** to call this a flake.

**Derail point (verified):** seq-34 request = 2 items, both `role=user`, **zero assistant turns** → this is
turn-1 / the very first initial. Degeneration does NOT require prior cache-hit turns. `served cached_tokens=0`
(vLLM prefix-cache miss on the opening request), but EXACT_SEED SSM-state restore/capture is active from
token 0 (see §3). Repetition (`"function":"function_promote"` ×~60 to the 500 cap) is on continuation seq-36,
not seq-34.

---

## 2. Cache-correlated (history) or cache-independent (like char-8)?

**HISTORY verified as internally consistent; this is a DIFFERENT mode from char-8.**

Two cleanly separable modes in the banked corpora (bigdenom + tree_cache_matrix):

| Mode | cache-ON | cache-OFF | Interpretation |
|---|---|---|---|
| **byte-garble / spaceless-run** | 4/59 | **0/80** | cache-ON-specific → lossiness signal |
| **char-8 "Unterminated string" 400** | 23/59 | 5/80 | **fires OFF → cache-INDEPENDENT** → known flake |

- char-8 fires cache-OFF (`cat55221_b1`, `validate_clean`) → confirms the prior host-verified finding that
  char-8 is a coherent tool-call-arg JSON re-parse flake, **cache-independent** (FR13_APC_SESSION_FINDINGS_20260629.md;
  inference_proxy.py:2013-2017 retryable via LUMO_PROXY_RETRY_UPSTREAM_400). **13033 spaceless is NOT char-8.**
- The spaceless/byte-garble mode is **0/80 on cache-OFF** and only appears on prefix-caching=True arms.
- **Confound refuted:** OFF arms ran the identical hard task set (12907×27, 13398×15, 13033×16, 13236×15) and
  produced **zero** garble. So ON-vs-OFF is a cache effect, not a task-difficulty artifact.

**Caveat on the tally:** the 3 older bigdenom garble traces (cat9_apc*) predate EXACT_SEED — they are the
*old* lossy prefix-cache path. Only the 13033 trace is on the *proven* EXACT_SEED=1 config. So the "4/59"
mixes two different cache implementations; the EXACT_SEED-specific N is effectively **1** so far. This is why
the verdict is inconclusive, not decided.

---

## 3. Most plausible CODE mechanism — and the two recent suspects

**Config of the garbled run** (`run_20260701T063919Z/m_e5_ON`, boot log verified):
`EXACT_SEED=1`, `enable_prefix_caching=True`, MTP spec `num_spec=5` (tree `[(0,),(0,0),…]`), `TREE_ATTN`,
`mamba_ssm_cache_dtype=float32`, `block_size=1024`, `cudagraph_mode=FULL_AND_PIECEWISE`, B=1, temp 0.6.

### TIMING (the load-bearing finding)

The garbled run booted **06:39:34Z**. Two changes to the hot ES checkpoint path landed **just before it**:

- **`1e1df386` FIX A LRU cap** — committed **06:04:39Z** — bounds `BlockPool._fr13_es_ckpt` to
  `FR13_ES_CKPT_CAP=64` via `OrderedDict.popitem(last=False)` at the write site.
- **`faecc88d` .cpu()-drop** — committed **06:16:34Z** — `_fr13_es_d[layer] = _fr13_es_ck_state.cpu()` →
  `_fr13_es_ck_state` (keep the checkpoint GPU-resident on GB10 unified memory).

**This is the FIRST run to carry both changes.** The garble appeared on that first run. That temporal
coincidence is the strongest reason not to dismiss the .cpu()-drop / FIX-A as innocent.

### .cpu()-drop analysis (suspect: PLAUSIBLE, revertible)

Traced the capture and restore:

- Capture (line ~6691): `_fr13_es_ck_state = _fr13_es_final_s[0].detach().to(float32).clone()` → the stored
  value is **already an independent clone**, so the drop does NOT create a capture-side view/alias of live
  kernel state. Good.
- Restore (line ~5942): `initial_state[r2] = stored.to(device=initial_state.device, dtype=initial_state.dtype)`.
  `initial_state` is a per-forward `ssm_state` slice; `[r2] = …` is an indexed **copy-in**, source read-only.

**Net:** on this exact path the .cpu()-drop is *value-preserving* — both variants copy the same float32
numbers into `initial_state`. I did **not** find a definite aliasing/mutation bug.

**HOWEVER, two real reductions in safety margin the drop introduces, worth reverting to test:**
1. With `mamba_ssm_cache_dtype=float32`, `stored.to(same_device, float32)` returns **the stored tensor itself**
   (PyTorch `.to()` is a no-op → returns self). The `.cpu()` variant *always* forced a fresh temporary
   (host→device copy). Any future/adjacent code that reads the `.to()` result expecting a private copy now
   shares storage with the persistent checkpoint. The current single consumer is a copy-in, so it's safe
   *today*, but the defensive copy that `.cpu()` guaranteed is gone.
2. GPU-resident checkpoints now live in the same 117 GiB unified pool as the live KV/SSM cache and are subject
   to CUDA-graph capture/replay timing; a device→device `.clone()` at capture is async (no stream sync),
   whereas `.cpu()` forced a sync. If capture races replay under FULL_AND_PIECEWISE graphs, a GPU-resident
   checkpoint could capture a **partially-written** state that the host copy would have serialized. This is
   speculative but is exactly the class of bug that produces "restored a slightly-wrong recurrent seed →
   blended hidden state → merged/doubled tokens + inflated P(EOS) early stop."

### FIX-A LRU cap analysis (suspect: LOW, but not zero)

FIX A evicts the **oldest** block-hash at cap=64. The commit claims "cap ≥ one task's live blocks so an
in-task restore never misses." If a single task's distinct prefill block-hashes exceed 64 (a 131k-ctx agent
loop with many turns can), a **still-live** checkpoint could be evicted, and the restore is None-safe (falls
back to no-seed) → *not* a corruption, but a silent loss of the seed → recompute-from-scratch → could shift
decode. This would degrade *quality* subtly, not produce byte-garble; lower suspicion than the .cpu()-drop.
Worth confirming cap=64 ≥ 13033's live-block count.

### Baseline mechanism (independent of the two recent changes)

Even without the recent commits, the literature-matched mechanism is **recurrent/SSM-state mis-restore on a
prefix-cache/spec-decode boundary** (the classic "reused one block too many" GDN/Mamba APC bug, ×MTP
accept-verify). The recent .cpu()-drop and LRU cap are **new amplifiers layered on top of that surface**, not
the sole hypothesis.

---

## 4. What the literature says

RESEARCH is decisive on the split and I concur with its reading:

- **Sampling degeneration** (Qwen3 "don't use greedy decoding") is documented as **coherent** endless
  repetition at LOW entropy (whole-word/phrase loops, `!!!!`), NOT intra-word corruption. temp 0.6 / top_p 0.95
  / top_k 20 is Qwen's *recommended anti-degeneration* thinking preset — arguing **against** sampling as cause.
  Sampling does not delete mid-token spaces, splice subword fragments (`prourpose`), or force early EOS.
- **Spaceless + doubled-subword + early-stop** matches **spec-decode / recurrent-state correctness** reports:
  vLLM #40831/#40880 (MTP×KV-quant×CUDA-graph → degenerate loops on Qwen3-Next hybrid, not closed on the MTP
  path), #39809 (Mamba prefix-cache + MTP: Triton kernel doesn't roll back state on spec-reject → corruption),
  the Yifei-Hu "reused one block too many" GDN/Mamba APC-hit bug, and mlx-lm #1292 (1-token completions on
  MTP when system prompt reused + user differs → "correctness bug in spec decoding, not sampling").
- The universal discriminator in every cited report: **disable spec and/or prefix-cache with sampling
  unchanged** — if garble/early-stop vanishes, it's the lossy path; if it persists at temp 0.6, only then
  suspect sampling.

---

## 5. Is the LIVE run the decisive experiment? — YES

**The LIVE run IS the decisive ON-vs-OFF re-test of 13033.**

- Live process (`ps`, read-only): `run_20260701T072605Z`, driver `fr13_tree_cache_matrix.sh`, subset
  `subset_b4_sixteen.json` (16 tasks; **13033 is task #2**). Currently executing **astropy-13033 (retry)** on
  `m_e5_ON` (a live dcgm sampler is on `astropy__astropy-13033/dcgm_samples_retry.jsonl`).
- The matrix runs 6 arms: `e5/cat6/cat8 × {OFF: EXACT_SEED=0, ON: EXACT_SEED=1+APC}`. Env flags verified:
  `m_cat8_ON` = `FR13_APC_EXACT_SEED=1` (+SNAP_FIX/CONV_SNAPSHOT); `m_cat8_OFF` = `EXACT_SEED=0`. Each arm
  re-runs 13033.
- **Because ON and OFF run the identical task/prompt/sampling at temp 0.6, they hold sampling fixed and vary
  only the cache path — exactly the discriminator the literature demands.**

Read-out when it lands:
- **13033 garbles/early-stops on cache-ON but is clean on cache-OFF** (repeatably, ≥2/N) → **LOSSINESS**
  confirmed; the byte-criterion needs amending to include intra-word/token-stream corruption.
- **13033 clean on both, or garbles on OFF too, or resolves on ON** → **FLAKE / sampling noise** (consistent
  with the §4 EXACT_SEED resolve-flip already banked: e5_ON RESOLVED while e5_OFF failed).

**Note:** the live run carries the SAME .cpu()-drop + FIX-A code as the garbled run. So a clean live ON result
would also (partially) exonerate those two commits; a dirty live ON result does NOT by itself distinguish
"baseline APC lossiness" from "the new .cpu()-drop regression" — for that, see the cheap check.

---

## 6. The ONE cheap check that settles it

**Grep the live run's per-turn served dumps for the spaceless/doubled-subword signature, gated on cache
state — no GPU, no new run:**

```
# for each arm dir under run_20260701T072605Z/m_{e5,cat6,cat8}_{ON,OFF}:
#   scan proxy_pair_dumps/*_initial.json output[].content[].text for
#   (a) long alpha runs with no spaces  AND  (b) doubled-subword bigrams
#   then bucket the hit-count by the arm's FR13_APC_EXACT_SEED flag.
# ON-only hits (0 on every OFF arm) across 13033 replicas  => LOSSINESS.
# hits on any OFF arm, or 0 on all ON arms                  => FLAKE.
```

**If a stronger causal test is wanted (needs a run, so it is BLOCKED right now — do NOT do it on the live
GPU): revert `faecc88d` (restore `.cpu()`) and, separately, raise `FR13_ES_CKPT_CAP` well above 13033's
live-block count, then re-run 13033 cache-ON.** If garble vanishes with `.cpu()` restored, the GPU-resident
checkpoint is the carrier and **`faecc88d` should be reverted**. Both are one-line/one-env reverts.

**`.cpu()`-drop suspect statement:** the .cpu()-drop (`faecc88d`) is a **plausible, cleanly-revertible
suspect**. I did not find a definite aliasing bug on the current single consumer, but it (i) removed the
defensive host-copy that guaranteed checkpoint isolation, (ii) makes `.to()` a no-op that returns the shared
stored tensor under float32, and (iii) trades a capture-time stream sync for an async device→device clone
under FULL_AND_PIECEWISE CUDA graphs — and it shipped in the exact run that first garbled. Revert it to test.

---

## Files

- Garbled trace: `output/fr13_tree_cache_matrix/run_20260701T063919Z/m_e5_ON/proxy_pair_dumps/pair_01782889020934756976_000034_initial.json` (seq 34; +35/36/37)
- Task artifacts: `.../m_e5_ON/swe_out/verified/per_task/astropy__astropy-13033/{codex_trace*.jsonl,runner_metadata.json}` (empty_patch)
- Boot config: `.../m_e5_ON/boot_log_snapshot.txt` (EXACT_SEED=1, APC=True, MTP-5, TREE_ATTN)
- LIVE decisive run: `output/fr13_tree_cache_matrix/run_20260701T072605Z/` (subset_b4_sixteen, 13033=task#2)
- Suspect commits: `faecc88d` (.cpu()-drop, 06:16Z), `1e1df386` (FIX-A LRU cap, 06:04Z) — both < run boot 06:39Z
- Code: `scripts/fr10_phase4_patch_vllm_tree_gdn.py` capture ~6691, restore ~5942, kernel call ~5983
- Criterion anchor: `FR13_PAPER_RESULTS_AND_VERDICTS.md:144-145`
