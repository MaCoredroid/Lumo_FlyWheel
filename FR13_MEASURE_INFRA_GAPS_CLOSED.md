# FR13 — Canonical measure infra: GAP-1 + GAP-2 CLOSED

CPU extend (no GPU boot in this pass). EXTENDS the one canonical module
`scripts/fr13_measure.py` + `scripts/fr13_measure_orchestrate.sh` (never a fork). The
GPU re-validation (Phase 2) boots native E5 + cat9 and runs the new reducers; this pass
CPU-validates them on the banked + synthetic data. vLLM citations read fresh from the
pinned image (`scripts/vllm_src.sh`, sha `3dbe092e…`, never `/tmp`). int-view never atol.

These two gaps were flagged by the GPU validation of the just-built infra (`2b4313e3`):
the `temp06-drift` TV was VACUOUS (`n_positions_scored=0`) and the free-running accept
FORKS cross-boot (not apples-to-apple). Both are now closed by EXTENDING the module.

---

## GAP-1 — `temp06-drift` TV was VACUOUS (id/string key mismatch)

### The bug (root-caused, code + artifact verified)
`capture-q` emitted `q` keyed by the **decoded token STRING** (vLLM's `/v1/completions`
`top_logprobs` lose the token id in JSON serialization), while the recurrent oracle `p` is
keyed by **token ID** (`oracle_topk_ids`). The `temp06-drift` reduce compared `set(q_str)`
against `set(p_id)` → **zero overlap → `n_positions_scored=0`, every position
`align_status=id_string_mismatch`** (`output/fr13_measure/native_e5_temp06_drift.json`).

The `0.738 / p95 0.9998` in `native_e5_temp06_drift_idkeyed.json` was an OUT-OF-BAND re-key
(its own `note` already says it: q top-20 STRINGS vs p top-20 IDs partial overlap at the
SAMPLED-stream flip positions) — a key-misalignment artifact scored at only ~158
positions, NOT a lossless verdict.

A second, separate defect: the recurrent oracle `p` only retained the full top-K on **flip
positions** in the reduced `per_prompt[].positions[]` (to save space); the non-flip
positions had no top-K — so even with a correct re-key the TV could only score flips, not
the full stream.

### The fix (two seams)

1. **`capture-q` emits q ID-KEYED at capture time** (`top_logprobs_ids` per position),
   re-keyed via the served model's own tokenizer:
   - `_build_dec2id(tokenizer_path)` builds a `decoded_string -> token_id` reverse map ONCE
     over the full vocab (`tok.batch_decode([[i]] for i in vocab)`). **MEASURED on this
     model (247130-entry vocab): the map is COLLISION-FREE over the captured top-K support —
     0 decode collisions / 1506 distinct keys.** Any decoded string produced by >1 id is
     DROPPED from the unambiguous map (routed to the unmapped bucket), so a many-to-one
     decode never mis-assigns mass.
   - `rekey_q_to_ids(...)` re-keys one position's `top_logprobs`, honoring the **served-token
     ANCHOR**: the served token's exact id is known from `return_token_ids`, so the served
     candidate is forced to its true id even in the (unobserved) collision case. Strings with
     no unambiguous id go to an `unmapped::<s>` bucket so their probability mass is NEVER
     dropped from the TV. **Full banked stream: 10230 mapped / 6 unmapped (0.059%); the only
     distinct unmapped string is `''` (an empty-decode byte fragment).**
   - The artifact records `q_id_keyed`, `rekey_tokenizer`, `rekey_mapped_total`,
     `rekey_unmapped_total`, `rekey_unmapped_frac`.

2. **The recurrent oracle `p` is captured over the FULL served stream** — new
   `--full-topk-all-positions` flag on `fr13_recurrent_decode_oracle.py rescore` retains the
   top-K on EVERY position (not just flips); the artifact also records `sink_dir`. The
   reduce reads the full top-K either from `positions[]` (when that flag was set) or, for
   OLD oracle artifacts, from the per-prompt SINK JSONL files (which the forced-decode LP
   already wrote for all 128 steps).

3. **`temp06-drift` aligns by TOKEN ID** (`fr13.measure.temp06_drift.v2`): builds the
   id-keyed q (from `top_logprobs_ids`, or on-the-fly re-keys an OLD string-keyed artifact)
   vs the id-keyed full-stream p, and scores a REAL per-position
   `TV(softmax(q/0.6), softmax(p/0.6))` + `KL(p||q)` + over-floor count + max-TV over the
   **union** of id supports (mass on the symmetric difference is counted, not dropped). The
   per-position `q_tail_mass_T1` (`1 - sum exp(top-K logprob)`) is the truncation error bar.

### CPU validation (banked `native_e5` q + recurrent p)

| metric | OLD (vacuous) | NEW (id-aligned) |
|---|---|---|
| `n_positions_scored` | **0** | **512** (4 prompts × 128, FULL stream) |
| `mean_tv_q_p_at_temp` | null | 0.3089 |
| positions with `TV < 0.02` (q≈p) | — | **175** |
| pos 0/1/2/3 TV | — | 0.0041 / 0.0013 / 0.0003 / ~0 |

The low-TV early positions (`q_served ≈ p_served ≈ 0.98–1.0`, overlap 15–19/20) PROVE the
id alignment is real — impossible under a string/id mismatch. The non-trivial mean (0.309)
is the temp-0.6 **SAMPLED** q vs the **GREEDY** recurrent p (sampling spread), which the
record's `interpretation_note` documents: a high raw mean TV alone is NOT a lossless miss;
the LOSSLESS gate is the per-position **over-floor count vs the native temp-0.6 self-floor**
+ the multi-seed **bag-TV** (`cmd_bag_tv`), paired with the per-position vector
(`reference_scalar_metric_per_token_blindspot`).

---

## GAP-2 — accept/event FORKS cross-boot (not apples-to-apple) → paired teacher-forced accept

### The bug
Free-running accept/event is TRAJECTORY-BOUND (bug-class #12). On GB10 the same-prefill
greedy stream FORKS cross-boot at the token-6 near-tie (the autotune realization floor,
`feedback_no_cross_boot_byte_gate`; the no-spec oracle ranks the gold argmax by ~11 nats =
the gold trajectory is correct, our boot just landed the other side). So a cross-boot
cat9-vs-native accept compares two DIFFERENT served contents — not apples-to-apple.

### The fix — `paired-accept` subcommand (fork-immune)
Pin ONE reference trajectory (the **no-spec RECURRENT oracle GREEDY stream** = the
deployment-correct ground truth) and score BOTH arms on that SAME fixed token sequence.
`_load_reference_streams` binds the run to a `reference_fingerprint`; both arms reduce
against the identical content.

Two modes, the distinction documented on every record:
- **`mode=structural` (CPU, validatable now):** along the fixed reference, the GREEDY verify
  accepts reference token `i` iff it equals the arm's VERIFIER argmax (from the id-keyed q).
  `_structural_accept_on_reference` segments the reference into depth-`D` spec events
  (native MTP-N → D=N; tree → D=len(TREE) spine) and sums the per-event accepted run lengths
  (each run = consecutive reference tokens the arm's verifier argmax-confirms, then +1 bonus
  token, then advance). Uses ONLY the captured per-position verifier dist q — no GPU,
  fork-immune (BOTH arms anchored to the SAME reference). The verifier argmax MUST come from
  the id-keyed q; `paired-accept` FAILS LOUD (class-9) if an arm-q is not id-keyed
  (`--allow-served-fallback` to score the served stream as a labelled proxy instead).
- **`mode=force` (GPU, deferred — orchestrate hook):** boot the arm, force the reference
  stream as the served sequence so the spec-verify commits the reference tokens, and read
  `d(num_accepted)/d(num_drafts)` live. This is the ground-truth paired-accept; structural
  is its cheap fork-immune proxy. The tree's BRANCH superset edge (a sibling holding the
  reference token after a spine miss) is only realized in `mode=force`; the structural
  number is the SPINE (linear-chain) proxy and is labelled
  `is_tree_spine_proxy` / `verifier_argmax_source`.

**Distinction (printed on every record, `deployment_vs_paired`):** paired-accept = the
apples-to-apple STRUCTURAL edge on a COMMON reference (drives the break-even);
deployment-accept (`cmd_speed.accept_per_event`) = free-running, trajectory-bound floor.
**NEVER cross-compare the two.**

### CPU validation
- **Fail-loud verified:** a non-id-keyed arm-q raises class-9 (prevents the meaningless
  served-fallback that conflates the served stream with the verify decision).
- **Synthetic apples-to-apple** (2 prompts, len-40 reference, both arms forced to the SAME
  0.85 verifier accuracy): native depth-5 → structural accept/event **3.05**; cat9 depth-9 →
  **4.06** on the SAME `reference_fingerprint` — the deeper verification window accepts more
  consecutive reference tokens per event = exactly the fork-immune structural edge.
- **Reducer unit checks:** all-match d5 → 4.25, d9 → 9.0 (deeper accepts more/event, fewer
  events); mismatch-every-3rd → run lengths truncate at each mismatch;
  `_arm_verifier_argmax_ids` picks the max-logprob id per position (the verify decision, not
  the served token).

---

## Files changed (EXTEND, not fork)

- `scripts/fr13_measure.py`
  - `_build_dec2id` / `rekey_q_to_ids` — host-tokenizer decoded-string→id re-key (GAP-1).
  - `cmd_capture_q` — emits `top_logprobs_ids` (id-keyed q) + re-key provenance; new
    `--tokenizer` / `--no-rekey`.
  - `cmd_temp06_drift` (`v2`) — id-aligned full-stream TV/KL; `_p_topk_by_pos` reads
    full-stream p from `positions[]` or the sink JSONL; new `--tokenizer` / `--no-rekey`.
  - `cmd_paired_accept` (+ `_load_reference_streams`, `_arm_verifier_argmax_ids`,
    `_structural_accept_on_reference`) — GAP-2 fork-immune paired accept; new `paired-accept`
    subcommand with `--reference` / `--arm-q` / `--default-depth` / `--allow-served-fallback`.
- `scripts/fr13_recurrent_decode_oracle.py`
  - `--full-topk-all-positions` (retain oracle top-K on EVERY position = full-stream p);
    records `sink_dir`.
- `scripts/fr13_measure_orchestrate.sh`
  - `drift ARM` — GAP-1: boots the recurrent oracle (`--full-topk-all-positions`) on the
    arm's capture-q served stream, then runs the id-aligned `temp06-drift`.
  - `paired REF_ARM ARM [ARM…]` — GAP-2: paired teacher-forced accept (CPU structural) of
    the arms on the common reference oracle stream.

### How to invoke
```
# GAP-1 (per arm; reuses <arm>_q_temp06_on.json from `native`/`tree`):
scripts/fr13_measure_orchestrate.sh native e5      # -> native_e5_q_temp06_on.json
scripts/fr13_measure_orchestrate.sh drift native_e5  # oracle p (full top-K) + id-aligned TV
# GAP-2 (CPU; reference = a recurrent-oracle greedy stream):
scripts/fr13_measure_orchestrate.sh paired native_e5 native_e5 cat9
```
Or directly:
```
python3 scripts/fr13_measure.py capture-q --arm native_e5 --tokenizer /models/qwen3.6-27b-fp8 ...
python3 scripts/fr13_recurrent_decode_oracle.py rescore --arm native_e5 \
  --src <capture-q.json> --full-topk-all-positions --out <p.json>
python3 scripts/fr13_measure.py temp06-drift --q <capture-q.json> --p <p.json> \
  --per-position-floor 0.05 --tokenizer /models/qwen3.6-27b-fp8 --out <drift.json>
python3 scripts/fr13_measure.py paired-accept --reference <p.json> \
  --arm-q <native_q.json> <cat9_q.json> --out <paired.json>
```

### CPU-validated bits (this pass)
- GAP-1 reduce: 0 → **512** positions scored on the banked native artifacts; 175 positions
  at `TV<0.02` (q≈p) prove the id alignment is real; re-key is collision-free (0 / 1506) and
  loses 0.059% mass to a labelled `unmapped` bucket.
- GAP-2: fail-loud on non-id-keyed q; synthetic apples-to-apple (same accuracy) gives
  d9 cat9 4.06 > d5 native 3.05 on one `reference_fingerprint`; reducer unit checks pass.
- Not yet on GPU: the real `temp06-drift` TV with a FRESH id-keyed capture-q + the
  `--full-topk-all-positions` oracle p, and the `mode=force` ground-truth paired-accept
  (Phase 2). The structural paired-accept is the documented fork-immune proxy until then.
