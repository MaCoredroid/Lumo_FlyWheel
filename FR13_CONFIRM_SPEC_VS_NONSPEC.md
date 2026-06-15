# FR13 — Confirm: is the big-denom a genuine SPEC-decode (served) vs NON-SPEC recurrent (oracle) comparison?

CPU read-only confirmation on BANKED data + source (no GPU boot). All vLLM citations read fresh
from the pinned image (`scripts/vllm_src.sh`, sha `3dbe092e…`). int-view equality reasoning, never atol.

**THE RESULT UNDER TEST** (`output/fr13_bigdenom_rescore/consolidated.json`):
- cat9   = 1181/8717 = 13.548%  Wilson95 [12.846%, 14.283%]
- native = 1224/8752 = 13.985%  Wilson95 [13.275%, 14.728%]
- CIs OVERLAP; cat9 slightly LOWER; `ci_separated_cat9_above_native=false`.
- Each arm scored vs its OWN no-spec RECURRENT decode oracle. clear-margin = served≠recurrent_argmax
  AND (served out-of-top20 OR deviation_nat>1.0).

Bug-class playbook rows in force (`FR13_BUG_CLASS_PLAYBOOK.md`):
- **#9 Silent fallback / vacuous instrument** — "a run passes while measuring nothing"; remedy =
  engagement asserts + flag-state headers + fail-loud on disengagement BEFORE trusting any number.
- **#10 Shared-source ≠ shared-SASS** — byte A/B, int-view equality NEVER atol.
- **#11 Batch-composition / BI-flag sensitivity** — native-vs-native control; pin BI on BOTH arms.
- **#12 Measurement traps** — raw counters only; capture-once pinned prompts; validated denominators.

---

## Per-link CONFIRMED / REFUTED table

| # | Link | Verdict | Basis |
|---|------|---------|-------|
| 1 | **SERVED = SPEC-DECODE, both arms** | **CONFIRMED** | CODE-READ |
| 2 | **ORACLE = NON-SPEC RECURRENT single-token decode, both arms** | **CONFIRMED** | CODE-READ |
| 3 | **BOTH ARMS IDENTICAL FRAMING** | **CONFIRMED** | CODE-READ + artifact |
| 4 | **NON-VACUOUS** (spec frozen, recurrent ran, denom = validated round-trip) | **CONFIRMED** | CODE-READ + artifact |

---

### LINK 1 — SERVED streams are the ACTUAL spec-decode serve output (CONFIRMED, CODE-READ)

The served streams are captured by the proxy pair-dump from a real spec serve, with engagement asserts
gating the draft shape per arm.

- **cat9 boots the forked-FA2 9-node tree spec serve.** `scripts/fr13_bigdenom_swe_serve.sh:90-97`
  routes `KIND=cat9` to `scripts/fr13_launch_locked.sh` → which sets `FR10_DECODE_MODE_DEFAULT=tree_mtp`
  and `exec fr13_launch_forked_fa2_tree_server.sh` (`fr13_launch_locked.sh:14,16,50-51`: comment
  `cat9 num_spec=9 TREE_ATTN | pipeline ON`, "9-node: 5-spine + top-2 leaf on depths 1-4").
- **native boots E5 MTP-5.** `fr13_bigdenom_swe_serve.sh:98-109` routes `KIND=native` to
  `scripts/fr10_launch_speed_server.sh` with `SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}`,
  `FR10_ENABLE_TREE_GDN=0`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`.
- **Spec ENGAGEMENT is asserted from RAW /metrics deltas (class #9), not inferred.**
  `fr13_bigdenom_swe_serve.sh:199-220`: after an identical warmup probe it reads
  `vllm:spec_decode_num_drafts_total` and `vllm:spec_decode_num_draft_tokens_total` deltas and asserts
  `drafts>0` AND `draft_tokens/drafts == EXPECT_RATIO` (9 for cat9, 5 for native, `abs(ratio-expect)<1e-9`).
  → cat9's 9-node tree-verify and native's 5-token linear spec are PROVEN engaged on each arm before any
  served traffic is dumped. Worker `/proc/<pid>/environ` needle (`:151-174`) additionally proves the flags
  reached the EngineCore (not just the container env), and `:176-179` requires "Graph capturing finished"
  (CUDA-captured serve, not eager fallback).
- **The served tokens being rescored came from THAT serve.** The proxy
  (`src/lumo_flywheel_serving/inference_proxy.py`) only FORWARDS to the booted vLLM container — it runs no
  inference. `_pair_dump_upstream("initial", payload, parsed)` (`:2066`) and the auto-continue path
  (`:2148`) dump the parsed UPSTREAM /v1/responses response (`:45-69`, schema `lumo.fr13.proxy_pair_dump.v1`),
  i.e. the served output of the real spec engine. The pair-dump is gated ON for both arms
  (`fr13_bigdenom_swe_serve.sh:226-251`: `LUMO_PROXY_PAIR_DUMP_DIR` set, pin asserted), temperature forced
  to 0.0 on both, and proxy is diagnostic-only ("served traffic is unchanged", `inference_proxy.py:40`).

INFERRED caveat (does not break the link): the rescore token ids are RE-TOKENIZED from the served TEXT,
not the engine's emitted ids (the /v1/responses SSE path emits no per-token ids;
`fr13_swe_stream_to_oracle_src.py:1-12`). This is sound because every kept turn passes a class-#12
byte-exact round-trip gate `tok.decode(ids)==served_text` (`fr13_swe_stream_to_oracle_src.py:107`); a
failing turn is DROPPED and counted, never mis-scored. Both arms have 0 dropped (see Link 4), so the
re-tokenized ids reconstruct exactly the served content stream the spec serve produced.

### LINK 2 — ORACLE is the ACTUAL non-spec single-token RECURRENT decode (CONFIRMED, CODE-READ)

The rescore (`scripts/fr13_recurrent_decode_oracle.py rescore`) loads the model ONCE in-process with NO
speculation and forces each served id one token at a time, so every scored distribution is produced by
the deployment recurrent decode path — NOT chunked re-prefill, NOT streamed logprobs, NOT a serial-torch
reference, NOT a backend NAME, NOT a fallback.

- **No speculation → recurrent decode path.** `_build_llm` (`fr13_recurrent_decode_oracle.py:166-188`)
  sets `FR12_NO_SPECULATIVE_CONFIG=1` and constructs `LLM(...)` with NO `speculative_config` kwarg,
  `max_num_seqs=1`, eager. With spec off and `query_len==1` per generated token, vLLM's
  `split_decodes_and_prefills(decode_threshold=1)` yields `num_decodes>0, num_prefills==0`.
- **The vLLM dispatch is exactly as claimed** (read fresh from the pinned image,
  `model_executor/layers/mamba/gdn_linear_attn.py`):
  - `_forward_core` **L806-818**: `if enable_packed_recurrent_decode and spec_sequence_masks is None and
    num_prefills==0 and num_decodes>0: return self._forward_core_decode_non_spec(...)` — the pure
    recurrent rank-1 decode fast path.
  - `_forward_core_decode_non_spec` **L1045-1097**: `causal_conv1d_update` (L1075) +
    `fused_recurrent_gated_delta_rule_packed_decode` (L1085), carrying `conv_state`/`ssm_state` forward in
    the KV cache (`self_kv_cache`, L1057-1065) — the genuine recurrent roll.
  - The CHUNKED path is reached ONLY when `num_prefills>0` (**L880-895**, `causal_conv1d_fn` +
    `chunk_gated_delta_rule` at L982-996); the per-position re-prefill instrument the oracle docstring
    warns against (`:6-27`) would have hit it. The single-token decode never does.
  - (Even with `enable_packed_recurrent_decode` off, `num_decodes>0` falls to the **L1007**
    `fused_sigmoid_gating_delta_rule_update` branch — still recurrent; only `num_prefills>0` is chunked.)
- **Recurrent engagement is PROVEN, fail-loud (class #9).**
  `_install_recurrent_path_counter` (`:197-225`) monkeypatches `_forward_core_decode_non_spec` to a
  call counter; the engine core runs IN-PROCESS (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `:173`) so the
  driver sees the worker's GDN calls. `cmd_rescore` raises SystemExit (`:463-466`) if
  `_RECUR_DECODE_CALLS["n"]==0`. Banked: cat9 = 830,496 calls, native = 834,048 calls
  (`rescore_cat9/native.json`, `RECURRENT_PATH_ENGAGED:true`) ≈ 1 recurrent call per layer per decoded
  token (96 GDN layers × ~8.7k positions ≈ ~830k). NOT a 0-call fallback.
- **Clean argmax is recorded IN-PROCESS, NOT streamed logprobs.** The per-request logits processor
  (`:103-134`) reads the CLEAN recurrent argmax + top-k from the logits row BEFORE forcing
  (`top=torch.topk(row,k)`, `log_softmax`), writes the record, THEN forces served[i] as the unique argmax
  so greedy commits it and the recurrent state advances. The consolidator independently asserts the
  rescore schema is `fr13.recurrent_decode_oracle.rescore.v1` (NOT a streamed-logprob HTTP instrument)
  (`fr13_bigdenom_rescore_consolidate.py:58-61`).

### LINK 3 — BOTH ARMS use IDENTICAL framing (CONFIRMED, CODE-READ + artifact)

The rescore is invoked twice with EVERYTHING pinned except `--arm`/`--src`:
- `fr13_bigdenom_phase3_rescore.sh:79-101` calls `fr13_recur_rescore_in_container.sh <arm> <src> <out>`
  for native then cat9 with the SAME env `SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88` and
  `--attn-backend FLASH_ATTN` (`fr13_recur_rescore_in_container.sh:33-38`, same pinned IMAGE
  `3dbe092e…`).
- Same SOURCE for served streams both arms: served-from-spec-serve → reduced by the SAME
  `fr13_swe_stream_to_oracle_src.py` (`phase3_rescore.sh:52-55`) → rescored by the SAME oracle. No arm is
  chunked while the other is recurrent; no per-arm oracle.
- Same clear-margin def + threshold/top-k both arms (the `_lp`/`cmd_rescore` logic is arm-agnostic;
  `clear = flip and ((not in_topk) or dev > threshold)`, `:381`).
- **Artifact cross-check** (banked, both arms): schema `fr13.recurrent_decode_oracle.rescore.v1`,
  seed 1313, top_k 20, threshold_nat 1.0, attn_backend FLASH_ATTN, `RECURRENT_PATH_ENGAGED:true`,
  `within_proc_det_all_prompts:true` — identical on cat9 and native.
- COMPARATOR-RULE note (depth axis): cat9 is depth-5 (5-spine) tree vs native E5 (5-token) — depth-matched
  for the speed axis. The lossless axis here is depth-AGNOSTIC: each arm is scored against ITS OWN
  recurrent oracle, so the 13.55% vs 13.99% comparison is apples-to-apples (per-token argmax-vs-own-clean-
  decode rate), consistent with the standing rule "US lossless iff its flip-vs-own-no-spec-oracle matches
  E5 within floor".

### LINK 4 — NON-VACUOUS (CONFIRMED, CODE-READ + artifact)

- **Spec config FROZEN during the oracle.** `spec_frozen_evidence.json` records
  `FR12_NO_SPECULATIVE_CONFIG_set:true`, `no_speculative_config_kwarg:true`,
  `in_process_engine_core:true`, `recurrent_path_counter_failloud:true`. This is GENERATED by grepping the
  oracle source (`phase3_rescore.sh:61-73`: checks `setdefault("FR12_NO_SPECULATIVE_CONFIG","1")` present
  and `speculative_config` ABSENT from the kwargs block) — so spec_decode counters CANNOT advance during
  the oracle (there is no spec engine to advance them). The recurrent-decode-path counter (>0) is the
  binding engagement gate, not a spec counter.
- **The recurrent path actually RAN** (not a 0-call fallback): 830,496 / 834,048 calls, both
  `RECURRENT_PATH_ENGAGED:true`; both rescore scripts would have raised on zero.
- **Denominator = validated round-trip tokens, dropped turns accounted (class #12).** The reducer drops any
  turn that fails `tok.decode(ids)==served_text` and counts it (`fr13_swe_stream_to_oracle_src.py:107,
  116-137`). Banked srcs: cat9 = 66/66 turns kept, 0 dropped, 8717 positions; native = 64/64 kept, 0
  dropped, 8752 positions. The consolidator ASSERTS `total_positions_rescored == n_positions_scored_src`
  per arm (`fr13_bigdenom_rescore_consolidate.py:71-78`) — so the Wilson CI is on the validated round-trip
  token count, not text length and not `usage.output_tokens`. Both arms satisfy it (8717==8717, 8752==8752).
- **Non-empty served-stream gates at every stage**: arm-level pair-dump non-vacuity (`swe_serve.sh:321-360`,
  fails if no served text), autoadvance pre-check (`fr13_bigdenom_autoadvance.sh:78-83`), phase-3 pre-flight
  (`phase3_rescore.sh:26-32`), and reducer refusal to emit empty src (`stream_to_oracle_src.py:148-153`).
  The consolidated `non_vacuity` block is all-true.

---

## VERDICT

**VALID.** The big-denominator result is a genuine spec-decode (served) vs non-spec recurrent (oracle)
lossless comparison, with IDENTICAL framing on both arms and no vacuity:

- SERVED is the real spec serve on both arms — cat9 = forked-FA2 9-node tree-verify (draft_tokens/drafts
  ratio asserted == 9), native = MTP-5 linear (ratio == 5) — captured via the proxy pair-dump from a real
  CUDA-captured spec serve (Link 1, CODE-READ).
- ORACLE is the ACTUAL deployment non-spec single-token RECURRENT GDN decode
  (`_forward_core_decode_non_spec` → `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`,
  vLLM `gdn_linear_attn.py` L806-818 / L1045-1097), spec disabled (`FR12_NO_SPECULATIVE_CONFIG=1`, no
  `speculative_config`), recurrent path PROVEN engaged ~830k calls, clean argmax recorded in-process — NOT
  chunked re-prefill, streamed logprobs, serial-torch ref, backend name, or fallback (Link 2, CODE-READ).
- Both arms use the same recurrent oracle, same seed/top-k/threshold/backend, same clear-margin def, same
  reducer; only `--arm`/`--src` differ (Link 3).
- Non-vacuous: spec frozen, recurrent ran, denominator = round-trip-validated tokens with 0 dropped on
  both arms, consolidator asserts denominator integrity (Link 4).

**Therefore the 13.548% (cat9) vs 13.985% (native) clear-margin rate, CIs overlapping with cat9 LOWER, is a
real lossless-vs-native PASS at scale**: cat9's per-token deviation from its own no-spec recurrent decode is
statistically indistinguishable from (and numerically slightly below) native E5's deviation from its own no-
spec recurrent decode. cat9 ≈ native within floor ⇒ lossless is met at the big-denominator scale.

### Framing caveats (do NOT overturn the verdict; honest scope)
1. **Re-tokenized served ids (INFERRED, mitigated).** Scored ids come from re-tokenizing served TEXT, not
   the engine's emitted ids. Protected by the class-#12 byte-exact round-trip gate; 0 turns dropped on both
   arms, so the reconstruction is exact for the scored content stream. (The content stream excludes harmony
   channel/structural tokens by design — both arms identically; the comparison is over generated content.)
2. **Both rates ~13.5–14% is a same-frame DELTA, not an absolute lossless-vs-greedy claim.** The non-zero
   floor is the GDN recurrent realization vs the spec serve's per-token argmax; native E5 sits at the SAME
   ~14% floor, which is exactly why the cat9≈native equality is the binding within-floor result (consistent
   with class #11: floors are measured, not assumed; native is the control). This is a relative
   lossless-vs-native PASS, which is the stated bar — not an independent byte-exact-vs-greedy proof.
3. **Single SWE task (astropy-12907), B=1.** 66/64 turns, ~8.7k positions/arm — a large denominator from one
   uncapped agentic task at B=1. The within-floor verdict is scoped to this regime; B=4/CUDA-captured
   superset is a separate axis.
