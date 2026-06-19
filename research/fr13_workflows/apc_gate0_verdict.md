# APC Gate-0 verdict (red-teamed) — 2026-06-19

Boot of the forked FA2 tree server with `FR13_ENABLE_APC=1` (align + chunked + fp32 ssm cache,
`mamba_block_size=1024`). Run: `output/fr13_apc_gate0/run_20260619T195222Z/`. Driver:
`scripts/fr13_apc_gate0.sh` (detached, self-teardown). Branch `fr13-prefix-cache`.

## Verdict: Gate-0 effectively PASSES (all 4 gates green on substance)
| Gate | Raw | Red-teamed verdict |
|---|---|---|
| A boot/health | PASS | PASS — healthy after 422s |
| B APC engaged | PASS | PASS — `enable_prefix_caching=True`, `mamba_cache_mode='align'` (auto-forced), `mamba_ssm_cache_dtype='float32'`, `enable_chunked_prefill=True`; experimental-align warning; **no NotImplementedError** (SD conv layout confirmed) |
| C cache hits | PASS | **PASS, non-vacuous** — `prefix_cache_queries_total=12964`, `prefix_cache_hits_total=4992`; live hit-rate 38.5%→51.3%. NOT the #45238 silent 0-hit trap. |
| D no-crash | "FAIL" | **PASS in substance** — the FAIL is a SCRIPT BUG, not an APC failure (below). Server alive, real-crash-count 0, rc=0, healthy spec-decode after. |

## Two apparent problems — both ARTIFACTS, neither is an APC defect

### 1. GATE-D "FAIL" = script arithmetic bug (FIXED)
`run2.log`: `scripts/fr13_apc_gate0.sh: line 356: ((: 0\n0: syntax error in expression`.
`CRASH_HITS=$(grep -cnE "..." file || echo 0)` → `grep -c` already prints `0` on no-match AND
exits 1, so `|| echo 0` appended a second `0` → the two-line string `"0\n0"` → `(( CRASH_HITS == 0 ))`
threw a syntax error → fell to the FAIL branch. The *actual* survival signals were all green
(`alive=1`, real-crash-count `0`, `rc=0`). **Fixed** (`grep -cE ... | head -1` + int-guard); re-run
will report GATE-D PASS.

### 2. The garbled `output_text` generation = degenerate-PROMPT artifact, NOT #43559 APC-poisoning
The gen output was `</think>\n\noutput_text\n\noutput_text…` (runaway). This is **not** APC
corruption:
- The banked warm-prompt (`long_prompt.txt`) literally contains harmony channel markers
  (`output_text`, `reasoning_text`) and ends mid-structure (`…reasoning_text\n\nLet me read…`),
  sent to **raw `/v1/completions`** (no chat template) at temp 0.6 → the model faithfully
  continues the degenerate channel-marker pattern. The text echo is a prompt-format artifact.
- **Decisive counter-evidence to poisoning:** spec-decode acceptance is HEALTHY under live APC —
  `Mean acceptance length: 5.00`, per-position `1.0, 0.9, 0.75, 0.7, 0.65` at **51.3% cache-hit
  rate**. #43559 cache-poisoning would COLLAPSE acceptance (draft ≠ verify); instead draft/verify
  agree strongly *with the cache hitting*. 
- Cold (req1) ≡ warm (req2) produced **identical** output — the cache reproduced the state, did
  not corrupt it.

The garble is therefore informative only about the warm-prompt (a raw harmony dump is a poor
canary); it says nothing bad about APC. The binding #43559 question is settled definitively by the
4-arm lossless A/B on **clean SWE streams** (where harmony markers don't appear), not by this
canary.

## So: APC is feasible on our build. Next = the binding lossless A/B
Gate-0's job (boots + engages + non-vacuous hits + survives num_accepted>1) is met. The remaining
question — is APC LOSSLESS for our tree-spec path (does #43559 reproduce on commit gfe9c3d6c5)? —
is NOT answered by Gate-0 and MUST go through the 4-arm same-boot A/B vs the no-spec RECURRENT
oracle within the E5 floor (NEVER a proxy): B cache-OFF / control chunked-ON / A cache-ON /
native-E5 cache-ON. AWAIT user steer before launching that multi-hour GPU campaign.

Evidence: `output/fr13_apc_gate0/run_20260619T195222Z/{completion_*.json, docker_log_after_specgen.txt,
metrics_after_req*.txt}`; `output/fr13_apc_gate0_run2.log`.
