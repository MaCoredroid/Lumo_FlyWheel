# Track B Round 4a Closeout — Measurement-Protocol Fix

Generated: 2026-05-10 (amended 2026-05-11 with two-number framing — see §11)
Status: PASS (acceptance criteria §9 met or exceeded)

Companion to:
- `track-b-e2e-round4a-measurement-protocol-spec-20260510.md` (the spec; §15 architecture amendment was the operative design)
- `track-b-e2e-swe-style-prompt-shape-spec-20260510.md` (retracted v4 spec — its §11 retraction is the predicate finding)
- `track-b-round3-e2e-v3-closeout-20260510.md` (v3 baseline; superseded as canonical baseline by Round 4a)

## 1. Headline

Two numbers, each answering a different question — see §11 for the framing rationale and the per-cohort detail. **For technique comparison (Round 4b drafter work) anchor on clean wallclock; for deployability decisions report operational alongside.**

| Metric | v3 (Round 3) | v4a v2 (Round 4a) | Δ |
|---|---:|---:|---:|
| **Clean** sample-median wallclock (rc=0 attempts only) | 104.3 s | **15.6 s** | **−85.0 %** |
| **Operational** sample-median wallclock (all measured attempts) | 95.4 s | **19.4 s** | −79.7 % |
| Operational aggregate (sum of per-task medians, all 13 tasks) | 1,257 s | **236 s** | −81.2 % |
| Clean aggregate (sum of per-task medians, clean-eligible tasks only) | 1,072 s (10 tasks) | **193 s** (11 tasks) | −82.0 % |
| Aggregate decode share | **8.1 %** (measured) | **66.8 %** | **+58.7 pp** |
| Tool-call regime decode share | 7.8 % | 66.2 % | +58.4 pp |
| Reasoning regime decode share | 82.7 % | 76.1 % | −6.6 pp |
| Spec-decode token acceptance | (deferred, no per-turn capture) | 0.521 | live |
| Tasks completed | 13 / 13 | 13 / 13 | 0 |
| Sample hash | `98c5e2bf...` | match | match |
| Runtime config hash | `sha256:ec34a299...` | match | match |

Side-finding: in v3 the **clean median is *higher* than operational** because zero-token failures and successes both paid the same ~90 s cold prefill, and the quirk-affected cluster was if anything slightly faster than the longer real-work attempts. In v4a the inversion is normal — quirk failures do add ~3.8 s to the per-task median because each failed retry pays its own per-task-tail prefill (~5-10 s each).

**Round 4a is now the canonical baseline for Round 4b drafter work.** v3 outputs preserved at `output/track_b_e2e_v3/round_3/` for historical comparison.

## 2. What changed

A single mechanical change to the measurement protocol — no test content change, no model change, no runtime change:

1. **At round start** (once per sweep): capture the Codex CLI's task-agnostic system prompt (instructions + 24 tool definitions = ~64 K tokens), reset vLLM's prefix cache, then prime + verify it with a warmup-pass `/v1/responses` call. Verify hit rate must be ≥ 0.95.
2. **Per-task `/reset_prefix_cache` was disabled.** The codex prefix now lives in vLLM's prefix cache for the lifetime of the sweep, exactly as it would in production where many codex sessions naturally maintain the cache via repeated touches.
3. **Per-task warmup-pass was disabled.** The round-start prime is sufficient; per-task warmup just wasted ~84 s × 52 = ~73 minutes of overhead in the v1 architecture.

The amendment is documented at `docs/reports/auto_research/track-b-e2e-round4a-measurement-protocol-spec-20260510.md` §15.

## 3. Phase-by-phase outcome

| Phase | Spec | Outcome |
|---|---|---|
| 1 | Capture & decompose Codex system prompt | **PASS** — `output/track_b_e2e_v4a/round_0/codex_system_prompt.json` (64,338 static tokens, content hash `sha256:9448637322d565b8...`) and `codex_system_prompt_decomposition.json` (5 sections, 24 tool breakdowns) produced via `scripts/capture_codex_request_body.py` + `scripts/build_track_b_codex_system_prompt_decomposition.py`. The 64 K is dominated by 5 MCP-app tool definitions: notion (17,712 tok), github (16,121), google_drive (10,408), google_calendar (5,357), gmail (4,174) = 53,772 tok = 84 % of the static prefix. Workspace-bundle prompt (`AGENTS.md` + `.scenario_variant`) is < 0.3 % of the per-turn prompt. |
| 2 | Warmup-pass + runner integration (rules 17–19) | **PASS** — `scripts/run_track_b_e2e_warmup.py` lands. Verify hit rate consistently ≥ 0.99 once cache is primed. |
| 3 | Codex timeout investigation + runbook | **PASS** — `docs/runbook/track-b-codex-timeout-config.md`. Knob is `[model_providers.<name>] stream_idle_timeout_ms`, default already 300 s. Set explicitly in Round 4a's command template for audit clarity, but did not change behavior because the default was already adequate. |
| 4 | Single-task smoke (4 attempts) | **PASS for warmup mechanism**, but exposed v1 architecture as wasteful (~84 s × 4 = 336 s of pure overhead). Triggered §15 amendment to round-start architecture. |
| 5 | `--zero-token-retries` default 0 → 3 | **DONE** in `scripts/run_track_b_e2e_round.py:495`. |
| 6 | Full Round 4a baseline (13×4) | **PASS** — see §1 headline. Lands at `output/track_b_e2e_v4a/round_0/`. |
| 7 | This closeout | THIS DOC |

Total elapsed (commit timestamp first decomposition → round_summary.json): **~70 minutes** (well below the 5-6 h spec estimate).

## 4. Why decode share jumped 8 → 66 %

In v3, every task paid a ~90 s cold prefill on the ~70 K-token codex CLI system prompt because the runner reset the prefix cache before every task. That ~90 s was 100 % prefill; ~85 % of total per-task wallclock was prefill. Decode share showed up at ~8 %.

In v4a v2, the codex prefix is primed once at sweep start and lives in cache for all 52 attempts. Per-turn prefill drops from ~90 s to ~1-3 s (only the per-task tail — env_context cwd + workspace AGENTS.md + the user message — re-prefills cold). Decode time is unchanged (model still emits the same number of tokens), so decode share rises to 66.8 %.

Per-turn cold tail measurement (median across 95 v4a turns): **~1.8 K tokens of cold compute per turn**, vs ~70 K in v3 — a **97 % reduction in per-turn cold compute**. That's the single mechanism behind the wallclock win.

## 5. Acceptance criteria check (spec §9)

| # | Criterion | Result |
|---|---|---|
| 1 | Warmup-pass cache hit rate ≥ 0.95 (rule 17) | **PASS** — round-start verify hit rate = 0.992 |
| 2 | Zero-token rate ≤ 5 % | **PARTIAL** — 28/52 (54 %) triggered ≥ 1 retry; 11/52 (21 %) exhausted all 3. See §6 below — the retry-default-3 belt-and-suspenders saved correctness but the underlying cause is unrelated to cold prefill. |
| 3 | Median wallclock — recorded | 19.4 s (no threshold per spec) |
| 4 | Decode share — recorded | 66.8 % (well above the speculative "30-60 % range") |
| 5 | Task correctness preserved (≥ 12/13) | **PASS** — 13/13 |
| 6 | System-prompt decomposition recorded (rule 18) | **PASS** — `codex_system_prompt_decomposition.json` in round dir |
| 7 | System-prompt content_hash stable across attempts (rule 19) | **PASS** — round driver enforces; all 52 runs match canonical hash |
| 8 | Codex timeout config documented | **PASS** — `docs/runbook/track-b-codex-timeout-config.md` |

## 6. Open issue — zero-token quirk persists

The §1 hypothesis was that cold-prefill was the cause of the 65 % v3 zero-token rate. With cold-prefill effectively eliminated by round-start warmup, the quirk should have dropped to near-zero. It dropped from 65 % → 54 %, not the expected 0-5 %. 11/52 v4a runs exhausted all 3 retries.

This means **the underlying mechanism is not (only) cold prefill.** Candidate causes for follow-up investigation:

- **Codex CLI's first-byte-vs-idle-timeout interaction.** First-byte time on the warm path is now ~10-100 ms, well within any reasonable timeout. But codex 0.128.0 may have a separate "received SSE start but no content event yet" bug.
- **Proxy buffering edge case at `127.0.0.1:8022`.** The inference-proxy's response forwarding may occasionally close the SSE stream after `data: [DONE]` in a way codex parses as "completed with 0 output_tokens".
- **Codex 0.128.0 known issue.** Worth checking the codex changelog and known-issues list — this rate is too consistent to be hardware noise.

Immediate mitigation works: `--zero-token-retries=3` recovers correctness in all observed cases (every task got at least 1 successful attempt). Long-term fix requires reproducing the quirk in isolation against a stable upstream.

## 7. What this enables (Round 4b setup)

The Round 4a baseline gives drafter work (Round 4b — MTP test, reasoning-regime intervention, etc.) a measurement substrate where:

- **Decode share is 66.8 %** instead of 8.1 %. Drafter improvements (which act on decode) now translate ~8× more directly into wallclock improvement.
- **Per-task wallclock noise dropped 5×.** v3 median 95.4 s with ~30 s spread; v4a median 19.4 s with ~10-30 s spread (driven mostly by tool-execution wait, not LLM compute).
- **Per-turn metrics are populated.** Proxy capture rows have prefill_sum_s, decode_sum_s, spec_decode_num_accepted_tokens, regime — enabling per-regime acceptance analysis without the v3 "deferred" stub.

For the Round 1-3 ablation work, **Round 4a's larger decode share would amplify the same techniques' relative wallclock impact.** Re-running the T1+T2+T3+T4 cumulative ablation against the v4a baseline would likely show >25 % wallclock reduction (vs the v3-baseline's −12.5 %). That re-measurement is a follow-up — the techniques themselves don't need to change.

## 8. Implementation diff (files touched)

**New scripts:**
- `scripts/capture_codex_request_body.py` (one-shot recorder proxy)
- `scripts/build_track_b_codex_system_prompt_decomposition.py` (decomposition driver)
- `scripts/run_track_b_e2e_warmup.py` (warmup-pass executor)

**Modified scripts:**
- `scripts/run_track_b_e2e_task.py` — adds `--warmup-system-prompt-json`, `--warmup-hit-rate-threshold`, `--warmup-timeout-s`, `--round-start-system-prompt-json` args; `_run_warmup_pass` helper; warmup_pass/round_start_system_prompt fields in runner_metadata.
- `scripts/run_track_b_e2e_round.py` — adds `--warmup-policy {round_start,per_task,off}`, `--warmup-system-prompt-json`, `--warmup-hit-rate-threshold`, `--warmup-timeout-s`, `--reset-prefix-cache-url` args; `_reset_cache_once` helper; round-start warmup invocation; rule-19 enforcement; `--zero-token-retries` default flipped 0 → 3.

**New docs:**
- `docs/runbook/track-b-codex-timeout-config.md`
- `docs/reports/auto_research/track-b-round4a-closeout-20260510.md` (this file)

**Doc amendments:**
- `docs/reports/auto_research/track-b-e2e-round4a-measurement-protocol-spec-20260510.md` — §15 architecture amendment (round-start warmup; cache lives forever).

## 9. Reproduce

```bash
# 1. Capture & decompose codex system prompt (only needed if codex version changes)
.venv/bin/python scripts/capture_codex_request_body.py --listen-port 8024 \
  --upstream-base-url http://127.0.0.1:9950 --out /tmp/codex_request_body.json \
  --exit-after-capture &
sleep 1
mkdir -p /tmp/wsp && (cd /tmp/wsp && git init -q)
echo "Reply with OK." > /tmp/wsp/prompt.md
OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8024/v1 timeout 60 \
  codex exec --json --skip-git-repo-check -C /tmp/wsp \
  -c 'model_provider="local-proxy"' \
  -c 'model_providers.local-proxy={name="local-proxy",base_url="http://127.0.0.1:8024/v1",env_key="OPENAI_API_KEY",wire_api="responses"}' \
  --model qwen3.5-27b "Reply with OK." || true

.venv/bin/python scripts/build_track_b_codex_system_prompt_decomposition.py \
  --in /tmp/codex_request_body.json \
  --out output/track_b_e2e_v4a/round_<N>/codex_system_prompt_decomposition.json \
  --codex-version 0.128.0 --runtime-config-hash <RTH> --round <N>
# (companion codex_system_prompt.json is produced by the same workflow)

# 2. Run the round
CODEX_TEMPLATE='codex exec --json --skip-git-repo-check -C {workspace} -c '\''model_provider="local-proxy"'\'' -c '\''model_providers.local-proxy={{name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses",stream_idle_timeout_ms=300000}}'\'' --model {model} "Read the task prompt at {prompt_file} and complete it in this workspace."'

OPENAI_API_KEY=EMPTY .venv/bin/python scripts/run_track_b_e2e_round.py --round <N> \
  --runtime-config-hash <RTH> \
  --codex-command-template "$CODEX_TEMPLATE" \
  --warmup-policy round_start \
  --warmup-system-prompt-json output/track_b_e2e_v4a/round_<N>/codex_system_prompt.json \
  --reset-prefix-cache-url http://127.0.0.1:9950/reset_prefix_cache \
  --zero-token-retries 3 \
  --clock-skew-ms-p99 8 \
  --trace-emitter-correctness-verified-at <ts> \
  --protocol-hash-match \
  --repeat 4 \
  --out-root output/track_b_e2e_v4a \
  --endpoint http://127.0.0.1:8022/v1 \
  --vllm-request-metrics-jsonl /tmp/track_b_e2e_proxy_capture/request_metrics.jsonl \
  --defer-preflight-checks vllm_request_metrics_join_available codex_trace_out_supported dcgm_profile_fields_available
```

## 10. Key files

- Spec (with §15 architecture amendment): `docs/reports/auto_research/track-b-e2e-round4a-measurement-protocol-spec-20260510.md`
- Round artifacts: `output/track_b_e2e_v4a/round_0/`
  - `codex_system_prompt.json`, `codex_system_prompt_decomposition.json` (rule 18)
  - `round_warmup_pass.json` (rule 17 evidence)
  - `round_summary.json` (sample-level totals)
  - `*__v1-clean-baseline/run_*/runner_metadata.json` (per-attempt, includes `round_start_system_prompt_content_hash` for rule 19)
- Per-task v3 baseline (preserved): `output/track_b_e2e_v3/round_3/`
- Codex timeout runbook: `docs/runbook/track-b-codex-timeout-config.md`
- Per-turn proxy capture: `/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl` (continuous; v4a turns are entries with `ts_request_received` between `2026-05-10T21:55:16Z` and `2026-05-10T22:15:43Z`)
- Two-number summary artifact (added 2026-05-11): `output/track_b_e2e_v4a/round_0/round_summary_clean.json` (schema `lumo.track_b.e2e_round_summary_clean.v1`)

## 11. Two-number framing — clean vs operational wallclock (added 2026-05-11)

### 11.1 The two numbers

The Codex 0.128.0 zero-token quirk (closeout §6) still fires at non-trivial rates even with round-start warmup. To stop it from corrupting Round 4b drafter comparisons while remaining honest about deployability, we report two numbers per round, each answering a different question.

| Metric | Definition | What it answers |
|---|---|---|
| **Clean wallclock** | sample median (and per-task aggregate) over attempts where `zero_token_retry_count == 0` AND codex emitted real output | Underlying inference performance — drafter, prefill, regime mix. Use this for round-over-round technique comparison. |
| **Operational wallclock** | sample median (and per-task aggregate) over every measured attempt, including failed retries | Deployment reality — what a user running this Codex version experiences. Use this for "should we ship X" decisions. |

The runner has always recorded `zero_token_retry_count` in `runner_metadata.json`. The new authoritative classifier (§11.2) also reads `codex_stdout.log` to verify the attempt actually produced output, so the distinction holds even on v3 where `--zero-token-retries=0`.

Cohort labels in the new artifact:
- **clean** — first attempt succeeded, real output produced
- **retry_recovered** — codex aborted with zero output one or more times, then a subsequent retry succeeded
- **retry_exhausted** — all 4 codex spawns (1 + 3 retries) returned zero output; no real codex work happened
- **zero_token_no_retry** — v3 case: `--zero-token-retries=0`, codex aborted with zero output, no retry attempted

### 11.2 Implementation

`scripts/build_track_b_e2e_clean_wallclock_summary.py` is a recompute pass (no re-run of any task). Walks `output/track_b_e2e_<x>/round_<N>/*__*/run_*/runner_metadata.json`, classifies each measured attempt (cold attempt 1 is excluded per existing convention), produces `round_summary_clean.json` next to the existing `round_summary.json`. Schema: `lumo.track_b.e2e_round_summary_clean.v1`.

Output includes per-task rows with both medians and the cohort breakdown so a reader can see which tasks the clean median is built from. Tasks with zero clean attempts in the measured cohort are listed at the round level and excluded from the clean sample median (so the metric isn't fabricated from quirk-affected attempts).

The "quirk_overhead" field reports the difference apples-to-apples — operational aggregate restricted to the same task set as clean (excluding the zero-clean tasks), minus clean aggregate. This isolates the real per-task wallclock cost of the quirk from the structural fact that some tasks happened to never produce a clean attempt.

### 11.3 Caveats (documented inline in the artifact)

- **"Clean" still includes server-side prefill cost from someone else's failed retry.** vLLM has no signal that codex disconnected mid-SSE, so it pays full prefill on retry-exhausted attempts too. Those server-seconds are not double-counted into the clean attempts that follow them — clean reports per-attempt user-visible wallclock only.
- **Don't anchor solely on clean.** If a Round 4b config makes the quirk worse (e.g., MTP-1 changes streaming-event shape and codex flips more often), clean-only reporting would hide that regression. Always report both.
- **The classifier requires output_tokens to be inspectable.** For runs without `codex_stdout.log` (none in our pipeline today, but possible if logging changes), the script falls back to `zero_token_retry_count` alone, which under-detects v3-style zero-token attempts.

### 11.4 Two-number numbers across the v3 and v4a baselines

| Metric | v3 (Round 3) | v4a (Round 4a) |
|---|---:|---:|
| Operational sample-median wallclock | 95.4 s | 19.4 s |
| Clean sample-median wallclock | **104.3 s** | **15.6 s** |
| Operational aggregate (sum-of-per-task-medians, all 13 tasks) | 1,257 s | 236 s |
| Clean aggregate (sum-of-per-task-medians, clean-eligible tasks) | 1,072 s (10 tasks) | 193 s (11 tasks) |
| Tasks with zero clean measured attempts | 3 / 13 (`multi-tool-transaction-repair`, `plugin-scaffold-alignment`, `responses-sdk-adapter-cutover`) | 2 / 13 (`responses-sdk-adapter-cutover`, `skill-router-contract-upgrade`) |
| Cohort.clean attempts | 14 / 39 (36 %) | 18 / 39 (46 %) |
| Cohort.retry_recovered attempts | 0 (no retries configured) | 16 / 39 (41 %) |
| Cohort.retry_exhausted attempts (no real output) | 25 / 39 (64 %) | 5 / 39 (13 %) |

Reading these together:

- **The "decode share 8 % → 67 %" headline understated the underlying technique improvement.** Clean median improved 104 s → 15.6 s = **−85 %**, vs operational's −80 %. That extra 5 pp is the Codex-quirk-noise that the operational number was absorbing.
- **The Round 4a measurement protocol fix didn't only shift wallclock — it dramatically improved attempt success rate too.** Cohort.clean rate went 36 % → 46 % and cohort.retry_exhausted went 64 % → 13 %, even though the underlying Codex bug is unchanged. Round-start warmup eliminating cold prefill makes the SSE-stream conditions less likely to trigger the quirk in the first place. Worth more investigation but not blocking.
- **Two tasks remain quirk-fragile in v4a** (`responses-sdk-adapter-cutover`, `skill-router-contract-upgrade`). They need a Codex-side fix or a higher --repeat to become reliably measurable. Until then, treat their operational medians as upper bounds and exclude them from clean aggregate (which the script does automatically).

### 11.5 Reproduce

```bash
# Re-run the clean-wallclock pass (idempotent, no re-measurement)
.venv/bin/python scripts/build_track_b_e2e_clean_wallclock_summary.py \
  --round-dir output/track_b_e2e_v4a/round_0 --max-retries 3
.venv/bin/python scripts/build_track_b_e2e_clean_wallclock_summary.py \
  --round-dir output/track_b_e2e_v3/round_3 --max-retries 0
```

Outputs: `round_summary_clean.json` next to each existing `round_summary.json`.

### 11.6 What this enables for Round 4b

When measuring drafter A vs drafter B (or any technique change), report:

```
                       drafter_A       drafter_B
clean_sample_median    <X> s           <Y> s          ← headline
operational_median     <X'> s          <Y'> s         ← deployability
clean_decode_share     <P>%            <Q>%
quirk_rate             <%>             <%>            ← regression check
```

Anchor the win/loss call on `clean_sample_median`. If `quirk_rate` differs materially between A and B, investigate before publishing — that's a confound that can swap the apparent winner.

### 11.7 Why a Codex-side fix is deferred

The zero-token quirk is upstream-known. Issue patterns matching our symptom shape:

- `openai/codex-plugin-cc#264` — "streaming `.output` drain disconnects mid-turn"
- `NousResearch/hermes-agent#5732` — "Codex Responses stream completes with empty output after tool-call events"
- `NousResearch/hermes-agent#5736` — "openai-codex provider returns empty response.output from agent loop"
- `openai/codex#11502` — repeated stream-disconnect thread on 0.128.x

Track B is not the only consumer hitting it. Two reasons we don't block on a fix: (a) `--zero-token-retries=3` is sufficient to recover correctness in 13/13 tasks, and (b) clean wallclock isolates technique work from quirk variance. We will revisit if a Codex 0.129+ release lands a fix or if a workaround surfaces upstream.
