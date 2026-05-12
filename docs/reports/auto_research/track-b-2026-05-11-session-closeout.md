# Track B Session Closeout — 2026-05-11 → 2026-05-12

Status: **CLOSED, with one blocking operator decision left open.**

This session set out to run two queued Round 4b workstreams (per-regime
acceptance + e2e ablation against v4a baseline). The work landed both
artifacts, but the ablation null result triggered a benchmark-validity
audit that found Track B's entire wallclock measurement program has been
measuring an agent that doesn't actually do the work. Subsequent
investigation pinpointed the bottleneck (qwen3.5-27b on this corpus,
not the harness, confirmed by a gpt-5.5 swap that produced real
artifacts), and one external-published fix was attempted and falsified.

## 1. What landed

| Commit | Subject | What it adds |
|---|---|---|
| `8864cf7` | Round 4b per-regime acceptance | Reasoning acceptance still 0.23 (vs v2 0.21); reasoning is 6.8% of decode-time on this corpus. Deprioritized MTP-1 test on this corpus. |
| `6a5ed0b` | Round 4b e2e ablation | 4-point ablation A/B/C/D, clean median range 14.85–16.19 s — all four points statistically indistinguishable. Round 3 synthetic microbench's 33→79 gradient does not reproduce in real e2e. |
| `c7855d9` | Round 4b ablation artifacts | Round summaries + per-regime captures for D + A/B/C points. |
| `8e7adec` | Benchmark-validity audit | Across 208 measured runs, 0 apply_patch calls, 0 agent messages with text, 0 workspace mutations. Every "rc=0 task completion" was codex reading the prompt and running 1-5 read-only `cat`/`find`/`ls`. |
| `3c52faf` | Audit §10 — codex iterates after all | Correction: codex DOES iterate (mean 2.29 proxy calls per v4a run). The audit's "1 turn" was a misreading of the event model. Bottleneck is model output per round, not codex loop. |
| `03bf269` | Audit §11 — gpt-5.5 PASSES | gpt-5.5 high on the same workspace produced both required artifacts, 20 tool calls in one codex turn, 3,653 output tokens, 169 reasoning tokens. **Harness works.** Identified known qwen3.5/3.6 27B chat-template + tool-call-parser bug via NVIDIA DGX Spark forum (this machine) + 5 GitHub issues. |
| `542c8e1` | Audit §12 — chat template fix DID NOT WORK | Swapped to enhanced template, patched for `role: developer`, re-tested. Same broken behavior: 1 turn, 1 cat call, 0 patches, 0 files. The documented fix targets one specific failure mode that's not the binding constraint on this corpus. |

All commits pushed to `origin/main`.

## 2. Trajectory of the discovery

The session is a worked example of why "honest measurement protocol"
catches things that "shipping numbers" doesn't:

1. **Round 4b ablation, expected outcome:** T2/T3/T4 contribute
   monotone wallclock improvement, per the Round 3 synthetic
   microbench (T1 only = 33.5% → all on = 78.9% acceptance).
2. **Actual outcome:** No measurable difference between any of the
   four ablation points (15–16 s clean median across all four). Wrote
   that up as "techniques have hit a real-traffic ceiling," which it
   nominally is.
3. **The pre-audit comment from the user** ("15 s/task feels too easy
   to be doing real work; how were graders even passing?") forced the
   benchmark-validity check.
4. **Audit found:** `tasks_correctness_deferred_to_exit_code: 13` in
   every round_summary. The truthful-measurement contract's rule 8
   was being satisfied by accepting `codex_exit_code == 0` as proof
   of task completion. The family graders never ran.
5. **What the agent actually did, across 208 runs:** read the prompt
   via `cat`, maybe `ls`/`find` once or twice, exit. **Zero workspace
   mutations across the entire ablation matrix.** No drafter technique
   could have moved a number that was already 0.
6. **First-order conclusion:** all Track B wallclock measurements
   (Round 3, 4a, 4b) measured the time for codex to read prompt.md +
   exit. The decode-share jump in Round 4a (8% → 67%) is real but
   describes the right thing measured on the wrong workload.
7. **The §10 correction:** initial framing said "codex doesn't
   iterate." Proxy capture data showed otherwise — codex iterates
   2.29 times per run on average. The bottleneck is model behavior:
   qwen3.5-27b emits one read-only tool call per round and stops.
8. **gpt-5.5 swap:** same workspace, same codex command, swapped only
   the model. Result: 20 tool calls in one turn, 3,653 output tokens,
   2/2 required artifacts written, substantive content (correctly
   identified the failed guardrail with multi-source evidence
   references in the incident packet). **Harness is fine.**
9. **External-knowledge fix:** NVIDIA DGX Spark forum + 5 GitHub
   issues converge on a chat-template + tool-call-parser bug in qwen
   3.5/3.6 27B. Documented fix: swap in `qwen3.5-enhanced.jinja` +
   `--tool-call-parser qwen3_xml`.
10. **Fix attempted (§12):** Container relaunched with patched
    enhanced template (developer-role compatibility patches required).
    Re-tested qwen+codex on the same workspace. **Same broken
    behavior.** The documented fix was for a different failure mode
    on a different hardware+quantization+workload combination.

## 3. What we now know — clearly

| Claim | Confidence |
|---|---|
| qwen3.5-27b emits one read-only tool call and stops on Codex CLI prompts on this hardware | High — 208 measured runs + post-fix attempt all show the same pattern |
| The harness (codex CLI, bench proxy, runtime, prompts) works | High — gpt-5.5 swap produced real artifacts on the same harness |
| Chat-template fix alone doesn't unblock qwen | High — empirically tested on the recommended fix |
| `tool-call-parser qwen3_xml` and `reasoning-parser qwen3` are correctly wired | High — verified in launch_cmd and via Qwen3XMLToolParser init logs |
| Round 4a "−80% wallclock" reflects cold-prefill removal | Yes — mechanism is real, but the workload it measured is the degenerate one |
| Round 4b ablation null result reflects technique-on-degenerate-corpus | High — re-running ablation on a fixed corpus is the only valid signal |
| Round 3 synthetic microbench overstated technique value by ~30 pp | High — see §10.4 and §11.3 of audit |

## 4. What we don't know — the still-open questions

- **Why does qwen3.5-27b emit one-shot tool calls on Codex prompts?**
  Could be training-distribution shape (qwen never saw Codex's tool
  surface during instruction tuning), could be the specific reasoning
  prompt format, could be tokenizer / chat-template subtlety that the
  fix template didn't address. Unclear without more probes.
- **Would gpt-5.5 (or comparable cloud model) sustain the win across
  the full 13-task corpus?** Only tested on `incident-evidence-synthesis`.
- **Is there a finetuning path for qwen3.5-27b on Codex-shape
  traces?** Plausible but a separate research effort.
- **Do family graders even work?** They were never invoked. Need a
  smoke test against one gpt-5.5-produced workspace to confirm the
  grader path is functional before relying on it.

## 5. Open decisions for the operator

These are the load-bearing decisions left after this session:

### 5.1 Anchor model choice — required before any Round 5 work

Pick one:

- **(a) Switch to gpt-5.5 (or other capable agentic model) as the
  Track B harness anchor.** Re-baseline v4a on this model. All
  drafter / suffix work going forward measures against this baseline.
  Loses qwen-specific suffix-decoding focus but unblocks real
  measurement immediately.
- **(b) Investigate qwen3.5-27b one-shot behavior.** Could include:
  prompt-engineering experiments at the codex user-message level,
  scaffolding via codex's `update_plan` tool, finetuning probes, or
  switching to a different open model (Qwen3-Coder-30B-A3B is one
  candidate per the unsloth fix repo). Larger time investment, may
  not converge.
- **(c) Both in parallel.** Use gpt-5.5 for harness-validation now
  (so other Round 4b/5 work can proceed against a real baseline),
  and pursue qwen unblock as a separate research thread.

The audit recommends (a) or (c). (b) alone keeps the harness blocked
indefinitely.

### 5.2 Family grader wiring — required regardless of (5.1)

Replace `tasks_correctness_deferred_to_exit_code` with real `task_score`
from `verifiers/<family>/score_ranking.py`. Round summary should fail
loud if `task_score == 0` on the majority of tasks. Without this, the
next regression hides.

### 5.3 Loosen acceptance criterion

The §10.7 acceptance bar ("≥1 `apply_patch` call") didn't anticipate
that capable models would use `python3 - <<PY` heredocs or
`cat > file <<EOF` constructs instead. Loosen to "any workspace
mutation by the agent" — apply_patch is one path among several.

### 5.4 Container state — what's running now

The vLLM container `lumo-vllm-track-b-suffix` is currently running
the **enhanced + developer-patched chat template** (md5
`7a6059bb08728e06d028cb27e96aa02e`). The original Lumo template is
preserved at
`docker/chat_templates/qwen3-openai-codex.jinja.backup-pre-enhanced-20260511`.

If the operator wants to revert to the v4a baseline runtime exactly:

```bash
set -a; source /home/mark/shared/lumoFlyWheel/.lumo.local.env; set +a
cp docker/chat_templates/qwen3-openai-codex.jinja.backup-pre-enhanced-20260511 \
   docker/chat_templates/qwen3-openai-codex.jinja
docker stop lumo-vllm-track-b-suffix
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' sync
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' sysctl -w vm.drop_caches=3
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' swapoff -a
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' swapon -a
docker start lumo-vllm-track-b-suffix
```

The container restart cycle is now well-tested (executed 3× this
session). Each cycle is ~4 min wall, dominated by the model load
(11 shards × 15 s each + warmup).

## 6. What stays valid from prior closeouts

- **Round 4a measurement protocol** (round-start warmup, prefix-cache
  pinning, clean-vs-operational two-number framing). Sound
  infrastructure. Doesn't need to be rebuilt; just needs a working
  agent loop on top of it.
- **Round 4a runtime configuration** (vLLM 0.19.0 image, suffix
  decoding speculative-config, batch sizes). Unchanged across the
  session except for the chat-template swap.
- **`scripts/run_track_b_v4a_e2e_ablation.py`** (the ablation driver
  built this session). Reusable for any future ablation as long as
  the agent loop is producing real work.
- **`scripts/build_track_b_per_regime_acceptance.py`** (the
  per-regime aggregator + None-format bug fix). Reusable.

## 7. Where to read more

- `docs/reports/auto_research/track-b-benchmark-validity-audit-20260511.md`
  — full audit with §10/§11/§12 addenda. The definitive document on
  the benchmark-validity finding and what's been ruled out.
- `docs/reports/auto_research/track-b-round4b-e2e-ablation-20260511.md`
  — the ablation null result, written before the audit but now reads
  as "ablation-on-degenerate-corpus."
- `docs/reports/auto_research/track-b-round4b-per-regime-acceptance-20260511.md`
  — per-regime acceptance on v4a baseline. Stands on its own.
- `docs/reports/auto_research/track-b-round4a-closeout-20260510.md`
  — v4a baseline. The wallclock numbers there are now contextualized
  as cold-prefill-removal-on-degenerate-corpus.

## 8. One-paragraph summary for anyone landing on this cold

Track B measures Codex CLI wallclock on 13 SWE-style tasks. This
session ran a 4-point ablation against the v4a baseline expecting a
monotone gradient from prior techniques; got a null result. Audited
why: discovered the agent had been emitting one read-only tool call
per task and exiting since at least Round 3, while every round
summary reported `13/13 correctness` by deferring to `codex_exit_code
== 0` rather than running family graders. Verified harness works by
swapping qwen3.5-27b for gpt-5.5 on one task (produced real
artifacts). Verified documented qwen chat-template fix does NOT
unblock qwen on this hardware+workload. Decisions left for operator:
pick anchor model (gpt-5.5 vs. continued qwen investigation vs.
both), wire family graders into round summary regardless. No Round 5
or further drafter work until one of these lands — the wallclock
substrate is currently measuring nothing.
