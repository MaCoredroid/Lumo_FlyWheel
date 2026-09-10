# Experiment card — controlled GB10 reproduction of #54928 (Plan v3 item P5) — v2 (Codex round-1 applied)

> For Mark's FUNDING decision. Nothing runs and nothing posts without it.
> Funding the run is separate from approving its report. v2 = Codex's
> replacement: pins verified to full commits and the official aarch64
> v0.28.0 wheel; eager must be paired on both sides; instability and path
> divergence can coexist; ties in top-5 do not identify a unique verifier
> argmax; token IDs (incl. reasoning tokens), not parsed text; prioritized
> initial matrix, one chosen extension; unstable baselines are results.

---

## Question and scope

On one GB10, does a pinned stock v0.28.0 FP8 GDN target with DFlash2 differ
from target-only at the same generated prefix? Separately measure repeat
instability, launch sensitivity, and target-versus-verification ranking.
These can coexist. This is a related reproduction of #54928, matching
Windless84's release/model revisions on a different device; it is not the
original multi-GPU BF16 reproduction or a claim about current main.

## Immutable artifacts and environment

- Target: `Qwen/Qwen3.8-27B-FP8` at
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Draft: `incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- Official PyPI wheel:
  `vllm-0.28.0-cp38-abi3-manylinux_2_28_aarch64.whl`, SHA256
  `817b8181f7f61b4a62dc1d5d9ab39f2bfb60a6cb86c29879a78a147b85756787`.
- Isolated environment; campaign installation untouched. Record Python,
  driver, torch/CUDA, FlashInfer, all installed versions, wheel provenance,
  resolved model/tokenizer snapshots, chat-template hash, runner, dtypes,
  and selected backends. Save a dependency lock before comparative runs;
  change no dependencies between arms. Pin revisions in the actual loader
  arguments or immutable local snapshots, not only in this document.
- Runtime compatibility is unverified. No source patches or backend
  workarounds are part of this stock experiment.

## Initial matrix: two server arms, two fixed request cases

A: target-only. B: the same settings plus DFlash2 with K=7.
Use TP=1, prefix caching off, max_num_seqs=4, max_model_len=4096,
serial requests, default compilation, and the same resource settings.
Record resolved settings and argv; allow only service plumbing and the
speculative configuration to differ. Use the same raw-logprobs mode.

Case G: Windless84's exact git-squash prompt, enable_thinking=false,
seed=1234. Case T: the exact Spanish user message from the original issue,
reasoning_effort=medium, preserve_thinking=true, thinking enabled,
seed=20260902. Save each complete request JSON and rendered prompt-token
IDs; preserve the same request within each comparison.
Both cases: temperature=0, top_p=1, top_k=-1, min_p=0,
repetition_penalty=1, presence_penalty=frequency_penalty=0,
min_tokens=0, ignore_eos=false, max_tokens=256, n=1, stream=false,
logprobs=true, top_logprobs=5. Explicit settings override model defaults.

Start A and B twice each, one server at a time; use order A/B then B/A.
In each launch submit each case five times in a fixed recorded order.
Keep every result, including warmup/compile effects and unstable runs;
do not discard a baseline because its repeats differ.

## Token-level attribution

Use the stock return_token_ids and return_tokens_as_token_ids options;
confirm that full generated IDs, including reasoning tokens, align with
returned per-position logprobs before interpreting the first divergence.
Compare token IDs, not decoded or reasoning-parser-filtered strings.
If stock outputs cannot establish this alignment, report that limitation;
new tracing or an analysis harness needs a separate new-code decision.

At the first differing generated position with identical preceding token
IDs, record A (target-only emitted token and reported maximizers), E
(speculative emitted token), and the set of reported verifier maximizers V.
Only use a single-token V when the returned maximum is unique. With ties,
report set membership and "argmax unresolved from returned logprobs"; do not
infer an acceptance bug from arbitrary top-5 ordering. Retain both top-5
lists and emitted-token logprobs. Report each side's top-two margin in nats;
if a requested comparison token is absent, report its value as unavailable.

Report within-launch and between-launch stability separately. E matching a
unique V that differs from A localizes a ranking discrepancy; it does not
identify its kernel cause or rule out incorrect state. No divergence before
EOS/the output limit is a scoped non-reproduction, not proof of equivalence.

## Budget, extensions, and stop rules

Mark funds one half-day including setup/download/build time. Four server
launches and the initial matrix are the first target, not a guaranteed
runtime. One further half-day requires a named unresolved question.
Choose one extension: K=1 against the matching baseline; an eager pair
with BOTH target-only and DFlash2 eager; or the Spanish thinking-off
control. Do not automatically run their Cartesian product.
A current-main comparison requires its own pinned paired baseline and
budget; never compare a new speculative build only against an old baseline.

Stop the performance/correctness comparison if the stock wheel or draft
cannot load, or the speculative path is not actually exercised. Record
startup logs and acceptance/proposal counters. A startup failure is a
compatibility observation, not reproduction of greedy divergence. Verify
its relevance before deciding whether any public report belongs in #54928.
No competing GPU workloads, source patches, or unapproved instrumentation.

## Deliverable and approval

Provide arm/launch/case/repetition, token IDs, first-divergence index,
stability, reported maximizers, margins, finish reason, raw JSON responses,
resolved configuration, exact commands and immutable artifacts.
If existing tools cannot produce this record, classify the missing tooling
as new-code before proceeding. Positive, negative, unstable, and
unattributable outcomes are all retained with their scope.
Mark's funding GO authorizes the bounded run, not publication; the actual
report receives red-team and a separate Mark GO. No tree pitch or
oracle-floor relaxation in any eventual #54928 comment.
