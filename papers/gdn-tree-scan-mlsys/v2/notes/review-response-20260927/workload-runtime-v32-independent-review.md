# Workload runtime v3.2: independent CPU review

2026-09-28 UTC. **R1 and R3 are closed at the CPU component level. R2 and R4 each retain one concrete integration blocker below.** Keep WP closed; workload progress remains **0/4**. Exactly `scikit-learn__scikit-learn-9288`, AR → CHAIN_MTP → SGLANG_EAGLE → LUMOTREE, one attempt each, request seed absent. No task, tuning, retry, budget or gate changes were made.

## Authenticated review input

Read-only SSH copied the package into `p0/monitor/review-response-20260927/workload-runtime-v32-reviewed-20260928T165658Z/`. The root `workload-plan/MANIFEST-WORKLOAD-RUNTIME-PACKAGE-v3.2.json` SHA256 is **726f29dc36c4a51c0c8cff7ffc83bc38802af7994ad4453703c2a37ea2a0913f**. Initial recursive component traversal checked 241 references and 165 unique files without mismatch. The separately nested `v3_manifest` link was then followed and its 19 references authenticated, including its pinned manifest `f8116ba387fddecd16d6a245042436c212bb17c85f0df62d44b13aed13d49ea2`. Supporting immutable dependency bundles and the already-extracted SGLang emitter were copied read-only; import of the actual successor chain verified its dependency seals/member sizes and hashes.

The complete review snapshot inventory contains 323 source/evidence files, distinguishing package references from supplemental source copies. It is bound by `REVIEW-EVIDENCE.json`, SHA256 **8200a20d093d6353e143f6a3fe018edf1ecedf1400fb706bf8bc0c534f579308**. No existing accepted bundle or remote file was changed.

CPU controls used the configured bundled Python. Since pytest is unavailable locally, a small decorator/assertion compatibility shim loaded selected original fixture functions; this is **not a claimed pytest rerun**. Actual runtime/collector/closure functions and injected Docker/Popen seams were used. Sixteen selected controls passed: twelve boot/setup/cleanup paths and four rate controls. The two independent counterexamples below were then reproduced. Source and full output are retained as `REVIEW-controls.py` / `REVIEW-controls.log` in the snapshot (SHA256 `6d033298795e31b5d4cf3c5372545784c03292fd837fe7a55ad9edc6beb98d8c` / `271230d86cd1c361d1588a44a28d1b04e8d9025cf8aebb0b63aadfc75fdff587`). Controls launched no real process, Docker operation, model request or evaluator. SSH was used only to read/copy existing artifacts.

All source references below are relative to the snapshot's `repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/`.

## Closed findings and retained accepted checks

**R1 — prior evidence map:** `workload-plan/tools/attempt-runtime/sole_executor_v3_2.py:114–205` reconstructs prior closure/record/raw-blob evidence, authenticates packet-supplied approval files, verifies terminal executor ledgers, and gives the same authenticated population to preflight and `Runtime.own`. The real closure-successor path now admits ordinal 2 after verified terminal boot/setup failure, and refuses ordinal 2 when cleanup is unverified. The tested positive and negative paths exercise the actual dependency chain, not a fake closed-prefix return.

**R3 — zero-output and count/time pairing:** `workload-case-study-v1/pooled_rate_v3_2.py:131–164,218–244` refuses each zero-output completed request. It uses cumulative completion counts at the first and last token-bearing events, paired with those same event timestamps. The `[10,0]` counterexample is NOT_REDUCED; completed requests with an aborted request contributing 100 aggregate tokens produce **9**, not 59 tokens/s. The engine aggregate is a cross-check, never the numerator. Deleting the slow request from both proxy lists is rejected by the independent engine request count. Raw abort/error/compaction/retry requests remain in the inventory; completed requests alone enter the ratio.

Accepted F1 boot/stage and F2 pinned-dataset/official-TestSpec behavior is not reopened. The closure successor adds an explicit pre-agent setup-failure path: verified owned cleanup permits INFRASTRUCTURE_FAILED with unknown official task result; unverified cleanup retains INCOMPLETE_CLEANUP_UNVERIFIED. It does not invent agent execution, patch, evaluator completion or success. The remaining cancellation race below occurs before this cleanup evidence becomes sufficient.

## Remaining concrete blocker R2: launcher cancellation can outlive terminal cleanup

`workload-plan/tools/attempt-runtime/boot_v3_2.py:65–81` catches `subprocess.TimeoutExpired`, but not other exceptions from `proc.wait`. For example, KeyboardInterrupt escapes without terminating or waiting for the launcher process group. `boot_server` wraps it as BootFailure (`325–330`); `fail_boot` can subsequently observe an empty attempt-label query and report verified `no_owned_container` (`371–382`). That empty query is not durable absence while the launcher remains able to create a container.

**Exact injected reproduction:** the real `SubprocessRunner` with only Popen/killpg replaced by mocks receives `KeyboardInterrupt` from its wait. It calls `killpg` **zero** times and waits once. Through the actual v3.2 caller/harness, the result is:

```text
state=BOOT_FAILED_CLOSED
terminal=True
cleanup=no_owned_container
killpg_calls=0
preflight(2).next_ordinal=2
```

No real child process was launched for this control. The unobserved real-world launcher could still create the server after that empty query, overlapping the next arm. Timeout escalation has a related omission: after SIGKILL the code neither waits again nor records confirmed process termination.

**Minimal repair/admission criteria:** after Popen succeeds, all abnormal exit/cancellation paths must terminate the owned launcher process group and reap/observe the launcher before treating label absence as final. Retain the actual PID/group identity, exception/timeout, signals and wait result alongside partial stdout/stderr. A failed termination or unobserved launcher/group quiescence must retain an incomplete state and refuse the next arm, even if the instantaneous container query is empty. After SIGKILL, confirm termination; do not infer it merely from sending the signal. Re-raise cancellation after retaining the bounded cleanup result. Keep exact-CID server cleanup and full ownership checks; never kill by name or alter unrelated containers. No retry or additional workload attempt is needed to implement/test this with injected processes.

## Remaining concrete blocker R4: artifact bytes can be relabeled as another attempt/arm

The reducer is now called by `sole_executor_v3_2.measure` (`209–231`), and it retains raw bytes, population and reduction. This closes the prior disconnected-library issue. However, its artifact paths still come directly from CLI options (`373–396`). `instrumentation_v3_1.measure` reads these files and assigns the supplied current `attempt_id`/`arm` to the new bracket and result (`206–239`); there is no freeze-bound capture ownership, boot/server identity, or producer-capture receipt proving that those files were generated for that attempt. No actual pre/post `/metrics` capture occurs in this caller. Hashing files after reading them preserves bytes but does not establish their origin or interval.

**Exact CPU reproduction:** one unchanged set of four fixture raw files was passed to the actual instrumentation once as `attempt-A/AR` and once as `attempt-B/LUMOTREE`. Both returned **REDUCED=14.0**, with identical raw hashes. Their request/count populations match internally, but belong to neither independently established attempt boundary. This is the narrow remaining source/population binding gap, not a request to redesign the reducer.

**Minimal repair:** enforce a fresh attempt-owned capture directory and bind its path/proxy producer to the frozen attempt/config/boot; retain owned-server pre/post metric observations at the actual before-agent/after-quiescence boundaries. The caller must invoke those collectors and pass their authenticated receipt to reduction, rather than merely accepting file paths and `--proxy-stopped`. Require the actual pinned capture producer hash and server identity; refuse a copied prior capture or another attempt's bracket. Retain raw bytes and all-request equality checks unchanged. Missing observations remain NOT_REDUCED, without dropping the attempt or fabricating a rate.

## Seed-tamper failure: refusal works; exception classes differ

Independently reran `test_seed_probe_raw_tamper_refuses` through the v3.1 closure-successor fixture. The changed raw seed is rejected by `contracts.validate_runtime` at `runtime-collectors/contracts.py:76`:

```text
accepted_e3_adapter.Refusal:
  delivered seed vs independent frozen expectation differs from bound identity
```

`contracts.py:9` imports the accepted collector's adapter, while the successor test expects `successor_e3_adapter_v3.Refusal`. These are different classes, so unittest reports an error instead of recognizing the expected refusal. Both derive from ValueError, and the caller's error handling includes ValueError. **No false pass/integrity bypass was observed.** Preserve the failed log. A narrowly reviewed successor exception normalization or an explicit test for both bound refusal classes can make the test meaningful; do not silently relabel it passed or change accepted bundles. This probe check remains distinct from the prospectively declared absence of seeds in workload requests.

## Recommendations on the seven preparation decisions

These are source-grounded preparation recommendations; the parent owns the configuration freeze and WP.

1. **Agent bundle:** read and bind the actually mounted qwen-code bundle on the evaluator host; use that same digest for all four arms. The image's 0.19.6 compaction thresholds do not establish the mounted bundle's behavior. Preserve the common limits and obtain the thresholds from that actual bundle.
2. **Per-event usage:** proceed with a minimal versioned common upstream `continuous_usage_stats=true` option, while preserving agent-visible final-usage semantics. SGLang's copied `StreamOptions` and `should_include_usage` support the field; copied `serving_chat.py:1503–1517` uses engine `meta_info.completion_tokens`, and `710–792,2413–2472` attaches cumulative usage to emitted chunks. No source evidence here opposes this option. The corresponding pinned vLLM path must remain bound in the implementation receipt. One engine update can split into multiple parser/tool SSE events carrying the same count: repeated cumulative values are legitimate, **SSE event count is not generated token count**. A final token-bearing event with no usage leaves the endpoint undefined; never fill it from a later usage-only event or assume one token per event. This is a telemetry/serialization change, not a model-kernel change, but record it as a shared serving configuration.
3. **SGLang KV precision:** use explicit bf16 KV for the comparable configuration, retaining the same bound target weights/tokenizer. This aligns the numerical precision with vLLM; it is not a lower-precision shortcut. It changes the historical as-shipped SGLang FP8-KV route, so that historical qualification/result cannot be silently inherited. Record the resolved dtype under the existing route qualification.
4. **Prompt parity:** adopt the versioned common text-array newline flattening and explicit tool-object template already CPU-tested in `PROMPT-TOKEN-PARITY-v1.1.json`. All six tested vLLM streams remain identical and SGLang aligns to them. Pin both new source/template hashes and disclose that equivalence covers those captured requests, not every possible message. For identical input IDs this changes no accepted vLLM numerical input; new template identity still needs an explicit successor binding.
5. **Cache/FA2:** for same-stack AR/CHAIN_MTP comparisons, prefer the same pinned FA2 artifact and a common supported APC policy, preserving the already-qualified AR/LumoTree deployment as the reference. The current records expressly say AR uses patched FA2 + APC align while CHAIN_MTP uses stock FA2 + APC off; do not call these a controlled mechanism-only comparison. Confirm support in the launcher source before preparing the common config; changing CHAIN_MTP's binary/cache policy requires its existing route qualification to bind the new choice. SGLang may retain its source-declared engine cache implementation. This review chooses no new qualification experiment or resource value.
6. **LumoTree launcher:** derive the owned wrapper from the candidate route the parent actually accepts, preserving its numerical source/env/image contract and **0.7** memory-utilization binding. Add ownership/cidfile, tokenizer/template and shared instrumentation explicitly; do not substitute an old launcher default or an unqualified attention path.
7. **Never-started agent cleanup:** add exact-CID removal after proving the agent's full attempt ownership and created/never-started state; retain the observation/cleanup receipt. This is resource cleanup, not agent execution or a new attempt. Unverified ownership must remain incomplete, never trigger broad/name-based deletion.

The present event-aligned rate is `sum(last cumulative count − first cumulative count) / sum(last token-event time − first token-event time)`. It excludes the first event batch and any untimed tail. Report that actual definition; do not mix it with historical aggregate-token/subset-wall proxies or silently treat it as an unchanged rate instrument.

Only the two reproduced R2/R4 blockers require another narrow caller successor review. Actual serving telemetry, clock transport, final route configuration and deployment observations remain unobserved preparation requirements. No gate was opened and no workload/evaluator attempt occurred during this review.
