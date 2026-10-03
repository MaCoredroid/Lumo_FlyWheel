# Hidden publication v8 connection — initial independent review

Disposition: **two connected-path blockers**, limited to the submitted hooks `8db9041940802723cffeadca60e0d72c6752a81c04742d677568300d68231621` and patcher `c0d1cf74cd6e443cb433d5add0ac6447b911b26a28351e3e57a4934e83d5591b`. Parent received both findings and owns repairs. This note does not evaluate later moving edits, approve a launch or reopen accepted hidden helper internals.

## F1 — seal expects a field absent from the production census

`q1_candidate_hooks_v8.py:656–658` reads top-level `event['request_ids']`. Generated GDN `_fr13_fixed32_observed_build_record` at8226–8454 computes request hashes and emits `fr13-fixed32-work-census-v12`; its record at8338–8372 contains `drafter_runtime`, not raw top-level request IDs. Consequently every actual event with that schema raises `KeyError('request_ids')` at the new hidden seal.

Reproduction executes the exact pinned record-constructor AST with synthetic section values and the actual submitted `_hidden_event_sealed` AST. The resulting source-shaped census lacks the requested field and the actual seal fails as described. No engine/model is involved.

Small repair: compare case request identity using the production canonical SHA256 of `json.dumps([request_id], ensure_ascii=True, separators=(',', ':')).encode('ascii')` and the per-request UTF8 SHA256 list inside `event['drafter_runtime']`. Keep mode/batch/forward-step/event-index matching and validate actual census schema, `event_complete` and current `producer_pid`. Preserve the actual census structure; do not insert invented raw request IDs into production evidence. Build the retained expected owner from the live binding/case and observed event fields.

## F2 — original-operation and seal exceptions bypass hidden fail-stop retention

The patcher adds callbacks around the original Eagle operations but no failure boundary enclosing them. At actual Eagle832, if `self.model(**model_kwargs)` raises after a successful before-capture, no after callback or hidden exception handler executes. The same gap applies to original input/copy/gather or intervening metadata failures within the admitted proposal. The partial bridge remains in memory, with no new-hook path to persist it through `_mark_unusable`/case sealing.

The CPU control executes the actual patched first-forward statement block with a model stub that raises. The before callback executes and retains a partial marker; the original error propagates without any failure callback. This establishes the missing connection, not a claim that an actual model failure occurred.

Separately, `_hidden_event_sealed` runs inside `on_sealed:680`; its exception is caught by the generic existing branch at693–695. The submitted S4 code seals INVALID and returns, without latching or re-raising `ProcessUnusable`. A control using F1's source-shaped event confirms `case.sealed=True`, an invalid reason, and `unusable=None` after normal return. That is inconsistent with the accepted hidden collector's fail-stop contract.

Small repair: add an exception/context boundary around the actual admitted Runner→`drafter.propose` call, leaving the original production operations and arguments intact. On any escape, retain the same-case partial record, latch unusable, seal best-effort and re-raise; warmup/no-active-case should keep its original behavior. Cover hidden seal failures with the same latch and explicit `ProcessUnusable` propagation so the enclosing generic S4 catch cannot swallow it. Do not retry or synthesize missing after-evidence. Preserve the original error chain.

## Other bounded checks

The new source anchors otherwise match the intended source map: actual Runner target slice before8065, post-TSR copy797–807, first actual model kwargs/return832–838, sampling gather1011 and continuation gather1031. The original continuation source is retained inside the accepted bridge before Python rebinds `hidden_states`. Forward-after runs before MTP KV remap; gathers run after payload clear and completion flags, matching the accepted stage-aware owner.

Warmup/disabled/no-case/sealed/prefill behavior passed28 independent public-hook no-op calls using dummy arguments that would fail if dereferenced. Active duplicate/foreign handling delegates to the accepted owner and bridge; it was not re-audited here.

Patcher prevalidates mandatory hashes and anchors for all three files and compiles all generated outputs before writes. Parent's retained attempt1 receipt reports all21 anchors and compile success; the independent exception control additionally applies/checks Eagle edits in memory and parses the result. No generated source was installed or executed in vLLM.

## Evidence and next bounded check

Artifacts: `p0/monitor/review-response-20260927/hidden-v8-connection-independent-20260929/`. Exact submitted sources, executable `initial-controls.py`, `INITIAL-RESULTS.json`, source hash bindings and manifest are retained. Initial controls use stdlib AST extraction, synthetic census fields and stubs; no Torch, GPU, model, container, remote mutation or gate was used.

After parent repair, the necessary follow-up is narrow: source-shaped completed census positive plus wrong-request/PID/incomplete negatives; original forward/copy/gather escape retains partial evidence and latches/rethrows; hidden seal failures also latch/rethrow; positive path and28 no-op controls remain valid. No new scientific criterion, cohort or experiment is requested.
