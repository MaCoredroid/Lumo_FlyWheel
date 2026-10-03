# Probe boundary and raw evidence reader — bounded source review

**Verdict: PASS for the bounded source review after the counter-coverage repair.** One concrete false pass was reproduced and is closed in the reviewed successor. No further blocking defect was found in this pass. This is not a frozen recipe, complete launcher qualification, or workload gate approval. No HTTP, Docker/container, SSH, GPU, engine, agent, or evaluator operation was performed.

## Reviewed bytes

Paths below are relative to `experiments/review-response-20260927/workload-plan/tools/`.

| File | Reviewed SHA256 |
| --- | --- |
| `attempt-runtime/probe_boundary_v1.py` | `5d5825a0da31e6ebce17b78f4054949ecc85d1075fac4865f4929ec43aca29f9` |
| `runtime-collectors/runtime_evidence_reader_v1.py` | `0d0a286b6eeb7136cf45994ce562ad0b6dfaa479628dc9491ee223d174cef213` |
| `runtime-collectors/contracts_v2.py` | `b6c623d1afc641f0ae738d0fea876b9e4f2324dc59d087cb54b41e0a05e44393` |
| `runtime-collectors/collectors_v3.py` — caller context | `2c70218d2947b5b3ee212d6eabacfbf2ed187d37f1c86c5f6dbc69b908c7e3ed` |

Original drafts, final source copies, controls and raw logs are retained under `p0/monitor/review-response-20260927/probe-boundary-reader-independent-20260928T235838Z/`; repaired-source evidence is in `repair1/`. Dependency hashes and the source-stability check are in `repair1/DEPENDENCIES.json`. Existing bootstrap v2, request bundle and producer v1.2 reviews remain accepted; this review does not reopen their unchanged logic.

## F1 — incomplete counter coverage falsely certified equality; repaired

The preserved initial boundary (`1c6f59fd2573df9157f8c33750fe21b7809fcc4b4454ddc6ea3e0598d80dea74`) selected whichever recognized counter series existed, required only a nonempty equal map, then certified `request_and_token_counters_equal: true`. Two snapshots containing only `vllm:request_success_total 1` passed without either token counter. Two snapshots containing only `vllm:generation_tokens_total 32` also passed without request or prompt counters. The first case was compatible with the producer's completed-request accounting requirement; token observation was missing.

The repaired boundary at lines 186–187 and repeated contract check at lines 111–112 require completion, prompt-token and generation-token families in **both** snapshots. Equality also preserves every selected series' labels and values. Both original counterexamples now refuse. Complete equal vLLM and SGLang examples pass; a generation increment, changed label population, and empty map refuse.

The SGLang names are source-supported by the retained current-image `workload-plan/inspections/codex-counter-source-20260929T0007Z/metrics_collector.py`, SHA256 `04662d1f5c6723298b76d8333f496a997b40f61a12b577d46f226648114eced6`: prompt/generation definitions at lines 1513–1524, request count at 1598 onward, and increments at 1756–1759 and 1786. The `is_streaming` labels remain part of the compared series, rather than being discarded.

## Source and trust-boundary conclusions

- The boundary validates a parent-frozen source/policy recipe, derives attempt/boot/request identities from the reservation and independent pre-agent seed map, and writes helpers, policy and preparation records once. These are declarations, not observations. The one seeded probe remains distinct from the declared-absent seed policy for measured requests.
- Actual Docker bind destinations, source paths, read/write modes and required startup environment are checked before the probe. The actual bootstrap/request installation records must come from the allocated target PID and the mapped API PID, with the matching attempt, boot and policy/source hashes. These records complement the already reviewed source-only bootstrap policy; a mount specification alone does not prove that a wrapper executed.
- `run_probe` refuses measured-phase markers, requires the exact prepared directory/policy, calls the reviewed producer once, requires all observations, rehashes retained producer output, and retains the boundary in the packet. The producer's existing output-directory claim and failure retention enforce the no-retry boundary. The draft connected caller places this before metrics-pre and agent creation.
- The reader retains the source index and every successful indexed raw body, checks hashes/lengths and field-source membership, and compares prospective policies to the independently derived caller recipe. Allocation/request bundle identities, pre/post allocation stability, original-record bindings, target metadata projection, named request and effective sampling/seed projection are checked. It consumes the approved producer's live join; it is not an independent re-execution of Docker/process ownership observation.
- The repaired boundary retains both metric bodies and its exclusion receipt by hash. `contracts_v2.verify_probe_baseline` repeats the raw coverage/equality checks, binds the probe runtime and source index, and checks ordering before the agent gate. The connected source distinguishes coordinator producer bytes from engine source files. No requested or configured precision is inserted into the allocated worker observations by this reader.

## CPU evidence and remaining boundary

The final **15 independent controls pass**: complete vLLM and source-named SGLang baselines, missing-family and changed-series refusals, retained-baseline contract acceptance/raw-tamper refusal, plus a reader positive path built through the actual allocation/request bundle functions and negative cases for raw member/index mutation, missing raw evidence, independent policy mismatch and projected seed disagreement. Docker ownership was injected, records were synthetic, and the contract's approved-source-role check was a fixture. No live engine or request was used.

The initial 11-control run preserves both deliberate false-pass reproductions. An earlier test-fixture attempt lacked the packet store and newly required source-index schema; its script/output are retained as `*.attempt1`. Correcting those fixtures did not modify implementation. Final source hashes were rechecked against disk.

A future frozen recipe must still bind the actual engine/post-patch paths, helper/source maps, layer/route identities, independent probe seed entries, and declared policy. The launcher must establish private empty read-only cache/source mounts before Python starts and preserve per-role installation evidence. Successful live probe/raw accounting and subsequent admission remain unobserved here. No additional task, probe repetition, seed choice, serving setting, or scientific gate change is proposed.
